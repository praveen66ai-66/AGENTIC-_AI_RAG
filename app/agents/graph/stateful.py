"""
Stateful graph runner — single entry point for the full agent pipeline.

Write order after every request:
  Redis  working    → session.init_meta, bump_query_count, append_message (×2)
  PG     registry   → store.upsert_session
  PG     episodic   → store.log_turn (user turn + assistant turn)
  PG     semantic   → store.store_semantic_fact (topic extracted from question)
  PG     audit      → store.log_query (full pipeline trace)
  Redis  sem-cache  → semantic_cache.store (answer cache for future reuse)

Idempotency
  All Redis writes use msg_id / request_id SETNX guards.
  All PostgreSQL writes use ON CONFLICT (idempotency_key) DO NOTHING.
  The idempotency_key for the full request is stored in Redis for 5 min —
  a retried request returns the stored result without re-running the pipeline.

Timestamps
  started_at captured at function entry; duration_ms computed at the end
  and written to both the result dict and the PG audit log.
  PostgreSQL created_at columns use the DB clock (DEFAULT NOW()).
"""

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone

from app.agents.graph.builder import rag_graph
from app.memory import semantic_cache
from app.memory import session as session_mem
from app.memory import store
from app.observability import splunk

_IDEM_TTL = 300   # seconds — covers any realistic client retry window


def run(
    question:        str,
    session_id:      str | None = None,
    tenant_id:       str        = "default",
    user_id:         str        = "default",
    idempotency_key: str | None = None,
) -> dict:
    """
    Invoke the full planner → retriever(+rerank) → reasoner → generator pipeline.

    Parameters
    ----------
    question         : The user's finance question.
    session_id       : Reuse across turns to maintain MemorySaver + Redis context.
    tenant_id        : Tenant namespace — isolates all Redis keys and PG rows.
    user_id          : User within the tenant.
    idempotency_key  : Per-request UUID. Retries with the same key within 5 min
                       return the stored result without re-running the pipeline.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())

    trajectory_id = str(uuid.uuid4())
    prefix        = session_mem.make_prefix(tenant_id, user_id, session_id)
    r             = session_mem._get_client()
    idem_redis    = f"{prefix}:idem:{idempotency_key}"
    started_at    = time.time()

    # ── 1. Idempotency replay check (Redis) ───────────────────────────────────
    stored = r.get(idem_redis)
    if stored:
        result = json.loads(stored)
        result["idempotency_replay"] = True
        return result

    # ── 2. Init working memory + session registry ─────────────────────────────
    session_mem.init_meta(prefix, tenant_id, user_id, session_id)
    session_mem.bump_query_count(prefix, request_id=idempotency_key)
    store.upsert_session(tenant_id, user_id, session_id)

    # ── 3. Semantic cache check ───────────────────────────────────────────────
    cached = semantic_cache.check(tenant_id, user_id, question)
    if cached:
        answer     = cached["answer"]
        confidence = cached["confidence"]
        cache_type = cached["cache_type"]
        duration   = round((time.time() - started_at) * 1000, 2)

        # Working memory (Redis)
        session_mem.append_message(prefix, "user",      question, msg_id=f"{idempotency_key}:user")
        session_mem.append_message(prefix, "assistant", answer,   msg_id=f"{idempotency_key}:asst")

        # Episodic memory (PG)
        store.log_turn(
            tenant_id=tenant_id, user_id=user_id, session_id=session_id,
            trajectory_id=trajectory_id, role="user", content=question,
            idempotency_key=f"{idempotency_key}:user",
            cache_hit=True, cache_type=cache_type,
        )
        store.log_turn(
            tenant_id=tenant_id, user_id=user_id, session_id=session_id,
            trajectory_id=trajectory_id, role="assistant", content=answer,
            idempotency_key=f"{idempotency_key}:asst",
            confidence=confidence, cache_hit=True, cache_type=cache_type,
        )

        # Audit log (PG)
        store.log_query(
            tenant_id=tenant_id, user_id=user_id, session_id=session_id,
            trajectory_id=trajectory_id, query=question, retrieved="",
            idempotency_key=f"{idempotency_key}:query",
            answer=answer, duration_ms=duration, confidence=confidence,
            cache_hit=True, cache_type=cache_type,
        )

        result = {
            "answer":             answer,
            "sources":            [],
            "confidence":         confidence,
            "plan":               [],
            "iteration_count":    0,
            "session_id":         session_id,
            "trajectory_id":      trajectory_id,
            "cache_hit":          True,
            "cache_type":         cache_type,
            "similarity":         cached.get("similarity", 1.0),
            "idempotency_replay": False,
            "created_at":         datetime.now(timezone.utc).isoformat(),
            "duration_ms":        duration,
        }
        splunk.rag_query(
            tenant_id=tenant_id, user_id=user_id,
            session_id=session_id, trajectory_id=trajectory_id,
            question_length=len(question), answer_length=len(answer),
            confidence=confidence, cache_hit=True, cache_type=cache_type,
            iteration_count=0, duration_ms=duration, source_count=0,
        )
        _store_idem(r, idem_redis, result)
        return result

    # ── 4. Full agent pipeline ────────────────────────────────────────────────
    initial_state = {
        "messages":           [],
        "question":           question,
        "session_id":         prefix,
        "plan":               [],
        "retrieved_chunks":   [],
        "context_sufficient": False,
        "retrieval_gap":      "",
        "answer":             "",
        "sources":            [],
        "confidence":         0.0,
        "iteration_count":    0,
        "trajectory_id":      trajectory_id,
        "retrieval_empty":    False,
    }

    config      = {"configurable": {"thread_id": session_id}}
    final_state = rag_graph.invoke(initial_state, config=config)

    # Persist retrieval gap so the planner can address it on the next question
    last_gap = final_state.get("retrieval_gap", "")
    if last_gap:
        session_mem.set_last_gap(prefix, last_gap)

    answer          = final_state.get("answer", "")
    confidence      = final_state.get("confidence", 0.5)
    plan            = final_state.get("plan", [])
    iteration_count = final_state.get("iteration_count", 0)
    sources         = final_state.get("sources", [])
    duration        = round((time.time() - started_at) * 1000, 2)

    citation_summary = ", ".join(
        f"p{s.get('page')} {s.get('section','')}" for s in sources[:3]
    )

    # ── 5. Episodic memory — both turns ──────────────────────────────────────
    store.log_turn(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id,
        trajectory_id=trajectory_id, role="user", content=question,
        idempotency_key=f"{idempotency_key}:user",
    )
    store.log_turn(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id,
        trajectory_id=trajectory_id, role="assistant", content=answer,
        idempotency_key=f"{idempotency_key}:asst",
        sources=sources, confidence=confidence,
    )

    # ── 6. Semantic memory — topic extracted from question ────────────────────
    topic   = _extract_topic(question)
    fact_id = hashlib.md5(f"{tenant_id}:{user_id}:{topic}".encode()).hexdigest()
    store.store_semantic_fact(
        tenant_id=tenant_id, user_id=user_id,
        fact_type="topic", subject=topic,
        content=question,
        source_session=session_id,
        confidence=confidence,
        idempotency_key=f"topic:{fact_id}",
    )

    # ── 7. Audit log ──────────────────────────────────────────────────────────
    store.log_query(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id,
        trajectory_id=trajectory_id, query=question,
        idempotency_key=f"{idempotency_key}:query",
        plan=plan, retrieved=citation_summary, answer=answer,
        iteration_count=iteration_count, duration_ms=duration,
        confidence=confidence, cache_hit=False,
    )

    # ── 8. Semantic answer cache — skip LLM next time ─────────────────────────
    if answer and confidence >= 0.4:
        semantic_cache.store(tenant_id, user_id, question, answer, confidence)

    result = {
        "answer":             answer,
        "sources":            sources,
        "confidence":         confidence,
        "plan":               plan,
        "iteration_count":    iteration_count,
        "session_id":         session_id,
        "trajectory_id":      trajectory_id,
        "cache_hit":          False,
        "cache_type":         None,
        "similarity":         None,
        "idempotency_replay": False,
        "created_at":         datetime.now(timezone.utc).isoformat(),
        "duration_ms":        duration,
    }
    splunk.rag_query(
        tenant_id=tenant_id, user_id=user_id,
        session_id=session_id, trajectory_id=trajectory_id,
        question_length=len(question), answer_length=len(answer),
        confidence=confidence, cache_hit=False, cache_type=None,
        iteration_count=iteration_count, duration_ms=duration,
        source_count=len(sources),
    )
    _store_idem(r, idem_redis, result)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _store_idem(r, key: str, result: dict) -> None:
    """Store result in Redis with TTL — never surfaces failure to caller."""
    try:
        r.setex(key, _IDEM_TTL, json.dumps(result, default=str))
    except Exception:
        pass


def _extract_topic(question: str) -> str:
    """
    Simple topic normalisation: lowercase + first 60 chars.
    Replace with LLM extraction when budget allows.
    """
    return " ".join(question.lower().split())[:60].strip()
