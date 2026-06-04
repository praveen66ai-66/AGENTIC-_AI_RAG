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
  All app-level timestamps use UAE time (UTC+4, Gulf Standard Time).
  started_at captured at function entry; duration_ms computed at the end.
  PostgreSQL created_at columns use the DB clock (DEFAULT NOW()).
"""

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone, timedelta

# UAE — Gulf Standard Time (UTC+4, no daylight saving)
UAE_TZ = timezone(timedelta(hours=4))

from app.agents.graph.builder import rag_graph, pre_graph
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
    _t = time.time()
    session_mem.init_meta(prefix, tenant_id, user_id, session_id)
    session_mem.bump_query_count(prefix, request_id=idempotency_key)
    _outer_timings = {"redis_session_init_ms": round((time.time() - _t) * 1000, 2)}

    _t = time.time()
    store.upsert_session(tenant_id, user_id, session_id)
    _outer_timings["pg_upsert_session_ms"] = round((time.time() - _t) * 1000, 2)

    # ── 3. Semantic cache check ───────────────────────────────────────────────
    _t = time.time()
    cached = semantic_cache.check(tenant_id, user_id, question)
    _outer_timings["redis_cache_check_ms"] = round((time.time() - _t) * 1000, 2)
    if cached:
        answer      = cached["answer"]
        confidence  = cached["confidence"]
        cache_type  = cached["cache_type"]
        cached_srcs = cached.get("sources", [])
        duration    = round((time.time() - started_at) * 1000, 2)

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
            "sources":            cached_srcs,
            "confidence":         confidence,
            "plan":               [],
            "iteration_count":    0,
            "session_id":         session_id,
            "trajectory_id":      trajectory_id,
            "cache_hit":          True,
            "cache_type":         cache_type,
            "similarity":         cached.get("similarity", 1.0),
            "idempotency_replay": False,
            "created_at":         datetime.now(UAE_TZ).isoformat(),
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
        "stage_tokens":       {},
        "node_timings":       {},
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
    stage_tokens    = final_state.get("stage_tokens", {})
    node_timings    = final_state.get("node_timings", {})
    duration        = round((time.time() - started_at) * 1000, 2)

    total_tokens_used = sum(v.get("total", 0) for v in stage_tokens.values() if isinstance(v, dict))
    citation_summary  = ", ".join(f"p{s.get('page')} {s.get('section','')}" for s in sources[:3])

    # ── 5. Episodic memory — both turns ──────────────────────────────────────
    _t = time.time()
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
    _outer_timings["pg_log_turns_ms"] = round((time.time() - _t) * 1000, 2)

    # ── 6. Semantic memory ────────────────────────────────────────────────────
    topic   = _extract_topic(question)
    fact_id = hashlib.md5(f"{tenant_id}:{user_id}:{topic}".encode()).hexdigest()
    store.store_semantic_fact(
        tenant_id=tenant_id, user_id=user_id,
        fact_type="topic", subject=topic, content=question,
        source_session=session_id, confidence=confidence,
        idempotency_key=f"topic:{fact_id}",
    )

    # ── 7. Audit log ──────────────────────────────────────────────────────────
    _t = time.time()
    store.log_query(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id,
        trajectory_id=trajectory_id, query=question,
        idempotency_key=f"{idempotency_key}:query",
        plan=plan, retrieved=citation_summary, answer=answer,
        iteration_count=iteration_count, duration_ms=duration,
        confidence=confidence, cache_hit=False,
        token_usage=stage_tokens, total_tokens=total_tokens_used,
    )
    _outer_timings["pg_log_query_ms"] = round((time.time() - _t) * 1000, 2)

    # ── 8. Semantic cache store ───────────────────────────────────────────────
    _t = time.time()
    if answer and confidence >= 0.4:
        semantic_cache.store(tenant_id, user_id, question, answer, confidence, sources=sources)
    _outer_timings["redis_cache_store_ms"] = round((time.time() - _t) * 1000, 2)

    # Merge outer timings with per-node timings from the graph
    latency_trace = {"_pipeline": _outer_timings, **node_timings, "_total_ms": duration}

    # Single flat Splunk event — all sub-steps in one record for easy SPL queries
    splunk.latency_summary(
        trajectory_id=trajectory_id,
        session_id=prefix,
        tenant_id=tenant_id,
        latency_trace=latency_trace,
    )

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
        "created_at":         datetime.now(UAE_TZ).isoformat(),
        "duration_ms":        duration,
        "token_usage":        stage_tokens,
        "total_tokens":       total_tokens_used,
        "latency_trace":      latency_trace,
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


def stream(
    question:        str,
    session_id:      str | None = None,
    tenant_id:       str        = "default",
    user_id:         str        = "default",
    idempotency_key: str | None = None,
):
    """
    Streaming variant of run().
    Yields dicts:
      {"type": "metadata", "plan": [...], "confidence": float, "cache_hit": bool}
      {"type": "token",    "content": str}   ← one per LLM token
      {"type": "done",     "sources": [...], "confidence": float, "duration_ms": float}
    """
    from app.agents.nodes.generator import stream_tokens
    from app.guardrails.output_guard import check as output_guard_check

    if not session_id:
        session_id = str(uuid.uuid4())
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())

    trajectory_id = str(uuid.uuid4())
    prefix        = session_mem.make_prefix(tenant_id, user_id, session_id)
    started_at    = time.time()

    session_mem.init_meta(prefix, tenant_id, user_id, session_id)
    session_mem.bump_query_count(prefix, request_id=idempotency_key)
    store.upsert_session(tenant_id, user_id, session_id)

    # ── Cache hit: stream cached answer word-by-word ──────────────────────────
    cached = semantic_cache.check(tenant_id, user_id, question)
    if cached:
        answer      = cached["answer"]
        confidence  = cached["confidence"]
        cache_type  = cached["cache_type"]
        cached_srcs = cached.get("sources", [])
        yield {"type": "metadata", "plan": [], "confidence": confidence,
               "cache_hit": True, "cache_type": cache_type}
        for word in answer.split():
            yield {"type": "token", "content": word + " "}
        duration = round((time.time() - started_at) * 1000, 2)
        try:
            session_mem.append_message(prefix, "user",      question, msg_id=f"{idempotency_key}:user")
            session_mem.append_message(prefix, "assistant", answer,   msg_id=f"{idempotency_key}:asst")
            store.log_turn(tenant_id=tenant_id, user_id=user_id, session_id=session_id,
                           trajectory_id=trajectory_id, role="user", content=question,
                           idempotency_key=f"{idempotency_key}:user", cache_hit=True, cache_type=cache_type)
            store.log_turn(tenant_id=tenant_id, user_id=user_id, session_id=session_id,
                           trajectory_id=trajectory_id, role="assistant", content=answer,
                           idempotency_key=f"{idempotency_key}:asst",
                           confidence=confidence, cache_hit=True, cache_type=cache_type)
        except Exception as _e:
            print(f"[stream] cache-hit memory write failed (non-fatal): {_e}")
        yield {"type": "done", "sources": cached_srcs, "confidence": confidence,
               "duration_ms": duration, "cache_hit": True}
        return

    # ── Phase 1: planner → retriever → reasoner (no streaming needed) ─────────
    initial_state = {
        "messages": [], "question": question, "session_id": prefix,
        "plan": [], "retrieved_chunks": [], "context_sufficient": False,
        "retrieval_gap": "", "answer": "", "sources": [], "confidence": 0.0,
        "iteration_count": 0, "trajectory_id": trajectory_id, "retrieval_empty": False,
    }
    config      = {"configurable": {"thread_id": f"{session_id}_stream"}}
    mid_state   = pre_graph.invoke(initial_state, config=config)

    plan       = mid_state.get("plan", [])
    confidence = mid_state.get("confidence", 0.0)

    # Persist last gap for next turn's planner
    last_gap = mid_state.get("retrieval_gap", "")
    if last_gap:
        session_mem.set_last_gap(prefix, last_gap)

    yield {"type": "metadata", "plan": plan, "confidence": confidence,
           "cache_hit": False, "cache_type": None}

    # ── Phase 2: stream generator tokens ─────────────────────────────────────
    from app.agents.nodes.generator import _strip_llm_scaffolding
    full_answer = ""
    for token in stream_tokens(mid_state):
        full_answer += token
        yield {"type": "token", "content": token}

    # Strip ## Sources / ## Confidence scaffolding the LLM may have output
    cleaned = _strip_llm_scaffolding(full_answer)
    if cleaned != full_answer:
        # Send a correction token that replaces the displayed text
        yield {"type": "replace", "content": cleaned}
        full_answer = cleaned

    # Output guardrail on completed answer
    chunks = mid_state.get("retrieved_chunks") or []
    guard  = output_guard_check(full_answer, chunks)
    if guard.blocked:
        caveat       = f"\n\n> **Quality note:** {guard.reason}"
        full_answer += caveat
        yield {"type": "token", "content": caveat}
        splunk.security_event(event_type="output_guard_block", guard_type="output",
                              reason=guard.reason, trajectory_id=trajectory_id,
                              session_id=prefix)

    sources = [
        {
            "page":      c.get("page"),
            "section":   c.get("section", ""),
            "source":    c.get("source", ""),
            "score":     c.get("rerank_score", c.get("score", 0.0)),
            "rrf_score": c.get("score", 0.0),
        }
        for c in chunks[:5]
    ]
    duration = round((time.time() - started_at) * 1000, 2)

    # ── Memory + audit writes ─────────────────────────────────────────────────
    # Each write is isolated in its own try/except so a single Redis timeout or
    # PG connection error cannot prevent the done event from reaching the UI.
    try:
        session_mem.append_message(prefix, "user",      question,    msg_id=f"{idempotency_key}:user")
        session_mem.append_message(prefix, "assistant", full_answer, msg_id=f"{idempotency_key}:asst")
    except Exception as _e:
        print(f"[stream] Redis append failed (non-fatal): {_e}")

    try:
        store.log_turn(tenant_id=tenant_id, user_id=user_id, session_id=session_id,
                       trajectory_id=trajectory_id, role="user", content=question,
                       idempotency_key=f"{idempotency_key}:user")
        store.log_turn(tenant_id=tenant_id, user_id=user_id, session_id=session_id,
                       trajectory_id=trajectory_id, role="assistant", content=full_answer,
                       idempotency_key=f"{idempotency_key}:asst",
                       sources=sources, confidence=confidence)
    except Exception as _e:
        print(f"[stream] PG log_turn failed (non-fatal): {_e}")

    try:
        topic   = _extract_topic(question)
        fact_id = hashlib.md5(f"{tenant_id}:{user_id}:{topic}".encode()).hexdigest()
        store.store_semantic_fact(tenant_id=tenant_id, user_id=user_id, fact_type="topic",
                                  subject=topic, content=question, source_session=session_id,
                                  confidence=confidence, idempotency_key=f"topic:{fact_id}")
    except Exception as _e:
        print(f"[stream] PG store_semantic_fact failed (non-fatal): {_e}")

    try:
        citation_summary = ", ".join(f"p{s.get('page')} {s.get('section','')}" for s in sources[:3])
        store.log_query(tenant_id=tenant_id, user_id=user_id, session_id=session_id,
                        trajectory_id=trajectory_id, query=question,
                        idempotency_key=f"{idempotency_key}:query",
                        plan=plan, retrieved=citation_summary, answer=full_answer,
                        iteration_count=mid_state.get("iteration_count", 0),
                        duration_ms=duration, confidence=confidence, cache_hit=False)
    except Exception as _e:
        print(f"[stream] PG log_query failed (non-fatal): {_e}")

    try:
        if full_answer and confidence >= 0.4:
            semantic_cache.store(tenant_id, user_id, question, full_answer, confidence, sources=sources)
    except Exception as _e:
        print(f"[stream] semantic_cache.store failed (non-fatal): {_e}")

    try:
        splunk.rag_query(tenant_id=tenant_id, user_id=user_id, session_id=session_id,
                         trajectory_id=trajectory_id, question_length=len(question),
                         answer_length=len(full_answer), confidence=confidence,
                         cache_hit=False, cache_type=None,
                         iteration_count=mid_state.get("iteration_count", 0),
                         duration_ms=duration, source_count=len(sources))
    except Exception as _e:
        print(f"[stream] Splunk rag_query failed (non-fatal): {_e}")

    total_tok    = sum(v.get("total", 0) for v in mid_state.get("stage_tokens", {}).values() if isinstance(v, dict))
    node_t       = mid_state.get("node_timings", {})
    lat_trace    = {"_pipeline": _outer_timings if "_outer_timings" in dir() else {}, **node_t, "_total_ms": duration}
    yield {"type": "done", "sources": sources, "confidence": confidence,
           "duration_ms": duration, "cache_hit": False,
           "token_usage":    mid_state.get("stage_tokens", {}),
           "total_tokens":   total_tok,
           "latency_trace":  lat_trace}


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
