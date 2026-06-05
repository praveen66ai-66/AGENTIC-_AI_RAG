"""POST /api/v1/query — retrieval endpoint with citations and guardrails."""

import logging
import time
import uuid
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

UAE_TZ = timezone(timedelta(hours=4))  # Gulf Standard Time, UTC+4
from fastapi import APIRouter, Depends, HTTPException, Request

from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunk, Citation
from app.api.middleware.context import RequestContext, get_context
from app.memory import session as session_mem
from app.memory import store as pg_store
from app.guardrails.input_guard import check as guard_check
from app.observability import splunk
from app.retrieval.search import hybrid_search

router = APIRouter()

_SCORE_THRESHOLD = 0.4   # minimum RRF score to include a chunk in the response


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, ctx: RequestContext = Depends(get_context)):
    started_at   = time.time()
    request_id   = str(uuid.uuid4())   # stable for this request — used for idempotency + tracing

    # ── Guardrail check ───────────────────────────────────────────────────────
    # Blocked queries stop here — Qdrant, Redis, and PostgreSQL are never touched.
    result = guard_check(req.question)
    if result.blocked:
        splunk.security_event(
            event_type="input_guard_block",
            guard_type="input",
            reason=result.reason,
            tenant_id=ctx.tenant_id,
            session_id=ctx.session_id,
            question_length=len(req.question),
        )
        raise HTTPException(status_code=400, detail=result.reason)

    # ── Hybrid retrieval (RRF fusion) ─────────────────────────────────────────
    try:
        raw_content   = hybrid_search(req.question, "finance_content",   top_k=req.top_k)
        raw_structure = hybrid_search(req.question, "finance_structure", top_k=2)
    except Exception as exc:
        logger.exception("Qdrant retrieval failed")
        splunk.node_step(
            node="query_route", phase="qdrant_error",
            session_id=ctx.session_id,
            duration_ms=round((time.time() - started_at) * 1000, 2),
            error=str(exc),
        )
        raise HTTPException(status_code=503, detail="Retrieval service unavailable. Please try again.")

    # Deduplicate and assign citation numbers
    seen:      set[str]            = set()
    chunks:    list[RetrievedChunk] = []
    citations: list[Citation]      = []
    num = 1

    for r in raw_content + raw_structure:
        text = r.get("text", "")
        if text in seen:
            continue
        seen.add(text)

        chunks.append(RetrievedChunk(
            citation_number=num,
            text=text,
            score=r.get("score", 0.0),
            page=r.get("page"),
            section=r.get("section", ""),
            source=r.get("source", ""),
            collection=r.get("collection", ""),
        ))
        citations.append(Citation(
            number=num,
            page=r.get("page"),
            section=r.get("section", ""),
            source=r.get("source", ""),
            collection=r.get("collection", ""),
        ))
        num += 1

    # Drop low-confidence chunks — always keep at least 1
    if chunks:
        above = [c for c in chunks if c.score >= _SCORE_THRESHOLD]
        if above:
            kept_nums = {c.citation_number for c in above}
            chunks    = above
            citations = [c for c in citations if c.number in kept_nums]

    # ── Short-term memory (Redis) ─────────────────────────────────────────────
    session_mem.append_message(ctx.redis_prefix, "user", req.question)

    # ── Long-term log (PostgreSQL) ────────────────────────────────────────────
    citation_summary = ", ".join(
        f"[{c.number}] p{c.page} {c.section}" for c in citations[:3]
    )
    duration_ms = round((time.time() - started_at) * 1000, 2)
    pg_store.log_query(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        query=req.question,
        retrieved=citation_summary,
        idempotency_key=request_id,
        duration_ms=duration_ms,
    )
    splunk.retrieval_query(
        tenant_id=ctx.tenant_id,
        session_id=ctx.session_id,
        question_length=len(req.question),
        chunk_count=len(chunks),
        duration_ms=duration_ms,
    )

    return QueryResponse(
        question=req.question,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        chunks=chunks,
        citations=citations,
        answer=None,
        llm_ready=False,
        created_at=datetime.now(UAE_TZ).isoformat(),
        duration_ms=duration_ms,
    )
