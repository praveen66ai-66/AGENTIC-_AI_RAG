"""POST /api/v1/agent — full LangGraph pipeline: planner→retriever→reasoner→generator.
   POST /api/v1/agent/stream — same pipeline with token-by-token SSE streaming.
"""

import json
import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

from app.models.schemas import AgentRequest, AgentResponse, AgentSource
from app.api.middleware.context import RequestContext, get_context
from app.guardrails.input_guard import check as guard_check
from app.agents.graph.stateful import run, stream as agent_stream
from app.observability import splunk

router = APIRouter()


@router.post("/agent", response_model=AgentResponse)
def agent_query(req: AgentRequest, ctx: RequestContext = Depends(get_context)):
    # Input guardrail — same as /query
    guard = guard_check(req.question)
    if guard.blocked:
        splunk.security_event(
            event_type="input_guard_block",
            guard_type="input",
            reason=guard.reason,
            tenant_id=ctx.tenant_id,
            session_id=ctx.session_id,
            question_length=len(req.question),
        )
        raise HTTPException(status_code=400, detail=guard.reason)

    try:
        result = run(
            question=req.question,
            session_id=req.session_id or ctx.session_id,
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            idempotency_key=str(uuid.uuid4()),
        )
    except Exception as exc:
        logger.exception("Agent pipeline error")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    return AgentResponse(
        answer=result["answer"],
        sources=[AgentSource(**s) for s in result["sources"]],
        confidence=result["confidence"],
        plan=result["plan"],
        iteration_count=result["iteration_count"],
        session_id=result["session_id"],
        trajectory_id=result["trajectory_id"],
        cache_hit=result["cache_hit"],
        cache_type=result["cache_type"],
        similarity=result["similarity"],
        idempotency_replay=result["idempotency_replay"],
        created_at=result["created_at"],
        duration_ms=result["duration_ms"],
    )


@router.post("/agent/stream")
def agent_stream_endpoint(req: AgentRequest, ctx: RequestContext = Depends(get_context)):
    """
    SSE streaming endpoint. Yields newline-delimited JSON events:
      {"type": "metadata", "plan": [...], "confidence": float, "cache_hit": bool}
      {"type": "token",    "content": str}   ← one per LLM token
      {"type": "done",     "sources": [...], "confidence": float, "duration_ms": float}
    """
    guard = guard_check(req.question)
    if guard.blocked:
        splunk.security_event(
            event_type="input_guard_block", guard_type="input",
            reason=guard.reason, tenant_id=ctx.tenant_id,
            session_id=ctx.session_id, question_length=len(req.question),
        )
        raise HTTPException(status_code=400, detail=guard.reason)

    def generate():
        try:
            for event in agent_stream(
                question=req.question,
                session_id=req.session_id or ctx.session_id,
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                idempotency_key=str(uuid.uuid4()),
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.exception("Stream error")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
