import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._utils import format_chunks, format_history, get_llm, get_tokens, load_prompt
from app.agents.graph.state import AgentState
from app.guardrails.output_guard import check as output_guard_check
from app.memory import session as session_mem
from app.observability import splunk


def _split_prefix(prefix: str) -> tuple[str, str, str]:
    parts = prefix.split(":", 2)
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else ("", "", "")


def stream_tokens(state: dict):
    """
    Stream LLM tokens for the streaming API path.
    Yields raw string fragments as the LLM generates them.
    Does NOT write to memory — stateful.stream() handles that after collecting
    the full answer.
    """
    chunks             = state.get("retrieved_chunks") or []
    context_sufficient = state.get("context_sufficient", True)
    retrieval_gap      = state.get("retrieval_gap", "")
    sid                = state.get("session_id", "")

    if not chunks:
        yield (
            "I could not find relevant information in the textbook to answer this question. "
            "Please try rephrasing, or check that the topic is covered in the corporate finance material."
        )
        return

    prompt  = load_prompt("generator")
    history = session_mem.get_history(sid, *_split_prefix(sid)) if sid else []

    system_text = prompt["system"].format(
        retrieved_chunks=format_chunks(chunks),
        session_history=format_history(history),
    )

    for chunk in get_llm().stream([
        SystemMessage(content=system_text),
        HumanMessage(content=state["question"]),
    ]):
        yield chunk.content

    if not context_sufficient and retrieval_gap:
        yield f"\n\n> **Note:** The retrieved context may not fully cover this question. Missing: {retrieval_gap}"


def generator_node(state: AgentState) -> dict:
    t0            = time.time()
    trajectory_id = state.get("trajectory_id", "")
    sid           = state.get("session_id", "")
    iteration     = state.get("iteration_count", 0)

    splunk.node_step(node="generator", phase="enter", trajectory_id=trajectory_id,
                     session_id=sid, iteration_count=iteration)

    chunks             = state.get("retrieved_chunks") or []
    context_sufficient = state.get("context_sufficient", True)
    retrieval_gap      = state.get("retrieval_gap", "")
    guard_blocked      = False
    # Timing defaults (overwritten in else branch if LLM is called)
    _redis_history_ms = 0.0
    _llm_ms           = 0.0
    _guard_ms         = 0.0
    _redis_append_ms  = 0.0
    tokens            = {}

    # Premature abandonment guard: when retrieval returned nothing at all, skip
    # the LLM entirely — generating from an empty context produces hallucination.
    if not chunks:
        answer     = (
            "I could not find relevant information in the textbook to answer this question. "
            "Please try rephrasing, or check that the topic is covered in the corporate finance material."
        )
        confidence = 0.0
        sources    = []
    else:
        prompt = load_prompt("generator")

        _t = time.time()
        history = session_mem.get_history(sid, *_split_prefix(sid)) if sid else []
        _redis_history_ms = round((time.time() - _t) * 1000, 2)

        system_text = prompt["system"].format(
            retrieved_chunks=format_chunks(chunks),
            session_history=format_history(history),
        )

        _t = time.time()
        response = get_llm().invoke([
            SystemMessage(content=system_text),
            HumanMessage(content=state["question"]),
        ])
        _llm_ms = round((time.time() - _t) * 1000, 2)
        answer  = response.content.strip()
        tokens  = get_tokens(response)

        if not context_sufficient and retrieval_gap:
            answer += f"\n\n> **Note:** The retrieved context may not fully cover this question. Missing: {retrieval_gap}"

        _t = time.time()
        guard = output_guard_check(answer, chunks)
        _guard_ms = round((time.time() - _t) * 1000, 2)

        guard_blocked = guard.blocked
        if guard.blocked:
            answer += f"\n\n> **Quality note:** {guard.reason}"
            splunk.security_event(
                event_type="output_guard_block", guard_type="output",
                reason=guard.reason, trajectory_id=trajectory_id, session_id=sid,
            )

        raw_confidence = state.get("confidence", 0.5)
        confidence = min(raw_confidence, 0.4) if not context_sufficient else raw_confidence

        sources = [
            {"page": c.get("page"), "section": c.get("section", ""),
             "source": c.get("source", ""), "score": c.get("score", 0.0)}
            for c in chunks[:5]
        ]

    # Persist exchange
    _t = time.time()
    if sid:
        session_mem.append_message(sid, "user",      state["question"], msg_id=f"{trajectory_id}:user")
        session_mem.append_message(sid, "assistant", answer,            msg_id=f"{trajectory_id}:asst")
    _redis_append_ms = round((time.time() - _t) * 1000, 2)

    stage_tokens = dict(state.get("stage_tokens") or {})
    stage_tokens["generator"] = tokens

    _total_ms = round((time.time() - t0) * 1000, 2)

    splunk.node_step(
        node="generator", phase="exit",
        trajectory_id=trajectory_id, session_id=sid,
        iteration_count=iteration,
        duration_ms=_total_ms,
        answer_length=len(answer),
        source_count=len(sources),
        guard_blocked=guard_blocked,
        tokens_in=tokens.get("in", 0),
        tokens_out=tokens.get("out", 0),
        tokens_total=tokens.get("total", 0),
        redis_history_ms=_redis_history_ms,
        llm_ms=_llm_ms,
        guard_ms=_guard_ms,
        redis_append_ms=_redis_append_ms,
    )

    return {
        "answer":       answer,
        "sources":      sources,
        "confidence":   confidence,
        "stage_tokens": stage_tokens,
        "node_timings": {"generator": {
            "redis_history_ms": _redis_history_ms,
            "llm_ms":           _llm_ms,
            "output_guard_ms":  _guard_ms,
            "redis_append_ms":  _redis_append_ms,
            "total_ms":         _total_ms,
        }},
    }
