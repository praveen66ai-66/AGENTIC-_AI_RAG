import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._utils import format_history, get_llm, get_tokens, load_prompt, parse_json
from app.agents.graph.state import AgentState
from app.memory import session as session_mem
from app.observability import splunk


def _split_prefix(prefix: str) -> tuple[str, str, str]:
    """Split '{tenant}:{user}:{session}' → (tenant, user, session). Returns empty strings on error."""
    parts = prefix.split(":", 2)
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else ("", "", "")

_DOCUMENT_DESCRIPTION = (
    "Corporate Finance and Accounting textbook covering GAAP/IFRS standards, "
    "financial statements (income statement, balance sheet, cash flow), ratio "
    "analysis, valuation, working capital, and end-of-chapter exercises."
)


def planner_node(state: AgentState) -> dict:
    t0            = time.time()
    trajectory_id = state.get("trajectory_id", "")
    sid           = state.get("session_id", "")

    splunk.node_step(node="planner", phase="enter", trajectory_id=trajectory_id, session_id=sid)

    prompt = load_prompt("planner")

    _t = time.time()
    history   = session_mem.get_history(sid, *_split_prefix(sid)) if sid else []
    prior_gap = session_mem.get_last_gap(sid) if sid else ""
    _redis_ms = round((time.time() - _t) * 1000, 2)

    system_text = prompt["system"].format(
        document_description=_DOCUMENT_DESCRIPTION,
        session_summary=format_history(history),
        prior_gap=prior_gap or "None",
    )
    human_text = prompt["human"].format(
        question=state["question"],
        additional_context="",
    )

    _t = time.time()
    response = get_llm().invoke([
        SystemMessage(content=system_text),
        HumanMessage(content=human_text),
    ])
    _llm_ms = round((time.time() - _t) * 1000, 2)

    parsed = parse_json(response.content)
    plan   = parsed.get("sub_tasks") or [state["question"]]
    tokens = get_tokens(response)
    _total_ms = round((time.time() - t0) * 1000, 2)

    splunk.node_step(
        node="planner", phase="exit",
        trajectory_id=trajectory_id, session_id=sid,
        duration_ms=_total_ms,
        plan_count=len(plan),
        tokens_in=tokens.get("in", 0),
        tokens_out=tokens.get("out", 0),
        tokens_total=tokens.get("total", 0),
        redis_ms=_redis_ms, llm_ms=_llm_ms,
    )

    stage_tokens = dict(state.get("stage_tokens") or {})
    stage_tokens["planner"] = tokens

    return {
        "plan":            [str(t) for t in plan],
        "iteration_count": 0,
        "stage_tokens":    stage_tokens,
        "node_timings":    {"planner": {
            "redis_history_ms": _redis_ms,
            "llm_ms":           _llm_ms,
            "total_ms":         _total_ms,
        }},
    }
