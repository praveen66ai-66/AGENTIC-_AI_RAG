import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents._utils import format_history, get_llm, load_prompt, parse_json
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

    prompt    = load_prompt("planner")
    history   = session_mem.get_history(sid, *_split_prefix(sid)) if sid else []
    prior_gap = session_mem.get_last_gap(sid) if sid else ""

    system_text = prompt["system"].format(
        document_description=_DOCUMENT_DESCRIPTION,
        session_summary=format_history(history),
        prior_gap=prior_gap or "None",
    )
    human_text = prompt["human"].format(
        question=state["question"],
        additional_context="",
    )

    response = get_llm().invoke([
        SystemMessage(content=system_text),
        HumanMessage(content=human_text),
    ])

    parsed = parse_json(response.content)
    plan   = parsed.get("sub_tasks") or [state["question"]]

    splunk.node_step(
        node="planner", phase="exit",
        trajectory_id=trajectory_id, session_id=sid,
        duration_ms=round((time.time() - t0) * 1000, 2),
        plan_count=len(plan),
    )

    return {
        "plan":            [str(t) for t in plan],
        "iteration_count": 0,
    }
