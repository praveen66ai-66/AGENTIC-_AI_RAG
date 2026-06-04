import time

from langchain_core.messages import SystemMessage

from app.agents._utils import format_chunks, get_llm, get_tokens, load_prompt, parse_json
from app.agents.graph.state import AgentState
from app.observability import splunk


def reasoner_node(state: AgentState) -> dict:
    t0            = time.time()
    trajectory_id = state.get("trajectory_id", "")
    session_id    = state.get("session_id", "")
    iteration     = state.get("iteration_count", 0)

    splunk.node_step(node="reasoner", phase="enter", trajectory_id=trajectory_id,
                     session_id=session_id, iteration_count=iteration)

    chunks = state.get("retrieved_chunks") or []

    # Assumption drift guard: skip the LLM call entirely when there are no chunks.
    # Calling the LLM against "No chunks retrieved." risks a spurious
    # context_sufficient=True or wastes a Groq token budget.
    if not chunks:
        gap = f"No relevant chunks found for: {state['question'][:80]}"
        splunk.node_step(
            node="reasoner", phase="exit",
            trajectory_id=trajectory_id, session_id=session_id,
            iteration_count=iteration,
            duration_ms=round((time.time() - t0) * 1000, 2),
            context_sufficient=False,
            confidence=0.0,
        )
        return {
            "context_sufficient": False,
            "retrieval_gap":      gap,
            "confidence":         0.0,
        }

    prompt = load_prompt("reasoner")

    system_text = prompt["system"].format(
        question=state["question"],
        retrieved_chunks=format_chunks(chunks),
    )

    response = get_llm().invoke([SystemMessage(content=system_text)])
    parsed   = parse_json(response.content)

    context_sufficient = bool(parsed.get("context_sufficient", False))
    confidence         = float(parsed.get("confidence", 0.0))
    tokens             = get_tokens(response)

    splunk.node_step(
        node="reasoner", phase="exit",
        trajectory_id=trajectory_id, session_id=session_id,
        iteration_count=iteration,
        duration_ms=round((time.time() - t0) * 1000, 2),
        context_sufficient=context_sufficient,
        confidence=confidence,
        tokens_in=tokens.get("in", 0),
        tokens_out=tokens.get("out", 0),
        tokens_total=tokens.get("total", 0),
    )

    stage_tokens = dict(state.get("stage_tokens") or {})
    key = f"reasoner_{iteration}" if iteration > 1 else "reasoner"
    stage_tokens[key] = tokens

    return {
        "context_sufficient": context_sufficient,
        "retrieval_gap":      str(parsed.get("retrieval_gap") or ""),
        "confidence":         confidence,
        "stage_tokens":       stage_tokens,
    }
