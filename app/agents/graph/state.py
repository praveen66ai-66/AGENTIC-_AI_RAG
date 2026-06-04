from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]
    question: str
    session_id: str

    # Planner output
    plan: list[str]

    # Retriever output
    retrieved_chunks: list[dict]

    # Reasoner judgment
    context_sufficient: bool
    retrieval_gap: str        # what's still missing, fed back to retriever

    # Generator output
    answer: str
    sources: list[dict]
    confidence: float

    # Control
    iteration_count: int      # guards against infinite retrieval loops
    trajectory_id: str
    retrieval_empty: bool     # True when retriever found 0 chunks after all filtering
    stage_tokens: dict        # {"planner": {"in":N,"out":M}, "reasoner": {...}, "generator": {...}}
