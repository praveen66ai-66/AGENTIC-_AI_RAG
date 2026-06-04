from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from app.agents.graph.state import AgentState
from app.agents.graph.edges import route_after_reasoner
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.reasoner import reasoner_node
from app.agents.nodes.generator import generator_node


def build_rag_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner",   planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reasoner",  reasoner_node)
    graph.add_node("generator", generator_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner",   "retriever")
    graph.add_edge("retriever", "reasoner")

    graph.add_conditional_edges(
        "reasoner",
        route_after_reasoner,
        {"retrieve": "retriever", "generate": "generator"},
    )

    graph.add_edge("generator", END)

    return graph.compile(checkpointer=MemorySaver())


def build_pre_graph() -> StateGraph:
    """
    Planner → retriever → reasoner only — no generator.
    Used by the streaming path: run retrieval first, then stream LLM tokens
    separately so the user sees output as it is generated.
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner",   planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reasoner",  reasoner_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reasoner")

    # When context is sufficient or iteration cap hit → END (generator runs separately)
    graph.add_conditional_edges(
        "reasoner",
        route_after_reasoner,
        {"retrieve": "retriever", "generate": END},
    )

    return graph.compile(checkpointer=MemorySaver())


# Singletons
rag_graph = build_rag_graph()
pre_graph = build_pre_graph()
