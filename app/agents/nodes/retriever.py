import time

from app.agents.graph.state import AgentState
from app.memory import session as session_mem
from app.observability import splunk
from app.retrieval.reranker import rerank
from app.retrieval.search import hybrid_search

_FETCH_K        = 10
_RERANK_K       = 5
_RERANK_THRESHOLD = 0.4


def retriever_node(state: AgentState) -> dict:
    t0            = time.time()
    trajectory_id = state.get("trajectory_id", "")
    iteration     = state.get("iteration_count", 0)

    splunk.node_step(node="retriever", phase="enter", trajectory_id=trajectory_id,
                     session_id=state.get("session_id", ""), iteration_count=iteration)

    gap    = state.get("retrieval_gap") or ""
    plan   = state.get("plan") or [state["question"]]
    prefix = state.get("session_id", "")

    # Re-entry: use the reasoner's gap as the query; first pass: use plan sub-tasks
    queries = [gap] if gap else plan[:3]

    seen:       set[str]   = set()
    candidates: list[dict] = []

    for query in queries:
        for chunk in hybrid_search(query, "finance_content", top_k=_FETCH_K):
            if chunk["text"] not in seen:
                seen.add(chunk["text"])
                candidates.append(chunk)

    # Structure chunks for section context (first query only, not reranked)
    structure: list[dict] = []
    for chunk in hybrid_search(queries[0], "finance_structure", top_k=2):
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            structure.append(chunk)

    # Cross-encoder rerank content candidates
    reranked = rerank(state["question"], candidates, top_k=_RERANK_K)

    # Drop low-confidence chunks — keep at least 1 so the pipeline never stalls
    above = [c for c in reranked if c.get("rerank_score", 0.0) >= _RERANK_THRESHOLD]
    reranked = above if above else reranked[:1]

    all_chunks = reranked + structure

    # Dedup only on the first pass (iteration==0).
    # On re-entry (gap-based search, iteration>0) we skip dedup — the reasoner
    # already said the first-pass chunks were insufficient, so filtering them
    # again would remove exactly the chunks we most need to find.
    if prefix and iteration == 0:
        all_chunks = session_mem.filter_new_chunks(prefix, all_chunks)

    # Assumption drift guard: gap-based re-entry returned nothing → fall back to
    # the original question so the generator has something to work with rather
    # than proceeding on an invalid "we have context" assumption.
    if not all_chunks and gap:
        fallback: list[dict] = []
        for chunk in hybrid_search(state["question"], "finance_content", top_k=_FETCH_K):
            if chunk["text"] not in seen:
                seen.add(chunk["text"])
                fallback.append(chunk)
        reranked_fb = rerank(state["question"], fallback, top_k=_RERANK_K)
        if prefix:
            reranked_fb = session_mem.filter_new_chunks(prefix, reranked_fb)
        all_chunks = reranked_fb

    retrieval_empty = len(all_chunks) == 0

    new_iteration = iteration + 1
    splunk.node_step(
        node="retriever", phase="exit",
        trajectory_id=trajectory_id, session_id=state.get("session_id", ""),
        iteration_count=new_iteration,
        duration_ms=round((time.time() - t0) * 1000, 2),
        chunk_count=len(all_chunks),
        retrieval_empty=retrieval_empty,
    )

    return {
        "retrieved_chunks": all_chunks,
        "iteration_count":  new_iteration,
        "retrieval_empty":  retrieval_empty,
    }
