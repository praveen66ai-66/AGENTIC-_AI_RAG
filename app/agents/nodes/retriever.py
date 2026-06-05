import time

from app.agents.graph.state import AgentState
from app.memory import session as session_mem
from app.observability import splunk
from app.retrieval.reranker import rerank
from app.retrieval.search import hybrid_search, section_lookup

_FETCH_K          = 15   # wider candidate pool so tables ranked 6-15 reach the reranker
_RERANK_K         = 5    # reranker is cheap; keep top 5 for richer generator context
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

    _t = time.time()
    for query in queries:
        for chunk in hybrid_search(query, "finance_content", top_k=_FETCH_K):
            if chunk["text"] not in seen:
                seen.add(chunk["text"])
                candidates.append(chunk)
    structure: list[dict] = []
    for chunk in hybrid_search(queries[0], "finance_structure", top_k=2):
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            structure.append(chunk)

    # Section-body lookup: when a structure header matches well (score ≥ 0.3),
    # fetch the body paragraphs under that section via a payload filter.
    # This handles "paste the section title as the query" cases where the header
    # is in finance_structure but the content is in finance_content.
    _SECTION_SCORE_THRESHOLD = 0.3
    for s_chunk in structure:
        if s_chunk.get("score", 0.0) >= _SECTION_SCORE_THRESHOLD:
            section_name = s_chunk.get("section") or s_chunk.get("text", "").strip()
            for chunk in section_lookup(queries[0], section_name, top_k=6):
                if chunk["text"] not in seen:
                    seen.add(chunk["text"])
                    candidates.append(chunk)

    _qdrant_ms = round((time.time() - _t) * 1000, 2)

    _t = time.time()
    reranked = rerank(state["question"], candidates, top_k=_RERANK_K)
    _rerank_ms = round((time.time() - _t) * 1000, 2)

    above    = [c for c in reranked if c.get("rerank_score", 0.0) >= _RERANK_THRESHOLD]
    reranked = above if above else reranked[:1]
    all_chunks = reranked + structure

    _t = time.time()
    if prefix and iteration == 0:
        all_chunks = session_mem.filter_new_chunks(prefix, all_chunks)
    _redis_dedup_ms = round((time.time() - _t) * 1000, 2)

    # Assumption drift guard
    _fallback_ms = 0.0
    if not all_chunks and gap:
        _t = time.time()
        fallback: list[dict] = []
        for chunk in hybrid_search(state["question"], "finance_content", top_k=_FETCH_K):
            if chunk["text"] not in seen:
                seen.add(chunk["text"])
                fallback.append(chunk)
        reranked_fb = rerank(state["question"], fallback, top_k=_RERANK_K)
        if prefix:
            reranked_fb = session_mem.filter_new_chunks(prefix, reranked_fb)
        all_chunks   = reranked_fb
        _fallback_ms = round((time.time() - _t) * 1000, 2)

    retrieval_empty = len(all_chunks) == 0
    new_iteration   = iteration + 1
    _total_ms       = round((time.time() - t0) * 1000, 2)

    splunk.node_step(
        node="retriever", phase="exit",
        trajectory_id=trajectory_id, session_id=state.get("session_id", ""),
        iteration_count=new_iteration,
        duration_ms=_total_ms,
        chunk_count=len(all_chunks),
        retrieval_empty=retrieval_empty,
        qdrant_ms=_qdrant_ms, rerank_ms=_rerank_ms,
        redis_dedup_ms=_redis_dedup_ms,
    )

    suffix = f"_{iteration}" if iteration > 0 else ""
    return {
        "retrieved_chunks": all_chunks,
        "iteration_count":  new_iteration,
        "retrieval_empty":  retrieval_empty,
        "node_timings": {f"retriever{suffix}": {
            "qdrant_ms":      _qdrant_ms,
            "rerank_ms":      _rerank_ms,
            "redis_dedup_ms": _redis_dedup_ms,
            "fallback_ms":    _fallback_ms,
            "total_ms":       _total_ms,
        }},
    }
