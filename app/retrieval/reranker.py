"""
Cross-encoder reranker — precision pass after hybrid retrieval.

Model selection (RERANKER_MODEL env var):
  cross-encoder/ms-marco-MiniLM-L-2-v2  ← default, ~200ms CPU, 2-layer, fast
  cross-encoder/ms-marco-MiniLM-L-6-v2  ← ~500ms CPU, 6-layer, better quality
  BAAI/bge-reranker-base                 ← ~20s CPU, 12-layer BERT, best quality

Disable entirely: RERANKER_ENABLED=false  (falls back to RRF score ordering)
"""

import os

from sentence_transformers import CrossEncoder

_model:   CrossEncoder | None = None
_ENABLED: bool = os.getenv("RERANKER_ENABLED", "true").lower() != "false"
_MODEL_NAME: str = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-2-v2",  # 100x faster than bge-reranker-base on CPU
)


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME, max_length=512)
    return _model


def warmup() -> None:
    """Load the cross-encoder at startup so the first real request isn't slow."""
    if _ENABLED:
        _get_model()


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Score each (query, chunk_text) pair with the cross-encoder and return
    the top_k chunks sorted by reranker score descending.

    Falls back to RRF score ordering if disabled or on error.
    """
    if not chunks:
        return chunks

    if not _ENABLED:
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
        return [{**c, "rerank_score": c.get("score", 0.0)} for c in sorted_chunks[:top_k]]

    try:
        model  = _get_model()
        pairs  = [(query, c.get("text", "")) for c in chunks]
        scores = model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        return [
            {**chunk, "rerank_score": round(float(score), 4)}
            for score, chunk in ranked[:top_k]
        ]

    except Exception:
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
        return [{**c, "rerank_score": c.get("score", 0.0)} for c in sorted_chunks[:top_k]]
