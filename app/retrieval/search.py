"""
Shared hybrid retrieval singletons — BAAI/bge-base-en-v1.5 (768d) + BM25.
Imported by both the FastAPI query route and the agent retriever node so the
embedding models are loaded once per process, not once per module.
"""

import os

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FusionQuery, Fusion, Prefetch, SparseVector
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# ── Page number mapping ───────────────────────────────────────────────────────
# build_page_map.py writes data/processed/page_map.json with the exact
# {pdf_page: book_page} mapping extracted from the PDF headers.
# Falls back to the static offset when the map is missing (e.g. before first run).

_PDF_PAGE_OFFSET = int(os.getenv("PDF_PAGE_OFFSET", "16"))
_PAGE_MAP: dict[int, int] = {}

def _load_page_map() -> None:
    global _PAGE_MAP
    import json
    from pathlib import Path
    map_path = Path("data/processed/page_map.json")
    if map_path.exists():
        try:
            raw      = json.loads(map_path.read_text(encoding="utf-8"))
            _PAGE_MAP = {int(k): int(v) for k, v in raw.items()}
            print(f"[search] Loaded page map: {len(_PAGE_MAP)} entries")
        except Exception as e:
            print(f"[search] Could not load page map: {e} — using offset fallback")

def _to_book_page(pdf_page: int | None) -> int | None:
    if pdf_page is None:
        return None
    return _PAGE_MAP.get(pdf_page, pdf_page - _PDF_PAGE_OFFSET)

_load_page_map()

_embed  = SentenceTransformer("BAAI/bge-base-en-v1.5")
_sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
_qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _query_qdrant(collection: str, prefetch: list, query, limit: int):
    """Retried Qdrant network call — 3 attempts, 1→2→4→8s backoff."""
    return _qdrant.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=query,
        limit=limit,
    ).points


def hybrid_search(query: str, collection: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid dense+sparse search with RRF fusion.
    Returns list of dicts: {text, page, section, source, score, collection}.
    Qdrant call is retried up to 3 times on transient errors.
    """
    dense_vec  = _embed.encode(query).tolist()
    sparse_raw = list(_sparse.query_embed(query))[0]
    sparse_vec = SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist(),
    )
    points = _query_qdrant(
        collection,
        prefetch=[
            Prefetch(query=dense_vec,  using="dense",  limit=top_k * 2),
            Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )
    results = []
    for r in points:
        pdf_page  = r.payload.get("page")
        book_page = _to_book_page(pdf_page)
        results.append({
            "text":       r.payload.get("text", ""),
            "page":       book_page,   # printed book page shown in citations
            "pdf_page":   pdf_page,    # raw PDF page kept for debugging
            "section":    r.payload.get("section", ""),
            "source":     r.payload.get("source", ""),
            "score":      round(r.score, 4),
            "collection": collection,
        })
    return results
