"""
Build Qdrant indexes from processed chunks — Hybrid (Dense + BM25 Sparse).

Reads:
    data/processed/text_chunks.json
    data/processed/table_chunks.json
    data/processed/image_refs.json

Writes:
    data/processed/chunks_text.json      — theory text chunks
    data/processed/chunks_tables.json    — theory table chunks
    data/processed/chunks_images.json    — image description chunks
    data/processed/chunks_exercises.json — exercise/Q&A chunks (separate collection)
    data/processed/chunks_structure.json — header/front-matter chunks

Qdrant collections (never deleted — upsert with stable IDs is idempotent):
    finance_content   — theory text + tables + images  (queried at runtime)
    finance_exercises — exercise/Q&A content           (excluded from main queries)
    finance_structure — section headers + front matter (structural lookup)

Stable chunk IDs:
    ID = MD5(file_hash | page | content_type | text[:80]) as UUID
    Running build_index.py twice on the same data is safe — identical chunks
    overwrite themselves. New pages from a larger ingest run are simply added.

Chunking strategy:
    Text   — section-aware merge, 150-char overlap, 1200-char cap
    Tables — section + context_before + table_markdown, never split
    Images — LLM description text, 1200-char cap, skip if empty

Dense:  BAAI/bge-base-en-v1.5 (768 dims, 512-token limit ≈ 1200 chars)
Sparse: Qdrant/bm25 via fastembed (keyword matching, no token limit)
Fusion: RRF at query time

Usage:
    uv run python scripts/build_index.py
"""

import hashlib
import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TEXT_CHUNKS_PATH  = Path("data/processed/text_chunks.json")
TABLE_CHUNKS_PATH = Path("data/processed/table_chunks.json")
IMAGE_REFS_PATH   = Path("data/processed/image_refs.json")

CHUNKS_TEXT_OUT      = Path("data/processed/chunks_text.json")
CHUNKS_TABLES_OUT    = Path("data/processed/chunks_tables.json")
CHUNKS_IMAGES_OUT    = Path("data/processed/chunks_images.json")
CHUNKS_EXERCISES_OUT = Path("data/processed/chunks_exercises.json")
CHUNKS_STRUCTURE_OUT = Path("data/processed/chunks_structure.json")

DENSE_MODEL   = "BAAI/bge-base-en-v1.5"
DENSE_SIZE    = 768
MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS   = 150

COLLECTION_CONTENT   = "finance_content"
COLLECTION_EXERCISES = "finance_exercises"
COLLECTION_STRUCTURE = "finance_structure"
UPLOAD_BATCH_SIZE    = 20
QDRANT_TIMEOUT       = 120


# ── Stable chunk ID ───────────────────────────────────────────────────────────

def _chunk_id(file_hash: str, page: int | None, content_type: str, text: str) -> str:
    """
    Deterministic UUID from chunk identity.
    Safe to upsert repeatedly — identical chunks overwrite themselves in Qdrant.
    Different file_hash → different IDs → new document adds new points cleanly.
    """
    key = f"{file_hash}|{page}|{content_type}|{text[:200]}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


# ── Point factory ─────────────────────────────────────────────────────────────

def _make_point(
    text: str,
    page: int | None,
    section: str,
    source: str,
    content_type: str,
    item_type: str,
    file_hash: str,
    is_exercise: bool = False,
) -> dict:
    return {
        "id":   _chunk_id(file_hash, page, content_type, text),
        "text": text,
        "metadata": {
            "page":         page,
            "type":         item_type,
            "section":      section,
            "source":       source,
            "content_type": content_type,
            "file_hash":    file_hash,
            "is_exercise":  is_exercise,
        },
    }


# ── Text chunking ─────────────────────────────────────────────────────────────

def merge_text_chunks(
    raw_items: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Merge atomic text items into section-aware chunks with overlap.

    Returns (content_chunks, exercise_chunks, structure_chunks).

    Rules:
    - Front matter + SectionHeaderItems → structure collection only
    - Items where is_exercise=True → exercise collection
    - Everything else → content collection
    - Sliding window with OVERLAP_CHARS carry-over within each section
    """
    content_chunks:   list[dict] = []
    exercise_chunks:  list[dict] = []
    structure_chunks: list[dict] = []

    # Default file_hash from first item that has one
    file_hash = next(
        (i.get("file_hash", "") for i in raw_items if i.get("file_hash")), ""
    )

    # ── Headers and front matter → structure ──────────────────────────────────
    for item in raw_items:
        if item.get("is_front_matter") or item["type"] == "SectionHeaderItem":
            text = item.get("text", "").strip()
            if text:
                structure_chunks.append(_make_point(
                    text=text,
                    page=item.get("page"),
                    section=item.get("section", ""),
                    source="text",
                    content_type="text",
                    item_type=item["type"],
                    file_hash=item.get("file_hash", file_hash),
                    is_exercise=item.get("is_exercise", False),
                ))

    # ── Body items → group by (section, is_exercise) then slide window ────────
    # Key: (section, is_exercise) so exercise and theory items in the same
    # section name don't accidentally merge together.
    seen_keys:    list[tuple] = []
    section_map:  dict[tuple, list[dict]] = defaultdict(list)

    for item in raw_items:
        if item.get("is_front_matter"):
            continue
        if item["type"] == "SectionHeaderItem":
            continue
        text = item.get("text", "").strip()
        if not text:
            continue
        key = (item.get("section", ""), item.get("is_exercise", False))
        if key not in section_map:
            seen_keys.append(key)
        section_map[key].append(item)

    for (section, is_exercise) in seen_keys:
        items  = section_map[(section, is_exercise)]
        prefix = f"[{section}]\n" if section else ""
        budget = MAX_CHUNK_CHARS - len(prefix)
        target = exercise_chunks if is_exercise else content_chunks

        window:     list[str]  = []
        window_len: int        = 0
        first_page: int | None = items[0].get("page") if items else None
        chunk_fh = items[0].get("file_hash", file_hash)

        def flush(w: list[str], pg: int | None) -> dict | None:
            if not w:
                return None
            return _make_point(
                text=prefix + " ".join(w),
                page=pg,
                section=section,
                source="text",
                content_type="text",
                item_type="MergedChunk",
                file_hash=chunk_fh,
                is_exercise=is_exercise,
            )

        for item in items:
            t       = item["text"].strip()
            add_len = len(t) + 1

            if window and window_len + add_len > budget:
                pt = flush(window, first_page)
                if pt:
                    target.append(pt)

                overlap: list[str] = []
                ol = 0
                for prev in reversed(window):
                    if ol + len(prev) + 1 <= OVERLAP_CHARS:
                        overlap.insert(0, prev)
                        ol += len(prev) + 1
                    else:
                        break
                window     = overlap
                window_len = ol
                first_page = item.get("page", first_page)

            if not window:
                first_page = item.get("page", first_page)

            window.append(t)
            window_len += add_len

        pt = flush(window, first_page)
        if pt:
            target.append(pt)

    return content_chunks, exercise_chunks, structure_chunks


# ── Table chunking ────────────────────────────────────────────────────────────

_ROWS_PER_CHUNK = 5   # rows per NL-sentence chunk


def _parse_markdown_table(table_md: str) -> tuple[str, list[str], list[list[str]]]:
    """
    Parse a markdown table into (title, headers, data_rows).

    title      — first non-pipe, non-blank line (e.g. 'Brazil (in €bn)')
    headers    — column names, deduplicated with _1/_2 suffix when repeated
    data_rows  — list of cell-value lists, one per body row
    """
    title   = ""
    headers: list[str] = []
    rows:    list[list[str]] = []

    for line in table_md.splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith("|"):
            if not title:
                title = s
            continue
        if all(c in "|-: " for c in s):   # separator row
            continue
        cells = [c.strip() for c in s.split("|") if c.strip() != ""]
        if not headers:
            # Build deduplicated header list
            seen: dict[str, int] = {}
            for c in cells:
                if c in seen:
                    seen[c] += 1
                    headers.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    headers.append(c)
        else:
            rows.append(cells)

    return title, headers, rows


def _row_to_sentence(title: str, headers: list[str], cells: list[str]) -> str:
    """
    Serialize one table row as a natural-language sentence.

    Example output:
        'Brazil (in €bn) — Ambev: Market Capitalisation 87, Beta 0.18, P/E ratio 2014 20.6.'
    """
    _SKIP = {"n.s.", "na", "-", ""}
    pairs = []
    for h, v in zip(headers, cells):
        v = v.strip()
        if v.lower() not in _SKIP:
            pairs.append(f"{h} {v}")
    body = ", ".join(pairs)
    return f"{title} — {body}." if title else f"{body}."


def _table_summary(
    title: str, headers: list[str], rows: list[list[str]],
    prefix: str, page, section: str, fh: str, is_ex: bool,
) -> dict | None:
    """
    Build one summary chunk per table for high-level semantic queries.

    Covers: table name, row count, all entity names, numeric ranges.
    A query like 'top companies in Brazil' or 'what is in the Brazil table'
    hits this chunk directly without needing to rank across row-batch chunks.
    """
    if not rows or not headers:
        return None

    # Identify the entity-name column: most unique non-numeric values
    name_col = 0
    max_unique = 0
    for i in range(len(headers)):
        vals = {r[i] for r in rows if i < len(r)}
        non_num = {v for v in vals if v and not v.replace(".", "").replace("-", "").isnumeric()}
        if len(non_num) > max_unique:
            max_unique, name_col = len(non_num), i

    entities = [r[name_col].strip() for r in rows if name_col < len(r) and r[name_col].strip()]

    # Numeric ranges for up to 5 columns
    ranges: list[str] = []
    for i, h in enumerate(headers):
        nums: list[float] = []
        for r in rows:
            if i < len(r):
                try:
                    nums.append(float(r[i].replace(",", "").replace(" ", "")))
                except ValueError:
                    pass
        if nums:
            ranges.append(f"{h}: {min(nums):.4g}–{max(nums):.4g}")
        if len(ranges) == 5:
            break

    lines: list[str] = []
    if title:
        lines.append(f"Summary — {title}")
    lines.append(f"Entries: {len(rows)}. Columns: {', '.join(headers)}.")
    if entities:
        lines.append(f"Entities listed: {', '.join(entities)}.")
    if ranges:
        lines.append(f"Numeric ranges — {'; '.join(ranges)}.")

    text = (prefix + "\n".join(lines)).strip()
    return _make_point(
        text=text,
        page=page,
        section=section,
        source="table",
        content_type="table",
        item_type="TableSummary",
        file_hash=fh,
        is_exercise=is_ex,
    )


def process_table(chunk: dict) -> list[dict]:
    """
    Convert one raw table chunk into one or more index points.

    Strategy (highest-impact for semantic retrieval):
    - Parse markdown into title + column headers + data rows
    - Convert each row to a natural-language sentence
    - Group every ROWS_PER_CHUNK rows into one chunk, each prefixed with
      'Table: <title>\\nColumns: <headers>' so context is never lost
    - Falls back to the raw markdown when the table cannot be parsed

    Returns a list (may be empty); replaces the old dict | None return type.
    Old table chunks in Qdrant must be purged before uploading (IDs changed).
    """
    table_md = chunk.get("table_markdown", "").strip()
    if not table_md:
        return []

    section = chunk.get("section", "")
    fh      = chunk.get("file_hash", "")
    is_ex   = chunk.get("is_exercise", False)
    page    = chunk.get("page")
    prefix  = f"[{section}]\n" if section else ""

    title, headers, rows = _parse_markdown_table(table_md)

    # Context block prepended to every row-batch chunk
    col_str = ", ".join(headers)
    ctx = f"Table: {title}\nColumns: {col_str}\n\n" if title else f"Columns: {col_str}\n\n"

    if not rows:
        # Unparseable — fall back to raw markdown with context block
        text = f"{prefix}{ctx}{table_md}".strip()
        return [_make_point(text=text, page=page, section=section, source="table",
                            content_type="table", item_type="TableItem",
                            file_hash=fh, is_exercise=is_ex)]

    # ── Summary chunk (Point 3) ───────────────────────────────────────────────
    # One high-level chunk per table that answers broad queries like
    # "top companies in Brazil" without needing to rank across row batches.
    summary_pt = _table_summary(title, headers, rows, prefix, page, section, fh, is_ex)

    # ── Row-batch chunks ─────────────────────────────────────────────────────
    points: list[dict] = [summary_pt] if summary_pt else []
    for batch_start in range(0, len(rows), _ROWS_PER_CHUNK):
        batch     = rows[batch_start : batch_start + _ROWS_PER_CHUNK]
        sentences = "\n".join(_row_to_sentence(title, headers, r) for r in batch)
        text      = f"{prefix}{ctx}{sentences}".strip()
        points.append(_make_point(
            text=text,
            page=page,
            section=section,
            source="table",
            content_type="table",
            item_type="TableItem",
            file_hash=fh,
            is_exercise=is_ex,
        ))
    return points


# ── Image chunking ────────────────────────────────────────────────────────────

def process_image(chunk: dict) -> dict | None:
    desc = chunk.get("description", "").strip()
    if not desc:
        return None

    section = chunk.get("section", "")
    prefix  = f"[{section}]\n" if section else ""
    fh      = chunk.get("file_hash", "")
    text    = (prefix + desc)[:MAX_CHUNK_CHARS]

    return _make_point(
        text=text,
        page=chunk.get("page"),
        section=section,
        source="image",
        content_type="image_description",
        item_type="ImageItem",
        file_hash=fh,
        is_exercise=False,
    )


# ── Load + route all chunks ───────────────────────────────────────────────────

def load_chunks() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """
    Returns (text_chunks, table_chunks, image_chunks, exercise_chunks, structure_chunks).
    text_chunks + table_chunks + image_chunks → finance_content
    exercise_chunks                           → finance_exercises
    structure_chunks                          → finance_structure
    """
    with open(TEXT_CHUNKS_PATH, encoding="utf-8") as f:
        raw_text = json.load(f)
    text_chunks, exercise_text, structure_chunks = merge_text_chunks(raw_text)

    with open(TABLE_CHUNKS_PATH, encoding="utf-8") as f:
        raw_tables = json.load(f)

    table_chunks:    list[dict] = []
    exercise_tables: list[dict] = []
    for chunk in raw_tables:
        for pt in process_table(chunk):   # process_table now returns list[dict]
            (exercise_tables if pt["metadata"]["is_exercise"] else table_chunks).append(pt)

    exercise_chunks = exercise_text + exercise_tables

    image_chunks: list[dict] = []
    if IMAGE_REFS_PATH.exists():
        with open(IMAGE_REFS_PATH, encoding="utf-8") as f:
            raw_images = json.load(f)
        image_chunks = [pt for chunk in raw_images if (pt := process_image(chunk))]

    # ── Audit ─────────────────────────────────────────────────────────────────
    content_all = text_chunks + table_chunks + image_chunks
    all_lens    = [len(p["text"]) for p in content_all]
    over        = [p for p in content_all if len(p["text"]) > MAX_CHUNK_CHARS]

    print(f"\nChunk audit (BGE {DENSE_MODEL}, limit {MAX_CHUNK_CHARS} chars):")
    print(f"  theory text  : {len(text_chunks):4d} chunks")
    print(f"  theory tables: {len(table_chunks):4d} chunks")
    print(f"  images       : {len(image_chunks):4d} chunks")
    print(f"  exercises    : {len(exercise_chunks):4d} chunks  (separate collection)")
    print(f"  structure    : {len(structure_chunks):4d} chunks")
    if all_lens:
        print(f"  avg size     : {sum(all_lens)//len(all_lens)} chars")
    if over:
        types = {p["metadata"]["content_type"] for p in over}
        print(f"  INFO: {len(over)} chunks exceed limit ({', '.join(types)}) — BM25 covers all keywords.")
    else:
        print(f"  All content chunks within {MAX_CHUNK_CHARS}-char BGE safe limit.")

    return text_chunks, table_chunks, image_chunks, exercise_chunks, structure_chunks


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _purge_old_table_chunks(client) -> None:
    """
    Delete all content_type='table' points from finance_content before re-uploading.

    Row-to-sentence chunking changes chunk IDs (one table → many chunks), so old
    single-table chunks would persist as stale duplicates if not explicitly removed.
    Text and image chunks are unaffected — only table points are purged here.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
    try:
        client.delete(
            collection_name=COLLECTION_CONTENT,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="content_type",
                                                   match=MatchValue(value="table"))])
            ),
        )
        print(f"  Purged old table chunks from '{COLLECTION_CONTENT}'")
    except Exception as exc:
        print(f"  [warn] Could not purge old table chunks: {exc}")


def ensure_collection(client, name: str, dense_size: int, VectorParams, Distance,
                      SparseVectorParams, SparseIndexParams) -> None:
    """Create collection if it does not exist. Never deletes existing data."""
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": VectorParams(size=dense_size, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )
        print(f"  Created : {name}")
    else:
        print(f"  Exists  : {name}  (data preserved — upsert will add/overwrite by stable ID)")


def _upload(client, collection_name: str, points: list[dict],
            dense_model, sparse_model, PointStruct, SparseVector) -> None:
    if not points:
        print(f"  {collection_name}: 0 points, nothing to upload.")
        return

    print(f"\nEmbedding '{collection_name}' ({len(points)} points)...")
    texts = [p["text"] for p in points]

    dense_vecs  = dense_model.encode(texts, show_progress_bar=True, batch_size=32)
    print("  Generating BM25 sparse vectors...")
    sparse_vecs = list(sparse_model.embed(texts))

    qdrant_points = [
        PointStruct(
            id=p["id"],                          # stable deterministic ID
            vector={
                "dense":  dv.tolist(),
                "sparse": {
                    "indices": sv.indices.tolist(),
                    "values":  sv.values.tolist(),
                },
            },
            payload={"text": p["text"], **p["metadata"]},
        )
        for p, dv, sv in zip(points, dense_vecs, sparse_vecs)
    ]

    for i in range(0, len(qdrant_points), UPLOAD_BATCH_SIZE):
        batch = qdrant_points[i:i + UPLOAD_BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        done = min(i + UPLOAD_BATCH_SIZE, len(qdrant_points))
        print(f"  Uploaded {done}/{len(qdrant_points)}", end="\r")
    print(f"  Uploaded {len(qdrant_points)} points.        ")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for path in [TEXT_CHUNKS_PATH, TABLE_CHUNKS_PATH]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run ingest.py first.")
            sys.exit(1)

    text_chunks, table_chunks, image_chunks, exercise_chunks, structure_chunks = load_chunks()

    content_points  = text_chunks + table_chunks + image_chunks
    structure_points = structure_chunks

    print(f"\nfinance_content   : {len(content_points)} points")
    print(f"finance_exercises : {len(exercise_chunks)} points")
    print(f"finance_structure : {len(structure_points)} points")

    # Save each type for inspection
    for path, data in [
        (CHUNKS_TEXT_OUT,      text_chunks),
        (CHUNKS_TABLES_OUT,    table_chunks),
        (CHUNKS_IMAGES_OUT,    image_chunks),
        (CHUNKS_EXERCISES_OUT, exercise_chunks),
        (CHUNKS_STRUCTURE_OUT, structure_points),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    print("\nChunks saved to data/processed/:")
    print(f"  chunks_text.json      ({len(text_chunks)} chunks)")
    print(f"  chunks_tables.json    ({len(table_chunks)} chunks)")
    print(f"  chunks_images.json    ({len(image_chunks)} chunks)")
    print(f"  chunks_exercises.json ({len(exercise_chunks)} chunks)")
    print(f"  chunks_structure.json ({len(structure_points)} chunks)")

    # ── Models ────────────────────────────────────────────────────────────────
    print(f"\nLoading dense model '{DENSE_MODEL}'...")
    from sentence_transformers import SentenceTransformer
    dense_model = SentenceTransformer(DENSE_MODEL)
    print("Dense model loaded.")

    print("\nLoading sparse BM25 model...")
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    print("Sparse model loaded.")

    # ── Connect ───────────────────────────────────────────────────────────────
    print("\nConnecting to Qdrant Cloud...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        SparseVectorParams, SparseIndexParams,
        Prefetch, FusionQuery, Fusion, SparseVector,
    )
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=QDRANT_TIMEOUT,
    )
    print("Connected.")

    # ── Ensure collections exist (never delete) ───────────────────────────────
    print("\nEnsuring collections exist:")
    for name in [COLLECTION_CONTENT, COLLECTION_EXERCISES, COLLECTION_STRUCTURE]:
        ensure_collection(client, name, DENSE_SIZE,
                          VectorParams, Distance, SparseVectorParams, SparseIndexParams)

    # ── Purge old table chunks before re-uploading with new row-sentence IDs ─────
    print("\nPurging stale table chunks (row-to-sentence IDs changed):")
    _purge_old_table_chunks(client)

    # ── Embed + upsert ────────────────────────────────────────────────────────
    for col, pts in [
        (COLLECTION_CONTENT,   content_points),
        (COLLECTION_EXERCISES, exercise_chunks),
        (COLLECTION_STRUCTURE, structure_points),
    ]:
        _upload(client, col, pts, dense_model, sparse_model, PointStruct, SparseVector)

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\nSanity check — hybrid RRF on 'operating cash flow':")
    q_text    = "operating cash flow"
    dense_q   = dense_model.encode(q_text).tolist()
    sparse_q  = list(sparse_model.query_embed(q_text))[0]

    results = client.query_points(
        collection_name=COLLECTION_CONTENT,
        prefetch=[
            Prefetch(query=dense_q, using="dense", limit=10),
            Prefetch(
                query=SparseVector(
                    indices=sparse_q.indices.tolist(),
                    values=sparse_q.values.tolist(),
                ),
                using="sparse", limit=10,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=3,
    ).points

    for r in results:
        ex = "[exercise]" if r.payload.get("is_exercise") else ""
        print(f"  [{r.score:.3f}] p{r.payload.get('page')} {ex} | {r.payload['text'][:100]}")

    print("\nDone. Collections updated with stable-ID upsert.")
    print("  Run again on larger ingest → new pages added, existing pages overwritten (idempotent).")


if __name__ == "__main__":
    main()
