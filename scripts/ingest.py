"""
Ingest pipeline — extracts and separates PDF content by type.

Reads:  data/documents/CoreCourseFinancialAccounting.pdf
Writes:
    data/processed/text_chunks.json      — paragraphs, lists, headers
    data/processed/table_chunks.json     — tables + 8-line context before
    data/processed/image_refs.json       — image metadata + path to saved .png
    data/processed/images/               — actual image .png files

Each item carries:
    is_exercise: True  — content from exercise/Q&A sections (filtered at query time)
    is_exercise: False — theory/concept content (used for retrieval)

Pipeline order (do NOT skip steps):
    1. ingest.py          → extracts text, tables, images
    2. describe_images.py → converts images to text using Llama 4 Scout
    3. build_index.py     → chunks + embeds + uploads to Qdrant

Toggle SAMPLE_ONLY:
    True  → first FIRST_N_PAGES pages for testing
    False → full document (~4 hrs) for production
"""

import hashlib
import json
import sys
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PDF_PATH   = Path("data/documents/CoreCourseFinancialAccounting.pdf")
OUT_DIR    = Path("data/processed")
IMAGES_DIR = OUT_DIR / "images"

SAMPLE_ONLY = True
START_PAGE  = 1      # re-ingest full book with improved formula extraction
END_PAGE    = 1050   # last page of this batch (inclusive, covers end of book)

FRONT_MATTER_PAGES  = {1, 2, 3, 4, 5}
TABLE_CONTEXT_LINES = 8
# No size threshold — save every image Docling finds.
# Missing one real diagram means a full 4-hour re-ingest; capturing a tiny
# logo is harmless (LLM returns a short description that never matches queries).

# Section names that indicate exercise/Q&A content — tagged is_exercise=True
# so they can be routed to a separate Qdrant collection and excluded from
# theory retrieval queries.
EXERCISE_SECTION_NAMES = frozenset({
    "questions",
    "exercises",
    "exercise",
    "answers",
    "solutions",
    "problems",
    "further reading",
    "bibliography",
    "summary questions",
    "problems and solutions",
    "practice problems",
    "review questions",
    "discussion questions",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_file_hash(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_outputs(out_text, out_tables, out_images, text_chunks, table_chunks, image_refs):
    for path, data, name in [
        (out_text,   text_chunks,  "text_chunks"),
        (out_tables, table_chunks, "table_chunks"),
        (out_images, image_refs,   "image_refs"),
    ]:
        assert path.exists(),           f"[FAIL] Missing: {path}"
        assert path.stat().st_size > 0, f"[FAIL] Empty file: {path}"
        with open(path, encoding="utf-8") as f:
            disk = json.load(f)
        assert len(disk) == len(data), (
            f"[FAIL] {name} count mismatch — disk:{len(disk)} memory:{len(data)}"
        )
    print("[OK] Data integrity verified.")


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_elements(converter, pdf_path: Path, pages: list[int]) -> list[dict]:
    all_elements = []
    total = len(pages)

    for i, page_no in enumerate(pages, 1):
        print(f"  [{i}/{total}] page {page_no}...", end="\r")
        try:
            result = converter.convert(pdf_path, page_range=(page_no, page_no))
        except Exception as exc:
            print(f"\n  Page {page_no} failed: {exc}")
            continue

        doc     = result.document
        pic_idx = 0

        for item, level in doc.iterate_items():
            item_type = type(item).__name__
            item_page = None
            if hasattr(item, "prov") and item.prov:
                item_page = item.prov[0].page_no if item.prov else None

            if item_type == "PictureItem":
                image_path = ""
                try:
                    image = item.get_image(doc)
                    if image:
                        filename   = f"page{item_page}_img{pic_idx}.png"
                        saved_path = IMAGES_DIR / filename
                        image.save(saved_path)
                        image_path = str(saved_path)
                        print(f"\n  Saved image: {filename} ({image.width}x{image.height}px)")
                except Exception as exc:
                    print(f"\n  Image save failed p{item_page}: {exc}")
                pic_idx += 1

                all_elements.append({
                    "page":       item_page,
                    "level":      level,
                    "type":       item_type,
                    "text":       "",
                    "label":      "",
                    "image_path": image_path,
                })
                continue

            text = ""
            if hasattr(item, "text"):
                text = item.text or ""
            elif hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(doc)
                except TypeError:
                    text = item.export_to_markdown()

            all_elements.append({
                "page":       item_page,
                "level":      level,
                "type":       item_type,
                "text":       text,
                "label":      str(item.label) if hasattr(item, "label") else "",
                "image_path": "",
            })

    print()
    return all_elements


# ── PyMuPDF supplementary formula-text pass ──────────────────────────────────

def _is_formula_text(text: str) -> bool:
    """Heuristic: does this short block look like a formula rather than a heading?"""
    t = text.strip()
    if not t or len(t) > 250:
        return False
    math_chars = set("/×÷−±∑∫√≤≥≠≈∞·")
    math_kws   = {"fcf","fcff","wacc","ebit","ebitda","npv","irr","roe",
                  "roce","capm","eps","ev","tv","nopat","dscr","ltv"}
    has_math_char  = any(c in math_chars for c in t)
    has_eq_frac    = "=" in t and ("/" in t or any(c in math_chars for c in t))
    has_kw_eq      = any(k in t.lower() for k in math_kws) and "=" in t
    short_fragment = (len(t) < 30 and "\n" in t and
                      any(k in t.lower() for k in math_kws))
    return has_math_char or has_eq_frac or has_kw_eq or short_fragment


def extract_formula_text(
    pdf_path: Path,
    pages: list[int],
    docling_elements: list[dict],
    file_hash: str,
) -> list[dict]:
    """
    PyMuPDF pass: extract formula/equation text blocks that Docling dropped.
    Only adds blocks NOT already captured by Docling and that look mathematical.
    """
    try:
        import pymupdf
    except ImportError:
        print("  [formula pass] pymupdf not installed — skipping")
        return []

    # Page → section from Docling so formula items get correct section metadata
    page_section:     dict[int, str]  = {}
    page_is_exercise: dict[int, bool] = {}
    cur_sec = ""
    is_ex   = False
    for el in docling_elements:
        pg = el.get("page")
        if el["type"] == "SectionHeaderItem":
            cur_sec = el.get("text", "")
            is_ex   = cur_sec.lower().strip() in EXERCISE_SECTION_NAMES
        if pg:
            page_section[pg]     = cur_sec
            page_is_exercise[pg] = is_ex

    # Fingerprints of what Docling already captured (first 25 chars, no spaces)
    docling_fps: dict[int, set[str]] = {}
    for el in docling_elements:
        pg   = el.get("page")
        text = el.get("text", "").strip()
        if pg and text:
            fp = text[:25].lower().replace(" ", "")
            docling_fps.setdefault(pg, set()).add(fp)

    extras: list[dict] = []
    doc = pymupdf.open(str(pdf_path))

    for page_no in pages:
        try:
            page = doc[page_no - 1]
        except Exception:
            continue
        for block in page.get_text("blocks"):
            if len(block) < 7 or block[6] != 0:   # skip image blocks
                continue
            text = block[4].strip()
            if not text or len(text) < 4:
                continue
            fp = text[:25].lower().replace(" ", "")
            if fp in docling_fps.get(page_no, set()):
                continue                            # already captured by Docling
            if not _is_formula_text(text):
                continue
            # Collapse newlines inside formula fragments into spaces so the
            # chunk reads more naturally when embedded (e.g. "V\nFCFF\nk" → "V FCFF k")
            text = " ".join(text.split())
            extras.append({
                "page":            page_no,
                "type":            "FormulaItem",
                "level":           1,
                "section":         page_section.get(page_no, ""),
                "text":            text,
                "content_type":    "text",
                "is_front_matter": page_no in FRONT_MATTER_PAGES,
                "is_exercise":     page_is_exercise.get(page_no, False),
                "file_hash":       file_hash,
            })

    doc.close()
    if extras:
        print(f"  PyMuPDF formula pass: +{len(extras)} formula text block(s) added")
    return extras


# ── Separation ────────────────────────────────────────────────────────────────

def separate(elements: list[dict], file_hash: str):
    text_chunks:  list[dict] = []
    table_chunks: list[dict] = []
    image_refs:   list[dict] = []

    current_section    = ""
    is_exercise_section = False
    context_buffer: deque[str] = deque(maxlen=TABLE_CONTEXT_LINES)

    for item in elements:
        page      = item.get("page")
        item_type = item["type"]
        text      = item.get("text", "").strip()
        level     = item.get("level", 1)
        is_front  = page in FRONT_MATTER_PAGES

        # Skip table cell fragments
        if item_type == "TextItem" and level == 3:
            continue

        # Track section header + detect exercise sections
        if item_type == "SectionHeaderItem":
            current_section     = text
            is_exercise_section = current_section.lower().strip() in EXERCISE_SECTION_NAMES

        # Images → image_refs
        if item_type == "PictureItem":
            image_refs.append({
                "page":            page,
                "section":         current_section,
                "image_path":      item.get("image_path", ""),
                "description":     "",
                "content_type":    "image_description",
                "is_front_matter": is_front,
                "is_exercise":     is_exercise_section,
                "file_hash":       file_hash,
            })
            continue

        if not text:
            continue

        # Tables → table_chunks with context
        if item_type == "TableItem":
            context_before = "\n".join(context_buffer)
            table_chunks.append({
                "page":            page,
                "section":         current_section,
                "context_before":  context_before,
                "intro_text":      context_before,
                "table_markdown":  text,
                "content_type":    "table",
                "is_front_matter": is_front,
                "is_exercise":     is_exercise_section,
                "file_hash":       file_hash,
            })
            continue

        # All other items → text_chunks
        text_chunks.append({
            "page":            page,
            "type":            item_type,
            "level":           level,
            "section":         current_section,
            "text":            text,
            "content_type":    "text",
            "is_front_matter": is_front,
            "is_exercise":     is_exercise_section,
            "file_hash":       file_hash,
        })

        if item_type in ("TextItem", "ListItem", "SectionHeaderItem") and len(text) > 20:
            context_buffer.append(text)

    return text_chunks, table_chunks, image_refs


# ── Checkpoint helpers ────────────────────────────────────────────────────────

CHECKPOINT_FILE = OUT_DIR / "ingest_checkpoint.json"
BATCH_SIZE      = 50   # pages per batch — save progress after every 50 pages


def _load_checkpoint() -> int:
    """Return the last successfully completed page, or 0 if no checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            last = data.get("last_page", 0)
            print(f"  Checkpoint found — resuming from page {last + 1}")
            return last
        except Exception:
            pass
    return 0


def _save_checkpoint(last_page: int) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps({"last_page": last_page}, indent=2),
        encoding="utf-8",
    )


def _append_chunks(
    text_chunks: list, table_chunks: list, image_refs: list
) -> None:
    """Append this batch's output to the three JSON files on disk."""
    out_text   = OUT_DIR / "text_chunks.json"
    out_tables = OUT_DIR / "table_chunks.json"
    out_images = OUT_DIR / "image_refs.json"

    for path, new_data in [
        (out_text,   text_chunks),
        (out_tables, table_chunks),
        (out_images, image_refs),
    ]:
        existing: list = []
        if path.exists() and path.stat().st_size > 0:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
        merged = existing + new_data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing PDF file hash...")
    file_hash = compute_file_hash(PDF_PATH)
    print(f"  SHA256: {file_hash[:16]}...")

    # ── Checkpoint: skip already-processed pages ──────────────────────────────
    last_done = _load_checkpoint()

    if SAMPLE_ONLY:
        all_pages = list(range(START_PAGE, END_PAGE + 1))
        print(f"Mode: BATCH — pages {START_PAGE}–{END_PAGE}")
    else:
        all_pages = list(range(1, 10000))
        print("Mode: FULL DOCUMENT — this will take ~4 hours")

    # Filter out already-completed pages
    pages = [p for p in all_pages if p > last_done]
    if not pages:
        print("All pages already processed. Delete ingest_checkpoint.json to restart.")
        return

    if last_done:
        print(f"  Skipping pages 1–{last_done} (already done)")
    print(f"  Processing {len(pages)} pages in batches of {BATCH_SIZE}\n")

    # ── Load Docling ──────────────────────────────────────────────────────────
    print("Loading Docling...")
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr                  = False
    pipeline_options.do_table_structure      = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale            = 0.5   # 1.0 caused std::bad_alloc cascade from p180

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # ── Process in batches — save after each one ──────────────────────────────
    total_text = total_tables = total_images = 0
    num_batches = (len(pages) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        batch = pages[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        print(f"\nBatch {batch_idx + 1}/{num_batches}  "
              f"(pages {batch[0]}–{batch[-1]})  ━━━━━━━━━━━━━━━━━━━━━━━━━━")

        try:
            # Docling extraction
            elements = extract_elements(converter, PDF_PATH, batch)

            # PyMuPDF supplementary formula text pass
            formula_elements = extract_formula_text(PDF_PATH, batch, elements, file_hash)
            all_elements     = elements + formula_elements

            # Separate into text / table / image
            text_chunks, table_chunks, image_refs = separate(all_elements, file_hash)

            # Flush to disk immediately
            _append_chunks(text_chunks, table_chunks, image_refs)

            # Save checkpoint — this batch is done
            _save_checkpoint(batch[-1])

            saved_imgs = sum(1 for r in image_refs if r.get("image_path"))
            total_text   += len(text_chunks)
            total_tables += len(table_chunks)
            total_images += len(image_refs)

            print(f"  ✓ text={len(text_chunks)}  tables={len(table_chunks)}  "
                  f"images={len(image_refs)} ({saved_imgs} saved)  "
                  f"checkpoint → page {batch[-1]}")

        except Exception as exc:
            print(f"\n  ✗ BATCH FAILED at pages {batch[0]}–{batch[-1]}: {exc}")
            print(f"    Progress saved up to page {batch[0] - 1}.")
            print(f"    Fix the issue then re-run — it will resume from page {batch[0]}.")
            sys.exit(1)

    # ── All batches done — clean up checkpoint ────────────────────────────────
    CHECKPOINT_FILE.unlink(missing_ok=True)

    print(f"\n{'='*55}")
    print(f"  Ingestion complete!")
    print(f"  text_chunks  : {total_text}")
    print(f"  table_chunks : {total_tables}")
    print(f"  image_refs   : {total_images}")
    print(f"{'='*55}")
    print(f"\nNext steps:")
    print(f"  uv run python scripts/describe_images.py")
    print(f"  uv run python scripts/build_index.py")


if __name__ == "__main__":
    main()
