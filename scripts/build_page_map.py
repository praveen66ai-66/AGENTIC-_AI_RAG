"""
Build an exact PDF-page → book-page mapping using PyMuPDF.

The textbook PDF has front-matter (cover, TOC, preface) before page 1 of
the printed book. A static offset (e.g. -16) is approximate and wrong for
introductory pages. This script reads the actual page number printed in each
page's header/footer and saves the exact mapping to JSON.

Reads:  data/documents/CoreCourseFinancialAccounting.pdf
Writes: data/processed/page_map.json  ← {pdf_page: book_page}

Usage:
    uv run python scripts/build_page_map.py

After running this, search.py loads the map at startup and every citation
shows the exact book page number the user sees in the PDF viewer.
"""

import json
import re
from pathlib import Path

PDF_PATH = Path("data/documents/CoreCourseFinancialAccounting.pdf")
OUT_PATH = Path("data/processed/page_map.json")


def _extract_book_page(page) -> int | None:
    """
    Read the printed page number from the page header region.
    The Vernimmen textbook prints the page number in a running header at the
    top of every page, usually alongside the chapter title.
    Returns the integer page number, or None if not found.
    """
    h = page.rect.height
    w = page.rect.width

    # Check top 12% of page (header) and bottom 8% (footer)
    for clip in [
        (0, 0, w, h * 0.12),           # header
        (0, h * 0.92, w, h),           # footer
    ]:
        import pymupdf
        text = page.get_text("text", clip=pymupdf.Rect(*clip)).strip()
        # Find 2–4 digit numbers in range 1–1200
        numbers = [int(n) for n in re.findall(r'\b(\d{2,4})\b', text)
                   if 1 <= int(n) <= 1200]
        if numbers:
            return numbers[0]
    return None


def main():
    import pymupdf

    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening {PDF_PATH} ...")
    doc      = pymupdf.open(str(PDF_PATH))
    total    = len(doc)
    page_map: dict[int, int] = {}
    skipped  = 0

    print(f"Scanning {total} pages for printed page numbers...")

    for i in range(total):
        pdf_page  = i + 1
        page      = doc[i]
        book_page = _extract_book_page(page)

        if book_page:
            page_map[pdf_page] = book_page
        else:
            skipped += 1

        if pdf_page % 100 == 0:
            print(f"  {pdf_page}/{total} pages scanned...", end="\r")

    doc.close()
    print(f"\nScanned {total} pages")
    print(f"  Mapped  : {len(page_map)} pages with exact book page")
    print(f"  Skipped : {skipped} pages (front matter / blank / could not extract)")

    # Show a sample to verify accuracy
    print("\nSample (PDF page → book page):")
    samples = sorted(page_map.items())
    for pdf_pg, book_pg in samples[::total // 20][:10]:
        offset = pdf_pg - book_pg
        print(f"  PDF {pdf_pg:4d} → book page {book_pg:4d}  (offset {offset:+d})")

    # Save
    # JSON keys must be strings
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in page_map.items()}, f, indent=2)

    print(f"\nSaved to {OUT_PATH}")
    print("Now restart the API — search.py loads this map at startup.")


if __name__ == "__main__":
    main()
