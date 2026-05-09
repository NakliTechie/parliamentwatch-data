#!/usr/bin/env python3
"""CAG OCR slow lane — backfills scanned PDFs that the main scraper skipped.

The main `build_cag.py` scraper uses pypdf, which returns empty text for
scanned-only PDFs (no embedded text layer). It logs the skip and moves on.
This script is the catch-up: it walks `docs/cag/pdfs/` for PDFs without a
corresponding `docs/cag/text/<id>.txt`, runs them through tesseract OCR
via pdf2image, and writes the result.

Why a separate runner: tesseract is slow (~2-3s/page at 150 dpi). A 200-
page scanned PDF takes ~8 minutes. Folding that into the daily cron would
eat the budget and starve the network-bound scraper. This runs weekly,
processes a small batch per run, slowly catches up across the corpus.

Failures (OCR returned nothing, severely degraded scan, encrypted, etc.)
get a `text/<id>.ocr-failed` tombstone so we don't retry them every week.
The tombstone is invisible to manifest / bundle / index (those scan for
`*.txt` only).

System deps:
  apt-get install tesseract-ocr poppler-utils
  (or on macOS: brew install tesseract poppler)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Paths ───────────────────────────────────────────────────────────────────

ASSETS    = ROOT / "docs"
DOCS      = ASSETS / "cag"
TEXT_DIR  = DOCS / "text"
PDFS_DIR  = DOCS / "pdfs"
DOCS.mkdir(parents=True, exist_ok=True)

REPORTS_JSON  = DOCS / "reports.json"
MANIFEST_JSON = DOCS / "manifest.json"
META_JSON     = DOCS / "meta.json"
BUNDLE_JSON   = DOCS / "search-bundle.json"
INDEX_JSON    = DOCS / "search-index.json"

# ── Per-run budget ─────────────────────────────────────────────────────────

# How many scanned PDFs to OCR per weekly run. Default 5 — at ~8 min each
# for a typical audit report (~200 pages), that's ~40 minutes. Stays well
# under the 60-min GH Actions timeout. Bump via workflow_dispatch input
# for one-off catch-up runs.
MAX_OCR_PER_RUN = int(os.environ.get("MAX_OCR_PER_RUN", "5"))

# Hard wall-clock cap. After this, abort whatever's in flight (don't write
# partial text — we'd rather skip than lie).
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "2700"))   # 45 min

# Per-PDF page cap. CAG audit reports are typically 100-300 pages; a few
# range to 700+. Cap stops one bloated PDF from eating the entire run
# budget. Pages beyond this get OCR'd-then-truncated; the app shows a
# "first N pages OCR'd" disclaimer on those reports.
MAX_PAGES_PER_PDF = int(os.environ.get("MAX_PAGES_PER_PDF", "300"))

# Render DPI. 150 is the sweet spot for typeset government PDFs:
# tesseract gets 95%+ on clean English text at this resolution. Lower
# (100) is faster but quality drops noticeably; higher (300) is slower
# with marginal gains.
RENDER_DPI = int(os.environ.get("RENDER_DPI", "150"))

# Tesseract languages. eng + hin covers most CAG reports (some have Hindi
# sections). Add more language packs in the workflow's apt-install if
# specific corpora need them.
TESSERACT_LANGS = os.environ.get("TESSERACT_LANGS", "eng+hin")


def find_ocr_candidates() -> list[int]:
    """Walk pdfs/ for any <id>.pdf with no corresponding text/<id>.txt
    and no text/<id>.ocr-failed tombstone. Returns ids sorted newest-first
    (highest id wins — CAG IDs are monotonic)."""
    if not PDFS_DIR.exists():
        return []
    candidates: list[int] = []
    for pdf in PDFS_DIR.glob("*.pdf"):
        try:
            rid = int(pdf.stem)
        except ValueError:
            continue
        text_path = TEXT_DIR / f"{rid}.txt"
        tombstone = TEXT_DIR / f"{rid}.ocr-failed"
        if text_path.exists() or tombstone.exists():
            continue
        candidates.append(rid)
    candidates.sort(reverse=True)
    return candidates


def ocr_pdf(pdf_path: Path, *, deadline: float) -> str | None:
    """OCR one PDF. Returns extracted text on success, None if OCR produced
    nothing useful, or raises TimeoutError if deadline exceeded mid-render.

    Imports pdf2image and pytesseract lazily so that consumers (e.g. the
    main daily cron) don't need them installed.
    """
    from pdf2image import convert_from_path  # type: ignore
    import pytesseract  # type: ignore

    parts: list[str] = []
    # Render in batches to keep memory bounded — a 300-page PDF rendered
    # at 150 dpi as a single list is ~3 GB of PIL Images.
    BATCH = 10
    n_pages_done = 0
    page = 1
    while page <= MAX_PAGES_PER_PDF:
        if time.monotonic() > deadline:
            raise TimeoutError(f"deadline exceeded after {n_pages_done} pages")
        last = min(page + BATCH - 1, MAX_PAGES_PER_PDF)
        try:
            images = convert_from_path(
                str(pdf_path),
                dpi=RENDER_DPI,
                first_page=page,
                last_page=last,
            )
        except Exception as e:
            # Most likely cause: pdf2image past the actual last page.
            # Treat as "we've reached EOF" and return what we have.
            if "PDFPageCountError" in type(e).__name__ or "pdftoppm" in str(e):
                break
            raise
        if not images:
            break
        for img in images:
            if time.monotonic() > deadline:
                raise TimeoutError(f"deadline exceeded after {n_pages_done} pages")
            text = pytesseract.image_to_string(img, lang=TESSERACT_LANGS)
            if text:
                parts.append(text)
            n_pages_done += 1
        page = last + 1

    full = "\n\n".join(parts).strip()
    if not full:
        return None
    return full


def main():
    print("=== ParliamentWatch CAG OCR slow-lane ===")
    print(f"DOCS               : {DOCS}")
    print(f"MAX_OCR_PER_RUN    : {MAX_OCR_PER_RUN}")
    print(f"MAX_RUN_SECONDS    : {MAX_RUN_SECONDS}")
    print(f"MAX_PAGES_PER_PDF  : {MAX_PAGES_PER_PDF}")
    print(f"RENDER_DPI         : {RENDER_DPI}")
    print(f"TESSERACT_LANGS    : {TESSERACT_LANGS}")

    # Sanity-check tesseract is on PATH. Fail loudly if missing — the
    # workflow installs it via apt; a missing binary means the workflow
    # is misconfigured.
    import shutil
    if not shutil.which("tesseract"):
        print("ERROR: tesseract not on PATH. apt-get install tesseract-ocr (CI) "
              "or brew install tesseract (local).")
        sys.exit(1)

    print("\n[1/3] Finding OCR candidates (PDFs in pdfs/ with no .txt and no .ocr-failed)...")
    candidates = find_ocr_candidates()
    print(f"  candidates: {len(candidates)}")
    if not candidates:
        print("  nothing to do — all scanned PDFs already OCR'd or tombstoned")
        return

    target = candidates[:MAX_OCR_PER_RUN]
    print(f"  budget: OCR'ing up to {len(target)} this run (newest first)")

    deadline = time.monotonic() + MAX_RUN_SECONDS
    succeeded: list[int] = []
    failed: list[int] = []
    timed_out: list[int] = []

    print("\n[2/3] Running OCR...")
    for i, rid in enumerate(target, start=1):
        if time.monotonic() > deadline:
            print(f"  [BUDGET] wall-clock budget hit after {len(succeeded)} successes")
            break
        pdf_path = PDFS_DIR / f"{rid}.pdf"
        if not pdf_path.exists():
            print(f"  id={rid}: pdf missing (likely pruned) — skipping")
            continue
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"  [{i}/{len(target)}] id={rid} ({size_mb:.1f} MB)...", flush=True)
        t0 = time.time()
        try:
            text = ocr_pdf(pdf_path, deadline=deadline)
        except TimeoutError as e:
            elapsed = time.time() - t0
            print(f"    TIMEOUT after {elapsed:.0f}s ({e}) — leaving for next run")
            timed_out.append(rid)
            break
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    OCR FAILED after {elapsed:.0f}s: {e}")
            tombstone = TEXT_DIR / f"{rid}.ocr-failed"
            tombstone.parent.mkdir(parents=True, exist_ok=True)
            tombstone.write_text(f"OCR error: {e}\n", encoding="utf-8")
            failed.append(rid)
            continue
        elapsed = time.time() - t0
        if text is None:
            print(f"    EMPTY OCR after {elapsed:.0f}s — writing tombstone, won't retry")
            tombstone = TEXT_DIR / f"{rid}.ocr-failed"
            tombstone.parent.mkdir(parents=True, exist_ok=True)
            tombstone.write_text("OCR returned empty\n", encoding="utf-8")
            failed.append(rid)
            continue
        text_path = TEXT_DIR / f"{rid}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")
        kb = len(text) / 1024
        print(f"    OK in {elapsed:.0f}s · wrote {kb:.0f} KB to text/{rid}.txt")
        succeeded.append(rid)

    print(f"\n  succeeded: {len(succeeded)} · failed (tombstoned): {len(failed)} · timed out: {len(timed_out)}")

    print("\n[3/3] Rebuilding manifest + bundle + index + meta...")
    # Re-run the same builders the daily cron uses. Lazy-import so we
    # don't pay for the heavy pypdf path in this fast OCR-only runner.
    from build_cag import (
        load_existing_reports, build_manifest, build_search_bundle,
        build_search_index, write_meta,
    )
    reports = load_existing_reports()
    manifest = build_manifest()
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    bundle_stats = build_search_bundle(reports)
    index_stats = build_search_index()

    # Reuse write_meta but stuff our OCR-run stats into extract_stats so
    # they show up in meta.json. The daily cron's next run will overwrite
    # extract_stats with its own; that's fine — our data just lives until
    # then for visibility.
    extract_stats = {
        "extracted":  succeeded,
        "failed":     failed + timed_out,
        "rate_limited": False,
        "budget_hit":   len(timed_out) > 0,
        "candidates_total": len(candidates),
        "ocr_run":    True,
    }
    n_with_text = len(manifest["texts"])
    write_meta(
        total_reports=len(reports),
        total_with_text=n_with_text,
        extract_stats=extract_stats,
        bundle_stats=bundle_stats,
        index_stats=index_stats,
    )
    print(f"  manifest: {n_with_text} reports with extracted text")
    if bundle_stats:
        print(f"  search-bundle: {bundle_stats['total']} entries")
    if index_stats:
        print(f"  search-index: {index_stats['report_count']} docs, vocab {index_stats['vocab_size']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
