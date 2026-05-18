#!/usr/bin/env python3
"""Bills (sansad-side) OCR slow lane — backfills scanned PDFs that the
main scraper skipped.

The main `build_bills_sansad.py` scraper uses pypdf, which returns empty
text for scanned-only PDFs (no embedded text layer). Older bills (pre-
2000s especially) include a meaningful pool of scanned-only PDFs that
produce a `text/<composite_id>.pypdf-empty` marker. This script is the
catch-up: it walks records.json for those markers (or their bundled
equivalent in markers.json), runs the underlying PDFs through tesseract
OCR via pdf2image, and writes the result to `text/<composite_id>.txt`.

Why a separate runner: tesseract is slow (~2-3s/page at 150 dpi). A
typical 200-page bill PDF takes ~7 minutes. Folding that into the
scrape would eat the budget. This runs on demand (workflow_dispatch)
with sprint inputs, with the standard in-flight checkpointing + chain-
if-backlog-remains pattern.

Failures (OCR returned nothing, severely degraded scan, encrypted, etc.)
get a `text/<composite_id>.ocr-failed` tombstone so we don't retry them
every run. The tombstone is invisible to manifest / bundle / index
(those scan for `*.txt` only).

Structurally a near-twin of build_cag_ocr.py — same checkpoint cadence,
same defaults, same flat text/<cid>.<suffix> layout. Differences: paths
point to docs/bills/, composite_id is a string ("100_1952_L") rather
than an int, and PDF download goes through bills/sansad/scraper.py's
download_canonical_pdf which picks the highest-priority bill stage
(passed-in-both-houses → LS → RS → introduced) from the record's URLs.

System deps:
  apt-get install tesseract-ocr poppler-utils
  (or on macOS: brew install tesseract poppler)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from parliamentwatch_text_shards import load_markers
from bills.sansad.scraper import (
    download_canonical_pdf as _download_canonical_pdf,
    CANONICAL_PDF_FIELDS,
    RateLimited,
)

# ── Paths ───────────────────────────────────────────────────────────────────

ASSETS    = ROOT / "docs"
DOCS      = ASSETS / "bills"
TEXT_DIR  = DOCS / "text"
PDFS_DIR  = DOCS / "pdfs"
DOCS.mkdir(parents=True, exist_ok=True)

RECORDS_JSON = DOCS / "records.json"


# ── In-flight checkpointing ────────────────────────────────────────────────

CHECKPOINT_EVERY_N = int(os.environ.get("CHECKPOINT_EVERY_N", "2"))
CHECKPOINT_EVERY_S = int(os.environ.get("CHECKPOINT_EVERY_S", "300"))


def _git(*args) -> tuple[int, str, str]:
    p = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return p.returncode, p.stdout, p.stderr


def checkpoint_commit(message: str, paths: list[str]) -> bool:
    rc, _, err = _git("add", "--", *paths)
    if rc != 0:
        print(f"  [checkpoint] git add failed: {err.strip() or 'unknown'}")
        return False
    rc, _, _ = _git("diff", "--cached", "--quiet")
    if rc == 0:
        return True
    rc, _, err = _git("commit", "-m", message)
    if rc != 0:
        print(f"  [checkpoint] git commit failed: {err.strip()}")
        return False
    rc, _, err = _git("pull", "--rebase", "origin", "main")
    if rc != 0:
        print(f"  [checkpoint] git pull --rebase failed: {err.strip()} — aborting rebase")
        _git("rebase", "--abort")
        return False
    rc, _, err = _git("push")
    if rc != 0:
        print(f"  [checkpoint] git push failed: {err.strip()}")
        return False
    print(f"  [checkpoint] pushed: {message}")
    return True


# ── Per-run budget ─────────────────────────────────────────────────────────

MAX_OCR_PER_RUN = int(os.environ.get("MAX_OCR_PER_RUN", "5"))
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "2700"))   # 45 min
MAX_PAGES_PER_PDF = int(os.environ.get("MAX_PAGES_PER_PDF", "300"))
RENDER_DPI = int(os.environ.get("RENDER_DPI", "150"))
TESSERACT_LANGS = os.environ.get("TESSERACT_LANGS", "eng+hin")


# ── Candidate finder ────────────────────────────────────────────────────────

def _has_any_canonical_pdf(record: dict) -> bool:
    return any(record.get(f) for f in CANONICAL_PDF_FIELDS)


def find_ocr_candidates(records: list[dict]) -> list[dict]:
    """Walk records for any bill with a canonical PDF that has neither a
    text/<cid>.txt nor an .ocr-failed tombstone AND has a .pypdf-empty
    marker (i.e. pypdf already confirmed scanned-only).

    Returns list of record dicts sorted newest-first (by billYear desc,
    then billNumber desc — same order build_bills_sansad.select_candidates
    uses).

    Marker source-of-truth: docs/bills/markers.json (post-derive
    consolidation), with on-disk sidecars as a fallback for the brief
    window between a fresh extract commit and the next derive run.
    """
    bundled_markers = load_markers(DOCS)

    candidates: list[dict] = []
    for r in records:
        cid = r.get("compositeId")
        if not cid:
            continue
        if not _has_any_canonical_pdf(r):
            continue
        # Permanent skips: already extracted (.txt) or tombstoned.
        if (TEXT_DIR / f"{cid}.txt").exists():
            continue
        if bundled_markers.get(cid) == "ocr-failed":
            continue
        if (TEXT_DIR / f"{cid}.ocr-failed").exists():
            continue
        # Positive signal: pypdf-empty marker (in either source).
        is_pypdf_empty = (
            bundled_markers.get(cid) == "pypdf-empty"
            or (TEXT_DIR / f"{cid}.pypdf-empty").exists()
        )
        if not is_pypdf_empty:
            continue   # let backfill try pypdf first
        candidates.append(r)
    candidates.sort(
        key=lambda r: (r.get("billYear") or 0, r.get("billNumber") or ""),
        reverse=True,
    )
    return candidates


def ocr_pdf(pdf_path: Path, *, deadline: float) -> str | None:
    """OCR one PDF. Returns extracted text on success, None if OCR produced
    nothing useful, or raises TimeoutError if deadline exceeded mid-render.
    """
    from pdf2image import convert_from_path  # type: ignore
    import pytesseract  # type: ignore

    parts: list[str] = []
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


def _emit_chain_outputs(processed: int, remaining: int) -> None:
    gh_output_path = os.environ.get("GITHUB_OUTPUT")
    if gh_output_path:
        try:
            with open(gh_output_path, "a", encoding="utf-8") as f:
                f.write(f"processed={processed}\n")
                f.write(f"remaining={remaining}\n")
            print(f"  workflow outputs: processed={processed} remaining={remaining}")
        except OSError as e:
            print(f"  (couldn't write GITHUB_OUTPUT: {e})")
    else:
        print(f"  workflow outputs (local run): processed={processed} remaining={remaining}")


def _load_records() -> list[dict]:
    """Load records.json as a list. Tolerates both the dict-keyed-by-cid
    shape that build_bills_sansad.py persists and a list-shaped fallback.
    """
    if not RECORDS_JSON.exists():
        return []
    with open(RECORDS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Either {"records": [...]} wrapper or {<cid>: <record>, ...} flat dict.
        if "records" in data and isinstance(data["records"], list):
            return data["records"]
        return list(data.values())
    return []


def main():
    print("=== ParliamentWatch Bills OCR slow-lane ===")
    print(f"DOCS               : {DOCS}")
    print(f"MAX_OCR_PER_RUN    : {MAX_OCR_PER_RUN}")
    print(f"MAX_RUN_SECONDS    : {MAX_RUN_SECONDS}")
    print(f"MAX_PAGES_PER_PDF  : {MAX_PAGES_PER_PDF}")
    print(f"RENDER_DPI         : {RENDER_DPI}")
    print(f"TESSERACT_LANGS    : {TESSERACT_LANGS}")

    import shutil
    if not shutil.which("tesseract"):
        print("ERROR: tesseract not on PATH. apt-get install tesseract-ocr (CI) "
              "or brew install tesseract (local).")
        sys.exit(1)

    print("\n[1/3] Finding OCR candidates (scanned-only PDFs with no text yet)...")
    records = _load_records()
    candidates = find_ocr_candidates(records)
    print(f"  candidates: {len(candidates)} (out of {len(records)} total bills)")
    if not candidates:
        print("  nothing to do — all bills already have text or are tombstoned")
        _emit_chain_outputs(0, 0)
        return

    target = candidates[:MAX_OCR_PER_RUN]
    print(f"  budget: OCR'ing up to {len(target)} this run (newest year first)")
    print(f"  note: pdfs/ is gitignored — slow-lane downloads each PDF before OCR.")

    deadline = time.monotonic() + MAX_RUN_SECONDS
    succeeded: list[str] = []
    failed: list[str] = []
    timed_out: list[str] = []

    last_checkpoint_at = time.monotonic()
    units_since_checkpoint = 0

    print("\n[2/3] Downloading + OCR'ing...")
    for i, record in enumerate(target, start=1):
        cid = record["compositeId"]
        if time.monotonic() > deadline:
            print(f"  [BUDGET] wall-clock budget hit after {len(succeeded)} successes")
            break
        try:
            pdf_path_str, field = _download_canonical_pdf(record, str(PDFS_DIR))
        except RateLimited as rl:
            print(f"    RATE-LIMITED ({rl}) — stopping this run")
            break
        if not pdf_path_str:
            print(f"  [{i}/{len(target)}] {cid}: no downloadable canonical PDF — skipping")
            continue
        pdf_path = Path(pdf_path_str)
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"  [{i}/{len(target)}] {cid} ({field}, {size_mb:.1f} MB) — running OCR...", flush=True)
        t0 = time.time()
        try:
            text = ocr_pdf(pdf_path, deadline=deadline)
        except TimeoutError as e:
            elapsed = time.time() - t0
            print(f"    TIMEOUT after {elapsed:.0f}s ({e}) — leaving for next run")
            timed_out.append(cid)
            break
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    OCR FAILED after {elapsed:.0f}s: {e}")
            tombstone = TEXT_DIR / f"{cid}.ocr-failed"
            tombstone.parent.mkdir(parents=True, exist_ok=True)
            tombstone.write_text(f"OCR error: {e}\n", encoding="utf-8")
            failed.append(cid)
            units_since_checkpoint += 1
        else:
            elapsed = time.time() - t0
            if text is None:
                print(f"    EMPTY OCR after {elapsed:.0f}s — writing tombstone, won't retry")
                tombstone = TEXT_DIR / f"{cid}.ocr-failed"
                tombstone.parent.mkdir(parents=True, exist_ok=True)
                tombstone.write_text("OCR returned empty\n", encoding="utf-8")
                failed.append(cid)
                units_since_checkpoint += 1
            else:
                text_path = TEXT_DIR / f"{cid}.txt"
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(text, encoding="utf-8")
                kb = len(text) / 1024
                print(f"    OK in {elapsed:.0f}s · wrote {kb:.0f} KB to text/{cid}.txt")
                succeeded.append(cid)
                units_since_checkpoint += 1

        now = time.monotonic()
        if (units_since_checkpoint >= CHECKPOINT_EVERY_N or
            (units_since_checkpoint > 0 and now - last_checkpoint_at >= CHECKPOINT_EVERY_S)):
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            checkpoint_commit(
                f"Auto-checkpoint Bills OCR data (succeeded={len(succeeded)}, failed={len(failed)} this run) [{ts}]",
                ["docs/bills/text/"],
            )
            units_since_checkpoint = 0
            last_checkpoint_at = now

    if units_since_checkpoint > 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        checkpoint_commit(
            f"Auto-checkpoint Bills OCR data (final, succeeded={len(succeeded)}, failed={len(failed)} this run) [{ts}]",
            ["docs/bills/text/"],
        )

    print(f"\n  succeeded: {len(succeeded)} · failed (tombstoned): {len(failed)} · timed out: {len(timed_out)}")

    processed = len(succeeded) + len(failed)
    try:
        remaining = len(find_ocr_candidates(records))
    except Exception as e:
        print(f"  (couldn't recount candidates for chain decision: {e})")
        remaining = -1
    _emit_chain_outputs(processed, remaining)

    print("\nDone.")


if __name__ == "__main__":
    main()
