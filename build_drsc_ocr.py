#!/usr/bin/env python3
"""DRSC (Departmentally Related Standing Committees) OCR slow lane —
backfills scanned PDFs that the main scraper skipped.

The main `build_static.py` scraper uses pypdf, which returns empty text
for scanned-only PDFs (no embedded text layer). Older committee reports
(LS-14 / LS-15 era and earlier) include a meaningful pool of scanned-
only PDFs that produce a `text/<committee_key>/<file_id>.pypdf-empty`
marker. This script is the catch-up: it walks reports.json for those
markers (or their bundled equivalent in markers.json), runs the
underlying PDFs through tesseract OCR via pdf2image, and writes the
result to `text/<committee_key>/<file_id>.txt`.

Why a separate runner: tesseract is slow (~2-3s/page at 150 dpi).
A 300-page scanned committee report takes ~8 minutes. Folding that into
the daily cron would eat the budget and starve the network-bound
scraper. This runs on demand (workflow_dispatch) with sprint inputs,
with the standard in-flight checkpointing + chain-if-backlog-remains
pattern.

Failures (OCR returned nothing, severely degraded scan, encrypted, etc.)
get a `text/<committee_key>/<file_id>.ocr-failed` tombstone so we don't
retry them every run. The tombstone is invisible to manifest / bundle /
index (those scan for `*.txt` only).

Structurally a near-twin of build_fc_ocr.py — same checkpoint cadence,
same defaults, same committee-nested layout. Differences: paths point
to docs/drsc/, reports.json is keyed by `committee_key` (e.g.
"agriculture", "coal") rather than the smaller FC set of three, and
records include a `house` field (L/R) that we don't currently use for
identity (LS-only OCR coverage matches the scrape's current scope).

System deps:
  apt-get install tesseract-ocr poppler-utils
  (or on macOS: brew install tesseract poppler)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Paths ───────────────────────────────────────────────────────────────────

ASSETS    = ROOT / "docs"
DOCS      = ASSETS / "drsc"
TEXT_DIR  = DOCS / "text"
PDFS_DIR  = DOCS / "pdfs"
DOCS.mkdir(parents=True, exist_ok=True)

# Force scraper paths to docs/drsc/ before importing pdf_utils — without this
# config.PDFS_DIR defaults to data/pdfs/ and pdf_utils.download_pdf writes
# the PDF to a directory we never read from, triggering a FileNotFoundError
# at pdf_path.stat() below. Matches build_static.py:40.
os.environ["DATA_DIR"] = str(DOCS)

from parliamentwatch_text_shards import load_markers  # noqa: E402
from pdf_utils import download_pdf as _download_pdf, RateLimited  # noqa: E402

REPORTS_JSON = DOCS / "reports.json"


# ── file_id helper ─────────────────────────────────────────────────────────
#
# Same convention as build_static.py:_file_id. Inlined here to keep this
# module self-contained (build_static.py has top-level side effects on
# import we'd rather not pull in).

def _safe_num(report_num):
    if report_num is None:
        return None
    return str(report_num).replace("/", "-").replace(" ", "_")


def _file_id(lok_sabha, report_num) -> str | None:
    safe = _safe_num(report_num)
    if safe is None:
        return None
    if lok_sabha is None:
        return safe
    return f"LS{lok_sabha}_{safe}"


# ── In-flight checkpointing ────────────────────────────────────────────────

CHECKPOINT_EVERY_N = int(os.environ.get("CHECKPOINT_EVERY_N", "2"))
CHECKPOINT_EVERY_S = int(os.environ.get("CHECKPOINT_EVERY_S", "300"))


def _git(*args) -> tuple[int, str, str]:
    p = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return p.returncode, p.stdout, p.stderr


def checkpoint_commit(message: str, paths: list[str]) -> bool:
    """Stage `paths`, commit, pull-rebase, push. Best-effort; failures log
    but never abort the OCR loop. Caller's next checkpoint trigger retries.
    """
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

def find_ocr_candidates(
    reports: dict[str, list[dict]],
) -> list[tuple[str, int, int, str]]:
    """Walk reports.json for any record with a pdf_url that has neither a
    text/<cmt>/<fid>.txt nor an .ocr-failed tombstone AND has a
    .pypdf-empty marker (i.e. pypdf already confirmed scanned-only).

    Returns [(committee_key, lok_sabha, report_number, pdf_url), ...] sorted
    newest-first (highest LS, then highest report_number within LS).

    Marker source-of-truth: docs/drsc/markers.json (post-derive
    consolidation), with on-disk sidecars as a fallback for the brief
    window between a fresh extract commit and the next derive run.
    Matches the pattern in build_lc_ocr.py and build_fc_ocr.py.
    """
    bundled_markers = load_markers(DOCS)

    candidates: list[tuple[str, int, int, str]] = []
    for cmt, items in reports.items():
        cmt_text_dir = TEXT_DIR / cmt
        for r in items:
            pdf_url = r.get("pdf_url")
            if not pdf_url:
                continue
            ls = r.get("lok_sabha")
            num = r.get("report_number")
            if ls is None or num is None:
                continue
            fid = _file_id(int(ls), num)
            if fid is None:
                continue
            cid = f"{cmt}|{fid}"
            # Permanent skips: already extracted (.txt) or tombstoned.
            if (cmt_text_dir / f"{fid}.txt").exists():
                continue
            if bundled_markers.get(cid) == "ocr-failed":
                continue
            if (cmt_text_dir / f"{fid}.ocr-failed").exists():
                continue
            # Positive signal: pypdf-empty marker (in either source).
            is_pypdf_empty = (
                bundled_markers.get(cid) == "pypdf-empty"
                or (cmt_text_dir / f"{fid}.pypdf-empty").exists()
            )
            if not is_pypdf_empty:
                continue   # let the daily run try pypdf first
            candidates.append((cmt, int(ls), int(num), pdf_url))
    candidates.sort(key=lambda c: (-c[1], -c[2]))
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


def _load_existing_reports() -> dict[str, list[dict]]:
    """Load DRSC reports.json. Same shape as build_static.py expects:
    a committee-keyed dict of report lists. Returns empty dict on fresh
    install (shouldn't happen on a runner that just checked out main).
    """
    if not REPORTS_JSON.exists():
        return {}
    with open(REPORTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def main():
    print("=== ParliamentWatch DRSC OCR slow-lane ===")
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
    reports = _load_existing_reports()
    n_reports = sum(len(v) for v in reports.values())
    candidates = find_ocr_candidates(reports)
    print(f"  candidates: {len(candidates)} (out of {n_reports} total reports)")
    if not candidates:
        print("  nothing to do — all reports already have text or are tombstoned")
        _emit_chain_outputs(0, 0)
        return

    target = candidates[:MAX_OCR_PER_RUN]
    print(f"  budget: OCR'ing up to {len(target)} this run (newest LS first)")
    print(f"  note: pdfs/ is gitignored — slow-lane downloads each PDF before OCR.")

    deadline = time.monotonic() + MAX_RUN_SECONDS
    succeeded: list[tuple[str, int, int]] = []
    failed: list[tuple[str, int, int]] = []
    timed_out: list[tuple[str, int, int]] = []

    last_checkpoint_at = time.monotonic()
    units_since_checkpoint = 0

    print("\n[2/3] Downloading + OCR'ing...")
    for i, (cmt, ls, num, pdf_url) in enumerate(target, start=1):
        if time.monotonic() > deadline:
            print(f"  [BUDGET] wall-clock budget hit after {len(succeeded)} successes")
            break
        fid = _file_id(ls, num)
        cmt_pdfs_dir = PDFS_DIR / cmt
        pdf_path = cmt_pdfs_dir / f"{fid}.pdf"
        if not pdf_path.exists():
            print(f"  [{i}/{len(target)}] {cmt} {fid}: downloading {pdf_url}...", flush=True)
            try:
                got = _download_pdf(pdf_url, cmt, fid)
            except RateLimited as rl:
                print(f"    RATE-LIMITED ({rl}) — stopping this run")
                break
            if not got:
                print(f"    download failed — skipping (no tombstone; will retry next run)")
                continue
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"  [{i}/{len(target)}] {cmt} {fid} ({size_mb:.1f} MB) — running OCR...", flush=True)
        t0 = time.time()
        cmt_text_dir = TEXT_DIR / cmt
        try:
            text = ocr_pdf(pdf_path, deadline=deadline)
        except TimeoutError as e:
            elapsed = time.time() - t0
            print(f"    TIMEOUT after {elapsed:.0f}s ({e}) — leaving for next run")
            timed_out.append((cmt, ls, num))
            break
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    OCR FAILED after {elapsed:.0f}s: {e}")
            tombstone = cmt_text_dir / f"{fid}.ocr-failed"
            tombstone.parent.mkdir(parents=True, exist_ok=True)
            tombstone.write_text(f"OCR error: {e}\n", encoding="utf-8")
            failed.append((cmt, ls, num))
            units_since_checkpoint += 1
        else:
            elapsed = time.time() - t0
            if text is None:
                print(f"    EMPTY OCR after {elapsed:.0f}s — writing tombstone, won't retry")
                tombstone = cmt_text_dir / f"{fid}.ocr-failed"
                tombstone.parent.mkdir(parents=True, exist_ok=True)
                tombstone.write_text("OCR returned empty\n", encoding="utf-8")
                failed.append((cmt, ls, num))
                units_since_checkpoint += 1
            else:
                text_path = cmt_text_dir / f"{fid}.txt"
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(text, encoding="utf-8")
                kb = len(text) / 1024
                print(f"    OK in {elapsed:.0f}s · wrote {kb:.0f} KB to text/{cmt}/{fid}.txt")
                succeeded.append((cmt, ls, num))
                units_since_checkpoint += 1

        now = time.monotonic()
        if (units_since_checkpoint >= CHECKPOINT_EVERY_N or
            (units_since_checkpoint > 0 and now - last_checkpoint_at >= CHECKPOINT_EVERY_S)):
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            checkpoint_commit(
                f"Auto-checkpoint DRSC OCR data (succeeded={len(succeeded)}, failed={len(failed)} this run) [{ts}]",
                ["docs/drsc/text/"],
            )
            units_since_checkpoint = 0
            last_checkpoint_at = now

    if units_since_checkpoint > 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        checkpoint_commit(
            f"Auto-checkpoint DRSC OCR data (final, succeeded={len(succeeded)}, failed={len(failed)} this run) [{ts}]",
            ["docs/drsc/text/"],
        )

    print(f"\n  succeeded: {len(succeeded)} · failed (tombstoned): {len(failed)} · timed out: {len(timed_out)}")

    processed = len(succeeded) + len(failed)
    try:
        remaining = len(find_ocr_candidates(reports))
    except Exception as e:
        print(f"  (couldn't recount candidates for chain decision: {e})")
        remaining = -1
    _emit_chain_outputs(processed, remaining)

    print("\nDone.")


if __name__ == "__main__":
    main()
