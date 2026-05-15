#!/usr/bin/env python3
"""LC OCR slow lane — backfills scanned PDFs that the main scraper skipped.

The main `build_lc.py` scraper uses pypdf, which returns empty text for
scanned-only PDFs (no embedded text layer). Older Law Commission reports
(1950s-1990s) are typically scanned and produce a `text/<id>.pypdf-empty`
marker. This script is the catch-up: it walks `docs/lc/pdfs/` for PDFs
with the marker (and without a corresponding `text/<id>.txt`), runs them
through tesseract OCR via pdf2image, and writes the result.

Why a separate runner: tesseract is slow (~2-3s/page at 150 dpi). A
200-page scanned PDF takes ~8 minutes. Folding that into the daily cron
would eat the budget and starve the network-bound scraper. This runs
weekly, processes a small batch per run, slowly catches up across the
corpus.

Failures (OCR returned nothing, severely degraded scan, encrypted, etc.)
get a `text/<id>.ocr-failed` tombstone so we don't retry them every week.
The tombstone is invisible to manifest / bundle / index (those scan for
`*.txt` only).

Structurally a near-twin of build_cag_ocr.py — same checkpoint cadence,
same defaults. Differences: paths point to docs/lc/, scraper import is
`from lc.scraper import ...`, and the load-existing path goes through
`build_lc.load_existing_reports`.

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

from parliamentwatch_text_shards import load_markers

# ── Paths ───────────────────────────────────────────────────────────────────

ASSETS    = ROOT / "docs"
DOCS      = ASSETS / "lc"
TEXT_DIR  = DOCS / "text"
PDFS_DIR  = DOCS / "pdfs"
DOCS.mkdir(parents=True, exist_ok=True)

REPORTS_JSON  = DOCS / "reports.json"
MANIFEST_JSON = DOCS / "manifest.json"
META_JSON     = DOCS / "meta.json"
BUNDLE_JSON   = DOCS / "search-bundle.json"
INDEX_JSON    = DOCS / "search-index.json"


# ── In-flight checkpointing ────────────────────────────────────────────────
#
# OCR is slow (3-8 min per PDF). A runner-killed mid-stream OCR run that
# only commits at the end loses N-1 OCRs of work. With per-OCR checkpoints,
# we commit each successful OCR before starting the next. Worst case
# loss = at most one OCR's worth of work.
#
# Lower default than the extract scripts (N=2 vs 25) because OCR's
# per-unit cost is so high — we want commits as close to per-PDF as the
# git-overhead-vs-loss tradeoff allows.

import subprocess

CHECKPOINT_EVERY_N = int(os.environ.get("CHECKPOINT_EVERY_N", "2"))
CHECKPOINT_EVERY_S = int(os.environ.get("CHECKPOINT_EVERY_S", "300"))

# PDF archive sidecar — see build_lc.py for the full rationale. The OCR
# slow lane downloads scanned PDFs that pypdf couldn't read; those are
# exactly the rare / fragile / hardest-to-replace files in the corpus,
# so we want them archived even more than the text-extractable ones.
LC_ARCHIVE_DIR = os.environ.get("LC_ARCHIVE_DIR", "")


def _git(*args) -> tuple[int, str, str]:
    p = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False,
    )
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

# How many scanned PDFs to OCR per weekly run. Default 5 — at ~8 min each
# for a typical Law Commission report, that's ~40 minutes. Stays well
# under the 60-min GH Actions timeout. Bump via workflow_dispatch input
# for one-off catch-up runs.
MAX_OCR_PER_RUN = int(os.environ.get("MAX_OCR_PER_RUN", "5"))

# Hard wall-clock cap. After this, abort whatever's in flight (don't write
# partial text — we'd rather skip than lie).
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "2700"))   # 45 min

# Per-PDF page cap. Older LC reports range from ~50 to ~600 pages. Cap
# stops one bloated PDF from eating the entire run budget. Pages beyond
# this get OCR'd-then-truncated.
MAX_PAGES_PER_PDF = int(os.environ.get("MAX_PAGES_PER_PDF", "400"))

# Render DPI. 150 is the sweet spot for typeset government PDFs:
# tesseract gets 95%+ on clean English text at this resolution. Lower
# (100) is faster but quality drops noticeably; higher (300) is slower
# with marginal gains.
RENDER_DPI = int(os.environ.get("RENDER_DPI", "150"))

# Tesseract languages. eng + hin covers most LC reports (a few are
# bilingual or have Hindi appendices). Add more language packs in the
# workflow's apt-install if specific reports need them.
TESSERACT_LANGS = os.environ.get("TESSERACT_LANGS", "eng+hin")


def find_ocr_candidates(reports: dict[int, dict]) -> list[tuple[int, str]]:
    """Walk reports.json for any report with a pdf_url that has neither a
    text/<id>.txt nor an .ocr-failed tombstone AND has a .pypdf-empty
    marker (i.e. pypdf already confirmed scanned-only).
    Returns [(id, pdf_url), ...] sorted newest-first (highest report_number
    wins — LC numbers are monotonic across the whole corpus).

    Marker source-of-truth: docs/lc/markers.json (post-derive
    consolidation), with on-disk sidecars as a fallback for the brief
    window between a fresh scrape commit and the next derive run. The
    consolidation step in lc-derive.yml deletes the per-record sidecars
    from disk after merging them into markers.json — historically this
    function only checked sidecars, which made it find zero candidates
    once the corpus had been derive'd. Now it reads markers.json the
    same way build_lc.py does. See CONV.md "Per-attempt status markers"
    for the consolidation semantics.

    pdfs/ is gitignored, so on a fresh runner we don't have local PDFs
    cached. The slow lane downloads each candidate's PDF before OCR'ing.
    """
    # Consolidated markers — single source of truth post-derive. Empty
    # dict if markers.json doesn't exist yet (pre-first-derive corpus).
    bundled_markers = load_markers(DOCS)

    candidates: list[tuple[int, str]] = []
    for rid, meta in reports.items():
        pdf_url = meta.get("pdf_url")
        if not pdf_url:
            continue
        cid = str(rid)
        # Permanent skips: already extracted (.txt) or tombstoned
        # (.ocr-failed). Check both markers.json and on-disk sidecar —
        # sidecars only exist transiently in the pre-derive window after
        # a fresh OCR run wrote them but the next derive hasn't yet
        # consolidated.
        if (TEXT_DIR / f"{rid}.txt").exists():
            continue
        if bundled_markers.get(cid) == "ocr-failed":
            continue
        if (TEXT_DIR / f"{rid}.ocr-failed").exists():
            continue
        # Positive signal: pypdf-empty marker (in either source).
        is_pypdf_empty = (
            bundled_markers.get(cid) == "pypdf-empty"
            or (TEXT_DIR / f"{rid}.pypdf-empty").exists()
        )
        if not is_pypdf_empty:
            continue   # let the daily run try pypdf first
        candidates.append((rid, pdf_url))
    candidates.sort(key=lambda c: -c[0])
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
    # Render in batches to keep memory bounded — a 400-page PDF rendered
    # at 150 dpi as a single list is ~4 GB of PIL Images.
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
    print("=== ParliamentWatch LC OCR slow-lane ===")
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

    print("\n[1/3] Finding OCR candidates (scanned-only PDFs with no text yet)...")
    from build_lc import load_existing_reports as _load
    reports = _load()
    candidates = find_ocr_candidates(reports)
    print(f"  candidates: {len(candidates)} (out of {len(reports)} total reports)")
    if not candidates:
        print("  nothing to do — all reports already have text or are tombstoned")
        # Emit outputs even on early exit so the workflow's chain step
        # has explicit '0' values to compare (an empty output string
        # would let `outputs.x != '0'` evaluate true and create a loop).
        gh_output_path = os.environ.get("GITHUB_OUTPUT")
        if gh_output_path:
            try:
                with open(gh_output_path, "a", encoding="utf-8") as f:
                    f.write("processed=0\n")
                    f.write("remaining=0\n")
            except OSError:
                pass
        return

    target = candidates[:MAX_OCR_PER_RUN]
    print(f"  budget: OCR'ing up to {len(target)} this run (newest first)")
    print(f"  note: pdfs/ is gitignored — slow-lane downloads each PDF before OCR.")

    deadline = time.monotonic() + MAX_RUN_SECONDS
    succeeded: list[int] = []
    failed: list[int] = []
    timed_out: list[int] = []

    # In-flight checkpointing — see CHECKPOINT_EVERY_N / CHECKPOINT_EVERY_S.
    last_checkpoint_at = time.monotonic()
    units_since_checkpoint = 0   # successes + tombstones (both are commit-worthy)

    print("\n[2/3] Downloading + OCR'ing...")
    # Lazy-import the scraper's downloader (handles jitter + rate-limit)
    # and the archive helper (no-op when LC_ARCHIVE_DIR is unset).
    from lc.scraper import download_pdf as _download_pdf, RateLimited, archive_pdf
    for i, (rid, pdf_url) in enumerate(target, start=1):
        if time.monotonic() > deadline:
            print(f"  [BUDGET] wall-clock budget hit after {len(succeeded)} successes")
            break
        pdf_path = PDFS_DIR / f"{rid}.pdf"
        if not pdf_path.exists():
            print(f"  [{i}/{len(target)}] id={rid}: downloading {pdf_url}...", flush=True)
            try:
                got = _download_pdf(pdf_url, rid, pdfs_dir=str(PDFS_DIR))
            except RateLimited as rl:
                print(f"    RATE-LIMITED ({rl}) — stopping this run")
                break
            if not got:
                print(f"    download failed — skipping (no tombstone; will retry next run)")
                continue
        # Archive the PDF on every visit (idempotent — archive_pdf
        # short-circuits when sha256+size already match). OCR candidates
        # are by definition the rare, scanned-only PDFs that are hardest
        # to re-derive if upstream rotates URLs.
        if LC_ARCHIVE_DIR:
            try:
                archive_pdf(str(pdf_path), rid, pdf_url, archive_dir=LC_ARCHIVE_DIR)
            except Exception as ae:
                print(f"    [archive] failed to archive #{rid}: {ae}")
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"  [{i}/{len(target)}] id={rid} ({size_mb:.1f} MB) — running OCR...", flush=True)
        t0 = time.time()
        # NOTE: stale .pypdf-empty markers from earlier daily runs are NOT
        # removed when OCR resolves them. The audit logic in build_lc.py
        # compute_audit() reads markers in precedence order (.txt >
        # .ocr-failed > .pypdf-empty > ...) so counts stay correct.
        # Deleting on disk only would be a no-op for the published mirror
        # anyway — `git add docs/lc/text/` doesn't stage deletions of
        # tracked files.
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
            units_since_checkpoint += 1
        else:
            elapsed = time.time() - t0
            if text is None:
                print(f"    EMPTY OCR after {elapsed:.0f}s — writing tombstone, won't retry")
                tombstone = TEXT_DIR / f"{rid}.ocr-failed"
                tombstone.parent.mkdir(parents=True, exist_ok=True)
                tombstone.write_text("OCR returned empty\n", encoding="utf-8")
                failed.append(rid)
                units_since_checkpoint += 1
            else:
                text_path = TEXT_DIR / f"{rid}.txt"
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(text, encoding="utf-8")
                kb = len(text) / 1024
                print(f"    OK in {elapsed:.0f}s · wrote {kb:.0f} KB to text/{rid}.txt")
                succeeded.append(rid)
                units_since_checkpoint += 1

        # Checkpoint after each commit-worthy event (text or tombstone).
        # OCR is the slowest extractor — per-2-PDFs default keeps loss
        # bounded at one OCR's worth of work even on runner kill.
        now = time.monotonic()
        if (units_since_checkpoint >= CHECKPOINT_EVERY_N or
            (units_since_checkpoint > 0 and now - last_checkpoint_at >= CHECKPOINT_EVERY_S)):
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            checkpoint_commit(
                f"Auto-checkpoint LC OCR data (succeeded={len(succeeded)}, failed={len(failed)} this run) [{ts}]",
                ["docs/lc/text/"],
            )
            units_since_checkpoint = 0
            last_checkpoint_at = now

    # Final checkpoint of post-loop residue (anything since the last trigger).
    if units_since_checkpoint > 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        checkpoint_commit(
            f"Auto-checkpoint LC OCR data (final, succeeded={len(succeeded)}, failed={len(failed)} this run) [{ts}]",
            ["docs/lc/text/"],
        )

    print(f"\n  succeeded: {len(succeeded)} · failed (tombstoned): {len(failed)} · timed out: {len(timed_out)}")

    # Emit progress to GITHUB_OUTPUT so the workflow can chain another
    # OCR run when (a) we made progress AND (b) backlog remains. See
    # the "Chain another OCR run" step in .github/workflows/lc-ocr.yml.
    # Re-run find_ocr_candidates against the post-run disk state so the
    # remaining count reflects what we just wrote (.txt + .ocr-failed
    # files cause records to be excluded from the new candidate set).
    processed = len(succeeded) + len(failed)
    try:
        remaining = len(find_ocr_candidates(reports))
    except Exception as e:
        print(f"  (couldn't recount candidates for chain decision: {e})")
        remaining = -1
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

    # Note: this slow lane deliberately does NOT regenerate manifest /
    # bundle / index / meta. The lc-derive.yml workflow_run + cron backstop
    # picks up this commit and regenerates derived files within ~30 min,
    # so the new OCR'd texts get reflected automatically. Skipping the
    # regen here keeps OCR commits to a small, disjoint diff (text/<id>.txt
    # + text/<id>.ocr-failed only) — no rebase conflicts with the derive
    # workflow running concurrently against the same derived files.

    print("\nDone.")


if __name__ == "__main__":
    main()
