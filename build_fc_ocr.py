#!/usr/bin/env python3
"""FC (Financial Committees) OCR slow lane — backfills scanned PDFs that
the main scraper skipped.

The main `build_fc.py` scraper uses pypdf, which returns empty text for
scanned-only PDFs (no embedded text layer). Older PAC / Estimates / COPU
reports (LS-14..17) include a meaningful pool of scanned-only PDFs that
produce a `text/<committee>/<file_id>.pypdf-empty` marker. This script is
the catch-up: it walks reports.json for those markers (or their bundled
equivalent in markers.json), runs the underlying PDFs through tesseract
OCR via pdf2image, and writes the result to text/<committee>/<file_id>.txt.

Why a separate runner: tesseract is slow (~2-3s/page at 150 dpi). A
300-page scanned PDF takes ~8 minutes. Folding that into the daily cron
would eat the budget and starve the network-bound scraper. This runs on
demand (workflow_dispatch) with a sprint-sized budget, with the standard
in-flight checkpointing + chain-if-backlog-remains pattern.

Failures (OCR returned nothing, severely degraded scan, encrypted, etc.)
get a `text/<committee>/<file_id>.ocr-failed` tombstone so we don't retry
them every run. The tombstone is invisible to manifest / bundle / index
(those scan for `*.txt` only).

Structurally a near-twin of build_lc_ocr.py / build_cag_ocr.py — same
checkpoint cadence, same defaults. Differences: paths point to docs/fc/,
identity is composite `(committee, lok_sabha, report_number)` rather than
a single integer (matching DRSC's shape that FC borrowed), and the
download / file_id helpers come from `fc.scraper`.

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

from fc.scraper import (
    download_pdf as _download_pdf,
    file_id,
    RateLimited,
)

# ── Paths ───────────────────────────────────────────────────────────────────

ASSETS    = ROOT / "docs"
DOCS      = ASSETS / "fc"
TEXT_DIR  = DOCS / "text"
PDFS_DIR  = DOCS / "pdfs"
DOCS.mkdir(parents=True, exist_ok=True)


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

CHECKPOINT_EVERY_N = int(os.environ.get("CHECKPOINT_EVERY_N", "2"))
CHECKPOINT_EVERY_S = int(os.environ.get("CHECKPOINT_EVERY_S", "300"))


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

# How many scanned PDFs to OCR per run. Default 5 matches LC/CAG. For the
# one-shot FC backfill sprint, dispatch with a much higher number (the
# workflow's input override drives this). At ~5 min/PDF on a typical
# FC committee report, 100/run fits comfortably in MAX_RUN_SECONDS=20400.
MAX_OCR_PER_RUN = int(os.environ.get("MAX_OCR_PER_RUN", "5"))

# Hard wall-clock cap. After this, abort whatever's in flight (don't write
# partial text — we'd rather skip than lie).
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "2700"))   # 45 min

# Per-PDF page cap. FC committee reports are typically 50-300 pages, with
# a long tail to 700+. Cap stops one bloated PDF from eating the entire
# run budget. Pages beyond this get OCR'd-then-truncated.
MAX_PAGES_PER_PDF = int(os.environ.get("MAX_PAGES_PER_PDF", "300"))

# Render DPI. 150 is the sweet spot for typeset government PDFs:
# tesseract gets 95%+ on clean English text at this resolution. Lower
# (100) is faster but quality drops noticeably; higher (300) is slower
# with marginal gains.
RENDER_DPI = int(os.environ.get("RENDER_DPI", "150"))

# Tesseract languages. eng + hin covers most FC reports (a few have
# Hindi sections, especially older PAC reports). Add more language packs
# in the workflow's apt-install if specific reports need them.
TESSERACT_LANGS = os.environ.get("TESSERACT_LANGS", "eng+hin")


def find_ocr_candidates(
    reports: dict[str, list[dict]],
) -> list[tuple[str, int, int, str]]:
    """Walk reports.json for any record with a pdf_url that has neither a
    text/<cmt>/<fid>.txt nor an .ocr-failed tombstone AND has a
    .pypdf-empty marker (i.e. pypdf already confirmed scanned-only).

    Returns [(committee, lok_sabha, report_number, pdf_url), ...] sorted
    newest-first (highest LS, then highest report_number within LS — the
    same priority `build_fc.extract_missing_texts` already uses).

    Marker source-of-truth: docs/fc/markers.json (post-derive consolidation),
    with on-disk sidecars as a fallback for the brief window between a
    fresh extract commit and the next derive run. Matches the corrected
    pattern in build_lc_ocr.py.
    """
    # Consolidated markers — single source of truth post-derive. Empty
    # dict if markers.json doesn't exist yet (pre-first-derive corpus).
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
            fid = file_id(int(ls), int(num))
            cid = f"{cmt}|{fid}"
            # Permanent skips: already extracted (.txt) or tombstoned
            # (.ocr-failed). Check both markers.json and on-disk sidecar.
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
    # Priority: highest LS first, then highest report_number within LS.
    candidates.sort(key=lambda c: (-c[1], -c[2]))
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


def _emit_chain_outputs(processed: int, remaining: int) -> None:
    """Emit progress to $GITHUB_OUTPUT so the workflow can chain another
    OCR run when (a) we made progress AND (b) backlog remains. See the
    "Chain another OCR run" step in .github/workflows/fc-ocr.yml.
    """
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


def main():
    print("=== ParliamentWatch FC OCR slow-lane ===")
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
    from build_fc import load_existing_reports as _load
    reports = _load()
    n_reports = sum(len(v) for v in reports.values())
    candidates = find_ocr_candidates(reports)
    print(f"  candidates: {len(candidates)} (out of {n_reports} total reports)")
    if not candidates:
        print("  nothing to do — all reports already have text or are tombstoned")
        # Emit explicit '0' values for the chain step's compare.
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
    units_since_checkpoint = 0   # successes + tombstones (both are commit-worthy)

    print("\n[2/3] Downloading + OCR'ing...")
    for i, (cmt, ls, num, pdf_url) in enumerate(target, start=1):
        if time.monotonic() > deadline:
            print(f"  [BUDGET] wall-clock budget hit after {len(succeeded)} successes")
            break
        fid = file_id(ls, num)
        cmt_pdfs_dir = PDFS_DIR / cmt
        pdf_path = cmt_pdfs_dir / f"{fid}.pdf"
        if not pdf_path.exists():
            print(f"  [{i}/{len(target)}] {cmt} {fid}: downloading {pdf_url}...", flush=True)
            try:
                got = _download_pdf(pdf_url,
                                    committee=cmt, lok_sabha=ls,
                                    report_number=num,
                                    pdfs_dir=str(PDFS_DIR))
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
        # NOTE: stale .pypdf-empty markers from earlier extract runs are
        # NOT removed when OCR resolves them. The audit logic in build_fc.py
        # compute_audit() reads markers in precedence order (.txt >
        # .ocr-failed > .pypdf-empty > ...) so counts stay correct.
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
                f"Auto-checkpoint FC OCR data (succeeded={len(succeeded)}, failed={len(failed)} this run) [{ts}]",
                ["docs/fc/text/"],
            )
            units_since_checkpoint = 0
            last_checkpoint_at = now

    if units_since_checkpoint > 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        checkpoint_commit(
            f"Auto-checkpoint FC OCR data (final, succeeded={len(succeeded)}, failed={len(failed)} this run) [{ts}]",
            ["docs/fc/text/"],
        )

    print(f"\n  succeeded: {len(succeeded)} · failed (tombstoned): {len(failed)} · timed out: {len(timed_out)}")

    # Re-run find_ocr_candidates against the post-run disk state so the
    # remaining count reflects what we just wrote (.txt + .ocr-failed
    # files cause records to be excluded from the new candidate set).
    processed = len(succeeded) + len(failed)
    try:
        remaining = len(find_ocr_candidates(reports))
    except Exception as e:
        print(f"  (couldn't recount candidates for chain decision: {e})")
        remaining = -1
    _emit_chain_outputs(processed, remaining)

    # Note: this slow lane deliberately does NOT regenerate manifest /
    # bundle / index / meta. The fc-derive.yml workflow_run + cron backstop
    # picks up this commit and regenerates derived files within ~30 min,
    # so the new OCR'd texts get reflected automatically. Skipping the
    # regen here keeps OCR commits to a small, disjoint diff (text/<cmt>/
    # <fid>.txt + text/<cmt>/<fid>.ocr-failed only) — no rebase conflicts
    # with the derive workflow running concurrently against the same
    # derived files.

    print("\nDone.")


if __name__ == "__main__":
    main()
