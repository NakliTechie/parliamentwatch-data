#!/usr/bin/env python3
"""Debates OCR slow lane — RS-only.

The LS half of the corpus is HTML-from-API and never has scanned-PDF
content, so OCR is not relevant on that side. RS daily-proceedings
PDFs may need OCR for older sessions (pre-2010-ish) when the upstream
PDF is a scan rather than a typeset document. The main debates.yml
workflow uses pypdf and writes a `.pypdf-empty` marker for any RS
version that pypdf can't read — those are this script's targets.

Same shape as cag/build_cag_ocr.py + lc/build_lc_ocr.py:
  - Find .pypdf-empty markers under docs/debates/text/rs/
  - Re-download the PDF (transient)
  - OCR via pdf2image + pytesseract
  - On success: write text/rs/<file_id>.txt
  - On empty / error: write text/rs/<file_id>.ocr-failed tombstone
  - Per-OCR git checkpoint to bound runner-kill loss
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ASSETS    = ROOT / "docs"
DOCS      = ASSETS / "debates"
TEXT_DIR  = DOCS / "text"
RS_TEXT   = TEXT_DIR / "rs"
PDFS_DIR  = DOCS / "pdfs" / "rs"

REPORTS_JSON = DOCS / "reports.json"

# Per-run budget. OCR is slow (3-8 min per PDF). Default 5/run for
# a weekly cadence.
CHECKPOINT_EVERY_N  = int(os.environ.get("CHECKPOINT_EVERY_N", "2"))
CHECKPOINT_EVERY_S  = int(os.environ.get("CHECKPOINT_EVERY_S", "300"))
MAX_OCR_PER_RUN     = int(os.environ.get("MAX_OCR_PER_RUN", "5"))
MAX_RUN_SECONDS     = int(os.environ.get("MAX_RUN_SECONDS", "2700"))   # 45 min
MAX_PAGES_PER_PDF   = int(os.environ.get("MAX_PAGES_PER_PDF", "400"))
RENDER_DPI          = int(os.environ.get("RENDER_DPI", "150"))
TESSERACT_LANGS     = os.environ.get("TESSERACT_LANGS", "eng+hin")


# ── git checkpoint ─────────────────────────────────────────────────────────


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
        print(f"  [checkpoint] git pull --rebase failed: {err.strip()} — aborting")
        _git("rebase", "--abort")
        return False
    rc, _, err = _git("push")
    if rc != 0:
        print(f"  [checkpoint] git push failed: {err.strip()}")
        return False
    print(f"  [checkpoint] pushed: {message}")
    return True


# ── OCR ─────────────────────────────────────────────────────────────────────


def ocr_pdf(pdf_path: Path, *, deadline: float) -> str | None:
    """OCR one PDF. Returns text on success, None if OCR produced
    nothing, or raises TimeoutError if `deadline` exceeded.
    """
    from pdf2image import convert_from_path  # type: ignore
    import pytesseract  # type: ignore

    parts: list[str] = []
    BATCH = 10
    n_done = 0
    page = 1
    while page <= MAX_PAGES_PER_PDF:
        if time.monotonic() > deadline:
            raise TimeoutError(f"deadline exceeded after {n_done} pages")
        last = min(page + BATCH - 1, MAX_PAGES_PER_PDF)
        try:
            images = convert_from_path(
                str(pdf_path), dpi=RENDER_DPI,
                first_page=page, last_page=last)
        except Exception as e:
            if "PDFPageCountError" in type(e).__name__ or "pdftoppm" in str(e):
                break
            raise
        if not images:
            break
        for img in images:
            if time.monotonic() > deadline:
                raise TimeoutError(f"deadline exceeded after {n_done} pages")
            text = pytesseract.image_to_string(img, lang=TESSERACT_LANGS)
            if text: parts.append(text)
            n_done += 1
        page = last + 1
    full = "\n\n".join(parts).strip()
    return full if full else None


# ── Find candidates ────────────────────────────────────────────────────────


def find_ocr_candidates() -> list[tuple[int, str, str, str]]:
    """Walk text/rs/ for .pypdf-empty markers without .txt or
    .ocr-failed alongside. Returns [(session, date_iso, version, pdf_url), ...]
    sorted newest-session-first.

    Resolves each candidate's source PDF URL by reading reports.json's
    file_versions list (URLs persist there even though the on-runner
    pdfs/ dir is gitignored and transient).
    """
    if not REPORTS_JSON.exists():
        return []
    reports = json.loads(REPORTS_JSON.read_text(encoding="utf-8"))
    rs_records = reports.get("rs", []) or []
    # Map (session, date_iso) → versions dict
    url_by_key: dict[tuple[int, str], dict[str, str]] = {}
    for r in rs_records:
        ses = r.get("session"); iso = r.get("date_iso")
        if ses is None or iso is None: continue
        m: dict[str, str] = {}
        for v in (r.get("file_versions") or []):
            ver = v.get("version"); url = v.get("url")
            if ver and url: m[ver] = url
        url_by_key[(int(ses), iso)] = m

    candidates: list[tuple[int, str, str, str]] = []
    if not RS_TEXT.exists():
        return candidates
    import re
    for marker in RS_TEXT.glob("*.pypdf-empty"):
        stem = marker.stem
        # Parse: RS<ses>_<iso>_<version>
        m = re.match(r"RS(\d+)_(\d{4}-\d{2}-\d{2})_([a-z][a-z0-9]+)$", stem)
        if not m: continue
        ses, iso, ver = int(m.group(1)), m.group(2), m.group(3)
        # Skip if already resolved
        if (RS_TEXT / f"{stem}.txt").exists(): continue
        if (RS_TEXT / f"{stem}.ocr-failed").exists(): continue
        url_map = url_by_key.get((ses, iso))
        if not url_map: continue
        url = url_map.get(ver)
        if not url: continue
        candidates.append((ses, iso, ver, url))
    # Newest session first, then newest date, then floor before english before part1
    _VO = {"floor": 0, "english": 1, "part1": 2}
    candidates.sort(key=lambda c: c[1], reverse=True)
    candidates.sort(key=lambda c: _VO.get(c[2], 9))
    candidates.sort(key=lambda c: -c[0])
    return candidates


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    print("=== ParliamentWatch debates OCR slow-lane (RS only) ===")
    print(f"DOCS              : {DOCS}")
    print(f"MAX_OCR_PER_RUN   : {MAX_OCR_PER_RUN}")
    print(f"MAX_RUN_SECONDS   : {MAX_RUN_SECONDS}")
    print(f"MAX_PAGES_PER_PDF : {MAX_PAGES_PER_PDF}")
    print(f"RENDER_DPI        : {RENDER_DPI}")
    print(f"TESSERACT_LANGS   : {TESSERACT_LANGS}")

    if not shutil.which("tesseract"):
        print("ERROR: tesseract not on PATH (apt-get install tesseract-ocr OR brew install tesseract).")
        sys.exit(1)

    candidates = find_ocr_candidates()
    print(f"\n[1/3] Candidates: {len(candidates)} (RS PDFs with .pypdf-empty marker)")
    if not candidates:
        print("  nothing to do — all RS PDFs are extracted or tombstoned")
        return

    target = candidates[:MAX_OCR_PER_RUN]
    print(f"  budget: OCR up to {len(target)} this run (newest first)")

    deadline = time.monotonic() + MAX_RUN_SECONDS
    succeeded: list[tuple[int, str, str]] = []
    failed: list[tuple[int, str, str]] = []
    timed_out: list[tuple[int, str, str]] = []
    last_checkpoint_at = time.monotonic()
    units_since_checkpoint = 0

    # Import the downloader lazily — it's the same one the main scraper uses.
    from debates.scrapers.rajyasabha import (
        download_pdf, versioned_file_id, RateLimited,
    )

    print("\n[2/3] Downloading + OCR'ing...")
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    for i, (ses, iso, ver, url) in enumerate(target, start=1):
        if time.monotonic() > deadline:
            print(f"  [BUDGET] wall-clock budget hit after {len(succeeded)} successes")
            break
        fid = versioned_file_id(ses, iso, ver)
        pdf_path = PDFS_DIR / f"{fid}.pdf"
        if not pdf_path.exists():
            print(f"  [{i}/{len(target)}] {fid}: downloading from {url[:70]}...", flush=True)
            try:
                got = download_pdf(url, session_no=ses, date_iso=iso,
                                   version=ver, pdfs_dir=str(PDFS_DIR))
            except RateLimited as rl:
                print(f"    RATE-LIMITED ({rl}) — stopping this run")
                break
            if not got:
                print(f"    download failed — skipping (no tombstone; retry next run)")
                continue
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"  [{i}/{len(target)}] {fid} ({size_mb:.1f} MB) — running OCR...", flush=True)
        t0 = time.time()
        try:
            text = ocr_pdf(pdf_path, deadline=deadline)
        except TimeoutError as e:
            elapsed = time.time() - t0
            print(f"    TIMEOUT after {elapsed:.0f}s ({e}) — leaving for next run")
            timed_out.append((ses, iso, ver))
            break
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    OCR FAILED after {elapsed:.0f}s: {e}")
            tombstone = RS_TEXT / f"{fid}.ocr-failed"
            tombstone.parent.mkdir(parents=True, exist_ok=True)
            tombstone.write_text(f"OCR error: {e}\n", encoding="utf-8")
            failed.append((ses, iso, ver))
            units_since_checkpoint += 1
        else:
            elapsed = time.time() - t0
            if text is None:
                print(f"    EMPTY OCR after {elapsed:.0f}s — tombstone, won't retry")
                tombstone = RS_TEXT / f"{fid}.ocr-failed"
                tombstone.parent.mkdir(parents=True, exist_ok=True)
                tombstone.write_text("OCR returned empty\n", encoding="utf-8")
                failed.append((ses, iso, ver))
                units_since_checkpoint += 1
            else:
                text_path = RS_TEXT / f"{fid}.txt"
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(text, encoding="utf-8")
                kb = len(text) / 1024
                print(f"    OK in {elapsed:.0f}s · wrote {kb:.0f} KB")
                succeeded.append((ses, iso, ver))
                units_since_checkpoint += 1
        # Delete the transient PDF — fetch-extract-delete pattern.
        try:
            os.remove(pdf_path)
        except Exception:
            pass

        # Checkpoint
        now = time.monotonic()
        if (units_since_checkpoint >= CHECKPOINT_EVERY_N or
            (units_since_checkpoint > 0 and now - last_checkpoint_at >= CHECKPOINT_EVERY_S)):
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            checkpoint_commit(
                f"Auto-checkpoint debates OCR data (succeeded={len(succeeded)}, failed={len(failed)}) [{ts}]",
                ["docs/debates/text/rs/"],
            )
            units_since_checkpoint = 0
            last_checkpoint_at = now

    if units_since_checkpoint > 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        checkpoint_commit(
            f"Auto-checkpoint debates OCR data (final, succeeded={len(succeeded)}, failed={len(failed)}) [{ts}]",
            ["docs/debates/text/rs/"],
        )

    print(f"\n  succeeded: {len(succeeded)} · failed: {len(failed)} · timed out: {len(timed_out)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
