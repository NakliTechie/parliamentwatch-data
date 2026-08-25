"""sansad.in bills scraper — canonical bills + status timeline + per-stage PDFs.

Independent of DRSC and CAG scrapers. Per the Independence Principle
(sansadsaar-vision-001-v0.5.md, sansadsaar-spec-002-v0.2.md), this scraper
has no shared imports with the DRSC scraper at the repo root or with
cag/scraper.py. HTTP / jitter / rate-limit primitives are duplicated here
by design.

Source: sansad.in's public legislation API at /api_rs/legislation/getBills.
The endpoint serves both Lok Sabha and Rajya Sabha bills despite the
api_rs prefix — the LS-side Next.js chunk reaches into api_rs because
the schema is shared. Don't be fooled by the prefix.

Pagination shape: ?page=N&size=M, optional &billYear=YYYY. Default size=10,
max effective size=500. Total: ~10,069 records as of 2026-05; archive
goes back to 1952. Full corpus pull: ~21 pages at size=500 in ~35 s.

Anti-bot posture: same as DRSC. Run from CI or laptop, not from agent
runtimes.

Recon: plan/bills-recon-001.md (in SansadLocal repo, internal/gitignored).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Iterator, Optional

import requests
from pypdf import PdfReader

# ── Constants ───────────────────────────────────────────────────────────────

BASE_URL = "https://sansad.in"
BILLS_API = BASE_URL + "/api_rs/legislation/getBills"

# Adjacent enum-resolution endpoints (rarely needed — getBills already returns
# human-readable strings for most fields). Listed here so a future extender
# doesn't have to re-discover them.
BILL_CATEGORIES_API = BASE_URL + "/api_rs/legislation/getBillCategories"
BILL_STATUS_API     = BASE_URL + "/api_rs/legislation/getBillStatus"
BILL_TYPES_API      = BASE_URL + "/api_rs/legislation/getBillTypes"
MINISTRY_NAME_API   = BASE_URL + "/api_rs/legislation/getMinistryName"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,*/*;q=0.8",
}

# Politeness — random per-fetch sleep so we don't hammer sansad.in. Same
# defaults as the DRSC + CAG scrapers. Set JITTER_MIN_MS / JITTER_MAX_MS to 0
# to disable for a quick local test.
_JITTER_MIN_MS = int(os.environ.get("JITTER_MIN_MS", "250"))
_JITTER_MAX_MS = int(os.environ.get("JITTER_MAX_MS", "500"))

_DEFAULT_PAGE_SIZE = int(os.environ.get("BILLS_PAGE_SIZE", "500"))


class RateLimited(Exception):
    """Raised when sansad.in returns 429/403 — caller should stop the loop."""


def _jitter() -> None:
    if _JITTER_MAX_MS <= 0:
        return
    ms = random.randint(_JITTER_MIN_MS, _JITTER_MAX_MS)
    time.sleep(ms / 1000.0)


def _get(url: str, *, timeout: int = 30, retries: int = 4, **kwargs) -> requests.Response:
    """HTTP GET with rate-limit-aware error handling + one retry on transient
    timeouts. Honours Retry-After in the RateLimited message (per the
    parliamentwatch-data 1c09f17 fix; logic duplicated here per Independence).
    Pass stream=True via kwargs for PDF downloads.
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        _jitter()
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout, **kwargs)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        if resp.status_code in (429, 403):
            retry_after_hdr = resp.headers.get("Retry-After")
            suggested_wait: Optional[int] = None
            if retry_after_hdr:
                try:
                    suggested_wait = int(retry_after_hdr)
                except ValueError:
                    # Could be an HTTP-date instead of seconds; ignore for now.
                    pass
            msg = f"{resp.status_code} from {resp.url}"
            if suggested_wait is not None:
                msg += f" (Retry-After: {suggested_wait}s)"
            raise RateLimited(msg)
        resp.raise_for_status()
        return resp
    raise last_err if last_err else RuntimeError("_get exhausted")


# ── Index walker ────────────────────────────────────────────────────────────

# sansad.in pads many string fields with trailing whitespace
# ("XXXXIII             "). Strip on ingestion so downstream consumers don't
# have to.
_STRIP_KEYS = (
    "billNumber", "billName", "billType", "billCategory", "ministryName",
    "billIntroducedInHouse", "billIntroducedBy",
    "actNo", "status",
)


def _normalise(record: dict) -> dict:
    """Strip whitespace-padded fields, collapse empty strings to None.
    Returns a new dict."""
    out = dict(record)
    for k in _STRIP_KEYS:
        v = out.get(k)
        if isinstance(v, str):
            out[k] = v.strip()
    for k, v in list(out.items()):
        if v == "":
            out[k] = None
    return out


_SLUG_RE = re.compile(r'[^A-Za-z0-9._-]+')

# Recon-confirmed: only "Lok Sabha" and "Rajya Sabha" appear in
# billIntroducedInHouse across the full archive (verified 2026-05-10).
# X is a defensive fallback in case the API ever surfaces a third value.
_HOUSE_TO_CODE = {
    "Lok Sabha": "L",
    "Rajya Sabha": "R",
}


def composite_id(record: dict) -> str:
    """Build a stable, filesystem-safe composite ID for a bill record.

    Format: <billNumber-slug>_<billYear>_<houseCode>. The house suffix is
    needed because Roman-numeral bill numbers collide across houses (a
    "Bill XXXIII of 1990" can exist independently in LS and RS). Without
    the house code, ~1.55% of bills collide on (number, year) alone.
    """
    bn = (record.get("billNumber") or "").strip() or "NO_NUMBER"
    bn_slug = _SLUG_RE.sub('-', bn).strip('-') or "NO_NUMBER"
    by = record.get("billYear") or "NO_YEAR"
    house_raw = (record.get("billIntroducedInHouse") or "").strip()
    house_code = _HOUSE_TO_CODE.get(house_raw, "X")
    return f"{bn_slug}_{by}_{house_code}"


def fetch_bills_page(page: int, *, size: int = _DEFAULT_PAGE_SIZE,
                     bill_year: Optional[int] = None) -> tuple[list[dict], dict]:
    """Fetch one page of bills. Returns (records, _metadata).

    Pagination is 1-indexed. Default size=500 (max effective). totalPages in
    the metadata tells you when to stop.
    """
    params: dict = {"page": page, "size": size}
    if bill_year is not None:
        params["billYear"] = bill_year
    resp = _get(BILLS_API, params=params)
    payload = resp.json()
    return payload.get("records", []), payload.get("_metadata", {})


def walk_all_bills(*, size: int = _DEFAULT_PAGE_SIZE,
                   bill_year: Optional[int] = None,
                   max_pages: Optional[int] = None) -> Iterator[dict]:
    """Yield normalised bill records across all pages of the API.

    Pass `bill_year` to filter to one year. Pass `max_pages` for a small
    slice (dev / testing). Default: walk the entire archive.
    """
    page = 1
    total_pages: Optional[int] = None
    while True:
        records, meta = fetch_bills_page(page, size=size, bill_year=bill_year)
        if total_pages is None:
            total_pages = meta.get("totalPages")
        if not records:
            return
        for r in records:
            yield _normalise(r)
        if max_pages is not None and page >= max_pages:
            return
        if total_pages is not None and page >= total_pages:
            return
        page += 1


def collect_records(*, size: int = _DEFAULT_PAGE_SIZE,
                    bill_year: Optional[int] = None,
                    max_pages: Optional[int] = None) -> tuple[list[dict], int]:
    """Walk sansad.in bills API, normalise records, dedupe by compositeId.
    Returns (records, duplicates_dropped). Pure — no filesystem writes.
    The orchestrator decides how to shape the output (single file, sharded,
    or merged with PRS payload).
    """
    records: list[dict] = []
    seen_ids: set[str] = set()
    duplicates = 0
    for r in walk_all_bills(size=size, bill_year=bill_year, max_pages=max_pages):
        cid = composite_id(r)
        if cid in seen_ids:
            duplicates += 1
            continue
        seen_ids.add(cid)
        r["compositeId"] = cid
        records.append(r)
    return records, duplicates


def build_index(out_path: str, *, size: int = _DEFAULT_PAGE_SIZE,
                bill_year: Optional[int] = None,
                max_pages: Optional[int] = None) -> dict:
    """CLI smoke-test convenience: collect records, write a single flat file.
    Production use goes through collect_records() + the orchestrator's
    sharding step.
    """
    records, duplicates = collect_records(size=size, bill_year=bill_year, max_pages=max_pages)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "records": records,
            "count": len(records),
            "scraped_year": bill_year,
            "duplicate_composite_ids_dropped": duplicates,
            "scraper_version": SCRAPER_VERSION,
        }, f, ensure_ascii=False, indent=2)
    return {
        "count": len(records),
        "duplicates_dropped": duplicates,
        "out_path": out_path,
    }


# ── PDF download + text extraction ─────────────────────────────────────────

# Priority order for canonical text — most-final → least-final. Used by
# extract_canonical_text() to pick which PDF to extract per bill.
CANONICAL_PDF_FIELDS = (
    "billPassedInBothHousesFile",
    "billPassedInLSFile",
    "billPassedInRSFile",
    "billIntroducedFile",
)

# All PDF fields we mirror per bill. Includes canonical stages plus
# supplementary documents (errata, gazette notification, synopsis, the DRSC
# committee report when present). Per-bill folder layout:
#   docs/bills/pdfs/<compositeId>/<field>.pdf
ALL_PDF_FIELDS = CANONICAL_PDF_FIELDS + (
    "errataFile",
    "billGazettedFile",   # cross-corpus link to v1.2 Gazette corpus (sparse)
    "billSynopsisFile",
    "reportFile",         # cross-corpus link to DRSC corpus (very sparse — 0.3%)
)

SCRAPER_VERSION = "bills.sansad.scraper/0.2"


def _sanitize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    return url.replace("\\", "/")


def download_pdf(url: str, out_path: str) -> Optional[str]:
    """Download a single PDF to `out_path`. Returns out_path on success,
    None on non-rate-limit failure. Raises RateLimited on 429/403 — caller
    should stop the whole batch. Idempotent — skips if file already exists.
    """
    if os.path.exists(out_path):
        return out_path
    url = _sanitize_url(url) or ""
    if not url:
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        resp = _get(url, timeout=180, stream=True)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return out_path
    except RateLimited:
        # Partial file may exist — remove so the next run re-downloads cleanly.
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return None


def download_canonical_pdf(record: dict, pdfs_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Download just the highest-priority canonical PDF for a bill (most-final
    stage with a URL present). Returns (path, field_name) or (None, None) if
    the bill has no canonical PDF available. RateLimited propagates.

    This is the cheap path for text extraction — most bills don't need their
    full PDF set mirrored if we only want canonical text. Use download_all_pdfs
    for the full mirror.
    """
    cid = record["compositeId"]
    bill_dir = os.path.join(pdfs_dir, cid)
    for field in CANONICAL_PDF_FIELDS:
        url = record.get(field)
        if not url:
            continue
        path = os.path.join(bill_dir, f"{field}.pdf")
        result = download_pdf(url, path)
        if result:
            return result, field
        # If download failed (non-rate-limit), try the next priority stage.
    return None, None


def download_all_pdfs(record: dict, pdfs_dir: str) -> dict:
    """Download every PDF stage available for one bill into
    `pdfs_dir/<compositeId>/<field>.pdf`. Returns dict {field: path|None}.
    RateLimited propagates.
    """
    cid = record["compositeId"]
    bill_dir = os.path.join(pdfs_dir, cid)
    out: dict = {}
    for field in ALL_PDF_FIELDS:
        url = record.get(field)
        if not url:
            out[field] = None
            continue
        path = os.path.join(bill_dir, f"{field}.pdf")
        out[field] = download_pdf(url, path)
    return out


class _PypdfEmpty(Exception):
    """Sentinel: pypdf parsed the file but returned no text (scanned/encrypted)."""
    pass


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a single PDF.

    Returns the joined text on success. Raises `_PypdfEmpty` if pypdf
    returned an empty result (scanned/encrypted PDF, OCR target). Lets
    any other exception (corrupt file, pypdf bug) propagate so the caller
    can distinguish empty-but-fine from genuinely-failed.
    """
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    full = "\n\n".join(parts)
    if not full.strip():
        raise _PypdfEmpty(f"pypdf returned empty for {pdf_path}")
    return full


def extract_canonical_text(record: dict, pdfs_dir: str, text_dir: str) -> Optional[str]:
    """Extract canonical text for one bill: download the highest-priority
    available canonical PDF, run pypdf, write to `text_dir/<compositeId>.txt`.
    Returns the text-file path or None on no-PDF / extraction-failure.
    Idempotent — returns the existing text file if already present.
    RateLimited propagates.

    Per-attempt status markers (see CONV.md "Per-attempt status markers"):
      .txt           — successful extraction
      .pypdf-empty   — pypdf returned empty (scanned/encrypted) → OCR target
      .pypdf-error   — pypdf raised an exception → retryable

    Markers let select_candidates() skip re-downloading known-bad PDFs.
    """
    cid = record["compositeId"]
    text_path        = os.path.join(text_dir, f"{cid}.txt")
    pypdf_empty_path = os.path.join(text_dir, f"{cid}.pypdf-empty")
    pypdf_error_path = os.path.join(text_dir, f"{cid}.pypdf-error")
    if os.path.exists(text_path):
        return text_path

    pdf_path, _stage = download_canonical_pdf(record, pdfs_dir)
    if not pdf_path:
        return None

    os.makedirs(text_dir, exist_ok=True)
    try:
        full = extract_text_from_pdf(pdf_path)
    except _PypdfEmpty:
        # Scanned or encrypted PDF — drop a marker so future backfills
        # skip it (no point retrying pypdf) and so OCR's slow lane (when
        # one's added for Bills) can pick it up cleanly.
        print(f"  pypdf empty for {cid} — marking .pypdf-empty (OCR candidate)")
        with open(pypdf_empty_path, "w", encoding="utf-8") as f:
            f.write(f"pypdf returned empty for {cid} at {pdf_path}\n")
        return None
    except Exception as e:
        print(f"  Failed to extract from {pdf_path}: {e} — marking .pypdf-error (retryable)")
        try:
            with open(pypdf_error_path, "w", encoding="utf-8") as f:
                f.write(f"pypdf error for {cid} at {pdf_path}: {e}\n")
        except Exception:
            pass
        return None

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full)
    return text_path


# ── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test: build a small slice (1 page = 500 records) and print summary.
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bills_index_sample.json"
    res = build_index(out, max_pages=1)
    print(json.dumps(res, indent=2))
