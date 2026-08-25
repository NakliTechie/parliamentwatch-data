"""CAG (Comptroller and Auditor General of India) scraper.

Cross-origin browser fetches are blocked by upstream; this mirror runs
server-side and re-publishes the data with CORS-friendly headers.

Site shape:
  LISTING : https://cag.gov.in/en/audit-report?page=<N>      # 10/page, default newest first
  DETAIL  : https://cag.gov.in/en/audit-report/details/<id>
  PDF     : https://cag.gov.in/webroot/uploads/download_audit_report/<year>/<filename>.pdf
            (cag.gov.in moved the PDF asset root from /uploads/ to
             /webroot/uploads/ circa late 2025; older listing HTML and
             stale reports.json entries with /uploads/ now 302 → 404.
             _normalize_pdf_url() rewrites both shapes to the canonical
             form; the one-shot migration of docs/cag/reports.json was
             committed in this same change.)

The corpus is ~2,710 reports total (page 1..271). Year filtering via URL
params is broken (?year=N is silently ignored); we derive year from the
publication date in our index.

Independence from DRSC scraper.py: no shared imports, no shared state.
The HTTP / jitter / rate-limit primitives are duplicated by design — per
spec v0.4 §"Independence Principle". Refactor to a shared helper later
when a third corpus arrives and the duplication starts to bite.
"""

from __future__ import annotations

import html
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Iterable, Iterator, Optional

import requests
from pypdf import PdfReader

# ── Constants ───────────────────────────────────────────────────────────────

BASE_URL = "https://cag.gov.in"
LISTING_URL = BASE_URL + "/en/audit-report"
DETAIL_URL_FMT = BASE_URL + "/en/audit-report/details/{id}"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Politeness — random per-fetch sleep so we don't hammer cag.gov.in. Same
# defaults as the DRSC scraper. Set JITTER_MIN_MS / JITTER_MAX_MS to 0 for
# a quick local test.
_JITTER_MIN_MS = int(os.environ.get("JITTER_MIN_MS", "250"))
_JITTER_MAX_MS = int(os.environ.get("JITTER_MAX_MS", "500"))


class RateLimited(Exception):
    """Raised when cag.gov.in returns 429 / 403 — caller should stop the loop."""


def _jitter() -> None:
    if _JITTER_MAX_MS <= 0:
        return
    ms = random.randint(_JITTER_MIN_MS, _JITTER_MAX_MS)
    time.sleep(ms / 1000.0)


def _get(url: str, *, timeout: int = 30, retries: int = 4, **kwargs) -> requests.Response:
    """HTTP GET with rate-limit-aware error handling + one retry on transient
    timeouts. Default 30s — generous enough to ride out cag.gov.in's
    occasional slow-response blips (run 25619522300 caught a 15s+
    listing-page response and crashed the whole backfill workflow before
    this fix).

    RateLimited / HTTPError still short-circuit immediately (signal, not
    noise). Only ReadTimeout / ConnectionError trigger the retry path.
    Callers needing longer timeouts (PDF downloads, ~180s) pass
    `timeout=` explicitly.
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        _jitter()
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout, **kwargs)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))   # tiny backoff before retry
                continue
            raise
        if resp.status_code in (429, 403):
            retry_after_hdr = resp.headers.get("Retry-After")
            suggested_wait = None
            if retry_after_hdr:
                try:
                    suggested_wait = int(retry_after_hdr)
                except ValueError:
                    # Could be an HTTP-date instead of seconds; ignore for now.
                    pass
            msg = f"{resp.status_code} from {url}"
            if suggested_wait is not None:
                msg += f" (Retry-After: {suggested_wait}s)"
            raise RateLimited(msg)
        resp.raise_for_status()
        return resp
    # Unreachable — loop returns or raises.
    raise last_err if last_err else RuntimeError("_get exhausted")


# ── Listing walker ──────────────────────────────────────────────────────────

_DETAIL_ID_RE = re.compile(r'audit-report/details/(\d+)')


def fetch_listing_page(page: int) -> list[int]:
    """Return the list of detail-page IDs on listing page `page`. Empty list
    means the page is past the last one with content.

    cag.gov.in returns 404 for pages past the corpus end (not an empty
    200 like the local-probe via curl-piped-to-grep suggested during
    recon). Catch and treat as "no more". Backfill run 25608634028
    crashed on `?page=272` before this fix landed.
    """
    url = f"{LISTING_URL}?page={page}"
    try:
        resp = _get(url)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return []
        raise
    seen = set()
    ids: list[int] = []
    for m in _DETAIL_ID_RE.finditer(resp.text):
        rid = int(m.group(1))
        if rid in seen:
            continue
        seen.add(rid)
        ids.append(rid)
    return ids


def walk_listing(*, max_pages: Optional[int] = None) -> Iterator[int]:
    """Yield report IDs in listing order (newest first by default).

    `max_pages=None` walks until the listing returns an empty page (full
    archive — currently ~271 pages). Pass a small int for a quick slice
    (initial seeding, dev runs).
    """
    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            return
        ids = fetch_listing_page(page)
        if not ids:
            return
        for rid in ids:
            yield rid
        page += 1


# ── Detail page parser ─────────────────────────────────────────────────────

# Detail pages render metadata as <div class="labelTitleBold">LABEL</div>
# followed by <div class="labelItemBold">VALUE</div> pairs. The title is in
# an <h3 class="singTitle">. The PDF link is the first <a> with href to
# /uploads/download_audit_report/. The file size is in a "(X.XX MB)" string
# near the PDF link.
_TITLE_RE     = re.compile(r'<h3\s+class="singTitle">\s*(.+?)\s*</h3>', re.DOTALL)
_LABEL_RE     = re.compile(
    r'<div class="labelTitleBold">\s*(.+?)\s*</div>\s*'
    r'<div class="labelItemBold">\s*(.+?)\s*</div>',
    re.DOTALL,
)
# PDF link regex.
#
# Captures BOTH the historical absolute-URL form
#   https://cag.gov.in/uploads/download_audit_report/<year>/<file>.pdf
# AND the new path scheme cag.gov.in switched to circa late 2025
#   /webroot/uploads/download_audit_report/<year>/<file>.pdf  (relative)
#
# After capture, _normalize_pdf_url() rewrites either shape to the
# canonical absolute https://cag.gov.in/webroot/uploads/... form so
# downloads succeed. The non-capturing groups handle the optional
# scheme/domain and the optional /webroot/ prefix.
_PDF_RE       = re.compile(
    r'href="((?:https://cag\.gov\.in)?/(?:webroot/)?uploads/download_audit_report/[^"]+\.pdf)"',
    re.IGNORECASE,
)
_FILESIZE_RE  = re.compile(r'\(([0-9.]+)\s*(KB|MB|GB)\)', re.IGNORECASE)
_HTML_TAG_RE  = re.compile(r'<[^>]+>')
_WHITESPACE_RE = re.compile(r'\s+')


def _clean_text(s: str) -> str:
    """Strip HTML tags, decode HTML entities, collapse whitespace.

    HTML entity decoding matters because cag.gov.in's pages double-encode
    ampersands in some fields. Without it, a "J&K" report's PDF URL gets
    extracted as `J&amp;amp;K...` which 404s on download. Caught by OCR
    run 25608708774's failure on id 125145 (Jammu & Kashmir Revenues).
    """
    s = _HTML_TAG_RE.sub('', s)
    s = html.unescape(s)
    s = _WHITESPACE_RE.sub(' ', s).strip()
    return s


# Best-effort year inference from publication-date strings. CAG date strings
# vary: "30/06/2023", "30 June 2023", "2023-06-30", etc. Fall back to None.
_YEAR_FROM_DATE = re.compile(r'(19|20)\d{2}')


def _infer_year(date_str: str) -> Optional[int]:
    if not date_str:
        return None
    m = _YEAR_FROM_DATE.search(date_str)
    return int(m.group(0)) if m else None


# Best-effort report-type / report-number extraction from title. CAG title
# formats vary across years and gov-types:
#   "Report No. 6 of 2026 (Financial Audit)"
#   "Report 2 of 2026: ..."
#   "Report No. 3 of 2023 - State Finances"
# The regex tolerates "No.", "No", or no separator at all.
_REPORT_NO_RE     = re.compile(r'\bReport\s+(?:No\.?\s*)?([0-9]+)\s+of\s+(\d{4})\b', re.IGNORECASE)
_REPORT_TYPE_RE   = re.compile(r'\(([^)]+?Audit)\)', re.IGNORECASE)


@dataclass
class CAGReport:
    id: int                        # detail-page integer ID — stable, unique
    title: str = ""
    report_no: Optional[str] = None     # "6 of 2026" if extractable from title
    report_type: Optional[str] = None   # Financial / Compliance / Performance Audit
    gov_type: Optional[str] = None      # Union / State / Local Bodies
    department: Optional[str] = None
    sector: Optional[str] = None
    state: Optional[str] = None         # set when gov_type is State
    date_tabled: Optional[str] = None   # raw string from page (date format varies)
    date_sent: Optional[str] = None     # raw string from page
    year: Optional[int] = None          # inferred from date_tabled/date_sent/title
    pdf_url: Optional[str] = None
    file_size_mb: Optional[float] = None
    detail_url: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


def _html_unescape_loop(s: str) -> str:
    """html.unescape only undoes one level of encoding. cag.gov.in
    occasionally emits double-encoded entities (`&amp;amp;` for `&`),
    so we loop until no further change. Bounded to avoid pathological
    input loops.

    NOTE: do NOT use this on PDF URLs — cag.gov.in's filesystem has
    literal `&amp;` in J&K-style filenames, so the URL must KEEP the
    `&amp;` (not decode it to `&`). Use `html.unescape(url)` exactly
    once for PDF URLs (see _normalize_pdf_url below) so that `&amp;amp;`
    → `&amp;` (which the server expects) and stops there.
    """
    for _ in range(4):
        prev = s
        s = html.unescape(s)
        if s == prev:
            break
    return s


def _normalize_pdf_url(raw_href: str) -> str:
    """Normalize a PDF href captured from a CAG detail page into the
    canonical downloadable URL.

    Two shapes accepted:
      - https://cag.gov.in/[webroot/]uploads/download_audit_report/...  (absolute)
      - /[webroot/]uploads/download_audit_report/...                    (relative)

    Two transforms applied:
      1. Path migration: if /uploads/ without /webroot/, insert /webroot/.
         cag.gov.in restructured the path scheme circa late 2025; the
         old /uploads/ path now 302s to a non-existent /hi/ path → 404.
      2. HTML unescape ONCE only. Server filenames with `&` are stored
         as literal `&amp;`; the listing page HTML-encodes them once
         more to `&amp;amp;`. Single unescape: `&amp;amp;` → `&amp;`
         which matches the on-disk filename. Multi-level unescape would
         strip down to `&` and 404. See _html_unescape_loop docstring.
    """
    url = html.unescape(raw_href)

    # Make absolute.
    if url.startswith("/"):
        url = "https://cag.gov.in" + url

    # Insert /webroot/ if the path is the old scheme.
    if url.startswith("https://cag.gov.in/uploads/"):
        url = url.replace(
            "https://cag.gov.in/uploads/",
            "https://cag.gov.in/webroot/uploads/",
            1,
        )

    return url


def parse_detail_html(report_id: int, html: str) -> CAGReport:
    rep = CAGReport(id=report_id, detail_url=DETAIL_URL_FMT.format(id=report_id))

    m = _TITLE_RE.search(html)
    if m:
        rep.title = _clean_text(m.group(1))

    # Pull all label-value pairs; map known labels to fields.
    for label_raw, value_raw in _LABEL_RE.findall(html):
        label = _clean_text(label_raw).lower()
        value = _clean_text(value_raw)
        if not value:
            continue
        if 'date on which report tabled' in label:
            rep.date_tabled = value
        elif 'date of sending the report' in label:
            rep.date_sent = value
        elif label == 'government type':
            rep.gov_type = value
        elif 'union department' in label or label == 'department' or 'department/sector' in label:
            rep.department = value
        elif label == 'sector' or label == 'sector in india':
            rep.sector = value
        elif 'state' in label and 'union' not in label:
            rep.state = value

    # Year inference: prefer date_tabled, fall back to date_sent, then title.
    rep.year = _infer_year(rep.date_tabled) or _infer_year(rep.date_sent) or _infer_year(rep.title)

    # Report number + type from title
    m = _REPORT_NO_RE.search(rep.title or "")
    if m:
        rep.report_no = f"{m.group(1)} of {m.group(2)}"
    m = _REPORT_TYPE_RE.search(rep.title or "")
    if m:
        rep.report_type = m.group(1).strip()

    # PDF link. _normalize_pdf_url handles (a) the relative vs absolute
    # href shape, (b) the /uploads/ → /webroot/uploads/ migration cag.gov.in
    # rolled out in late 2025, and (c) single-shot html.unescape that
    # preserves `&amp;` in J&K-style filenames (the server's filesystem
    # stores `&` as literal `&amp;` chars). See _normalize_pdf_url docstring.
    m = _PDF_RE.search(html)
    if m:
        rep.pdf_url = _normalize_pdf_url(m.group(1))

    # File size — usually formatted as "(3.81 MB)" near the PDF link.
    m = _FILESIZE_RE.search(html)
    if m:
        n = float(m.group(1))
        unit = m.group(2).upper()
        rep.file_size_mb = n if unit == 'MB' else (n / 1024 if unit == 'KB' else n * 1024)

    return rep


def fetch_detail(report_id: int) -> CAGReport:
    resp = _get(DETAIL_URL_FMT.format(id=report_id))
    return parse_detail_html(report_id, resp.text)


# ── PDF download + text extraction ─────────────────────────────────────────

def _ensure_dirs(text_dir: str, pdfs_dir: str) -> None:
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(pdfs_dir, exist_ok=True)


def download_pdf(pdf_url: str, report_id: int, *, pdfs_dir: str) -> Optional[str]:
    """Download <id>.pdf into pdfs_dir. Returns filesystem path, or None on
    non-rate-limit failure. Idempotent — skips if already present."""
    os.makedirs(pdfs_dir, exist_ok=True)
    pdf_path = os.path.join(pdfs_dir, f"{report_id}.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    try:
        resp = _get(pdf_url, timeout=180, stream=True)
        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return pdf_path
    except RateLimited:
        raise
    except Exception as e:
        print(f"  Failed to download {pdf_url}: {e}")
        return None


def extract_text(pdf_path: str, report_id: int, *, text_dir: str) -> Optional[str]:
    """Extract text from pdf_path → text_dir/<id>.txt. Idempotent.

    Per-attempt status markers (sidecar files in text_dir):
      .txt           — successful extraction (this run or earlier)
      .pypdf-empty   — pypdf returned empty (scanned/encrypted) → OCR target
      .pypdf-error   — pypdf raised an exception (corrupt/unusual) → retryable
    The build script's candidate selection reads these to avoid re-downloading
    known-bad PDFs every hour. See CONV.md "Per-attempt status markers".
    """
    os.makedirs(text_dir, exist_ok=True)
    text_path        = os.path.join(text_dir, f"{report_id}.txt")
    pypdf_empty_path = os.path.join(text_dir, f"{report_id}.pypdf-empty")
    pypdf_error_path = os.path.join(text_dir, f"{report_id}.pypdf-error")
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            return f.read()

    try:
        reader = PdfReader(pdf_path)
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        full = "\n\n".join(parts)
        if not full.strip():
            # pypdf returned empty for the whole PDF — likely scanned/encrypted.
            # Don't write a 0-byte text file (would lie in manifest/bundle/index).
            # Drop a `.pypdf-empty` marker so this PDF gets queued for OCR
            # and skipped by future backfills (saves re-download cost).
            #
            # Stale `.pypdf-error` markers from prior attempts are left in
            # place rather than deleted — `git add docs/cag/text/` doesn't
            # stage deletions of tracked files (would need -A), so deleting
            # on disk only is a no-op for the published mirror. The audit
            # logic (compute_audit in build_cag.py) checks markers in
            # precedence order: .txt > .ocr-failed > .pypdf-empty >
            # .pypdf-error > never_attempted, so stale lower-priority
            # markers don't affect counts.
            print(f"  pypdf produced empty text for id={report_id} — marking .pypdf-empty (OCR candidate)")
            with open(pypdf_empty_path, "w", encoding="utf-8") as f:
                f.write(f"pypdf returned empty for id={report_id} at {pdf_path}\n")
            return None
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full)
        return full
    except Exception as e:
        print(f"  Failed to extract text from {pdf_path}: {e} — marking .pypdf-error (retryable)")
        # Write a `.pypdf-error` marker. Unlike .pypdf-empty, this is a
        # retryable signal: the next backfill WILL retry (transient corruption,
        # pypdf bug, etc could resolve itself). The marker is overwritten on
        # each attempt to capture the latest error message.
        try:
            with open(pypdf_error_path, "w", encoding="utf-8") as f:
                f.write(f"pypdf error for id={report_id} at {pdf_path}: {e}\n")
        except Exception:
            pass
        return None


def get_report_text(pdf_url: str, report_id: int, *, text_dir: str, pdfs_dir: str) -> Optional[str]:
    """Convenience: download + extract in one call. Returns extracted text or None."""
    pdf_path = download_pdf(pdf_url, report_id, pdfs_dir=pdfs_dir)
    if not pdf_path:
        return None
    return extract_text(pdf_path, report_id, text_dir=text_dir)
