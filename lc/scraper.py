"""Law Commission of India scraper.

Source: https://lawcommissionofindia.nic.in/

Site is WordPress + S3WaaS (the standardised gov-of-India CMS). Same theme
that powers cag.gov.in. The REST API at /wp-json/ is auth-locked (401);
we parse HTML directly. No anti-bot, no CF in front, no rate limiting
observed during recon — but we adopt the same defensive _get() shape as
cag/scraper.py for safety.

Structure of the corpus:

  /law-commission-reports/             Landing — lists 22 Commission terms.
  /report_first/  ... /report_twentysecond/   Per-term page. Contains an
                                        HTML table with one row per report:
                                          report_number | title | date | pdf_link
  PDFs live on cdnbbsr.s3waas.gov.in/.../uploads/YYYY/MM/<hash>.pdf

Total corpus: ~280 reports across all 22 terms. Tiny by mirror standards
(smaller than every other corpus we maintain by an order of magnitude).

Primary key: `report_number` — int, sequential across all Commission
terms (1..278+). Same number across re-scrapes = same report. Tracked
upstream consistently.

Independence Principle (per CONV.md): no imports from drsc/scraper.py or
cag/scraper.py — code duplication is the explicit choice. The HTTP layer,
RateLimited exception, jitter helper are intentionally re-implemented
here to keep this corpus's data pipeline self-contained.

See plan/law-commission-recon-001.md for the full recon writeup.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import random
import re
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

import requests


BASE_URL    = "https://lawcommissionofindia.nic.in"
LISTING_URL = f"{BASE_URL}/law-commission-reports/"

# Per-Commission page URLs. Built from the English-ordinal slug found on the
# listing page. Order is 1st → 22nd so the scraper walks newest-last;
# pdf-extraction order can independently reverse this if newest-first is
# preferred for the candidate list.
ORDINAL_SLUGS = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth", "twentyfirst", "twentysecond",
]
COMMISSION_PAGE_FMT = BASE_URL + "/report_{slug}/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}

# ── Rate-limit-aware HTTP layer ──────────────────────────────────────────


class RateLimited(Exception):
    """Raised when the upstream signals 429/403. The orchestrator's per-run
    cooldown gate uses this to short-circuit the extraction phase.
    """


def _jitter() -> None:
    """Sleep 250-500ms between requests to be polite even though the upstream
    doesn't enforce a rate limit. Same jitter as CAG.
    """
    lo = int(os.environ.get("JITTER_MIN_MS", "250"))
    hi = int(os.environ.get("JITTER_MAX_MS", "500"))
    time.sleep(random.uniform(lo / 1000.0, hi / 1000.0))


def _get(url: str, *, timeout: int = 30, retries: int = 1, **kwargs) -> requests.Response:
    """HTTP GET with rate-limit-aware error handling + one retry on transient
    network blips. Honors `Retry-After` if the server provides it on a 429
    (see CONV.md "Respect 429 / Retry-After"). Adopted directly from
    cag/scraper.py — see that file's docstring for the full design notes.
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
            wait_hint = None
            if retry_after_hdr:
                try:
                    wait_hint = int(retry_after_hdr)
                except ValueError:
                    pass
            msg = f"{resp.status_code} from {url}"
            if wait_hint is not None:
                msg += f" (Retry-After: {wait_hint}s)"
            raise RateLimited(msg)
        resp.raise_for_status()
        return resp
    raise last_err if last_err else RuntimeError("_get exhausted")


# ── HTML helpers ─────────────────────────────────────────────────────────


# Strip HTML tags from a single cell, normalise whitespace, and decode HTML
# entities. Cheap regex-based — we don't pull in BeautifulSoup for parsing
# 22 tables with regular shape.
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE  = re.compile(r"\s+")


def _clean_cell(s: str) -> str:
    s = _TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# Pull the first .pdf href out of a cell. Tolerant of attributes order and
# entity-encoded `&amp;` (cag.gov.in had the same gotcha — see the
# cleanup_html_entities pass in build_cag.py for the back-history).
_PDF_HREF_RE = re.compile(
    r'href=["\']([^"\']+\.pdf)["\']', re.IGNORECASE
)


def _extract_pdf_url(cell_html: str) -> Optional[str]:
    m = _PDF_HREF_RE.search(cell_html)
    if not m:
        return None
    return html.unescape(m.group(1))


# Date parser — table cells look like "17th April 2023" or "17th March 2023".
# Returns (year, full_normalised_iso) tuple, or (None, original) on parse fail.
_DATE_RE = re.compile(
    r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s*,?\s*(\d{4})",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_date(raw: str) -> tuple[Optional[int], str]:
    """Return (year, ISO-yyyy-mm-dd). Falls back to (None, raw) on parse fail —
    the orchestrator preserves the raw string in reports.json regardless so
    nothing is lost.
    """
    m = _DATE_RE.search(raw or "")
    if not m:
        return (None, raw or "")
    day = int(m.group(1))
    mo  = _MONTHS.get(m.group(2).lower())
    yr  = int(m.group(3))
    if not mo:
        return (yr, raw)
    return (yr, f"{yr:04d}-{mo:02d}-{day:02d}")


# ── Listing + per-Commission walk ────────────────────────────────────────


@dataclass
class LCReport:
    report_number: int
    title: str
    date_submitted: str           # ISO yyyy-mm-dd when parseable, else raw
    date_submitted_raw: str        # original e.g. "17th April 2023"
    year: Optional[int]
    commission_term: int           # 1..22
    pdf_url: Optional[str]


_TABLE_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_TD_RE        = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)


def fetch_commission_page(slug: str, commission_term: int) -> list[LCReport]:
    """Fetch one Commission-term page and return the list of reports parsed
    from its HTML table. Tolerates rows with 3 cells (no date or no link)
    or 4 cells; rows that don't parse to a valid report_number are skipped
    (header rows, blank rows, decorative rows).
    """
    url = COMMISSION_PAGE_FMT.format(slug=slug)
    resp = _get(url)
    html_body = resp.text
    out: list[LCReport] = []
    for row_html in _TABLE_ROW_RE.findall(html_body):
        cells = _TD_RE.findall(row_html)
        if len(cells) < 2:
            continue   # not a data row
        # Cell 0: report number. Tolerate any prefix junk; pull first integer.
        cell0 = _clean_cell(cells[0])
        num_match = re.search(r"\d+", cell0)
        if not num_match:
            continue
        try:
            number = int(num_match.group(0))
        except ValueError:
            continue
        # Cell 1: title.
        title = _clean_cell(cells[1])
        if not title:
            continue
        # Cell 2: date (may be empty or missing on older Commission pages).
        date_raw = _clean_cell(cells[2]) if len(cells) >= 3 else ""
        year, date_iso = _parse_date(date_raw)
        # Last cell (3 or higher): PDF link. Search the WHOLE row's HTML so
        # we catch the link even if it's wrapped in extra spans / br tags.
        pdf_url = _extract_pdf_url(row_html)
        out.append(LCReport(
            report_number=number,
            title=title,
            date_submitted=date_iso,
            date_submitted_raw=date_raw,
            year=year,
            commission_term=commission_term,
            pdf_url=pdf_url,
        ))
    return out


def walk_all_commissions() -> Iterator[LCReport]:
    """Walk every Commission-term page (1st → 22nd) and yield all reports.
    Failed page fetches are logged + skipped — one bad Commission page
    doesn't stop the run.
    """
    for term_idx, slug in enumerate(ORDINAL_SLUGS, start=1):
        try:
            rows = fetch_commission_page(slug, term_idx)
        except RateLimited:
            raise   # caller decides — usually means stop the whole run
        except Exception as e:
            print(f"  [walk_all] {term_idx}th Commission page failed: {e}")
            continue
        for r in rows:
            yield r


# ── PDF download ────────────────────────────────────────────────────────


def download_pdf(pdf_url: str, report_number: int, *, pdfs_dir: str) -> Optional[str]:
    """Download the PDF for a single report. Returns the local file path
    on success, None on failure. pdfs/ directory is gitignored — the PDF
    is cached locally just for the duration of the runner job; OCR slow
    lane re-downloads when needed.
    """
    os.makedirs(pdfs_dir, exist_ok=True)
    pdf_path = os.path.join(pdfs_dir, f"{report_number}.pdf")
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


# ── pypdf extraction with per-attempt markers ────────────────────────────


def extract_text(pdf_path: str, report_number: int, *, text_dir: str) -> Optional[str]:
    """Extract text from pdf_path → text_dir/<number>.txt. Idempotent.

    Per-attempt status markers (sidecar files in text_dir):
      .txt           — successful extraction (this run or earlier)
      .pypdf-empty   — pypdf returned empty (scanned/encrypted) → OCR target
      .pypdf-error   — pypdf raised an exception (corrupt/unusual) → retryable

    Matches the marker contract used by cag/scraper.py + pdf_utils.py +
    bills/sansad/scraper.py — the build script reads them to short-circuit
    re-downloads of known-bad PDFs. See CONV.md "Per-attempt status markers".
    """
    from pypdf import PdfReader   # lazy import so callers without pypdf can import this module

    os.makedirs(text_dir, exist_ok=True)
    text_path        = os.path.join(text_dir, f"{report_number}.txt")
    pypdf_empty_path = os.path.join(text_dir, f"{report_number}.pypdf-empty")
    pypdf_error_path = os.path.join(text_dir, f"{report_number}.pypdf-error")
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
            print(f"  pypdf produced empty text for #{report_number} — marking .pypdf-empty (OCR candidate)")
            with open(pypdf_empty_path, "w", encoding="utf-8") as f:
                f.write(f"pypdf returned empty for report #{report_number} at {pdf_path}\n")
            return None
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full)
        return full
    except Exception as e:
        print(f"  Failed to extract from #{report_number} ({pdf_path}): {e} — marking .pypdf-error (retryable)")
        try:
            with open(pypdf_error_path, "w", encoding="utf-8") as f:
                f.write(f"pypdf error for report #{report_number} at {pdf_path}: {e}\n")
        except Exception:
            pass
        return None


def get_report_text(pdf_url: str, report_number: int, *, text_dir: str, pdfs_dir: str) -> Optional[str]:
    """Convenience: download + extract in one call. Returns extracted text
    or None. Identical contract to cag/scraper.py:get_report_text.
    """
    pdf_path = download_pdf(pdf_url, report_number, pdfs_dir=pdfs_dir)
    if not pdf_path:
        return None
    return extract_text(pdf_path, report_number, text_dir=text_dir)


# ── PDF archive (separate repo: sansadsaar-lc) ─────────────────────────────


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# Process-wide lock around the index.json read/modify/write inside
# archive_pdf(). build_lc.py uses EXTRACT_WORKERS=4 ThreadPoolExecutor;
# without this lock, two workers writing index.json concurrently can
# lose each other's entries (worker A reads, worker B reads, A writes,
# B writes — A's entry is gone). v0 of the archive feature surfaced
# this in run 25659927513: extracted=3 on the mirror but only 1 entry
# in index.json on the archive repo.
_archive_lock = threading.Lock()


def archive_pdf(pdf_path: str, report_number: int, pdf_url: str,
                *, archive_dir: str) -> bool:
    """Copy a freshly-downloaded PDF into the sansadsaar-lc archive layout.

    Rationale: lawcommissionofindia.nic.in is on WordPress / S3WaaS — gov-of-
    India CDN that has occasionally rotated URLs or dropped older content.
    We snapshot each PDF we fetch into a separate public repo so:
      - the app can offer an "archive copy" fallback link in the detail panel
      - the corpus survives an upstream takedown / URL rotation
      - third-party researchers can pull the whole archive with `git clone`

    Layout written under `archive_dir`:
      pdfs/<report_number>.pdf
      index.json   — dict keyed by report_number with {sha256, size_bytes,
                     archived_at (ISO), source_url}; merge semantics preserve
                     archive_dir as the source of truth across runs.

    Idempotent: if pdfs/<n>.pdf already exists AND has matching sha256 +
    size, nothing is rewritten. Returns True when something new was
    persisted, False otherwise.

    No-op (returns False) when `archive_dir` is falsy or missing — local-dev
    runs that don't set LC_ARCHIVE_DIR get default behaviour with no
    archival side effects.
    """
    if not archive_dir:
        return False
    if not os.path.isdir(archive_dir):
        # Fail loudly: the workflow sets this to a cloned repo and a missing
        # dir means the clone step was skipped or failed. Silent skip would
        # let us silently lose the archival promise.
        print(f"  [archive] LC_ARCHIVE_DIR={archive_dir!r} doesn't exist; skipping archive of #{report_number}")
        return False
    if not pdf_path or not os.path.isfile(pdf_path):
        return False

    pdfs_subdir = os.path.join(archive_dir, "pdfs")
    os.makedirs(pdfs_subdir, exist_ok=True)
    dest_path = os.path.join(pdfs_subdir, f"{report_number}.pdf")
    index_path = os.path.join(archive_dir, "index.json")

    src_size = os.path.getsize(pdf_path)

    # Serialise the read-modify-write of index.json across all worker
    # threads. The PDF copy is also done inside the lock — cheap (file
    # is local, kernel page cache), and keeps the function's contract
    # simple (return True iff something landed).
    with _archive_lock:
        # Load existing index (best-effort; treat malformed/missing as empty).
        index_obj: dict = {}
        if os.path.isfile(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_obj = json.load(f)
            except Exception as e:
                print(f"  [archive] index.json unparseable ({e}); rebuilding from disk")
                index_obj = {}
        reports = index_obj.get("reports") or {}

        existing = reports.get(str(report_number))
        if (os.path.isfile(dest_path)
                and existing
                and existing.get("size_bytes") == src_size):
            # Cheap short-circuit: same file already there. Avoid the
            # sha256 cost on every run.
            return False

        digest = _sha256_of(pdf_path)
        if (os.path.isfile(dest_path)
                and existing
                and existing.get("sha256") == digest
                and existing.get("size_bytes") == src_size):
            return False

        shutil.copyfile(pdf_path, dest_path)

        reports[str(report_number)] = {
            "report_number": report_number,
            "sha256":        digest,
            "size_bytes":    src_size,
            "archived_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_url":    pdf_url,
        }
        # Sort by report_number for a deterministic, diff-friendly index.json.
        sorted_reports = {str(k): reports[str(k)] for k in sorted(
            (int(k) for k in reports.keys()))}
        index_obj = {
            "version":      "1.0",
            "corpus":       "lc",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total":        len(sorted_reports),
            "reports":      sorted_reports,
        }
        # Atomic-ish write: temp + rename. Avoids a half-written
        # index.json on runner kill mid-update.
        tmp_path = index_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(index_obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, index_path)
        return True
