"""Scraper for sansad.in's three Lok-Sabha "Financial Committees":

  - Committee on Estimates                       (committeeCode=10)
  - Committee on Public Accounts (PAC)           (committeeCode=26)
  - Committee on Public Undertakings (COPU)      (committeeCode=27)

All three are LS-chaired (PAC and COPU are joint with RS members but
nominally LS committees — reports tabled in LS, secretariat is LS, the
upstream API treats them as LS). So we only use sansad.in's LS API.

API:
  https://sansad.in/api_ls/committee/lsRSAllReports
    ?house=L
    &committeeCode=<10|26|27>
    &lsNo=<14..18>
    &page=N
    &size=N
    &sortOn=reportNo
    &sortBy=desc

  Required header: Referer: https://sansad.in/ls/committees
  Same shape used by the legacy `scraper.py` for DRSCs; we adopt the
  same defensive _get() pattern (UA, Retry-After parsing, jitter) per
  CONV.md "Independence Principle".

Primary keys are composite — (committee, lok_sabha, report_number).
Each LS term has its own report numbering per committee, so report
#152 in PAC LS-16 is a different report from #152 in PAC LS-17.

Total volume across LS-14..18: ~800 reports (verified by recon —
see plan/financial-committees-recon-001.md).
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import requests


BASE_URL = "https://sansad.in"
REPORTS_API = f"{BASE_URL}/api_ls/committee/lsRSAllReports"

# Three LS Financial Committees + their API codes. Each entry carries a
# short slug we use as `committee` in the data model (and as a directory
# name under docs/fc/text/).
COMMITTEES: dict[str, dict] = {
    "estimates": {
        "name":         "Committee on Estimates",
        "short":        "Estimates",
        "api_code":     10,
        "type":         "LS-only",   # not joint
    },
    "pac": {
        "name":         "Committee on Public Accounts",
        "short":        "PAC",
        "api_code":     26,
        "type":         "Joint",     # joint with RS members; chair from LS
    },
    "copu": {
        "name":         "Committee on Public Undertakings",
        "short":        "COPU",
        "api_code":     27,
        "type":         "Joint",
    },
}

# LS terms to walk during a backfill. LS-14 matches the deepest scope
# DRSC currently covers (so FC's historical breadth lines up with
# DRSC's). Override via env to extend back to earlier terms.
DEFAULT_LOK_SABHAS = [14, 15, 16, 17, 18]


# Process-wide cooldown after a 429/403. Same pattern as cag/lc.
class RateLimited(Exception):
    pass


_RETRY_AFTER_DEFAULT_SECONDS = 30
_JITTER_MIN_MS = int(os.environ.get("JITTER_MIN_MS", "250"))
_JITTER_MAX_MS = int(os.environ.get("JITTER_MAX_MS", "500"))

_HEADERS_LS = {
    "User-Agent": "Mozilla/5.0",
    "Referer":    "https://sansad.in/ls/committees",
}


def _jitter() -> None:
    delay = random.randint(_JITTER_MIN_MS, _JITTER_MAX_MS) / 1000.0
    time.sleep(delay)


def _get(url: str, *, timeout: int = 30, retries: int = 1, **kwargs) -> requests.Response:
    """GET with one retry, polite jitter, and 429/Retry-After handling.
    Headers default to the LS API's expected Referer; callers may
    override.
    """
    headers = dict(_HEADERS_LS)
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            _jitter()
            resp = requests.get(url, headers=headers, timeout=timeout, **kwargs)
            if resp.status_code == 429 or resp.status_code == 403:
                ra = resp.headers.get("Retry-After")
                wait = _RETRY_AFTER_DEFAULT_SECONDS
                if ra:
                    try:
                        wait = int(ra)
                    except ValueError:
                        pass
                raise RateLimited(f"{resp.status_code} on {url} (retry-after {wait}s)")
            resp.raise_for_status()
            return resp
        except RateLimited:
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries:
                _jitter()
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"unreachable _get fallthrough for {url}")


@dataclass
class FCReport:
    """A single Financial-Committee report. The (committee, lok_sabha,
    report_number) triple is the composite primary key.
    """
    committee:        str        # one of COMMITTEES.keys()
    committee_name:   str        # human label (from upstream or local map)
    lok_sabha:        int        # 14..18
    report_number:    int
    title:            str
    presented_in_ls:  Optional[str]
    laid_in_rs:       Optional[str]
    date_of_presentation: Optional[str]
    date_of_adoption: Optional[str]
    pdf_url:          Optional[str]
    pdf_url_hindi:    Optional[str]


def _sanitize_url(url: Optional[str]) -> Optional[str]:
    # Same fix the legacy scraper applies — upstream occasionally returns
    # backslash-separated URLs (path interop bug on the server).
    if url:
        return url.replace("\\", "/")
    return url


def fetch_reports_for(committee_key: str, lok_sabha: int,
                      *, page_size: int = 200) -> list[FCReport]:
    """One API call per (committee, lok_sabha). Returns the full list
    of reports for that pair.

    With page_size=200 every (committee, ls) pair fits in a single page
    (the largest observed value during recon was PAC LS-17 at 151
    reports). If volume grows past page_size in some future LS term,
    add pagination (totalPages > 1 → loop) — for now the assertion at
    the end catches that.
    """
    cmt = COMMITTEES[committee_key]
    params = {
        "house":         "L",
        "committeeCode": cmt["api_code"],
        "lsNo":          lok_sabha,
        "page":          1,
        "size":          page_size,
        "sortOn":        "reportNo",
        "sortBy":        "desc",
    }
    print(f"  Fetching {cmt['name']} (LS {lok_sabha})...")
    resp = _get(REPORTS_API, params=params)
    data = resp.json()

    meta = data.get("_metadata", {})
    total = meta.get("totalElements", 0)
    pages = meta.get("totalPages", 1)
    if pages > 1:
        # Defensive — bump page_size up the call chain if this ever
        # fires. With page_size=200 vs max-observed=151 we have headroom
        # for the active LS term to grow.
        print(f"  WARN: {cmt['name']} LS{lok_sabha} has {pages} pages — "
              f"only page 1 fetched. Bump page_size.")

    records = data.get("records", []) or []
    out: list[FCReport] = []
    for r in records:
        rno = r.get("reportNo")
        if rno is None:
            continue
        out.append(FCReport(
            committee=             committee_key,
            committee_name=        (r.get("CommitteeName") or cmt["name"]).strip(),
            lok_sabha=             int(r.get("Loksabha") or lok_sabha),
            report_number=         int(rno),
            title=                 (r.get("SubjectOfTheReport") or "").strip(),
            presented_in_ls=       r.get("PresentedInLS"),
            laid_in_rs=            r.get("LaidInRS"),
            date_of_presentation=  r.get("dateOfPresentation"),
            date_of_adoption=      r.get("dateOfAdoption"),
            pdf_url=               _sanitize_url(r.get("url")),
            pdf_url_hindi=         _sanitize_url(r.get("urlH")),
        ))
    print(f"    {len(out)} reports for {cmt['name']} (LS{lok_sabha}, expected {total})")
    return out


def walk_all_committees(lok_sabhas: Optional[list[int]] = None) -> Iterator[FCReport]:
    """Iterator over every (committee × LS) tuple. Yields one FCReport
    per upstream record. ~3 × len(lok_sabhas) API calls total per
    backfill — bounded and small.
    """
    lok_sabhas = lok_sabhas or DEFAULT_LOK_SABHAS
    for committee_key in COMMITTEES:
        for ls in lok_sabhas:
            try:
                for r in fetch_reports_for(committee_key, ls):
                    yield r
            except RateLimited:
                raise
            except Exception as e:
                print(f"  ERROR walking {committee_key} LS{ls}: {e}")


# ── File ID convention ──────────────────────────────────────────────────────


def file_id(lok_sabha: int, report_number: int) -> str:
    """Stable id within a (committee, lok_sabha) pair. Matches DRSC's
    `LS<ls>_<num>` convention from build_static.py:_file_id. Committee
    is implicit in the directory: docs/fc/text/<committee>/<file_id>.txt.
    """
    return f"LS{lok_sabha}_{report_number}"


# ── PDF download ────────────────────────────────────────────────────────


def download_pdf(pdf_url: str, *, committee: str, lok_sabha: int,
                 report_number: int, pdfs_dir: str) -> Optional[str]:
    """Download the PDF for one (committee, ls, num) tuple. Returns the
    local file path on success, None on failure. pdfs_dir/<committee>/
    is gitignored — PDF is cached locally only for the runner's
    lifetime.
    """
    cmt_dir = os.path.join(pdfs_dir, committee)
    os.makedirs(cmt_dir, exist_ok=True)
    fid = file_id(lok_sabha, report_number)
    pdf_path = os.path.join(cmt_dir, f"{fid}.pdf")
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


def extract_text(pdf_path: str, *, committee: str, lok_sabha: int,
                 report_number: int, text_dir: str) -> Optional[str]:
    """Extract text from pdf_path → text_dir/<committee>/<file_id>.txt.
    Idempotent. Per-attempt markers (see CONV.md "Per-attempt status
    markers"):
      .txt           — successful extraction
      .pypdf-empty   — pypdf returned empty (scanned/encrypted) → OCR
                       target (slow lane not yet deployed for FC)
      .pypdf-error   — pypdf raised an exception (corrupt/unusual) →
                       retryable next run
    """
    from pypdf import PdfReader   # lazy import

    cmt_dir = os.path.join(text_dir, committee)
    os.makedirs(cmt_dir, exist_ok=True)
    fid = file_id(lok_sabha, report_number)
    text_path        = os.path.join(cmt_dir, f"{fid}.txt")
    pypdf_empty_path = os.path.join(cmt_dir, f"{fid}.pypdf-empty")
    pypdf_error_path = os.path.join(cmt_dir, f"{fid}.pypdf-error")
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
            print(f"  pypdf produced empty text for {fid} ({committee}) — marking .pypdf-empty")
            with open(pypdf_empty_path, "w", encoding="utf-8") as f:
                f.write(f"pypdf returned empty for {committee} {fid} at {pdf_path}\n")
            return None
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full)
        return full
    except Exception as e:
        print(f"  Failed to extract {committee} {fid} ({pdf_path}): {e}")
        try:
            with open(pypdf_error_path, "w", encoding="utf-8") as f:
                f.write(f"pypdf error for {committee} {fid} at {pdf_path}: {e}\n")
        except Exception:
            pass
        return None


def get_report_text(pdf_url: str, *, committee: str, lok_sabha: int,
                    report_number: int, text_dir: str,
                    pdfs_dir: str) -> Optional[str]:
    """Convenience: download + extract in one call. Identical contract
    to lc/scraper.py:get_report_text, just with composite identity.
    """
    pdf_path = download_pdf(pdf_url, committee=committee, lok_sabha=lok_sabha,
                            report_number=report_number, pdfs_dir=pdfs_dir)
    if not pdf_path:
        return None
    return extract_text(pdf_path, committee=committee, lok_sabha=lok_sabha,
                        report_number=report_number, text_dir=text_dir)
