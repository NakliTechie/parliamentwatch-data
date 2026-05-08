"""Download report PDFs and extract text (vendored from upstream).

This fork adds politeness controls for the historical-backfill phase: a small
random jitter between fetches and a short-circuit on 429/403 (so we bail out
the moment sansad.in rate-limits us, instead of retry-storming).
"""

import os
import random
import time
import requests
from pypdf import PdfReader
from config import PDFS_DIR, TEXT_DIR

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Politeness — random per-fetch sleep so we don't hammer sansad.in. ~1 req/sec
# across all workers, well below anything that should trip bot defences. Set
# JITTER_MIN_MS / JITTER_MAX_MS to 0 to disable for a quick local test.
_JITTER_MIN_MS = int(os.environ.get("JITTER_MIN_MS", "250"))
_JITTER_MAX_MS = int(os.environ.get("JITTER_MAX_MS", "500"))


class RateLimited(Exception):
    """Raised when sansad.in returns 429/403 — caller should stop the loop."""


def ensure_dirs(committee_key):
    os.makedirs(os.path.join(PDFS_DIR, committee_key), exist_ok=True)
    os.makedirs(os.path.join(TEXT_DIR, committee_key), exist_ok=True)


def _safe_name(report_number):
    return str(report_number).replace("/", "-").replace(" ", "_")


def _jitter():
    if _JITTER_MAX_MS <= 0:
        return
    ms = random.randint(_JITTER_MIN_MS, _JITTER_MAX_MS)
    time.sleep(ms / 1000.0)


def download_pdf(pdf_url, committee_key, report_number):
    ensure_dirs(committee_key)
    safe_name = _safe_name(report_number)
    pdf_path = os.path.join(PDFS_DIR, committee_key, f"{safe_name}.pdf")

    if os.path.exists(pdf_path):
        return pdf_path

    pdf_url = pdf_url.replace("\\", "/")
    _jitter()
    try:
        resp = requests.get(pdf_url, timeout=180, stream=True, headers=_HEADERS)
        if resp.status_code in (429, 403):
            # Don't retry; tell the caller to stop the whole batch. The next
            # scheduled run picks up where we left off.
            raise RateLimited(f"{resp.status_code} from {pdf_url}")
        resp.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return pdf_path
    except RateLimited:
        raise
    except Exception as e:
        print(f"  Failed to download {pdf_url}: {e}")
        return None


def extract_text(pdf_path, committee_key, report_number):
    ensure_dirs(committee_key)
    safe_name = _safe_name(report_number)
    text_path = os.path.join(TEXT_DIR, committee_key, f"{safe_name}.txt")

    if os.path.exists(text_path):
        with open(text_path, "r") as f:
            return f.read()

    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        full_text = "\n\n".join(text_parts)
        with open(text_path, "w") as f:
            f.write(full_text)
        return full_text
    except Exception as e:
        print(f"  Failed to extract: {e}")
        return None


def get_report_text(pdf_url, committee_key, report_number):
    pdf_path = download_pdf(pdf_url, committee_key, report_number)
    if not pdf_path:
        return None
    return extract_text(pdf_path, committee_key, report_number)
