#!/usr/bin/env python3
"""Sync per-record text files from docs/<corpus>/text/ to the
sansadsaar-texts R2 bucket.

Why this exists: Workers Static Assets caps deployments at 20,000 files.
The text files are the bulk of that count (~14k of ~14.3k total). Moving
them to R2 keeps the static-asset bundle clear for shard / manifest
files. See plan/cloudflare-strategy-001.md §"Texts to R2" for the
strategy.

Idempotent: creates the bucket if it doesn't exist (ignores 409s),
HEADs each object first to skip uploads that are already there with
the same etag (size + content match). Safe to re-run after partial
runs.

Auth: reads CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID from env.
The token must include "Workers R2 Storage: Edit" permission. If it
was created with the "Edit Cloudflare Workers" template only, this
script will get 403s; the user needs to update the token's permissions
in the CF dashboard.

Usage:
    python3 scripts/r2_sync_texts.py             # sync all corpora
    python3 scripts/r2_sync_texts.py drsc cag    # sync specific corpora
"""

import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
BUCKET_NAME = "sansadsaar-texts"

# Tune for politeness against CF's R2 endpoint. Cloudflare's R2 REST
# API can absorb a few hundred req/s comfortably; we keep it modest
# so a backfill push doesn't burn through the daily class-A op budget.
PARALLELISM = int(os.environ.get("R2_SYNC_PARALLELISM", "16"))
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5


def _api_base() -> str:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not account_id:
        sys.exit("CLOUDFLARE_ACCOUNT_ID not set")
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}"


def _auth_headers() -> dict:
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        sys.exit("CLOUDFLARE_API_TOKEN not set")
    return {"Authorization": f"Bearer {token}"}


def _diagnose_token() -> None:
    """Print what the token can / can't do. Run on 403 to disambiguate
    'token missing' from 'token wrong perms' from 'wrong account scope'."""
    # 1. Token validity + identity
    vr = requests.get("https://api.cloudflare.com/client/v4/user/tokens/verify",
                      headers=_auth_headers())
    if vr.ok:
        body = vr.json()
        rsl = body.get("result", {}).get("status", "?")
        print(f"  [diag] token verify: {vr.status_code} status={rsl}")
    else:
        print(f"  [diag] token verify FAILED: {vr.status_code} {vr.text[:150]}")
        return

    # 2. Account-scoped R2 read (list buckets) — needs R2 Storage:Read
    lr = requests.get(f"{_api_base()}/r2/buckets", headers=_auth_headers())
    if lr.status_code == 200:
        names = [b["name"] for b in lr.json().get("result", {}).get("buckets", [])]
        print(f"  [diag] R2 list: 200 — {len(names)} bucket(s): {names}")
    elif lr.status_code == 403:
        print(f"  [diag] R2 list: 403 — token has NO R2 perms on account {os.environ.get('CLOUDFLARE_ACCOUNT_ID','?')[:8]}…")
    else:
        print(f"  [diag] R2 list: {lr.status_code} {lr.text[:200]}")


def ensure_bucket() -> None:
    """Idempotent bucket create. 409 = already exists = fine."""
    url = f"{_api_base()}/r2/buckets"
    r = requests.post(url, headers=_auth_headers(),
                      json={"name": BUCKET_NAME})
    if r.status_code in (200, 201):
        print(f"[bucket] created {BUCKET_NAME}")
        return
    if r.status_code == 409:
        print(f"[bucket] {BUCKET_NAME} already exists")
        return

    # Failure path — diagnose first.
    print(f"[bucket] create returned {r.status_code}: {r.text[:300]}")
    _diagnose_token()
    if r.status_code == 403:
        sys.exit(
            "\n[bucket] 403 from CF on POST. Common causes:\n"
            "  1. Token lacks 'Account · Workers R2 Storage · Edit' permission.\n"
            "     Fix: https://dash.cloudflare.com/profile/api-tokens → edit\n"
            "     the token, add the permission, save (no need to rotate the\n"
            "     GitHub secret if you edited in place).\n"
            "  2. Token was REPLACED rather than edited — GitHub secret still\n"
            "     holds the old value. Fix: `gh secret set CLOUDFLARE_API_TOKEN\n"
            "     --repo NakliTechie/parliamentwatch-data` and paste the new value.\n"
            "  3. Token's Account Resources scope doesn't include the account\n"
            "     specified by CLOUDFLARE_ACCOUNT_ID. Fix: edit the token's\n"
            "     scope to include the account that holds your R2 buckets.")
    sys.exit(f"[bucket] unrecoverable: {r.status_code}")


def head_object(key: str) -> Optional[dict]:
    """Return object metadata if it exists, None otherwise.

    Uses CF's REST API rather than the S3-compatible one so we can
    reuse the existing CF API token (no separate access-key creation).
    """
    url = f"{_api_base()}/r2/buckets/{BUCKET_NAME}/objects/{key}"
    r = requests.head(url, headers=_auth_headers())
    if r.status_code == 200:
        return {
            "etag": r.headers.get("etag", "").strip('"'),
            "content_length": int(r.headers.get("content-length", "0") or 0),
        }
    if r.status_code == 404:
        return None
    # Any other status: treat as unknown and re-upload to be safe.
    print(f"  [head] unexpected {r.status_code} for {key}; will re-upload")
    return None


def upload_object(key: str, body: bytes, content_type: str = "text/plain; charset=utf-8") -> bool:
    url = f"{_api_base()}/r2/buckets/{BUCKET_NAME}/objects/{key}"
    headers = {**_auth_headers(), "Content-Type": content_type}
    for attempt in range(MAX_RETRIES):
        r = requests.put(url, headers=headers, data=body)
        if r.status_code in (200, 201, 204):
            return True
        if r.status_code in (429, 500, 502, 503, 504):
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            print(f"  [upload] {key} got {r.status_code}; retry {attempt+1}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait)
            continue
        if r.status_code == 403:
            print(f"  [upload] 403 for {key} — token lacks R2 write perms")
            return False
        print(f"  [upload] {key} failed: {r.status_code} {r.text[:200]}")
        return False
    return False


def _content_md5_etag(data: bytes) -> str:
    """R2 etag for a single-part PUT is the MD5 hex digest of the body
    (no S3-style multipart suffix). Lets us short-circuit re-uploads
    of identical files cheaply."""
    return hashlib.md5(data).hexdigest()


def sync_corpus(corpus: str, stats: dict) -> None:
    text_dir = DOCS / corpus / "text"
    if not text_dir.is_dir():
        print(f"[{corpus}] no text/ dir — skipping")
        return
    files = sorted(p for p in text_dir.rglob("*.txt") if p.is_file())
    print(f"[{corpus}] {len(files)} files in {text_dir}")

    def _one(path: Path) -> str:
        rel = path.relative_to(text_dir).as_posix()
        key = f"{corpus}/{rel}"
        data = path.read_bytes()
        # Cheap dedupe: if the existing object's content-length AND
        # etag match what we'd upload, skip. ETag comparison is the
        # strong check; length is a fast pre-filter.
        existing = head_object(key)
        if existing is not None:
            expected_etag = _content_md5_etag(data)
            if (existing["content_length"] == len(data)
                    and existing["etag"].lower() == expected_etag.lower()):
                return "skipped"
        ok = upload_object(key, data)
        return "uploaded" if ok else "failed"

    with ThreadPoolExecutor(max_workers=PARALLELISM) as pool:
        futures = {pool.submit(_one, f): f for f in files}
        for fut in as_completed(futures):
            result = fut.result()
            stats[result] = stats.get(result, 0) + 1
            total = sum(stats.values())
            if total % 500 == 0:
                print(f"  progress: {total} processed ({stats})")


def main() -> int:
    selected = sys.argv[1:] or ["drsc", "cag", "bills", "lc", "fc", "debates"]
    print(f"Corpora to sync: {', '.join(selected)}")
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Parallelism: {PARALLELISM}")
    ensure_bucket()

    stats: dict = {}
    t0 = time.time()
    for corpus in selected:
        sync_corpus(corpus, stats)
    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"Totals: {stats}")
    failed = stats.get("failed", 0)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
