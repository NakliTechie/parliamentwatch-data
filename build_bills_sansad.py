#!/usr/bin/env python3
"""Bills (sansad-side) static data builder — orchestrator for parliamentwatch-data.

Per scheduled (or workflow_dispatch) run:
  1. Walk sansad.in's getBills API → in-memory list of normalised records
  2. Extract canonical text for as many missing PDFs as fit in the per-run
     budget (MAX_EXTRACTIONS_PER_RUN, MAX_RUN_SECONDS) — newest bills first
  3. Shard and write:
        docs/bills/index-meta.json       (shard manifest)
        docs/bills/index-<NN>.json       (~1000 records per shard)
        docs/bills/manifest.json         (deeper stats for diagnostics)
        docs/bills/meta.json             (status display for the chip)

Output goes under docs/bills/. CF Workers serves it at /bills/* on
sansadsaar-data.naklitechie.com.

Sharding rationale: full archive ~9.9k records ≈ 10.5 MB single-file. Under
the CF Workers 25 MiB cap today, but PRS-payload merge will grow the index
and we don't want to scramble the app once it's depending on a single
fixed shape. Sharded from day 1: predictable filenames, parallel app
fetch, smaller per-cron commit diffs. App-side: fetch index-meta.json
first, then fetch all (or selected) shards in parallel.

Independence Principle: this orchestrator does not import from
build_static.py (DRSC) or build_cag.py. The three corpora share a
docs/ tree but no Python state. HTTP / jitter / rate-limit / Retry-After
primitives are duplicated in bills/sansad/scraper.py by design.

Until v1.1.b lands a PRS-side scraper, the sansad-side records ARE the
merged index (no merge needed). When the PRS scraper joins, this
orchestrator gains a merge step that folds PRS payload into each record
before sharding — owned by this run as the canonical record (per spec
§"Source division of labour").
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))   # so `from bills.sansad.scraper import ...` works

from bills.sansad.scraper import (  # noqa: E402
    SCRAPER_VERSION,
    BILLS_API,
    RateLimited,
    collect_records,
    extract_canonical_text,
    CANONICAL_PDF_FIELDS,
)

# ── Paths ───────────────────────────────────────────────────────────────────

ASSETS    = ROOT / "docs"
DOCS      = ASSETS / "bills"
TEXT_DIR  = DOCS / "text"
PDFS_DIR  = DOCS / "pdfs"        # gitignored via docs/**/pdfs/
DOCS.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)
PDFS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_META_JSON = DOCS / "index-meta.json"
MANIFEST_JSON   = DOCS / "manifest.json"
META_JSON       = DOCS / "meta.json"
STATE_JSON      = DOCS / ".scraper_state.json"   # cooldown bookkeeping

LEGACY_INDEX_JSON = DOCS / "index.json"  # superseded by sharded layout

# ── Per-run budget ─────────────────────────────────────────────────────────

# How many missing PDFs to extract per run. Same shape as DRSC build_static
# / CAG build_cag.
MAX_EXTRACTIONS_PER_RUN = int(os.environ.get("MAX_EXTRACTIONS_PER_RUN", "200"))
MAX_RUN_SECONDS         = int(os.environ.get("MAX_RUN_SECONDS", "1800"))   # 30 min default
EXTRACT_WORKERS         = int(os.environ.get("EXTRACT_WORKERS", "4"))

# After a 429 / 403, skip the extraction phase for this many seconds so the
# next scheduled run doesn't immediately retry-storm. Tuned to be ≥ one
# cron interval (currently daily) so we always sit out at least one cycle
# after a rate-limit before attempting again.
RATE_LIMIT_COOLDOWN_SECONDS = int(os.environ.get("RATE_LIMIT_COOLDOWN_SECONDS", str(6 * 3600)))

# ── Sharding ───────────────────────────────────────────────────────────────

# Records per index shard. 1000 → ~1 MB per shard at current schema density.
# Aim to keep individual shards small enough that a single shard's diff
# stays readable in PRs and small enough that browser parallel-fetch is a
# meaningful speedup over one big file. Rebalance later if record schema
# grows materially (e.g. PRS payload adds large free-text fields).
SHARD_SIZE = int(os.environ.get("BILLS_SHARD_SIZE", "1000"))


# ── State helpers ──────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_state() -> dict:
    if not STATE_JSON.exists():
        return {}
    try:
        with open(STATE_JSON, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    with open(STATE_JSON, "w") as f:
        json.dump(state, f, indent=2)


def _is_in_cooldown(state: dict) -> tuple[bool, Optional[float]]:
    """Returns (in_cooldown, seconds_remaining_or_None)."""
    until = state.get("rate_limit_until")
    if not until:
        return False, None
    try:
        until_ts = datetime.fromisoformat(until.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return False, None
    now_ts = time.time()
    remaining = until_ts - now_ts
    return (remaining > 0), (remaining if remaining > 0 else None)


def _mark_rate_limited(state: dict, msg: str) -> None:
    until_ts = time.time() + RATE_LIMIT_COOLDOWN_SECONDS
    until_iso = datetime.fromtimestamp(until_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state["rate_limit_until"] = until_iso
    state["last_rate_limit_message"] = msg
    state["last_rate_limit_at"] = _now_iso()
    _save_state(state)


def _clear_rate_limit(state: dict) -> None:
    state.pop("rate_limit_until", None)
    state.pop("last_rate_limit_message", None)
    _save_state(state)


# ── Candidate selection ────────────────────────────────────────────────────

def _has_any_canonical_pdf(record: dict) -> bool:
    return any(record.get(f) for f in CANONICAL_PDF_FIELDS)


def select_candidates(records: list[dict]) -> list[dict]:
    """Select bills missing extracted text but having ≥1 canonical PDF URL.
    Sorted newest-first so the budget gets spent on the most-recent material.
    """
    candidates = []
    for r in records:
        cid = r.get("compositeId")
        if not cid:
            continue
        text_path = TEXT_DIR / f"{cid}.txt"
        if text_path.exists():
            continue
        if not _has_any_canonical_pdf(r):
            continue
        candidates.append(r)
    candidates.sort(
        key=lambda r: (r.get("billYear") or 0, r.get("billNumber") or ""),
        reverse=True,
    )
    return candidates


# ── Extraction phase ───────────────────────────────────────────────────────

def _process_one(record: dict) -> tuple[str, bool]:
    """Worker callable. Returns (compositeId, extracted). Raises RateLimited."""
    cid = record["compositeId"]
    text_path = extract_canonical_text(record, str(PDFS_DIR), str(TEXT_DIR))
    return cid, (text_path is not None)


def run_extractions(candidates: list[dict], state: dict) -> dict:
    """Process up to MAX_EXTRACTIONS_PER_RUN candidates in parallel. Hard
    wall-clock cap at MAX_RUN_SECONDS. Returns small status dict.
    """
    started = time.time()
    deadline = started + MAX_RUN_SECONDS
    submit_budget = min(MAX_EXTRACTIONS_PER_RUN, len(candidates))

    extracted = 0
    failed = 0
    rate_limited = False
    rate_limit_msg = ""

    if submit_budget == 0:
        return {"extracted": 0, "failed": 0, "rate_limited": False, "elapsed": 0.0}

    print(f"  Extracting up to {submit_budget} bills (workers={EXTRACT_WORKERS}, deadline={MAX_RUN_SECONDS}s)")

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
        futures = {
            ex.submit(_process_one, candidates[i]): candidates[i]
            for i in range(submit_budget)
        }
        for fut in as_completed(futures):
            if time.time() > deadline:
                print(f"  Wall-clock deadline reached at {extracted} extracted; bailing.")
                break
            record = futures[fut]
            try:
                cid, ok = fut.result()
            except RateLimited as e:
                rate_limited = True
                rate_limit_msg = str(e)
                print(f"  Rate-limited: {e}. Stopping the batch.")
                break
            except Exception as e:
                failed += 1
                print(f"  Worker error on {record.get('compositeId')}: {e}")
                continue
            if ok:
                extracted += 1
            else:
                failed += 1

    if rate_limited:
        _mark_rate_limited(state, rate_limit_msg)
    else:
        _clear_rate_limit(state)

    return {
        "extracted": extracted,
        "failed": failed,
        "rate_limited": rate_limited,
        "elapsed": time.time() - started,
    }


# ── Output: sharded index + meta + manifest ───────────────────────────────

def _shard_filename(idx: int) -> str:
    return f"index-{idx:02d}.json"


def write_sharded_index(records: list[dict]) -> dict:
    """Write the sharded index. Records sorted newest-first, chunked into
    SHARD_SIZE-record shards. Removes legacy single-file index.json and any
    orphan shards left over from a previous run with more records.

    Returns shard manifest entries (list of dicts).
    """
    # Stable sort: billYear DESC, billNumber DESC. None sorts last.
    sorted_records = sorted(
        records,
        key=lambda r: (r.get("billYear") or 0, r.get("billNumber") or ""),
        reverse=True,
    )

    # Clean previous shards + legacy single-file index.
    for p in glob.glob(str(DOCS / "index-*.json")):
        # Don't blow away index-meta.json (keep separate).
        if Path(p).name == INDEX_META_JSON.name:
            continue
        try:
            os.remove(p)
        except OSError:
            pass
    if LEGACY_INDEX_JSON.exists():
        try:
            os.remove(LEGACY_INDEX_JSON)
        except OSError:
            pass

    # Write shards.
    manifest_entries = []
    for i in range(0, len(sorted_records), SHARD_SIZE):
        chunk = sorted_records[i:i + SHARD_SIZE]
        idx = i // SHARD_SIZE
        fname = _shard_filename(idx)
        path = DOCS / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"records": chunk, "count": len(chunk),
                       "shard_index": idx, "scraper_version": SCRAPER_VERSION},
                      f, ensure_ascii=False, indent=2)
        years = [r["billYear"] for r in chunk if r.get("billYear")]
        manifest_entries.append({
            "file": fname,
            "count": len(chunk),
            "newest_billYear": max(years) if years else None,
            "oldest_billYear": min(years) if years else None,
        })
    return manifest_entries


def write_index_meta(records: list[dict], shard_entries: list[dict]) -> None:
    """Write index-meta.json — small file the app fetches first to discover
    shards. Includes top-level totals so the chip status can render without
    fetching any shard."""
    with_text = sum(1 for r in records if (TEXT_DIR / f"{r['compositeId']}.txt").exists())
    with_canonical_pdf = sum(1 for r in records if _has_any_canonical_pdf(r))
    meta = {
        "scraper_version": SCRAPER_VERSION,
        "scraped_at": _now_iso(),
        "source": BILLS_API,
        "shard_size": SHARD_SIZE,
        "totals": {
            "bills": len(records),
            "with_canonical_pdf": with_canonical_pdf,
            "with_text": with_text,
        },
        "shards": shard_entries,
    }
    with open(INDEX_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def write_manifest(records: list[dict], extract_status: dict) -> None:
    """Build manifest.json with deeper stats + per-bill `texts` map.

    The `texts` map is keyed by compositeId and used by the app to know
    which bills have extracted text without per-bill probing — same shape
    as DRSC's manifest.texts and CAG's manifest.texts. Without it, the
    app's "with text" count stays at 0 even when text/<id>.txt files exist
    on the mirror.
    """
    from collections import Counter
    by_status = Counter(r.get("status") or "(null)" for r in records)
    by_type   = Counter(r.get("billType") or "(null)" for r in records)
    by_year   = Counter(r.get("billYear") for r in records if r.get("billYear"))
    with_intro_pdf      = sum(1 for r in records if r.get("billIntroducedFile"))
    with_canonical_pdf  = sum(1 for r in records if _has_any_canonical_pdf(r))
    became_law          = sum(1 for r in records if r.get("actNo"))
    with_report_link    = sum(1 for r in records if r.get("reportFile"))
    with_gazette_link   = sum(1 for r in records if r.get("billGazettedFile"))

    # Per-bill texts map. Walk the records once + check existence per id.
    texts: dict = {}
    for r in records:
        cid = r["compositeId"]
        text_file = TEXT_DIR / f"{cid}.txt"
        if text_file.exists():
            texts[cid] = {"url": f"text/{cid}.txt"}
    with_text = len(texts)

    manifest = {
        "scraper_version": SCRAPER_VERSION,
        "scraped_at": _now_iso(),
        "source": BILLS_API,
        "totals": {
            "bills": len(records),
            "with_intro_pdf": with_intro_pdf,
            "with_canonical_pdf": with_canonical_pdf,
            "with_text": with_text,
            "became_law": became_law,
            "with_drsc_report_link": with_report_link,
            "with_gazette_link": with_gazette_link,
        },
        "by_status": dict(by_status.most_common()),
        "by_type": dict(by_type.most_common()),
        "year_range": {
            "min": min(by_year) if by_year else None,
            "max": max(by_year) if by_year else None,
        },
        "this_run": extract_status,
        "texts": texts,
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_meta(records: list[dict], extract_status: dict, state: dict) -> None:
    """Build meta.json — small status JSON for chip status display."""
    in_cooldown, remaining = _is_in_cooldown(state)
    with_text = sum(1 for r in records if (TEXT_DIR / f"{r['compositeId']}.txt").exists())
    meta = {
        "scraper_version": SCRAPER_VERSION,
        "last_update": _now_iso(),
        "last_run_status": "rate_limited" if extract_status.get("rate_limited") else "ok",
        "total": len(records),
        "with_text": with_text,
        "in_rate_limit_cooldown": in_cooldown,
        "cooldown_remaining_seconds": int(remaining) if remaining else None,
        "this_run_extracted": extract_status.get("extracted", 0),
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"[bills/sansad] starting — {SCRAPER_VERSION}")
    state = _load_state()

    # Phase 1: refresh index. Cheap (~17-35s) — always do this.
    print("  Building bill index from getBills API ...")
    t0 = time.time()
    records, duplicates = collect_records()
    print(f"  Index: {len(records)} records "
          f"({duplicates} composite-id collisions dropped) "
          f"in {time.time()-t0:.1f}s")

    # Phase 2: extract canonical text for missing bills, within budget.
    in_cooldown, remaining = _is_in_cooldown(state)
    if in_cooldown:
        print(f"  Skipping extraction phase — rate-limit cooldown active "
              f"({int(remaining)}s remaining)")
        extract_status = {"extracted": 0, "failed": 0, "rate_limited": True,
                          "elapsed": 0.0, "skipped_due_to_cooldown": True}
    else:
        candidates = select_candidates(records)
        print(f"  Candidates needing text extraction: {len(candidates)}")
        extract_status = run_extractions(candidates, state)
        print(f"  Extraction: {extract_status['extracted']} extracted, "
              f"{extract_status['failed']} failed, "
              f"rate_limited={extract_status['rate_limited']}, "
              f"elapsed={extract_status['elapsed']:.1f}s")

    # Phase 3: outputs.
    shard_entries = write_sharded_index(records)
    write_index_meta(records, shard_entries)
    write_manifest(records, extract_status)
    write_meta(records, extract_status, state)
    print(f"  Wrote {len(shard_entries)} shard(s) + index-meta.json + manifest.json + meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
