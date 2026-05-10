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

LEGACY_INDEX_JSON          = DOCS / "index.json"           # pre-sharding
LEGACY_SEARCH_BUNDLE_JSON  = DOCS / "search-bundle.json"   # pre-sharding (single-file v1.1.a session 2)
LEGACY_SEARCH_INDEX_JSON   = DOCS / "search-index.json"    # never existed for bills, defensive

# ── Search constants (shape-aligned with CAG / DRSC for cross-corpus v2) ──
# Tokenisation + freq cutoffs duplicated from build_cag.py per Independence
# Principle, NOT imported. Same values so cross-corpus search in v2 doesn't
# have to reconcile two different vocabs.

import re

_STOPWORDS = frozenset("""
a an and the of to in on at by for with from is are was were be been being
this that these those it its they them their there as or but if then so
not no nor have has had do does did will would should could may might must
can shall about above after again against all am any because before below
between both each few further here how i me my myself we our ours ourselves
you your yours yourself yourselves he him his himself she her hers herself
itself which who whom whose what when where why off out over under up down
into through during until while above below between because such own same
""".split())

_TOKEN_RE         = re.compile(r"\w+", re.UNICODE)
_FREQ_CUTOFF_HIGH = 0.9
_FREQ_CUTOFF_LOW  = 2
_MAX_TOKEN_LEN    = 25
_HEAD_CHARS       = 5000
_DOCS_PER_SHARD   = 2500

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


def write_manifest(records: list[dict]) -> None:
    """Build manifest.json with deeper stats + per-bill `texts` map.

    The `texts` map is keyed by compositeId and used by the app to know
    which bills have extracted text without per-bill probing — same shape
    as DRSC's manifest.texts and CAG's manifest.texts. Without it, the
    app's "with text" count stays at 0 even when text/<id>.txt files exist
    on the mirror.

    Pure function of disk state (text/) + the API-fetched records list.
    Owned by bills-derive.yml under the split-phase pattern; the writer
    workflows (bills-sansad.yml, bills-backfill.yml) do not call it.
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
        "texts": texts,
    }
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _delete_legacy(path: Path) -> None:
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def build_search_bundle(records: list[dict]) -> Optional[dict]:
    """Build sharded search-bundle-NN.json — title + first _HEAD_CHARS chars
    per bill that has extracted text. Sharded by sorted-key range so no shard
    exceeds CF Workers' 25 MiB cap. Mirrors build_cag.py's pattern (v1.0c).

    Returns shard-stats dict (used by meta.json), or None if no texts yet.
    """
    if not TEXT_DIR.exists():
        return None

    entries = []
    truncated = 0
    for r in records:
        cid = r["compositeId"]
        text_path = TEXT_DIR / f"{cid}.txt"
        if not text_path.exists():
            continue
        try:
            text = text_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        head = text[:_HEAD_CHARS]
        if len(text) > _HEAD_CHARS:
            truncated += 1
        entries.append({
            "key":   f"bills|{cid}",
            "title": r.get("billName") or "",
            "head":  head,
        })

    if not entries:
        return None

    entries.sort(key=lambda e: e["key"])
    n_shards = max(1, (len(entries) + _DOCS_PER_SHARD - 1) // _DOCS_PER_SHARD)
    shard_size = (len(entries) + n_shards - 1) // n_shards
    generated_at = _now_iso()
    shards_meta: list[dict] = []
    total_size_bytes = 0

    # Clean previous shards (the count may have shrunk if records dropped).
    import glob as _glob
    for p in _glob.glob(str(DOCS / "search-bundle-*.json")):
        try:
            os.remove(p)
        except OSError:
            pass

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end   = min(start + shard_size, len(entries))
        slc   = entries[start:end]
        bundle = {
            "version":      2,
            "generated_at": generated_at,
            "shard":        shard_idx,
            "shard_count":  n_shards,
            "head_chars":   _HEAD_CHARS,
            "total":        len(slc),
            "entries":      slc,
        }
        fname = f"search-bundle-{shard_idx:02d}.json"
        out_path = DOCS / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))
        size = out_path.stat().st_size
        total_size_bytes += size
        shards_meta.append({"name": fname, "size_bytes": size, "entries": len(slc)})

    _delete_legacy(LEGACY_SEARCH_BUNDLE_JSON)

    return {
        "shard_count":     n_shards,
        "shards":          [s["name"] for s in shards_meta],
        "shard_sizes":     {s["name"]: s["size_bytes"] for s in shards_meta},
        "total":           len(entries),
        "truncated":       truncated,
        "head_chars":      _HEAD_CHARS,
        "size_bytes":      total_size_bytes,
        "max_shard_bytes": max((s["size_bytes"] for s in shards_meta), default=0),
    }


def build_search_index(records: list[dict]) -> Optional[dict]:
    """Build sharded search-index-NN.json — inverted token index over the
    full body of every extracted-text bill. Each shard carries the FULL
    vocabulary + a slice of report_keys + per-token postings (delta-encoded,
    doc-local within the shard). Mirrors build_cag.py's pattern (v1.0c).

    Token rules: \\w+ regex (Unicode), 2-25 chars, lowercased, drop digits +
    stopwords. Tokens kept iff they appear in [_FREQ_CUTOFF_LOW,
    _FREQ_CUTOFF_HIGH × n_docs] documents — knock out hapax-legomena
    typos and corpus-wide near-stopwords.

    Returns shard-stats dict, or None if no texts yet.
    """
    if not TEXT_DIR.exists():
        return None

    # composite-id → record (for billName lookups not used here, but cheap).
    records_by_cid = {r["compositeId"]: r for r in records}

    docs: list[tuple[str, set[str]]] = []
    df: dict[str, int] = {}

    for text_file in sorted(TEXT_DIR.glob("*.txt")):
        cid = text_file.stem
        if cid not in records_by_cid:
            # Stale text file — bill was removed from the index. Skip.
            continue
        try:
            text = text_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        tokens: set[str] = set()
        for m in _TOKEN_RE.finditer(text.lower()):
            t = m.group(0)
            if len(t) < 2 or len(t) > _MAX_TOKEN_LEN:
                continue
            if t.isdigit() or t in _STOPWORDS:
                continue
            tokens.add(t)
            if len(tokens) >= 50000:
                break  # defensive: pathological PDF, skip the long tail
        docs.append((f"bills|{cid}", tokens))
        for t in tokens:
            df[t] = df.get(t, 0) + 1

    n_docs = len(docs)
    if n_docs == 0:
        return None

    high = int(n_docs * _FREQ_CUTOFF_HIGH)
    low  = _FREQ_CUTOFF_LOW
    keep_tokens = sorted(t for t, c in df.items() if low <= c <= high)
    token_to_idx = {t: i for i, t in enumerate(keep_tokens)}

    docs.sort(key=lambda d: d[0])

    n_shards = max(1, (n_docs + _DOCS_PER_SHARD - 1) // _DOCS_PER_SHARD)
    shard_size = (n_docs + n_shards - 1) // n_shards
    generated_at = _now_iso()

    def _delta(lst: list[int]) -> list[int]:
        if not lst:
            return lst
        out = [lst[0]]
        prev = lst[0]
        for x in lst[1:]:
            out.append(x - prev)
            prev = x
        return out

    # Clean previous shards.
    import glob as _glob
    for p in _glob.glob(str(DOCS / "search-index-*.json")):
        try:
            os.remove(p)
        except OSError:
            pass

    total_postings = 0
    shards_meta: list[dict] = []
    total_size_bytes = 0

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end   = min(start + shard_size, n_docs)
        slc   = docs[start:end]

        raw_postings: list[list[int]] = [[] for _ in keep_tokens]
        for local_idx, (_key, tokens) in enumerate(slc):
            for t in tokens:
                ti = token_to_idx.get(t)
                if ti is not None:
                    raw_postings[ti].append(local_idx)

        total_postings += sum(len(p) for p in raw_postings)
        postings = [_delta(p) for p in raw_postings]
        shard_keys = [k for k, _ in slc]

        index = {
            "version":          3,
            "encoding":         "delta",
            "generated_at":     generated_at,
            "shard":            shard_idx,
            "shard_count":      n_shards,
            "report_count":     len(slc),
            "vocab_size":       len(keep_tokens),
            "freq_cutoff_low":  low,
            "freq_cutoff_high": high,
            "stopwords_count":  len(_STOPWORDS),
            "report_keys":      shard_keys,
            "vocab":            keep_tokens,
            "postings":         postings,
        }
        fname = f"search-index-{shard_idx:02d}.json"
        out_path = DOCS / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, separators=(",", ":"))
        size = out_path.stat().st_size
        total_size_bytes += size
        shards_meta.append({"name": fname, "size_bytes": size, "report_count": len(slc)})

    _delete_legacy(LEGACY_SEARCH_INDEX_JSON)

    return {
        "shard_count":      n_shards,
        "shards":           [s["name"] for s in shards_meta],
        "shard_sizes":      {s["name"]: s["size_bytes"] for s in shards_meta},
        "report_count":     n_docs,
        "vocab_size":       len(keep_tokens),
        "total_postings":   total_postings,
        "size_bytes":       total_size_bytes,
        "max_shard_bytes":  max((s["size_bytes"] for s in shards_meta), default=0),
        "freq_cutoff_low":  low,
        "freq_cutoff_high": high,
    }


def write_meta(records: list[dict], state: dict,
               bundle_stats: Optional[dict] = None,
               index_stats: Optional[dict] = None) -> None:
    """Build meta.json — small status JSON for chip status display + the
    search_bundle / search_index shard listings the app needs to fetch the
    sharded deep-search artefacts.

    Split-phase: pure function of disk state + state.json's cooldown info.
    Per-run extract counters (last_run_status, this_run_extracted) live
    only in workflow logs now — putting them in derived files coupled the
    derive phase to extract-time state and reintroduced the rebase race.
    See CONV.md "Split-phase scraping pattern".
    """
    in_cooldown, remaining = _is_in_cooldown(state)
    with_text = sum(1 for r in records if (TEXT_DIR / f"{r['compositeId']}.txt").exists())
    meta = {
        "scraper_version": SCRAPER_VERSION,
        "last_update": _now_iso(),
        "total": len(records),
        "with_text": with_text,
        "in_rate_limit_cooldown": in_cooldown,
        "cooldown_remaining_seconds": int(remaining) if remaining else None,
        "search_bundle": bundle_stats,   # None if no extracted texts yet
        "search_index":  index_stats,
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────

def phase_extract() -> int:
    """Phase 1 — produce primary files: text/<compositeId>.txt extracted
    via PDF download + parse, plus .scraper_state.json (cooldown bookkeeping).

    Bills has no `records.json` analogue — the records list is fetched from
    sansad.in's getBills API every run. That API call is cheap (~17-35s)
    and re-runs in the derive phase too, so we don't need to persist it.
    """
    state = _load_state()

    print("  Building bill index from getBills API ...")
    t0 = time.time()
    records, duplicates = collect_records()
    print(f"  Index: {len(records)} records "
          f"({duplicates} composite-id collisions dropped) "
          f"in {time.time()-t0:.1f}s")

    in_cooldown, remaining = _is_in_cooldown(state)
    if in_cooldown:
        print(f"  Skipping extraction phase — rate-limit cooldown active "
              f"({int(remaining)}s remaining)")
        return 0

    candidates = select_candidates(records)
    print(f"  Candidates needing text extraction: {len(candidates)}")
    extract_status = run_extractions(candidates, state)
    print(f"  Extraction: {extract_status['extracted']} extracted, "
          f"{extract_status['failed']} failed, "
          f"rate_limited={extract_status['rate_limited']}, "
          f"elapsed={extract_status['elapsed']:.1f}s")
    return 0


def phase_derive() -> int:
    """Phase 2 — regenerate derived files (sharded index, index-meta.json,
    manifest.json, search-bundle-*.json, search-index-*.json, meta.json)
    from the on-disk primary state (text/) plus the API-fetched records list.

    Owned by bills-derive.yml; one run at a time via `cancel-in-progress: true`.
    """
    state = _load_state()

    print("  Building bill index from getBills API ...")
    t0 = time.time()
    records, duplicates = collect_records()
    print(f"  Index: {len(records)} records "
          f"({duplicates} composite-id collisions dropped) "
          f"in {time.time()-t0:.1f}s")

    print("  Writing sharded index + index-meta.json + manifest.json...")
    shard_entries = write_sharded_index(records)
    write_index_meta(records, shard_entries)
    write_manifest(records)

    print("  Building search bundle (title + first 5K chars per bill, sharded)...")
    bundle_stats = build_search_bundle(records)
    if bundle_stats:
        print(f"    {bundle_stats['shard_count']} shard(s), "
              f"{bundle_stats['total']} entries, "
              f"max shard {bundle_stats['max_shard_bytes'] // 1024} KB")
    else:
        print("    skipped — no extracted texts yet")

    print("  Building search index (full-body inverted token index, sharded)...")
    index_stats = build_search_index(records)
    if index_stats:
        print(f"    {index_stats['shard_count']} shard(s), "
              f"{index_stats['report_count']} docs, "
              f"vocab={index_stats['vocab_size']}, "
              f"max shard {index_stats['max_shard_bytes'] // 1024} KB")
    else:
        print("    skipped — no extracted texts yet")

    write_meta(records, state, bundle_stats, index_stats)
    print(f"  Wrote {len(shard_entries)} index shard(s) + index-meta.json + manifest.json + meta.json + search artefacts")
    return 0


def main() -> int:
    """Dispatch to extract / derive / both based on BUILD_PHASE.

    BUILD_PHASE=extract — for bills-sansad.yml + bills-backfill.yml. Produces
                          only text/<compositeId>.txt + .scraper_state.json.
    BUILD_PHASE=derive  — for bills-derive.yml. Reads disk state + re-fetches
                          records from API, regenerates all derived files.
    BUILD_PHASE=all     — legacy / local-dev convenience.

    See CONV.md "Split-phase scraping pattern" for the full architecture.
    """
    phase = os.environ.get("BUILD_PHASE", "all").lower()
    if phase not in ("extract", "derive", "all"):
        print(f"BUILD_PHASE={phase!r} is not one of extract|derive|all — aborting.", file=sys.stderr)
        return 2

    print(f"[bills/sansad] starting — {SCRAPER_VERSION} (BUILD_PHASE={phase})")

    if phase in ("extract", "all"):
        rc = phase_extract()
        if rc != 0:
            return rc

    if phase in ("derive", "all"):
        rc = phase_derive()
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
