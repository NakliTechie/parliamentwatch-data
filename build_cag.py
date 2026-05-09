#!/usr/bin/env python3
"""CAG static data builder — orchestrator for parliamentwatch-data.

Per scheduled (or workflow_dispatch) run:
  1. Walk cag.gov.in's listing pages, enumerate detail-page IDs
  2. Fetch detail-page metadata for any new IDs we don't have yet
  3. Extract text for as many missing PDFs as fit in the per-run budget
     (MAX_EXTRACTIONS_PER_RUN, MAX_RUN_SECONDS) — newest reports first
  4. Build manifest.json + reports.json + meta.json
  5. Build search-bundle.json + search-index.json (mirror DRSC's v0.6 ladder)

Output goes under docs/cag/. CF Workers serves it at /cag/* on
sansadsaar-data.naklitechie.com (and the legacy sansad-files alias).

Independence Principle: this orchestrator does not import from the DRSC
build_static.py. The two corpora share a docs/ tree but no Python state.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))   # so `from cag.scraper import ...` works

from cag.scraper import (
    BASE_URL, LISTING_URL, DETAIL_URL_FMT,
    CAGReport, RateLimited,
    walk_listing, fetch_detail, get_report_text,
)

# ── Paths ───────────────────────────────────────────────────────────────────

ASSETS    = ROOT / "docs"
DOCS      = ASSETS / "cag"
TEXT_DIR  = DOCS / "text"
PDFS_DIR  = DOCS / "pdfs"        # gitignored via docs/**/pdfs/
DOCS.mkdir(parents=True, exist_ok=True)

REPORTS_JSON      = DOCS / "reports.json"
MANIFEST_JSON     = DOCS / "manifest.json"
META_JSON         = DOCS / "meta.json"
BUNDLE_JSON       = DOCS / "search-bundle.json"
INDEX_JSON        = DOCS / "search-index.json"

# ── Per-run budget ─────────────────────────────────────────────────────────

# How many listing pages to walk for new IDs. Default: walk the entire
# archive (None ⇒ "until empty"). Pass MAX_LISTING_PAGES=N for a quick slice
# during development / initial seeding.
MAX_LISTING_PAGES = (
    int(os.environ["MAX_LISTING_PAGES"])
    if os.environ.get("MAX_LISTING_PAGES")
    else None
)

# How many missing PDFs to extract per run. Same shape as DRSC build_static.
MAX_EXTRACTIONS_PER_RUN = int(os.environ.get("MAX_EXTRACTIONS_PER_RUN", "100"))
MAX_RUN_SECONDS         = int(os.environ.get("MAX_RUN_SECONDS", "1800"))   # 30 min
EXTRACT_WORKERS         = int(os.environ.get("EXTRACT_WORKERS", "4"))

# Cooldown after a 429 / 403 — same idea as DRSC's: skip the extraction phase
# if the previous run was rate-limited within this many seconds.
RATE_LIMIT_COOLDOWN_SECONDS = int(os.environ.get("RATE_LIMIT_COOLDOWN_SECONDS", str(6 * 3600)))


# ── Search bundle + index ──────────────────────────────────────────────────

# Tokenizer + index params live alongside the DRSC implementation in
# build_static.py. We re-import them here for consistency. The two corpora
# share the constants (stopwords, freq cutoffs, head_chars) so cross-corpus
# search in v2 doesn't have to reconcile two different tokenisations.
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

_TOKEN_RE       = re.compile(r"\w+", re.UNICODE)
_FREQ_CUTOFF_HIGH = 0.9
_FREQ_CUTOFF_LOW  = 2
_MAX_TOKEN_LEN    = 25
_HEAD_CHARS       = 5000


def load_existing_reports() -> dict[int, dict]:
    """Returns int-keyed dict of CAG metadata. Disk format JSON-keys-as-strings;
    we parse back to ints for in-memory work."""
    if not REPORTS_JSON.exists():
        return {}
    with open(REPORTS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_reports(reports: dict[int, dict]) -> None:
    # Sort by id desc (newest first) for stable diffs and easier scanning.
    sorted_items = sorted(reports.items(), key=lambda kv: -kv[0])
    out = {str(k): v for k, v in sorted_items}
    with open(REPORTS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ── Phase 1: enumerate IDs from listing pages ───────────────────────────────

def enumerate_ids(*, max_pages: int | None) -> list[int]:
    print(f"[1/6] Walking listing pages (max_pages={max_pages or 'until empty'})...")
    t0 = time.time()
    ids = list(walk_listing(max_pages=max_pages))
    print(f"  enumerated {len(ids)} ids in {time.time()-t0:.1f}s")
    return ids


# ── Phase 2: fetch detail metadata for new IDs ──────────────────────────────

def fetch_new_metadata(known_ids: set[int], all_ids: list[int]) -> dict[int, dict]:
    """Fetch detail-page metadata for any IDs not already in known_ids.
    Returns {id: metadata_dict}. Stops on RateLimited."""
    new_ids = [i for i in all_ids if i not in known_ids]
    if not new_ids:
        print("  no new ids — skipping detail fetch")
        return {}
    print(f"  fetching detail metadata for {len(new_ids)} new ids...")

    out: dict[int, dict] = {}
    rate_limited = False
    t0 = time.time()
    for i, rid in enumerate(new_ids, start=1):
        try:
            rep = fetch_detail(rid)
        except RateLimited as rl:
            print(f"  [RATE-LIMITED] {rl} — stopping detail-fetch phase")
            rate_limited = True
            break
        except Exception as e:
            print(f"  detail fetch failed for id={rid}: {e}")
            continue
        out[rid] = rep.to_dict()
        if i % 20 == 0:
            print(f"  ...{i}/{len(new_ids)} ({time.time()-t0:.1f}s)")
    print(f"  fetched {len(out)} metadata entries in {time.time()-t0:.1f}s · rate_limited={rate_limited}")
    return out


# ── Phase 3: extract missing texts ─────────────────────────────────────────

def _check_cooldown_and_skip() -> bool:
    """If the previous run wrote rate_limited:true to meta.json within the
    last RATE_LIMIT_COOLDOWN_SECONDS, skip the extraction phase."""
    if not META_JSON.exists():
        return False
    try:
        with open(META_JSON, "r", encoding="utf-8") as f:
            prev = json.load(f)
    except Exception:
        return False
    rl = prev.get("extract_stats", {}).get("rate_limited")
    if not rl:
        return False
    gen = prev.get("generated_at", "")
    try:
        gen_dt = datetime.strptime(gen, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - gen_dt).total_seconds()
    except Exception:
        return False
    if age < RATE_LIMIT_COOLDOWN_SECONDS:
        print(f"  previous run rate-limited {int(age/60)}m ago "
              f"(cooldown {RATE_LIMIT_COOLDOWN_SECONDS//60}m) — skipping extraction this run")
        return True
    return False


def extract_missing_texts(reports: dict[int, dict]) -> dict:
    if _check_cooldown_and_skip():
        return {"extracted": [], "failed": [], "rate_limited": True,
                "budget_hit": False, "skipped_due_to_cooldown": True,
                "candidates_total": 0}

    candidates = []
    for rid, meta in reports.items():
        if not meta.get("pdf_url"):
            continue
        text_path = TEXT_DIR / f"{rid}.txt"
        if text_path.exists():
            continue
        candidates.append((rid, meta["pdf_url"]))
    # Newest first (highest id wins — CAG IDs are monotonic).
    candidates.sort(key=lambda c: -c[0])
    print(f"  candidates: {len(candidates)} reports missing extracted text")

    if not candidates:
        return {"extracted": [], "failed": [], "rate_limited": False,
                "budget_hit": False, "candidates_total": 0}

    target = candidates[:MAX_EXTRACTIONS_PER_RUN]
    print(f"  budget: extracting up to {len(target)} this run "
          f"(MAX_EXTRACTIONS_PER_RUN={MAX_EXTRACTIONS_PER_RUN}, "
          f"MAX_RUN_SECONDS={MAX_RUN_SECONDS}, EXTRACT_WORKERS={EXTRACT_WORKERS})")

    deadline = time.monotonic() + MAX_RUN_SECONDS
    extracted, failed = [], []
    rate_limited = False
    budget_hit = False

    def _do(rid_url):
        rid, url = rid_url
        try:
            text = get_report_text(url, rid, text_dir=str(TEXT_DIR), pdfs_dir=str(PDFS_DIR))
            return rid, text, None
        except RateLimited as rl:
            return rid, None, rl
        except Exception as e:
            return rid, None, e

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
        futures = {ex.submit(_do, c): c for c in target}
        for fut in as_completed(futures):
            rid, text, err = fut.result()
            if isinstance(err, RateLimited):
                print(f"  [RATE-LIMITED] id={rid}: {err}")
                rate_limited = True
                for f in futures:
                    if not f.done(): f.cancel()
                break
            elif text:
                extracted.append(rid)
            else:
                failed.append(rid)
            if time.monotonic() > deadline:
                print(f"  [BUDGET] wall-clock budget hit after {len(extracted)} extractions")
                budget_hit = True
                for f in futures:
                    if not f.done(): f.cancel()
                break

    return {
        "extracted": extracted,
        "failed":    failed,
        "rate_limited": rate_limited,
        "budget_hit":   budget_hit,
        "candidates_total": len(candidates),
    }


# ── Phase 4: build manifest ────────────────────────────────────────────────

def build_manifest() -> dict:
    """List every extracted text file with its size + relative URL."""
    manifest: dict[str, dict] = {}
    if not TEXT_DIR.exists():
        return {"texts": manifest}
    for text_file in sorted(TEXT_DIR.glob("*.txt")):
        rid = text_file.stem
        manifest[rid] = {
            "size": text_file.stat().st_size,
            "url":  f"text/{text_file.name}",   # relative; app prepends 'cag/'
        }
    return {"texts": manifest}


# ── Phase 5: search bundle ─────────────────────────────────────────────────

def build_search_bundle(reports: dict[int, dict]) -> dict | None:
    """Title + first 5K chars per report. Replaces deep-search per-text fan-out."""
    if not TEXT_DIR.exists():
        return None
    entries = []
    truncated = 0
    for rid, meta in reports.items():
        text_path = TEXT_DIR / f"{rid}.txt"
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
            "key":   f"cag|{rid}",
            "title": meta.get("title", ""),
            "head":  head,
        })

    if not entries:
        return None

    bundle = {
        "version":      1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head_chars":   _HEAD_CHARS,
        "total":        len(entries),
        "truncated":    truncated,
        "entries":      entries,
    }
    with open(BUNDLE_JSON, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))
    return {
        "total":      len(entries),
        "truncated":  truncated,
        "head_chars": _HEAD_CHARS,
        "size_bytes": BUNDLE_JSON.stat().st_size,
    }


# ── Phase 6: search index (inverted token index over full body) ────────────

def build_search_index() -> dict | None:
    """Identical shape to DRSC's index. Same tokeniser, stopwords, cutoffs,
    delta-encoded postings — so v2 cross-corpus search can union both."""
    if not TEXT_DIR.exists():
        return None

    report_keys: list[str] = []
    doc_token_sets: list[set[str]] = []
    df: dict[str, int] = {}

    for text_file in sorted(TEXT_DIR.glob("*.txt")):
        rid = text_file.stem
        try:
            text = text_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        tokens: set[str] = set()
        for m in _TOKEN_RE.finditer(text.lower()):
            t = m.group(0)
            if len(t) < 2 or len(t) > _MAX_TOKEN_LEN: continue
            if t.isdigit() or t in _STOPWORDS: continue
            tokens.add(t)
            if len(tokens) >= 50000: break
        report_keys.append(f"cag|{rid}")
        doc_token_sets.append(tokens)
        for t in tokens:
            df[t] = df.get(t, 0) + 1

    n_docs = len(report_keys)
    if n_docs == 0:
        return None

    high = int(n_docs * _FREQ_CUTOFF_HIGH)
    low  = _FREQ_CUTOFF_LOW
    keep_tokens = sorted(t for t, c in df.items() if low <= c <= high)
    token_to_idx = {t: i for i, t in enumerate(keep_tokens)}

    raw_postings: list[list[int]] = [[] for _ in keep_tokens]
    for doc_idx, tokens in enumerate(doc_token_sets):
        for t in tokens:
            ti = token_to_idx.get(t)
            if ti is not None:
                raw_postings[ti].append(doc_idx)

    total_postings = sum(len(p) for p in raw_postings)

    def _delta(lst: list[int]) -> list[int]:
        if not lst: return lst
        out = [lst[0]]; prev = lst[0]
        for x in lst[1:]:
            out.append(x - prev); prev = x
        return out
    postings = [_delta(p) for p in raw_postings]

    index = {
        "version":          2,
        "encoding":         "delta",
        "generated_at":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "report_count":     n_docs,
        "vocab_size":       len(keep_tokens),
        "total_postings":   total_postings,
        "freq_cutoff_low":  low,
        "freq_cutoff_high": high,
        "stopwords_count":  len(_STOPWORDS),
        "report_keys":      report_keys,
        "vocab":            keep_tokens,
        "postings":         postings,
    }
    with open(INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))
    return {
        "report_count":   n_docs,
        "vocab_size":     len(keep_tokens),
        "total_postings": total_postings,
        "size_bytes":     INDEX_JSON.stat().st_size,
        "freq_cutoff_low":  low,
        "freq_cutoff_high": high,
    }


# ── Phase 7: meta ──────────────────────────────────────────────────────────

def write_meta(*, total_reports: int, total_with_text: int,
               extract_stats: dict, bundle_stats: dict | None,
               index_stats: dict | None) -> dict:
    meta = {
        "version":      "1.0",
        "corpus":       "cag",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_extractions_per_run": MAX_EXTRACTIONS_PER_RUN,
        "extract_workers":         EXTRACT_WORKERS,
        "total_reports":   total_reports,
        "total_with_text": total_with_text,
        "extract_stats": {
            "extracted":     len(extract_stats.get("extracted", [])),
            "failed":        len(extract_stats.get("failed", [])),
            "rate_limited":  extract_stats.get("rate_limited", False),
            "budget_hit":    extract_stats.get("budget_hit", False),
            "candidates_remaining_after_run": max(
                0, extract_stats.get("candidates_total", 0) - len(extract_stats.get("extracted", []))
            ),
        },
        "search_bundle": bundle_stats,
        "search_index":  index_stats,
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=== ParliamentWatch CAG static builder ===")
    print(f"DOCS                    : {DOCS}")
    print(f"MAX_LISTING_PAGES       : {MAX_LISTING_PAGES or '(walk to empty)'}")
    print(f"MAX_EXTRACTIONS_PER_RUN : {MAX_EXTRACTIONS_PER_RUN}")
    print(f"MAX_RUN_SECONDS         : {MAX_RUN_SECONDS}")
    print(f"EXTRACT_WORKERS         : {EXTRACT_WORKERS}")

    print("\n[1/7] Walking cag.gov.in listing pages...")
    all_ids = enumerate_ids(max_pages=MAX_LISTING_PAGES)

    print("\n[2/7] Loading existing metadata + fetching new detail pages...")
    reports = load_existing_reports()
    new_meta = fetch_new_metadata(set(reports.keys()), all_ids)
    reports.update(new_meta)
    save_reports(reports)
    print(f"  reports.json: {len(reports)} total ({len(new_meta)} new this run)")

    print("\n[3/7] Extracting missing texts (priority: newest first)...")
    extract_stats = extract_missing_texts(reports)
    print(f"  extracted={len(extract_stats['extracted'])} "
          f"failed={len(extract_stats['failed'])} "
          f"rate_limited={extract_stats.get('rate_limited', False)} "
          f"budget_hit={extract_stats.get('budget_hit', False)}")

    print("\n[4/7] Building manifest.json...")
    manifest = build_manifest()
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    n_with_text = len(manifest["texts"])
    print(f"  manifest: {n_with_text} reports with extracted text")

    print("\n[5/7] Building search-bundle.json (title + first 5K chars per report)...")
    bundle_stats = build_search_bundle(reports)
    if bundle_stats:
        mb = bundle_stats["size_bytes"] / (1024 * 1024)
        print(f"  search-bundle.json: {bundle_stats['total']} entries · {mb:.1f} MB raw")
    else:
        print("  no extracted texts yet — skipping bundle build")

    print("\n[6/7] Building search-index.json (inverted token index, full body)...")
    index_stats = build_search_index()
    if index_stats:
        mb = index_stats["size_bytes"] / (1024 * 1024)
        print(f"  search-index.json: {index_stats['report_count']} docs, "
              f"vocab={index_stats['vocab_size']}, postings={index_stats['total_postings']} · {mb:.1f} MB raw")
    else:
        print("  no extracted texts yet — skipping index build")

    print("\n[7/7] Writing meta.json...")
    meta = write_meta(
        total_reports=len(reports),
        total_with_text=n_with_text,
        extract_stats=extract_stats,
        bundle_stats=bundle_stats,
        index_stats=index_stats,
    )
    print(json.dumps(meta, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
