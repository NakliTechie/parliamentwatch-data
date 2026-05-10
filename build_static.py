#!/usr/bin/env python3
"""ParliamentWatch — static data builder for naklitechie/parliamentwatch-data.

Per scheduled (or workflow_dispatch) run:
  1. Scrape metadata for all 24 DRSCs across configured Lok Sabhas
  2. Extract text for as many missing PDFs as fit in the per-run budget
     (MAX_EXTRACTIONS_PER_RUN, MAX_RUN_SECONDS) — newest reports first
  3. Write docs/{reports.json, manifest.json, committees.json, meta.json}

Politeness for the historical backfill (~14k missing PDFs as of 2026-05):
  - Concurrency capped at 4 (down from 10)
  - Random 250-500ms jitter between fetches (in pdf_utils._jitter)
  - Stops the moment sansad.in returns 429/403; next run resumes
  - Per-run budget caps how aggressive any single dispatch can be

Output is served by GitHub Pages from /docs.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Each corpus's outputs are scoped under `docs/<corpus>/`. DRSC is the
# first corpus (this file); CAG and the others land in their own sibling
# subfolders without touching this layout.
ASSETS = Path(__file__).resolve().parent / "docs"
DOCS   = ASSETS / "drsc"   # v1.0a phase 2 — corpus-scoped output
DOCS.mkdir(parents=True, exist_ok=True)

# Force scraper paths to docs/drsc/ before importing config-dependent modules.
os.environ["DATA_DIR"] = str(DOCS)

from scraper import scrape_all_committees, load_existing_reports  # noqa: E402
from pdf_utils import get_report_text, RateLimited  # noqa: E402
from config import DRSC_COMMITTEES  # noqa: E402

# Per-run caps. Tuned for the historical backfill — at ~10s/PDF avg with 4
# workers, MAX_EXTRACTIONS_PER_RUN=400 finishes in ~17 minutes wall clock.
# MAX_RUN_SECONDS provides a hard ceiling so a slow round doesn't spill into
# the GH Actions 6h job limit.
MAX_EXTRACTIONS_PER_RUN = int(os.environ.get("MAX_EXTRACTIONS_PER_RUN", "400"))
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "1800"))  # 30 min
EXTRACT_WORKERS = int(os.environ.get("EXTRACT_WORKERS", "4"))

# After a 429/403 from sansad.in, skip the extraction phase for this many
# seconds so the next scheduled run doesn't immediately retry-storm. Tuned
# to be ≥ one cron interval (currently 4h) so we always sit out at least
# one cycle after a rate-limit before attempting again.
RATE_LIMIT_COOLDOWN_SECONDS = int(os.environ.get("RATE_LIMIT_COOLDOWN_SECONDS", str(6 * 3600)))

LOK_SABHAS = [int(x) for x in os.environ.get("LOK_SABHAS", "18").split(",")]


def _safe_num(report_num):
    """Filesystem-safe encoding of a report number (without LS prefix)."""
    if report_num is None:
        return None
    return str(report_num).replace("/", "-").replace(" ", "_")


def _file_id(lok_sabha, report_num):
    """Filesystem id for a report's text file: LS<n>_<safe-num>.

    Pre-v0.4 we keyed by report_num alone — but the same number recurs
    across Lok Sabhas (LS18 #23, LS17 #23, ...). Same-named files were
    overwriting each other on disk and the manifest collapsed multi-LS
    rows into one. The LS prefix uniquely identifies a single report.
    """
    safe = _safe_num(report_num)
    if safe is None:
        return None
    if lok_sabha is None:
        return safe   # shouldn't happen for real data; defensive fallback
    return f"LS{lok_sabha}_{safe}"


_LS_PREFIX_RE = re.compile(r"^LS\d+_")


def migrate_unprefixed_text_files():
    """One-time migration of pre-v0.4 text files to the LS-prefixed naming.

    Walks text/<committee>/<num>.txt entries and renames each to
    text/<committee>/LS<n>_<num>.txt, picking the highest-LS variant
    that matches in reports.json (the old extraction order processed
    LS desc and skipped if file existed, so the file represents the
    highest LS most of the time).

    Idempotent — files already LS-prefixed are left alone.
    """
    text_root = DOCS / "text"
    if not text_root.exists():
        return {"migrated": 0, "ambiguous": 0, "missing": 0, "preexisting": 0}

    from scraper import load_existing_reports as _load
    reports = _load()
    migrated = ambiguous = missing = preexisting = 0

    for committee_dir in sorted(text_root.iterdir()):
        if not committee_dir.is_dir():
            continue
        committee_key = committee_dir.name
        committee_reports = reports.get(committee_key, [])
        # safe_num -> [lok_sabha, ...]
        num_to_ls = {}
        for r in committee_reports:
            num = r.get("report_number")
            ls = r.get("lok_sabha")
            if num is None or ls is None:
                continue
            safe = _safe_num(num)
            num_to_ls.setdefault(safe, []).append(ls)

        for old_path in list(committee_dir.glob("*.txt")):
            stem = old_path.stem
            if _LS_PREFIX_RE.match(stem):
                continue   # already migrated
            ls_options = num_to_ls.get(stem, [])
            if not ls_options:
                print(f"  [migrate] no metadata for {committee_key}/{stem}.txt — leaving in place")
                missing += 1
                continue
            ls = max(ls_options)   # extraction-order heuristic: highest LS first
            if len(set(ls_options)) > 1:
                ambiguous += 1
            new_path = committee_dir / f"LS{ls}_{stem}.txt"
            if new_path.exists():
                # An LS-prefixed file already exists for the same report —
                # this can happen if a partial migration ran earlier. The
                # un-prefixed copy is the older one; remove it.
                print(f"  [migrate] {new_path.name} already exists, removing stale {old_path.name}")
                old_path.unlink()
                preexisting += 1
                continue
            old_path.rename(new_path)
            migrated += 1

    return {"migrated": migrated, "ambiguous": ambiguous, "missing": missing, "preexisting": preexisting}


def _missing_reports_priority_order():
    """Walk every report we have metadata for and yield those without an
    extracted text file, ordered (lok_sabha desc, report_number desc) so the
    most recent reports go first.

    The on-disk text file path is text/<committee>/LS<n>_<num>.txt, which
    uniquely identifies a single (committee, lok_sabha, report_number) tuple.
    """
    reports = load_existing_reports()
    candidates = []
    for committee_key, committee_reports in reports.items():
        for report in committee_reports:
            num = report.get("report_number")
            url = report.get("pdf_url")
            ls = report.get("lok_sabha")
            if not num or not url or ls is None:
                continue
            file_id = _file_id(ls, num)
            text_path = DOCS / "text" / committee_key / f"{file_id}.txt"
            if text_path.exists():
                continue
            candidates.append({
                "committee_key": committee_key,
                "report_number": num,
                "pdf_url": url,
                "lok_sabha": ls or 0,
                "file_id": file_id,
            })
    # Most recent first (LS18 before LS17, higher report number first within an LS)
    candidates.sort(key=lambda c: (c["lok_sabha"], c["report_number"]), reverse=True)
    return candidates


def extract_missing_texts():
    """Extract up to MAX_EXTRACTIONS_PER_RUN missing PDFs. Stops early on
    rate-limit (429/403) or wall-clock budget exhaustion.

    Pre-flight: if the most recent run wrote `rate_limited: true` to
    meta.json within the last RATE_LIMIT_COOLDOWN_SECONDS, skip the
    extraction phase entirely. The cron fires every 4h; the cooldown is
    6h, so we always sit out at least one cycle after sansad.in 429s us.
    """
    prev_meta_path = DOCS / "meta.json"
    if prev_meta_path.exists():
        try:
            prev_meta = json.loads(prev_meta_path.read_text())
            prev_rl = prev_meta.get("extract_stats", {}).get("rate_limited", False)
            if prev_rl:
                gen_at = prev_meta.get("generated_at", "")
                gen_dt = datetime.strptime(gen_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - gen_dt).total_seconds()
                if age < RATE_LIMIT_COOLDOWN_SECONDS:
                    print(f"  Previous run was rate-limited {int(age/60)}m ago "
                          f"(cooldown is {RATE_LIMIT_COOLDOWN_SECONDS//60}m) — skipping extraction this run.")
                    return {"extracted": [], "failed": [], "rate_limited": True,
                            "budget_hit": False, "candidates_total": 0,
                            "skipped_due_to_cooldown": True}
                else:
                    print(f"  Previous run rate-limited {int(age/60)}m ago — cooldown elapsed, proceeding.")
        except Exception as e:
            print(f"  (rate-limit pre-flight check failed: {e} — proceeding anyway)")

    candidates = _missing_reports_priority_order()
    print(f"  candidates: {len(candidates)} reports missing extracted text")
    if not candidates:
        return {"extracted": [], "failed": [], "rate_limited": False, "budget_hit": False}

    target = candidates[:MAX_EXTRACTIONS_PER_RUN]
    print(f"  budget: extracting up to {len(target)} this run "
          f"(MAX_EXTRACTIONS_PER_RUN={MAX_EXTRACTIONS_PER_RUN}, MAX_RUN_SECONDS={MAX_RUN_SECONDS}, EXTRACT_WORKERS={EXTRACT_WORKERS})")

    deadline = time.monotonic() + MAX_RUN_SECONDS
    extracted, failed = [], []
    rate_limited = False
    budget_hit = False

    def _do(c):
        # pdf_utils saves the text to text/<committee>/<file_id>.txt — we pass
        # file_id (LS<n>_<num>) as the "report_number" arg so the filename
        # matches our manifest key. pdf_utils itself doesn't care about the
        # naming scheme.
        try:
            text = get_report_text(c["pdf_url"], c["committee_key"], c["file_id"])
            return c, text, None
        except RateLimited as rl:
            return c, None, rl
        except Exception as e:
            return c, None, e

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
        futures = {ex.submit(_do, c): c for c in target}
        for fut in as_completed(futures):
            c, text, err = fut.result()
            label = f"{c['committee_key']}/{c['file_id']}"
            if isinstance(err, RateLimited):
                print(f"  [RATE-LIMITED] {label}: {err}")
                rate_limited = True
                # Cancel any not-yet-started futures and stop
                for f in futures:
                    if not f.done():
                        f.cancel()
                break
            elif text:
                extracted.append(label)
            else:
                failed.append(label)
            if time.monotonic() > deadline:
                print(f"  [BUDGET] wall-clock budget hit after {len(extracted)} extractions")
                budget_hit = True
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

    return {
        "extracted": extracted,
        "failed": failed,
        "rate_limited": rate_limited,
        "budget_hit": budget_hit,
        "candidates_total": len(candidates),
    }


def build_manifest():
    """List every extracted text file with its size."""
    manifest = {"texts": {}}
    text_root = DOCS / "text"
    if not text_root.exists():
        return manifest
    for committee_dir in sorted(text_root.iterdir()):
        if not committee_dir.is_dir():
            continue
        committee_key = committee_dir.name
        manifest["texts"][committee_key] = {}
        for text_file in sorted(committee_dir.glob("*.txt")):
            manifest["texts"][committee_key][text_file.stem] = {
                "size": text_file.stat().st_size,
                "url": f"text/{committee_key}/{text_file.name}",
            }
    return manifest


def build_committees_index():
    out = {}
    for key, c in DRSC_COMMITTEES.items():
        out[key] = {"name": c["name"], "house": c.get("house", "L")}
    with open(DOCS / "committees.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# Search bundle: title + first N chars per extracted report. Used by the
# app for substring matching across the corpus scope / intro / findings,
# and as the snippet source in result rows.
#
# v1.0c: SHARDED. The hosting platform has a 25 MiB per-asset hard limit.
# At ~5 KB/doc the unsharded bundle crosses 25 MiB around 5,000 reports —
# this corpus is heading to 13k+. We split into N shards by sorted
# reportKey range; each shard <= DOCS_PER_SHARD reports, each file
# <= ~15 MiB. App fetches all shards in parallel, merges entries into
# one Map.
#
# Output: docs/drsc/search-bundle-00.json, search-bundle-01.json, ...
# Old single-file `search-bundle.json` is removed when present so it
# doesn't keep blocking the deploy.

DOCS_PER_SHARD = 2500   # safe ceiling: 2500 × 5 KB ≈ 12.5 MiB per bundle shard


def _delete_legacy(path: Path) -> None:
    """Drop a pre-sharding single file if it's still around — otherwise
    it stays in docs/ and keeps tripping the 25 MiB asset check."""
    if path.exists():
        path.unlink()


def build_search_bundle(head_chars=5000, docs_per_shard=DOCS_PER_SHARD):
    reports = load_existing_reports()
    text_root = DOCS / "text"
    if not text_root.exists():
        return None

    entries = []
    truncated = 0

    for committee_key, committee_reports in reports.items():
        for r in committee_reports:
            num = r.get("report_number")
            ls  = r.get("lok_sabha")
            if num is None:
                continue
            file_id = _file_id(ls, num)
            if file_id is None:
                continue
            text_path = text_root / committee_key / f"{file_id}.txt"
            if not text_path.exists():
                continue
            try:
                text = text_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            head = text[:head_chars]
            if len(text) > head_chars:
                truncated += 1
            key = f"{committee_key}|{ls if ls is not None else ''}|{num}"
            entries.append({"key": key, "title": r.get("title", ""), "head": head})

    if not entries:
        return None

    # Sort by key for stable, deterministic shard composition. With sorted
    # input each shard's keys form a contiguous lexical range — useful for
    # at-a-glance debugging of which reports landed where.
    entries.sort(key=lambda e: e["key"])

    n_shards = max(1, (len(entries) + docs_per_shard - 1) // docs_per_shard)
    shard_size = (len(entries) + n_shards - 1) // n_shards
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    shards_meta: list[dict] = []
    total_size_bytes = 0

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end   = min(start + shard_size, len(entries))
        slc   = entries[start:end]
        bundle = {
            "version":      2,
            "generated_at": generated_at,
            "shard":        shard_idx,
            "shard_count":  n_shards,
            "head_chars":   head_chars,
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

    _delete_legacy(DOCS / "search-bundle.json")

    return {
        "shard_count":   n_shards,
        "shards":        [s["name"] for s in shards_meta],
        "shard_sizes":   {s["name"]: s["size_bytes"] for s in shards_meta},
        "total":         len(entries),
        "truncated":     truncated,
        "head_chars":    head_chars,
        "size_bytes":    total_size_bytes,
        "max_shard_bytes": max((s["size_bytes"] for s in shards_meta), default=0),
    }


# Inverted index for full-body recall (v0.6 part C). Pairs with the search
# bundle: bundle is title + first 5K chars (snippet preview + substring),
# index is token-presence across the full body of every report.
#
# Tokenization: \w+ lowercased. Stopwords dropped (common English only;
# Hindi gets separate treatment if/when search-quality complaints surface).
# Frequency cutoff drops tokens that appear in >FREQ_CUTOFF of docs —
# prunes governmental boilerplate ("committee", "report", "shri" appear in
# nearly every doc and never help recall).
#
# Output shape (compact int postings):
#   { version, generated_at, report_count, vocab_size, total_postings,
#     freq_cutoff, stopwords_count,
#     report_keys: [<key>, ...],
#     vocab:       [<token>, ...]   # sorted alphabetically — supports prefix match via binary search
#     postings:    [[ri, ri, ...], ...]   # postings[i] = sorted ints into report_keys for vocab[i] }
#
# Bundle path moves to docs/drsc/search-index.json in v1.0a phase 2.

# Common English stopwords (keep small; aim is to drop the obvious filler,
# not to do full IR-grade stopword filtering — the freq cutoff catches the
# rest of the high-frequency tokens corpus-side).
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

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Drop tokens in >FREQ_CUTOFF_HIGH of docs (governmental near-universals like
# "report", "committee", "shri" — every doc has them, useless for search).
# Drop tokens in <FREQ_CUTOFF_LOW (default: <2 docs ⇒ singleton, usually OCR
# garbage or report-specific names that won't be queried).
_FREQ_CUTOFF_HIGH = 0.9
_FREQ_CUTOFF_LOW  = 2
_MAX_TOKEN_LEN    = 25   # 25+ char tokens are essentially always OCR junk


# v1.0c: SHARDED. Same per-asset 25 MiB ceiling as the bundle. We split
# by sorted reportKey range. Each shard carries the FULL vocabulary
# (~1.5-3 MiB) but only its slice's report_keys + postings. App keeps
# shards separate at query time and unions doc-key results across them.
# Vocab duplication adds ~10% disk overhead vs unsharded; clean code
# wins.

def build_search_index(docs_per_shard=DOCS_PER_SHARD):
    reports = load_existing_reports()
    text_root = DOCS / "text"
    if not text_root.exists():
        return None

    # First pass: collect (report_key, set_of_tokens) per doc. Sets dedupe
    # within-doc — we don't store positions or counts, just presence.
    docs: list[tuple[str, set[str]]] = []   # (report_key, tokens)
    df: dict[str, int] = {}

    for committee_key, committee_reports in reports.items():
        for r in committee_reports:
            num = r.get("report_number")
            ls  = r.get("lok_sabha")
            if num is None:
                continue
            file_id = _file_id(ls, num)
            if file_id is None:
                continue
            text_path = text_root / committee_key / f"{file_id}.txt"
            if not text_path.exists():
                continue
            try:
                text = text_path.read_text(encoding="utf-8", errors="replace")
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
                    break
            key = f"{committee_key}|{ls if ls is not None else ''}|{num}"
            docs.append((key, tokens))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

    n_docs = len(docs)
    if n_docs == 0:
        return None

    # Frequency cutoffs (corpus-wide — applied once across all shards so
    # vocab is consistent everywhere, app can search uniformly).
    high = int(n_docs * _FREQ_CUTOFF_HIGH)
    low  = _FREQ_CUTOFF_LOW
    keep_tokens = sorted(t for t, c in df.items() if low <= c <= high)
    token_to_idx = {t: i for i, t in enumerate(keep_tokens)}

    # Sort docs by key for stable, deterministic shard composition.
    docs.sort(key=lambda d: d[0])

    n_shards = max(1, (n_docs + docs_per_shard - 1) // docs_per_shard)
    shard_size = (n_docs + n_shards - 1) // n_shards
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _delta(lst):
        if not lst:
            return lst
        out = [lst[0]]; prev = lst[0]
        for x in lst[1:]:
            out.append(x - prev); prev = x
        return out

    total_postings = 0
    shards_meta: list[dict] = []
    total_size_bytes = 0

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end   = min(start + shard_size, n_docs)
        slc   = docs[start:end]

        # Per-shard postings: doc indices are LOCAL to this shard
        # (0..len(slc)-1). App applies per-shard offsets when merging.
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
            "version":          3,             # sharded
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

    _delete_legacy(DOCS / "search-index.json")

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


def write_meta(total_reports, total_with_text, bundle_stats=None, index_stats=None):
    """Write meta.json — purely a function of disk state. The split-phase
    architecture separates extraction (writers) from derivation (this
    function's caller); per-run extract counts live in workflow logs, not
    in meta. See CONV.md "Split-phase scraping pattern".
    """
    meta = {
        "version": "1.2",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lok_sabhas": LOK_SABHAS,
        "total_reports": total_reports,
        "total_with_text": total_with_text,
        "search_bundle": bundle_stats,   # None if bundle wasn't built this run
        "search_index":  index_stats,    # None if index wasn't built this run
    }
    with open(DOCS / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def phase_extract():
    """Phase 1 — produce primary files (reports.json + text/<committee>/<file_id>.txt).
    Safe to run concurrently with other extract-phase workflows.
    """
    print("\n[Extract 1/3] Scraping committee metadata...")
    for ls in LOK_SABHAS:
        print(f"  Lok Sabha {ls}:")
        scrape_all_committees(lok_sabha=ls)

    print("\n[Extract 2/3] Migrating any pre-v0.4 un-prefixed text files...")
    mig = migrate_unprefixed_text_files()
    print(f"  migrated={mig['migrated']} ambiguous={mig['ambiguous']} "
          f"missing_metadata={mig['missing']} preexisting={mig['preexisting']}")

    print("\n[Extract 3/3] Extracting missing texts (priority: newest first)...")
    extract_stats = extract_missing_texts()
    print(f"  extracted={len(extract_stats['extracted'])} "
          f"failed={len(extract_stats['failed'])} "
          f"rate_limited={extract_stats.get('rate_limited', False)} "
          f"budget_hit={extract_stats.get('budget_hit', False)} "
          f"remaining_after={max(0, extract_stats.get('candidates_total', 0) - len(extract_stats['extracted']))}")


def phase_derive():
    """Phase 2 — regenerate derived files (manifest.json, committees.json,
    search-bundle-*.json, search-index-*.json, meta.json) from the on-disk
    primary state. Pure function of reports.json + text/. Owned by
    drsc-derive.yml; one run at a time via `cancel-in-progress: true`.
    """
    print("\n[Derive 1/4] Building manifest + committees index...")
    manifest = build_manifest()
    with open(DOCS / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    build_committees_index()

    print("\n[Derive 2/4] Building search bundle (title + first 5K chars per report)...")
    bundle_stats = build_search_bundle()
    if bundle_stats:
        mb = bundle_stats["size_bytes"] / (1024 * 1024)
        print(f"  search-bundle: {bundle_stats['total']} entries across {bundle_stats['shard_count']} shards, "
              f"{bundle_stats['truncated']} truncated to {bundle_stats['head_chars']} chars · "
              f"{mb:.1f} MB raw (CF gzip serves ~30%)")
    else:
        print("  no text/ directory yet — skipping bundle build")

    print("\n[Derive 3/4] Building search index (inverted token index, full body)...")
    index_stats = build_search_index()
    if index_stats:
        mb = index_stats["size_bytes"] / (1024 * 1024)
        print(f"  search-index: {index_stats['report_count']} docs across {index_stats['shard_count']} shards, "
              f"vocab={index_stats['vocab_size']} (post-cutoff), "
              f"postings={index_stats['total_postings']} · {mb:.1f} MB raw")
    else:
        print("  no text/ directory yet — skipping index build")

    print("\n[Derive 4/4] Writing meta.json...")
    reports = load_existing_reports()
    total = sum(len(v) for v in reports.values())
    total_with_text = sum(len(v) for v in manifest.get("texts", {}).values())
    meta = write_meta(total, total_with_text, bundle_stats, index_stats)
    print(json.dumps(meta, indent=2))


def main():
    """Dispatch to extract / derive / both based on BUILD_PHASE.

    BUILD_PHASE=extract — for scrape.yml writers. Produces only reports.json
                          + text/<committee>/<file_id>.txt; never touches
                          derived files. Multiple writers can race-safely
                          commit because their changes are on different
                          filenames.
    BUILD_PHASE=derive  — for drsc-derive.yml. Reads disk state, regenerates
                          manifest.json + committees.json + search-bundle-*.json
                          + search-index-*.json + meta.json. Single owner of
                          derived files (concurrency cancel-in-progress: true).
    BUILD_PHASE=all     — legacy / local-dev convenience. Runs both phases
                          in one process; same output as a writer + derive
                          run back-to-back. Don't use in CI — the split-phase
                          workflows exist precisely to avoid this race.
    """
    phase = os.environ.get("BUILD_PHASE", "all").lower()
    if phase not in ("extract", "derive", "all"):
        print(f"BUILD_PHASE={phase!r} is not one of extract|derive|all — aborting.", file=sys.stderr)
        sys.exit(2)

    print("=== ParliamentWatch static builder ===")
    print(f"BUILD_PHASE             : {phase}")
    print(f"DATA_DIR                : {DOCS}")
    if phase in ("extract", "all"):
        print(f"LOK_SABHAS              : {LOK_SABHAS}")
        print(f"MAX_EXTRACTIONS_PER_RUN : {MAX_EXTRACTIONS_PER_RUN}")
        print(f"MAX_RUN_SECONDS         : {MAX_RUN_SECONDS}")
        print(f"EXTRACT_WORKERS         : {EXTRACT_WORKERS}")

    if phase in ("extract", "all"):
        phase_extract()

    if phase in ("derive", "all"):
        phase_derive()

    print("\nDone.")


if __name__ == "__main__":
    main()
