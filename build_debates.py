#!/usr/bin/env python3
"""Debates corpus builder — orchestrator.

Per scheduled (or workflow_dispatch) run:
  1. Walk the LS debate-search API for each LS term in LOK_SABHAS,
     newest-first. Merge metadata into reports.json under the "ls" key.
  2. For each record missing extracted text, call debate-details and
     persist the stripped-HTML body as text/ls/<file_id>.txt.
  3. (Derive phase) Build manifest + sharded bundle + sharded index +
     meta + audit.

Output goes under docs/debates/, served at
sansadsaar-data.naklitechie.com/debates/.

PHASE A scope: LS only. Phase B (Rajya Sabha) extends this orchestrator
with a parallel call into debates/scrapers/rajyasabha.py — the structure
here (house-keyed reports.json, house-stratified text/ subdirs) is
designed so the RS addition is strictly additive, no refactor needed.

Storage strategy (per plan/debates-recon-001.md §"Decisions"): fetch-
extract-delete. No PDF involvement on the LS side at all (the API
serves HTML bodies which we strip to plain text directly).

Independence Principle: no imports from cag, lc, fc, drsc, or bills
scrapers. The HTTP layer + RateLimited + jitter + checkpoint primitives
are re-implemented under debates/common.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from debates.common import RateLimited
from debates.scrapers.loksabha import (
    DEFAULT_LOK_SABHAS,
    LSDebate,
    fetch_page as ls_fetch_page,
    extract_text as ls_extract_text,
    file_id as ls_file_id,
    report_key as ls_report_key,
)

# ── Paths ───────────────────────────────────────────────────────────────────

ASSETS    = ROOT / "docs"
DOCS      = ASSETS / "debates"
TEXT_DIR  = DOCS / "text"
DOCS.mkdir(parents=True, exist_ok=True)

REPORTS_JSON  = DOCS / "reports.json"
MANIFEST_JSON = DOCS / "manifest.json"
META_JSON     = DOCS / "meta.json"

# Houses we cover. Phase A is LS-only; phase B adds "rs".
HOUSES = ["ls"]

# ── Per-run budget ─────────────────────────────────────────────────────────
#
# Per the recon doc's confirmed decisions: 50 records per burst, 12×/day
# during LS phase A = 600/day. ~107 days to clear LS-13..18's ~64K
# records. Per-burst budget = 50 (within Politeness policy upper bound).
# 12×/day = 2-hour gaps between bursts.

MAX_EXTRACTIONS_PER_RUN = int(os.environ.get("MAX_EXTRACTIONS_PER_RUN", "50"))
MAX_RUN_SECONDS         = int(os.environ.get("MAX_RUN_SECONDS", "900"))    # 15 min
EXTRACT_WORKERS         = int(os.environ.get("EXTRACT_WORKERS", "4"))

# LS terms to enumerate this run. Comma-separated env var. Default =
# all of LS-13..18. The merge-with-existing logic preserves entries
# from un-enumerated terms across runs, so narrowing LOK_SABHAS to
# (say) "18" for daily steady-state is safe.
_LS_RAW = os.environ.get("LOK_SABHAS", "").strip()
LOK_SABHAS = ([int(x) for x in _LS_RAW.split(",") if x.strip()]
              if _LS_RAW else list(DEFAULT_LOK_SABHAS))

# Cooldown after a 429 / 403 — defensive.
RATE_LIMIT_COOLDOWN_SECONDS = int(os.environ.get("RATE_LIMIT_COOLDOWN_SECONDS", str(6 * 3600)))

# ── In-flight checkpointing ────────────────────────────────────────────────

CHECKPOINT_EVERY_N = int(os.environ.get("CHECKPOINT_EVERY_N", "25"))
CHECKPOINT_EVERY_S = int(os.environ.get("CHECKPOINT_EVERY_S", "300"))


def _git(*args) -> tuple[int, str, str]:
    p = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return p.returncode, p.stdout, p.stderr


def checkpoint_commit(message: str, paths: list[str]) -> bool:
    rc, _, err = _git("add", "--", *paths)
    if rc != 0:
        print(f"  [checkpoint] git add failed: {err.strip() or 'unknown'}")
        return False
    rc, _, _ = _git("diff", "--cached", "--quiet")
    if rc == 0:
        return True
    rc, _, err = _git("commit", "-m", message)
    if rc != 0:
        print(f"  [checkpoint] git commit failed: {err.strip()}")
        return False
    rc, _, err = _git("pull", "--rebase", "origin", "main")
    if rc != 0:
        print(f"  [checkpoint] git pull --rebase failed: {err.strip()} — aborting")
        _git("rebase", "--abort")
        return False
    rc, _, err = _git("push")
    if rc != 0:
        print(f"  [checkpoint] git push failed: {err.strip()}")
        return False
    print(f"  [checkpoint] pushed: {message}")
    return True


# ── Phase 1: walk + merge LS records ───────────────────────────────────────


def load_existing_reports() -> dict[str, list[dict]]:
    """Load house → list-of-reports dict.

    Phase A only has "ls"; phase B adds "rs". The on-disk format is
    forward-compatible: missing house keys are tolerated.
    """
    if REPORTS_JSON.exists():
        with open(REPORTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {h: [] for h in HOUSES}


def save_reports(reports: dict[str, list[dict]]) -> None:
    """Sort each house's list deterministically:
       - LS: lok_sabha desc, session desc, db_slno desc (newest first).
    """
    out = {}
    for h in HOUSES:
        items = reports.get(h, [])
        if h == "ls":
            items.sort(key=lambda r: (
                -int(r.get("lok_sabha") or 0),
                -int(r.get("session")   or 0),
                -int(r.get("db_slno")   or 0)))
        out[h] = items
    # Preserve any unknown-house keys for forward compatibility.
    for h, items in reports.items():
        if h not in HOUSES:
            out[h] = items
    with open(REPORTS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def _record_from_lsdebate(d: LSDebate) -> dict:
    return {
        "house":            "ls",
        "lok_sabha":        d.lok_sabha,
        "session":          d.session,
        "db_slno":          d.db_slno,
        "title":            d.title,
        "debate_date":      d.debate_date,
        "debate_type":      d.debate_type,
        "debate_type_desc": d.debate_type_desc,
        "members":          d.members,
        "keywords":         d.keywords,
    }


def _ls_key_tuple(r: dict) -> tuple[int, int, int]:
    return (int(r["lok_sabha"]), int(r["session"]), int(r["db_slno"]))


def walk_ls_and_merge(existing: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    """Walk LS debate-search across LOK_SABHAS, merge with existing
    records. Fresh wins on conflict; entries in LS terms NOT walked
    this run are preserved (the merge contract).

    Newest-first ordering within each LS via the API's default sort
    (desc by dbSlno, which monotonically tracks date).
    """
    print(f"[Walk] LS terms {LOK_SABHAS} (newest-first within each)...")
    t0 = time.time()
    fresh_ls: dict[tuple[int, int, int], dict] = {}
    seen_terms: set[int] = set()
    page_size = 200
    for ls in sorted(LOK_SABHAS, reverse=True):
        seen_terms.add(ls)
        page = 1
        total_pages = 1
        per_term = 0
        while page <= total_pages:
            try:
                records, meta = ls_fetch_page(ls, page=page, size=page_size)
            except RateLimited:
                raise
            except Exception as e:
                print(f"  ERR LS{ls} page {page}: {e} — aborting this LS")
                break
            total_pages = meta.get("totalPages") or 1
            for rec in records:
                key = (rec.lok_sabha, rec.session, rec.db_slno)
                if key in fresh_ls:
                    continue
                fresh_ls[key] = _record_from_lsdebate(rec)
            per_term += len(records)
            if page == 1 or page % 10 == 0 or page == total_pages:
                print(f"  LS{ls} page {page}/{total_pages}: {len(records)} records "
                      f"(running={per_term})")
            page += 1
        print(f"  LS{ls} done: {per_term} records")
    print(f"  walked all in {time.time()-t0:.1f}s — {len(fresh_ls)} records across walked terms")

    # Merge: fresh-LS overrides; existing LS entries in NON-walked
    # terms are preserved; existing LS entries in walked terms that
    # aren't in fresh are also preserved (archival promise — upstream
    # delisting doesn't drop us).
    merged: dict[str, list[dict]] = {h: list(existing.get(h, [])) for h in HOUSES}
    old_ls_by_key = {_ls_key_tuple(r): r for r in merged.get("ls", [])}
    new_count = 0; updated_count = 0; kept_legacy = 0
    out_ls: list[dict] = []
    combined_keys = set(old_ls_by_key.keys()) | set(fresh_ls.keys())
    for key in combined_keys:
        if key in fresh_ls:
            new_rec = fresh_ls[key]
            if key in old_ls_by_key:
                if old_ls_by_key[key] != new_rec:
                    updated_count += 1
            else:
                new_count += 1
            out_ls.append(new_rec)
        else:
            # Old entry not in fresh — "kept_legacy" only if its LS was walked
            if key[0] in seen_terms:
                kept_legacy += 1
            out_ls.append(old_ls_by_key[key])
    merged["ls"] = out_ls
    # Preserve any non-known-house keys (forward compat).
    for h, items in existing.items():
        if h not in HOUSES:
            merged[h] = items
    stats = {"new": new_count, "updated": updated_count, "kept_legacy": kept_legacy}
    total = sum(len(merged.get(h, [])) for h in HOUSES)
    print(f"  merged: {total} total LS records "
          f"(new={new_count}, updated={updated_count}, kept_legacy={kept_legacy})")
    return merged, stats


# ── Phase 2: extract missing bodies ───────────────────────────────────────


def _check_cooldown_and_skip() -> bool:
    if not META_JSON.exists():
        return False
    try:
        with open(META_JSON, "r", encoding="utf-8") as f:
            prev = json.load(f)
    except Exception:
        return False
    if not prev.get("rate_limited"):
        return False
    last_at = prev.get("rate_limited_at")
    if not last_at:
        return False
    try:
        last_dt = datetime.fromisoformat(last_at.replace("Z", "+00:00"))
    except Exception:
        return False
    elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
    if elapsed < RATE_LIMIT_COOLDOWN_SECONDS:
        print(f"  COOLDOWN: previous run rate-limited {elapsed:.0f}s ago; skipping (limit {RATE_LIMIT_COOLDOWN_SECONDS}s)")
        return True
    return False


def extract_missing_bodies(reports: dict[str, list[dict]], *, deadline: float) -> dict:
    """For each LS record without a `.txt` and without an empty/error
    marker, fetch debate-details, strip HTML, save as text. Bounded by
    MAX_EXTRACTIONS_PER_RUN.
    """
    if _check_cooldown_and_skip():
        return {"extracted": [], "failed": [], "rate_limited": True,
                "budget_hit": False, "skipped_due_to_cooldown": True,
                "candidates_total": 0}

    ls_text_dir = TEXT_DIR / "ls"
    candidates: list[tuple[int, int, int]] = []  # (loksabha, session, dbslno)
    skipped_marked = 0
    for r in reports.get("ls", []):
        ls = r.get("lok_sabha"); ses = r.get("session"); dn = r.get("db_slno")
        if ls is None or ses is None or dn is None:
            continue
        fid = ls_file_id(int(ls), int(ses), int(dn))
        if (ls_text_dir / f"{fid}.txt").exists():
            skipped_marked += 1; continue
        if (ls_text_dir / f"{fid}.empty").exists():
            # Upstream had no body — don't retry. Empty bodies are
            # legitimate for procedural items (rulings, papers laid).
            skipped_marked += 1; continue
        # NB: .error markers are retryable — fall through to candidate.
        candidates.append((int(ls), int(ses), int(dn)))

    # Priority: newest LS first, then highest dbSlno (newest record).
    candidates.sort(key=lambda c: (-c[0], -c[1], -c[2]))
    print(f"  candidates: {len(candidates)} LS records missing text "
          f"({skipped_marked} skipped — already marked)")

    if not candidates:
        return {"extracted": [], "failed": [], "rate_limited": False,
                "budget_hit": False, "candidates_total": 0}

    remaining = deadline - time.monotonic()
    if remaining <= 60:
        print(f"  only {remaining:.0f}s left in budget — skipping extract phase")
        return {"extracted": [], "failed": [], "rate_limited": False,
                "budget_hit": True, "candidates_total": len(candidates)}

    target = candidates[:MAX_EXTRACTIONS_PER_RUN]
    print(f"  budget: extracting up to {len(target)} this run "
          f"(MAX_EXTRACTIONS_PER_RUN={MAX_EXTRACTIONS_PER_RUN}, "
          f"remaining={remaining:.0f}s, EXTRACT_WORKERS={EXTRACT_WORKERS})")

    extracted: list[tuple[int, int, int]] = []
    failed: list[tuple[int, int, int]] = []
    rate_limited = False; budget_hit = False
    last_checkpoint_at = time.monotonic()
    extracted_since_checkpoint = 0

    def _do(c):
        ls, ses, dn = c
        try:
            text = ls_extract_text(ls, ses, dn, text_dir=str(ls_text_dir))
            return c, text, None
        except RateLimited as rl:
            return c, None, rl
        except Exception as e:
            return c, None, e

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
        futures = {ex.submit(_do, c): c for c in target}
        for fut in as_completed(futures):
            c, text, err = fut.result()
            if isinstance(err, RateLimited):
                print(f"  [RATE-LIMITED] LS{c[0]} S{c[1]} #{c[2]}: {err}")
                rate_limited = True
                for f in futures:
                    if not f.done(): f.cancel()
                break
            elif text:
                extracted.append(c)
                extracted_since_checkpoint += 1
            else:
                # Empty body (.empty marker written) or fetch error
                # (.error marker written). Either way it's resolved-
                # for-this-run.
                failed.append(c)
                extracted_since_checkpoint += 1
            now = time.monotonic()
            if (extracted_since_checkpoint >= CHECKPOINT_EVERY_N or
                (extracted_since_checkpoint > 0 and now - last_checkpoint_at >= CHECKPOINT_EVERY_S)):
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
                checkpoint_commit(
                    f"Auto-checkpoint debates primary data (extracted={len(extracted)} this run) [{ts}]",
                    ["docs/debates/reports.json", "docs/debates/text/"],
                )
                extracted_since_checkpoint = 0
                last_checkpoint_at = now
            if now > deadline:
                print(f"  [BUDGET] wall-clock budget hit after {len(extracted)} successes")
                budget_hit = True
                for f in futures:
                    if not f.done(): f.cancel()
                break

    if extracted_since_checkpoint > 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        checkpoint_commit(
            f"Auto-checkpoint debates primary data (final, extracted={len(extracted)} this run) [{ts}]",
            ["docs/debates/reports.json", "docs/debates/text/"],
        )

    return {
        "extracted": extracted, "failed": failed,
        "rate_limited": rate_limited, "budget_hit": budget_hit,
        "candidates_total": len(candidates),
    }


# ── Phase 3: derived files ─────────────────────────────────────────────────


def build_manifest() -> dict:
    """House-keyed manifest. {texts: {ls: {file_id: {size, url}}, rs: {}}}.
    """
    out: dict[str, dict] = {}
    if not TEXT_DIR.exists():
        return {"texts": out}
    for h in HOUSES:
        h_dir = TEXT_DIR / h
        if not h_dir.exists():
            continue
        out[h] = {}
        for text_file in sorted(h_dir.glob("*.txt")):
            fid = text_file.stem
            out[h][fid] = {
                "size": text_file.stat().st_size,
                "url":  f"text/{h}/{text_file.name}",
            }
    return {"texts": out}


def compute_audit(reports: dict[str, list[dict]]) -> dict:
    """Per-record status across the corpus."""
    counts = {
        "records":               sum(len(reports.get(h, [])) for h in HOUSES),
        "with_text":             0,
        "empty_upstream":        0,
        "error_retryable":       0,
        "never_attempted":       0,
    }
    if not TEXT_DIR.exists():
        counts["never_attempted"] = counts["records"]
        return {
            "audited_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "totals":     counts,
        }
    ls_dir = TEXT_DIR / "ls"
    for r in reports.get("ls", []):
        ls = r.get("lok_sabha"); ses = r.get("session"); dn = r.get("db_slno")
        if ls is None or ses is None or dn is None:
            continue
        fid = ls_file_id(int(ls), int(ses), int(dn))
        if (ls_dir / f"{fid}.txt").exists():
            counts["with_text"] += 1
        elif (ls_dir / f"{fid}.empty").exists():
            counts["empty_upstream"] += 1
        elif (ls_dir / f"{fid}.error").exists():
            counts["error_retryable"] += 1
        else:
            counts["never_attempted"] += 1
    return {
        "audited_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totals":     counts,
    }


# ── Search bundle + index (same sharding as LC/FC) ─────────────────────────

DOCS_PER_SHARD = 2500


def _delete_legacy(path: Path) -> None:
    if path.exists():
        path.unlink()


def build_search_bundle(reports: dict[str, list[dict]],
                        docs_per_shard: int = DOCS_PER_SHARD) -> dict | None:
    """Title + first 5K chars per record. Sharded by sorted reportKey."""
    if not TEXT_DIR.exists():
        return None
    HEAD = 5000
    entries = []
    truncated = 0
    ls_dir = TEXT_DIR / "ls"
    for r in reports.get("ls", []):
        ls = r.get("lok_sabha"); ses = r.get("session"); dn = r.get("db_slno")
        if ls is None or ses is None or dn is None: continue
        fid = ls_file_id(int(ls), int(ses), int(dn))
        text_path = ls_dir / f"{fid}.txt"
        if not text_path.exists(): continue
        try:
            text = text_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        head = text[:HEAD]
        if len(text) > HEAD: truncated += 1
        entries.append({
            "k":    ls_report_key(int(ls), int(ses), int(dn)),
            "t":    r.get("title", ""),
            "head": head,
        })
    entries.sort(key=lambda e: e["k"])
    if not entries:
        return None
    _delete_legacy(DOCS / "search-bundle.json")
    shard_count = (len(entries) + docs_per_shard - 1) // docs_per_shard
    shard_sizes: dict[str, int] = {}
    max_shard = 0; total_bytes = 0
    for i in range(shard_count):
        chunk = entries[i*docs_per_shard:(i+1)*docs_per_shard]
        path = DOCS / f"search-bundle-{i:02d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "head_chars": HEAD,
                       "shard_index": i, "entries": chunk}, f, ensure_ascii=False)
        sz = path.stat().st_size
        shard_sizes[f"search-bundle-{i:02d}.json"] = sz
        max_shard = max(max_shard, sz); total_bytes += sz
    return {
        "shard_count":     shard_count,
        "shards":          list(shard_sizes.keys()),
        "shard_sizes":     shard_sizes,
        "total":           len(entries),
        "truncated":       truncated,
        "head_chars":      HEAD,
        "size_bytes":      total_bytes,
        "max_shard_bytes": max_shard,
    }


import re
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_search_index(docs_per_shard: int = DOCS_PER_SHARD) -> dict | None:
    """Full-body inverted token index, sharded."""
    if not TEXT_DIR.exists():
        return None
    # Gather (report_key, text) in deterministic order.
    docs: list[tuple[str, str]] = []
    ls_dir = TEXT_DIR / "ls"
    if ls_dir.exists():
        files = list(ls_dir.glob("*.txt"))
        def _sort_key(p: Path):
            stem = p.stem  # "LS18_S7_5183"
            try:
                _, ls_part, rest = stem.split("_", 2)
                ls = int(ls_part.removeprefix("LS")) if False else int(stem[2:stem.find("_")])
                ses_part, dn_part = rest.split("_", 1)
                ses = int(ses_part.removeprefix("S"))
                return (-ls, -ses, -int(dn_part))
            except Exception:
                return (0, 0, 0)
        files.sort(key=_sort_key)
        for path in files:
            stem = path.stem  # "LS18_S7_5183"
            try:
                # Parse LS<n>_S<s>_<d>
                m = re.match(r"LS(\d+)_S(\d+)_(\d+)$", stem)
                if not m: continue
                ls, ses, dn = m.group(1), m.group(2), m.group(3)
                rk = f"debates|ls|{ls}|{ses}|{dn}"
            except Exception:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            docs.append((rk, text))
    if not docs:
        return None
    token_docs: dict[str, set[int]] = {}
    for i, (_, text) in enumerate(docs):
        for tok in set(_tokenize(text)):
            token_docs.setdefault(tok, set()).add(i)
    n_docs = len(docs)
    high_cut = max(2, int(n_docs * 0.5))
    low_cut  = 2
    vocab = sorted(tok for tok, ds in token_docs.items()
                   if low_cut <= len(ds) <= high_cut)
    if not vocab:
        vocab = sorted(token_docs.keys())
        low_cut = 1; high_cut = n_docs
    _delete_legacy(DOCS / "search-index.json")
    shard_count = (n_docs + docs_per_shard - 1) // docs_per_shard
    shard_sizes: dict[str, int] = {}
    max_shard = 0; total_bytes = 0; total_postings = 0
    for i in range(shard_count):
        start = i * docs_per_shard; end = min(start + docs_per_shard, n_docs)
        shard_keys = [docs[j][0] for j in range(start, end)]
        local_postings: list[list[int]] = []
        for tok in vocab:
            ds = token_docs[tok]
            in_shard = sorted(d - start for d in ds if start <= d < end)
            local_postings.append(in_shard)
            total_postings += len(in_shard)
        path = DOCS / f"search-index-{i:02d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "version":     "1.0",
                "shard_index": i,
                "vocab":       vocab,
                "report_keys": shard_keys,
                "postings":    local_postings,
            }, f, ensure_ascii=False)
        sz = path.stat().st_size
        shard_sizes[f"search-index-{i:02d}.json"] = sz
        max_shard = max(max_shard, sz); total_bytes += sz
    return {
        "shard_count":      shard_count,
        "shards":           list(shard_sizes.keys()),
        "shard_sizes":      shard_sizes,
        "report_count":     n_docs,
        "vocab_size":       len(vocab),
        "total_postings":   total_postings,
        "size_bytes":       total_bytes,
        "max_shard_bytes":  max_shard,
        "freq_cutoff_low":  low_cut,
        "freq_cutoff_high": high_cut,
    }


def write_meta(*, total_records: int, total_with_text: int,
               bundle_stats: dict | None, index_stats: dict | None) -> dict:
    meta = {
        "version":         "1.0",
        "corpus":          "debates",
        "generated_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_records":   total_records,
        "total_with_text": total_with_text,
        "houses":          HOUSES,
        "lok_sabhas":      LOK_SABHAS,
        "search_bundle":   bundle_stats,
        "search_index":    index_stats,
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


# ── Main ────────────────────────────────────────────────────────────────────


def phase_extract() -> None:
    overall_deadline = time.monotonic() + MAX_RUN_SECONDS
    print(f"\n[Extract 1/2] Walking LS {LOK_SABHAS}...")
    existing = load_existing_reports()
    print(f"  existing on disk: {sum(len(existing.get(h, [])) for h in HOUSES)} records")
    merged, walk_stats = walk_ls_and_merge(existing)
    save_reports(merged)

    print(f"\n[Extract 2/2] Fetching debate-details bodies (priority: newest LS, highest dbSlno)...")
    stats = extract_missing_bodies(merged, deadline=overall_deadline)
    print(f"  extracted={len(stats['extracted'])} "
          f"failed={len(stats['failed'])} "
          f"rate_limited={stats.get('rate_limited', False)} "
          f"budget_hit={stats.get('budget_hit', False)} "
          f"remaining_after={max(0, stats.get('candidates_total', 0) - len(stats.get('extracted', [])) - len(stats.get('failed', [])))}")


def phase_derive() -> None:
    reports = load_existing_reports()
    total = sum(len(reports.get(h, [])) for h in HOUSES)

    print("\n[Derive 1/4] Building manifest.json...")
    manifest = build_manifest()
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    n_with_text = sum(len(v) for v in manifest["texts"].values())
    print(f"  manifest: {n_with_text} records with extracted text")

    print("\n[Derive 2/4] Building search-bundle...")
    bundle_stats = build_search_bundle(reports)
    if bundle_stats:
        mb = bundle_stats["size_bytes"] / (1024 * 1024)
        print(f"  bundle: {bundle_stats['total']} entries × {bundle_stats['shard_count']} shards · {mb:.1f} MB")
    else:
        print("  no extracted texts yet — skipping bundle")

    print("\n[Derive 3/4] Building search-index...")
    index_stats = build_search_index()
    if index_stats:
        mb = index_stats["size_bytes"] / (1024 * 1024)
        print(f"  index: {index_stats['report_count']} docs × {index_stats['shard_count']} shards, "
              f"vocab={index_stats['vocab_size']}, postings={index_stats['total_postings']} · {mb:.1f} MB")
    else:
        print("  no extracted texts yet — skipping index")

    print("\n[Derive 4/4] Writing meta.json + audit.json...")
    meta = write_meta(total_records=total, total_with_text=n_with_text,
                      bundle_stats=bundle_stats, index_stats=index_stats)
    print(json.dumps(meta, indent=2))

    audit = compute_audit(reports)
    with open(DOCS / "audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    t = audit["totals"]
    print(f"\n  audit: with_text={t['with_text']} "
          f"empty={t['empty_upstream']} "
          f"error={t['error_retryable']} "
          f"never_attempted={t['never_attempted']} (total={t['records']})")


def main():
    """Dispatch to extract / derive / both based on BUILD_PHASE."""
    phase = os.environ.get("BUILD_PHASE", "all").lower()
    if phase not in ("extract", "derive", "all"):
        print(f"BUILD_PHASE={phase!r} not in extract|derive|all — aborting.", file=sys.stderr)
        sys.exit(2)

    print("=== ParliamentWatch Debates (LS phase A) static builder ===")
    print(f"BUILD_PHASE             : {phase}")
    print(f"DOCS                    : {DOCS}")
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
