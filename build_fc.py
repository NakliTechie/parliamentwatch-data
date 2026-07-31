#!/usr/bin/env python3
"""Financial Committees (PAC + Estimates + COPU) static-data builder.

Per scheduled (or workflow_dispatch) run:
  1. Walk the 3 LS Financial Committees × N LS terms via sansad.in's
     api_ls/committee/lsRSAllReports endpoint.
  2. Save reports.json (committee-keyed dict, matching DRSC's shape).
  3. Extract text from missing PDFs as fits the per-run budget — newest
     first (highest LS, then highest report_number).
  4. (Derive phase) Build manifest + sharded bundle + sharded index +
     meta + audit.

Output goes under docs/fc/, served at sansadsaar-data.naklitechie.com/fc/.

Structurally a hybrid: shape borrowed from build_lc.py (split-phase,
checkpoint cadence, derive separation) + composite-key handling
borrowed from DRSC's build_static.py (committee → list-of-reports
structure, file_id = LS<ls>_<num> under committee subdir).

No archive sidecar (sansad.in is the most-stable upstream we touch
— see plan/financial-committees-recon-001.md §"No archive sidecar
needed" for the rationale).

Independence Principle (CONV.md): no imports from drsc, cag, lc, or
bills scrapers. The HTTP layer, RateLimited exception, jitter, and
checkpoint primitives are re-implemented in fc/scraper.py + here.
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
sys.path.insert(0, str(ROOT))   # so `from fc.scraper import ...` works

from parliamentwatch_text_shards import (
    write_text_shards, consolidate_markers, load_markers, write_json_idempotent,
    load_bundled_ids,
)
from fc.scraper import (
    BASE_URL, REPORTS_API,
    COMMITTEES, DEFAULT_LOK_SABHAS,
    FCReport, RateLimited,
    walk_all_committees, get_report_text,
    file_id,
)

# ── Paths ───────────────────────────────────────────────────────────────────

ASSETS    = ROOT / "docs"
DOCS      = ASSETS / "fc"
TEXT_DIR  = DOCS / "text"
PDFS_DIR  = DOCS / "pdfs"        # gitignored via docs/**/pdfs/
DOCS.mkdir(parents=True, exist_ok=True)

REPORTS_JSON  = DOCS / "reports.json"
MANIFEST_JSON = DOCS / "manifest.json"
META_JSON     = DOCS / "meta.json"

# ── Per-run budget ─────────────────────────────────────────────────────────

MAX_EXTRACTIONS_PER_RUN = int(os.environ.get("MAX_EXTRACTIONS_PER_RUN", "25"))
MAX_RUN_SECONDS         = int(os.environ.get("MAX_RUN_SECONDS", "900"))   # 15 min
EXTRACT_WORKERS         = int(os.environ.get("EXTRACT_WORKERS", "4"))

# LS terms to walk this run. Default = DEFAULT_LOK_SABHAS (14..18).
# Workflow dispatch sets this:
#   - LOK_SABHAS=18           — steady-state daily (only current term gets
#                                new reports)
#   - LOK_SABHAS=14,15,16,17,18 — one-shot backfill of the full ~800-report
#                                 historical corpus
_LS_RAW = os.environ.get("LOK_SABHAS", "").strip()
LOK_SABHAS = ([int(x) for x in _LS_RAW.split(",") if x.strip()]
              if _LS_RAW else list(DEFAULT_LOK_SABHAS))

# Cooldown after a 429 / 403 — defensive. sansad.in has never rate-
# limited in production, but we adopt the same pattern as the other
# corpora so future changes upstream don't break the pipeline silently.
RATE_LIMIT_COOLDOWN_SECONDS = int(os.environ.get("RATE_LIMIT_COOLDOWN_SECONDS", str(6 * 3600)))

# ── In-flight checkpointing ────────────────────────────────────────────────
#
# Layer 1 crash-safety, same pattern as build_lc.py. Bounded loss on
# runner kill = at most CHECKPOINT_EVERY_N extractions of work.

CHECKPOINT_EVERY_N = int(os.environ.get("CHECKPOINT_EVERY_N", "100"))
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


# ── Phase 1: walk + merge reports ──────────────────────────────────────────


def load_existing_reports() -> dict[str, list[dict]]:
    """Load committee → list-of-reports dict, matching DRSC's shape."""
    if REPORTS_JSON.exists():
        with open(REPORTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {k: [] for k in COMMITTEES}


def save_reports(reports: dict[str, list[dict]]) -> None:
    """Sort each committee's list by (lok_sabha desc, report_number desc)
    so reports.json is deterministic and reads naturally newest-first.
    """
    sorted_reports = {}
    for cmt in COMMITTEES:
        items = reports.get(cmt, [])
        items.sort(key=lambda r: (-int(r.get("lok_sabha") or 0),
                                  -int(r.get("report_number") or 0)))
        sorted_reports[cmt] = items
    with open(REPORTS_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted_reports, f, indent=2, ensure_ascii=False)


def _record_from_fcreport(r: FCReport) -> dict:
    return {
        "committee":             r.committee,
        "committee_name":        r.committee_name,
        "lok_sabha":             r.lok_sabha,
        "report_number":         r.report_number,
        "title":                 r.title,
        "presented_in_ls":       r.presented_in_ls,
        "laid_in_rs":            r.laid_in_rs,
        "date_of_presentation":  r.date_of_presentation,
        "date_of_adoption":      r.date_of_adoption,
        "pdf_url":               r.pdf_url,
        "pdf_url_hindi":         r.pdf_url_hindi,
    }


def walk_and_merge_reports(existing: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    """Walk LS-N for each of the 3 committees, merge with existing.
    Fresh wins on (committee, lok_sabha, report_number) overlap; old
    kept on delisting (archival promise — same contract as Bills/LC).
    """
    print(f"[Walk] Fetching {len(COMMITTEES)} committees × LSes {LOK_SABHAS}...")
    t0 = time.time()
    fresh: dict[str, dict[tuple[int, int], dict]] = {k: {} for k in COMMITTEES}
    for fc_report in walk_all_committees(lok_sabhas=LOK_SABHAS):
        key = (fc_report.lok_sabha, fc_report.report_number)
        if key in fresh[fc_report.committee]:
            print(f"  [walk] duplicate {fc_report.committee} {key}; keeping first")
            continue
        fresh[fc_report.committee][key] = _record_from_fcreport(fc_report)
    walked = sum(len(v) for v in fresh.values())
    print(f"  walked all in {time.time()-t0:.1f}s — {walked} reports across all (committee, LS) pairs")

    # Merge with existing (committee, ls, num) entries. Existing-only
    # entries are kept (covering both archival of upstream-removed
    # reports AND historical-LS entries when the workflow runs with
    # narrow LOK_SABHAS — e.g. daily LOK_SABHAS=18 shouldn't wipe out
    # the historical LS-14..17 entries that previous backfills loaded).
    merged: dict[str, list[dict]] = {k: [] for k in COMMITTEES}
    new_count = 0
    updated_count = 0
    kept_legacy = 0
    for cmt in COMMITTEES:
        old_by_key = {(r["lok_sabha"], r["report_number"]): r
                      for r in existing.get(cmt, [])
                      if r.get("lok_sabha") is not None and r.get("report_number") is not None}
        # Add merged set: fresh entries override old; old entries not
        # in fresh are preserved (archival).
        combined_keys = set(old_by_key.keys()) | set(fresh[cmt].keys())
        for key in combined_keys:
            if key in fresh[cmt]:
                new_rec = fresh[cmt][key]
                if key in old_by_key:
                    if old_by_key[key] != new_rec:
                        updated_count += 1
                else:
                    new_count += 1
                merged[cmt].append(new_rec)
            else:
                # Old entry not in fresh — only counts as "kept_legacy"
                # if this LS was actually walked (i.e. fresh would have
                # had it if it still existed upstream). If we ran with
                # a narrow LOK_SABHAS that doesn't include this LS,
                # this is just "wasn't queried", not "delisted".
                if key[0] in LOK_SABHAS:
                    kept_legacy += 1
                merged[cmt].append(old_by_key[key])
    stats = {"new": new_count, "updated": updated_count, "kept_legacy": kept_legacy}
    total = sum(len(v) for v in merged.values())
    print(f"  merged: {total} total (new={new_count}, updated={updated_count}, "
          f"kept_legacy={kept_legacy})")
    return merged, stats


# ── Phase 2: extract missing texts ─────────────────────────────────────────


def _check_cooldown_and_skip() -> bool:
    """Honour the rate-limit cooldown if a previous run was throttled."""
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
        print(f"  COOLDOWN: previous run was rate-limited "
              f"{elapsed:.0f}s ago; skipping (limit {RATE_LIMIT_COOLDOWN_SECONDS}s)")
        return True
    return False


def extract_missing_texts(reports: dict[str, list[dict]], *, deadline: float) -> dict:
    """Extract up to MAX_EXTRACTIONS_PER_RUN missing PDFs. Stops on
    RateLimited or when `deadline` (monotonic seconds) is exceeded.
    """
    if _check_cooldown_and_skip():
        return {"extracted": [], "failed": [], "rate_limited": True,
                "budget_hit": False, "skipped_due_to_cooldown": True,
                "candidates_total": 0}

    # Skip records already in texts-meta.json's record_to_shard map
    # (bundled = source of truth post-2026-05-14 cleanup). FC composite
    # key matches the build adapter: `<committee>|<file_id>`.
    bundled_ids: set = set()
    texts_meta_path = DOCS / "texts-meta.json"
    if texts_meta_path.exists():
        try:
            with open(texts_meta_path, "r", encoding="utf-8") as f:
                texts_meta = json.load(f)
            bundled_ids = load_bundled_ids(DOCS)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! couldn't read texts-meta.json — proceeding without shard skip ({e})")

    # Build the candidate set across all committees. Skip anything
    # already classified (.txt / .pypdf-empty / .ocr-failed) OR bundled
    # OR carries a permanent marker (.pypdf-empty/.ocr-failed) in
    # markers.json. .pypdf-error stays retryable.
    bundled_markers = load_markers(DOCS)
    candidates: list[tuple[str, int, int, str]] = []  # (committee, ls, num, pdf_url)
    skipped_marked = 0
    skipped_bundled = 0
    for cmt, items in reports.items():
        for r in items:
            ls = r.get("lok_sabha")
            num = r.get("report_number")
            url = r.get("pdf_url")
            if ls is None or num is None or not url:
                continue
            fid = file_id(int(ls), int(num))
            cid = f"{cmt}|{fid}"
            if cid in bundled_ids:
                skipped_bundled += 1; continue
            bm = bundled_markers.get(cid)
            if bm in ("pypdf-empty", "ocr-failed"):
                skipped_marked += 1; continue
            cmt_text_dir = TEXT_DIR / cmt
            if (cmt_text_dir / f"{fid}.txt").exists():
                skipped_marked += 1; continue
            if (cmt_text_dir / f"{fid}.pypdf-empty").exists():
                skipped_marked += 1; continue
            if (cmt_text_dir / f"{fid}.ocr-failed").exists():
                skipped_marked += 1; continue
            candidates.append((cmt, int(ls), int(num), url))

    # Priority: highest LS first, then highest report_number within LS.
    # This favours the most-recent reports (more topical / commonly
    # searched) over deep historical backlog.
    candidates.sort(key=lambda c: (-c[1], -c[2], c[0]))
    print(f"  candidates: {len(candidates)} reports missing extracted text "
          f"({skipped_marked} skipped — already marked; "
          f"{skipped_bundled} skipped — already in shards)")

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

    extracted: list[tuple[str, int, int]] = []
    failed: list[tuple[str, int, int]] = []
    rate_limited = False
    budget_hit = False
    last_checkpoint_at = time.monotonic()
    extracted_since_checkpoint = 0

    def _do(c):
        cmt, ls, num, url = c
        try:
            text = get_report_text(url, committee=cmt, lok_sabha=ls,
                                   report_number=num, text_dir=str(TEXT_DIR),
                                   pdfs_dir=str(PDFS_DIR))
            return c, text, None
        except RateLimited as rl:
            return c, None, rl
        except Exception as e:
            return c, None, e

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
        futures = {ex.submit(_do, c): c for c in target}
        for fut in as_completed(futures):
            c, text, err = fut.result()
            cmt, ls, num, _ = c
            if isinstance(err, RateLimited):
                print(f"  [RATE-LIMITED] {cmt} LS{ls} #{num}: {err}")
                rate_limited = True
                for f in futures:
                    if not f.done(): f.cancel()
                break
            elif text:
                extracted.append((cmt, ls, num))
                extracted_since_checkpoint += 1
            else:
                failed.append((cmt, ls, num))
            now = time.monotonic()
            if (extracted_since_checkpoint >= CHECKPOINT_EVERY_N or
                (extracted_since_checkpoint > 0 and now - last_checkpoint_at >= CHECKPOINT_EVERY_S)):
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
                checkpoint_commit(
                    f"Auto-checkpoint FC primary data (extracted={len(extracted)} this run) [{ts}]",
                    ["docs/fc/reports.json", "docs/fc/text/"],
                )
                extracted_since_checkpoint = 0
                last_checkpoint_at = now
            if now > deadline:
                print(f"  [BUDGET] wall-clock budget hit after {len(extracted)} extractions")
                budget_hit = True
                for f in futures:
                    if not f.done(): f.cancel()
                break

    if extracted_since_checkpoint > 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        checkpoint_commit(
            f"Auto-checkpoint FC primary data (final, extracted={len(extracted)} this run) [{ts}]",
            ["docs/fc/reports.json", "docs/fc/text/"],
        )

    return {
        "extracted": extracted,
        "failed":    failed,
        "rate_limited": rate_limited,
        "budget_hit":   budget_hit,
        "candidates_total": len(candidates),
    }


# ── Phase 3: build manifest ────────────────────────────────────────────────


def _load_bundled_ids() -> set[str]:
    """Read record_to_shard keys from texts-meta.json. Composite_ids
    are `<committee>|<file_id>` for FC.
    """
    texts_meta_path = DOCS / "texts-meta.json"
    if not texts_meta_path.exists():
        return set()
    try:
        with open(texts_meta_path, "r", encoding="utf-8") as f:
            tm = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    return load_bundled_ids(DOCS)


def build_manifest() -> dict:
    """Committee-keyed presence map. Reads from texts-meta.json's
    record_to_shard first (post-bundling), falls back to TEXT_DIR
    glob for transient text/<cmt>/<fid>.txt files between extract
    and derive.
    """
    out: dict[str, dict] = {}
    # Primary: bundled records from texts-meta.json. Composite IDs
    # are `<committee>|<file_id>`; unpack into the nested manifest.
    for composite_id in _load_bundled_ids():
        cmt, _, fid = composite_id.partition("|")
        if not cmt or not fid:
            continue
        out.setdefault(cmt, {})[fid] = {"bundled": True}
    # Additive: on-disk per-record .txt files (transient pre-bundle).
    if TEXT_DIR.exists():
        for cmt in COMMITTEES:
            cmt_dir = TEXT_DIR / cmt
            if not cmt_dir.exists():
                continue
            for text_file in sorted(cmt_dir.glob("*.txt")):
                fid = text_file.stem
                if fid in out.get(cmt, {}):
                    continue
                out.setdefault(cmt, {})[fid] = {
                    "size": text_file.stat().st_size,
                    "url":  f"text/{cmt}/{text_file.name}",
                }
    return {"texts": out}


def compute_audit(reports: dict[str, list[dict]]) -> dict:
    """Per-report status breakdown across the corpus. Same shape as
    LC's audit.json.
    """
    counts = {
        "reports":                    sum(len(v) for v in reports.values()),
        "with_text":                  0,
        "pypdf_empty_awaiting_ocr":   0,
        "pypdf_error_retryable":      0,
        "ocr_failed_permanent":       0,
        "never_attempted":            0,
        "no_pdf_url":                 0,
    }
    bundled_ids = _load_bundled_ids()
    markers = load_markers(DOCS)
    for cmt, items in reports.items():
        cmt_text_dir = TEXT_DIR / cmt
        for r in items:
            ls = r.get("lok_sabha")
            num = r.get("report_number")
            if ls is None or num is None:
                counts["no_pdf_url"] += 1; continue
            if not r.get("pdf_url"):
                counts["no_pdf_url"] += 1; continue
            fid = file_id(int(ls), int(num))
            cid = f"{cmt}|{fid}"
            if cid in bundled_ids:
                counts["with_text"] += 1
            elif cmt_text_dir.exists() and (cmt_text_dir / f"{fid}.txt").exists():
                counts["with_text"] += 1
            elif cid in markers:
                mt = markers[cid]
                if mt == "ocr-failed":
                    counts["ocr_failed_permanent"] += 1
                elif mt == "pypdf-empty":
                    counts["pypdf_empty_awaiting_ocr"] += 1
                elif mt == "pypdf-error":
                    counts["pypdf_error_retryable"] += 1
                else:
                    counts["never_attempted"] += 1
            elif (cmt_text_dir / f"{fid}.ocr-failed").exists():
                counts["ocr_failed_permanent"] += 1
            elif (cmt_text_dir / f"{fid}.pypdf-empty").exists():
                counts["pypdf_empty_awaiting_ocr"] += 1
            elif (cmt_text_dir / f"{fid}.pypdf-error").exists():
                counts["pypdf_error_retryable"] += 1
            else:
                counts["never_attempted"] += 1
    return {
        "audited_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totals":     counts,
    }


# ── Phase 4: search bundle + index (sharded) ──────────────────────────────
#
# Same sharding architecture as LC/DRSC: bundle = title + first 5K chars
# per report; index = full-body inverted token index. Both sharded by
# sorted reportKey range to stay under the 25-MiB-per-asset host cap.

DOCS_PER_SHARD = 2500


def _delete_legacy(path: Path) -> None:
    if path.exists():
        path.unlink()


def _report_key(cmt: str, r: dict) -> str:
    """App-side primary key shape: `fc|<cmt>|<ls>|<num>`. Sortable as
    a string for shard-range partitioning.
    """
    return f"fc|{cmt}|{r['lok_sabha']}|{r['report_number']}"


def build_search_bundle(reports: dict[str, list[dict]],
                        docs_per_shard: int = DOCS_PER_SHARD) -> dict | None:
    """Title + first N chars per report, sharded."""
    if not TEXT_DIR.exists():
        return None
    entries = []
    truncated = 0
    HEAD = 5000
    for cmt, items in reports.items():
        cmt_text_dir = TEXT_DIR / cmt
        for r in items:
            ls = r.get("lok_sabha"); num = r.get("report_number")
            if ls is None or num is None:
                continue
            fid = file_id(int(ls), int(num))
            text_path = cmt_text_dir / f"{fid}.txt"
            if not text_path.exists():
                continue
            try:
                text = text_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            head = text[:HEAD]
            if len(text) > HEAD: truncated += 1
            entries.append({
                "k":     _report_key(cmt, r),
                "t":     r.get("title", ""),
                "head":  head,
            })
    entries.sort(key=lambda e: e["k"])
    if not entries:
        return None
    # Delete legacy unsharded path if it exists.
    _delete_legacy(DOCS / "search-bundle.json")
    # Write shards.
    shard_count = (len(entries) + docs_per_shard - 1) // docs_per_shard
    shard_sizes: dict[str, int] = {}
    max_shard = 0
    total_bytes = 0
    for i in range(shard_count):
        chunk = entries[i*docs_per_shard:(i+1)*docs_per_shard]
        path = DOCS / f"search-bundle-{i:02d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "head_chars": HEAD,
                       "shard_index": i, "entries": chunk}, f,
                      ensure_ascii=False)
        sz = path.stat().st_size
        shard_sizes[f"search-bundle-{i:02d}.json"] = sz
        max_shard = max(max_shard, sz)
        total_bytes += sz
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
    """Full-body inverted token index, sharded. Index format:
       { vocab: [tokens...], report_keys: [keys...], postings: [[doc_idx,...], ...] }
    Postings encode the doc-indices (into report_keys) where each
    vocab token appears at least once.
    """
    if not TEXT_DIR.exists():
        return None
    # Gather all (text, report_key) pairs in deterministic order.
    docs: list[tuple[str, str]] = []
    for cmt in sorted(COMMITTEES):
        cmt_text_dir = TEXT_DIR / cmt
        if not cmt_text_dir.exists():
            continue
        # Order: LS desc, num desc within committee (matches reports.json).
        files = list(cmt_text_dir.glob("*.txt"))
        def _sort_key(p: Path):
            stem = p.stem  # "LS18_7"
            try:
                ls, num = stem.removeprefix("LS").split("_", 1)
                return (-int(ls), -int(num))
            except Exception:
                return (0, 0)
        files.sort(key=_sort_key)
        for path in files:
            stem = path.stem
            try:
                ls, num = stem.removeprefix("LS").split("_", 1)
                rk = f"fc|{cmt}|{ls}|{num}"
            except Exception:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            docs.append((rk, text))
    if not docs:
        return None

    # Build vocab + postings.
    # token → set of doc indices.
    token_docs: dict[str, set[int]] = {}
    for i, (_, text) in enumerate(docs):
        for tok in set(_tokenize(text)):
            token_docs.setdefault(tok, set()).add(i)

    # Frequency cutoffs (same heuristic as LC): drop ultra-common (>50%
    # of docs) + ultra-rare (single-doc) terms — minor index size win
    # with negligible recall hit.
    n_docs = len(docs)
    high_cut = max(2, int(n_docs * 0.5))
    low_cut  = 2
    vocab = sorted(
        tok for tok, ds in token_docs.items()
        if low_cut <= len(ds) <= high_cut
    )
    if not vocab:
        # Tiny corpus — fall back to permissive (>=1 doc).
        vocab = sorted(token_docs.keys())
        low_cut = 1; high_cut = n_docs

    # Now shard by sorted-key range. Each shard carries: full vocab,
    # slice of report_keys, postings filtered to that shard's doc range
    # (postings indices are LOCAL to the shard — app applies offset on
    # merge).
    _delete_legacy(DOCS / "search-index.json")
    shard_count = (n_docs + docs_per_shard - 1) // docs_per_shard
    shard_sizes: dict[str, int] = {}
    max_shard = 0
    total_bytes = 0
    total_postings = 0

    for i in range(shard_count):
        start = i * docs_per_shard
        end = min(start + docs_per_shard, n_docs)
        shard_keys = [docs[j][0] for j in range(start, end)]
        # Postings filtered to docs in [start, end), indices made local.
        local_postings: list[list[int]] = []
        for tok in vocab:
            ds = token_docs[tok]
            in_shard = sorted(d - start for d in ds if start <= d < end)
            local_postings.append(in_shard)
            total_postings += len(in_shard)
        path = DOCS / f"search-index-{i:02d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "version":      "1.0",
                "shard_index":  i,
                "vocab":        vocab,
                "report_keys":  shard_keys,
                "postings":     local_postings,
            }, f, ensure_ascii=False)
        sz = path.stat().st_size
        shard_sizes[f"search-index-{i:02d}.json"] = sz
        max_shard = max(max_shard, sz)
        total_bytes += sz

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


# ── Phase 5: meta ──────────────────────────────────────────────────────────


def write_meta(*, total_reports: int, total_with_text: int,
               bundle_stats: dict | None, index_stats: dict | None) -> dict:
    meta = {
        "version":         "1.0",
        "corpus":          "fc",
        "generated_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_reports":   total_reports,
        "total_with_text": total_with_text,
        "lok_sabhas":      LOK_SABHAS,
        "search_bundle":   bundle_stats,
        "search_index":    index_stats,
    }
    # Non-idempotent on purpose — see build_lc.py write_meta() for the
    # full rationale. Always bump generated_at so the app's staleness
    # indicator reflects "last successful derive".
    write_json_idempotent(META_JSON, meta, ignore_keys=())
    return meta


# ── Main ────────────────────────────────────────────────────────────────────


def phase_extract() -> None:
    overall_deadline = time.monotonic() + MAX_RUN_SECONDS
    print(f"\n[Extract 1/2] Walking committees × LS {LOK_SABHAS}...")
    existing = load_existing_reports()
    print(f"  existing on disk: {sum(len(v) for v in existing.values())} reports")
    merged, walk_stats = walk_and_merge_reports(existing)
    save_reports(merged)
    total = sum(len(v) for v in merged.values())
    print(f"  reports.json: {total} total "
          f"({walk_stats['new']} new, {walk_stats['updated']} updated, "
          f"{walk_stats['kept_legacy']} kept legacy)")

    print(f"\n[Extract 2/2] Extracting missing texts (priority: highest LS, highest num)...")
    extract_stats = extract_missing_texts(merged, deadline=overall_deadline)
    print(f"  extracted={len(extract_stats['extracted'])} "
          f"failed={len(extract_stats['failed'])} "
          f"rate_limited={extract_stats.get('rate_limited', False)} "
          f"budget_hit={extract_stats.get('budget_hit', False)} "
          f"remaining_after={max(0, extract_stats.get('candidates_total', 0) - len(extract_stats.get('extracted', [])))}")


def phase_derive() -> None:
    """Pure function of disk state — owned by fc-derive.yml. Single
    owner, cancel-in-progress, never races. See CONV.md "Split-phase
    scraping pattern".
    """
    reports = load_existing_reports()

    print("\n[Derive 1/4] Building manifest.json...")
    manifest = build_manifest()
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    n_with_text = sum(len(v) for v in manifest["texts"].values())
    print(f"  manifest: {n_with_text} reports with extracted text")

    # Bundle per-record text files. FC's manifest is nested by committee:
    # `texts[<committee>][<file_id>] = {size, url}` (same shape as DRSC).
    print("\n[Derive] Building text shards...")
    # Only pass entries with a fresh on-disk text file. Bundled-only
    # entries (post-49538544) have no `url` field — write_text_shards's
    # preservation logic carries them forward without needing them here.
    items = []
    for committee, by_id in sorted(manifest["texts"].items()):
        for key, entry in sorted(by_id.items()):
            if "url" not in entry:
                continue
            items.append((f"{committee}|{key}", DOCS / entry["url"]))
    text_meta = write_text_shards(DOCS, items)
    bundled_ids = load_bundled_ids(DOCS)
    # Nested layout: text/<committee>/<fid>.<suffix>. Composite id includes the committee.
    marker_stats = consolidate_markers(
        DOCS, TEXT_DIR,
        composite_id_from_path=lambda p: f"{p.parent.name}|{p.stem}",
        drop_record_ids=bundled_ids,
    )
    print(f"  markers: {marker_stats['totals']} consolidated "
          f"(removed {marker_stats.get('removed_sidecar_count', 0)} sidecars)")
    t = text_meta["totals"]
    print(f"  text-shards: {t['shards']} shard(s), {t['records_with_text']} records, "
          f"{t['total_text_bytes'] / 1024 / 1024:.1f} MB, "
          f"{t['r2_fallback']} via R2 sentinel, {t['skipped_oversize_no_r2']} skipped")

    print("\n[Derive 2/4] Building search-bundle (title + first 5K chars)...")
    bundle_stats = build_search_bundle(reports)
    if bundle_stats:
        mb = bundle_stats["size_bytes"] / (1024 * 1024)
        print(f"  bundle: {bundle_stats['total']} entries × {bundle_stats['shard_count']} shards · {mb:.1f} MB")
    else:
        print("  no extracted texts yet — skipping bundle")

    print("\n[Derive 3/4] Building search-index (inverted token, full body)...")
    index_stats = build_search_index()
    if index_stats:
        mb = index_stats["size_bytes"] / (1024 * 1024)
        print(f"  index: {index_stats['report_count']} docs × {index_stats['shard_count']} shards, "
              f"vocab={index_stats['vocab_size']}, postings={index_stats['total_postings']} · {mb:.1f} MB")
    else:
        print("  no extracted texts yet — skipping index")

    print("\n[Derive 4/4] Writing meta.json + audit.json...")
    total = sum(len(v) for v in reports.values())
    meta = write_meta(total_reports=total, total_with_text=n_with_text,
                      bundle_stats=bundle_stats, index_stats=index_stats)
    print(json.dumps(meta, indent=2))

    audit = compute_audit(reports)
    if not write_json_idempotent(DOCS / "audit.json", audit):
        print("  [skip] audit.json unchanged (besides timestamp)")
    t = audit["totals"]
    print(f"\n  audit: with_text={t['with_text']} "
          f"pypdf_empty={t['pypdf_empty_awaiting_ocr']} "
          f"pypdf_error={t['pypdf_error_retryable']} "
          f"never_attempted={t['never_attempted']} (total={t['reports']})")


def main():
    """Dispatch to extract / derive / both based on BUILD_PHASE.

    BUILD_PHASE=extract — for fc.yml writers. Produces reports.json +
                          text/<committee>/<file_id>.txt only; never
                          touches derived files.
    BUILD_PHASE=derive  — for fc-derive.yml. Reads disk state, regenerates
                          manifest.json + search-bundle-*.json +
                          search-index-*.json + meta.json + audit.json.
    BUILD_PHASE=all     — legacy / local-dev convenience.
    """
    phase = os.environ.get("BUILD_PHASE", "all").lower()
    if phase not in ("extract", "derive", "all"):
        print(f"BUILD_PHASE={phase!r} not in extract|derive|all — aborting.", file=sys.stderr)
        sys.exit(2)

    print("=== ParliamentWatch FC (Financial Committees) static builder ===")
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
