#!/usr/bin/env python3
"""CAG static data builder — orchestrator for parliamentwatch-data.

Per scheduled (or workflow_dispatch) run:
  1. Walk cag.gov.in's listing pages, enumerate detail-page IDs
  2. Fetch detail-page metadata for any new IDs we don't have yet
  3. Extract text for as many missing PDFs as fit in the per-run budget
     (MAX_EXTRACTIONS_PER_RUN, MAX_RUN_SECONDS) — newest reports first
  4. Build manifest.json + reports.json + meta.json
  5. Build search-bundle.json + search-index.json (mirror DRSC's v0.6 ladder)

Output goes under docs/cag/, served at sansadsaar-data.naklitechie.com/cag/.

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

from parliamentwatch_text_shards import (
    write_text_shards, consolidate_markers, load_markers, write_json_idempotent,
    load_bundled_ids,
)
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

# How many parallel detail-page fetches. cag.gov.in is plain Apache — can
# handle multiple concurrent requests without rate-limit issues. 4 workers
# × 250-500ms jitter ≈ 8-16 req/sec, polite enough but ~4× faster than
# sequential. Backfill run 25609052452 hit the 60-min GH Actions timeout
# fetching ~2,660 detail pages sequentially (~45 min just for metadata)
# before reaching the extract phase. Parallelising fixes that.
DETAIL_WORKERS          = int(os.environ.get("DETAIL_WORKERS", "4"))

# Cooldown after a 429 / 403 — same idea as DRSC's: skip the extraction phase
# if the previous run was rate-limited within this many seconds.
RATE_LIMIT_COOLDOWN_SECONDS = int(os.environ.get("RATE_LIMIT_COOLDOWN_SECONDS", str(6 * 3600)))


# ── In-flight checkpointing ────────────────────────────────────────────────
#
# Layer 1 crash-safety: periodically commit + push the in-flight extracted
# text files DURING the run (not just at the end) so a runner-killed mid-
# stream backfill doesn't lose all its work. Trigger: every
# CHECKPOINT_EVERY_N successful extractions OR CHECKPOINT_EVERY_S wallclock
# seconds since the last checkpoint, whichever comes first.
#
# Best-effort: a checkpoint failure (network blip, race against another
# writer's commit, transient git error) is logged and the extraction loop
# continues. The next checkpoint trigger retries. The workflow's final
# commit-and-push step is still the backstop for anything not yet
# checkpointed.
#
# Why not commit per-PDF? Each commit + pull-rebase + push round-trip is
# ~10s of overhead. At 25 extractions per checkpoint we amortise that down
# to <0.5s per PDF, while bounding the worst-case loss to ~25 PDFs of work
# if the runner gets killed.

import subprocess

CHECKPOINT_EVERY_N = int(os.environ.get("CHECKPOINT_EVERY_N", "100"))
CHECKPOINT_EVERY_S = int(os.environ.get("CHECKPOINT_EVERY_S", "300"))


def _git(*args) -> tuple[int, str, str]:
    """Run a git command. Returns (returncode, stdout, stderr)."""
    p = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False,
    )
    return p.returncode, p.stdout, p.stderr


def checkpoint_commit(message: str, paths: list[str]) -> bool:
    """Stage `paths`, commit if there's something to commit, pull-rebase, push.

    Returns True on a successful push (or a no-op when nothing was staged).
    Best-effort: failures log to stdout but DO NOT raise — the caller
    continues extracting and the next checkpoint retries. Never abort
    a backfill mid-stream because of a checkpoint hiccup.
    """
    rc, _, err = _git("add", "--", *paths)
    if rc != 0:
        print(f"  [checkpoint] git add failed: {err.strip() or 'unknown'}")
        return False

    rc, _, _ = _git("diff", "--cached", "--quiet")
    if rc == 0:
        # No staged changes; nothing to commit.
        return True

    rc, _, err = _git("commit", "-m", message)
    if rc != 0:
        print(f"  [checkpoint] git commit failed: {err.strip()}")
        return False

    rc, _, err = _git("pull", "--rebase", "origin", "main")
    if rc != 0:
        print(f"  [checkpoint] git pull --rebase failed: {err.strip()} — aborting rebase")
        _git("rebase", "--abort")
        return False

    rc, _, err = _git("push")
    if rc != 0:
        print(f"  [checkpoint] git push failed: {err.strip()}")
        return False

    print(f"  [checkpoint] pushed: {message}")
    return True


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


def cleanup_html_entities(reports: dict[int, dict]) -> int:
    """One-shot idempotent cleanup. Walks reports and applies HTML-entity
    decoding to fields that may have been captured pre-fix (before
    cag/scraper.py's html.unescape support). Idempotent: re-running on
    already-clean entries is a no-op.

    Returns the number of entries modified.
    """
    from cag.scraper import _html_unescape_loop
    n_changed = 0
    for rid, meta in reports.items():
        before = dict(meta)
        for field in ("title", "pdf_url", "department", "sector", "state",
                      "report_type", "report_no", "date_tabled", "date_sent"):
            v = meta.get(field)
            if isinstance(v, str) and "&amp;" in v:
                meta[field] = _html_unescape_loop(v)
        if meta != before:
            n_changed += 1
    return n_changed


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

def fetch_new_metadata(known_ids: set[int], all_ids: list[int],
                       *, deadline: float | None = None) -> dict[int, dict]:
    """Fetch detail-page metadata for any IDs not already in known_ids.
    Parallelised with DETAIL_WORKERS threads to stay under the GH Actions
    60-min runner timeout. Stops on RateLimited or when deadline (monotonic
    seconds) is exceeded.

    Returns {id: metadata_dict}.
    """
    new_ids = [i for i in all_ids if i not in known_ids]
    if not new_ids:
        print("  no new ids — skipping detail fetch")
        return {}
    print(f"  fetching detail metadata for {len(new_ids)} new ids "
          f"(workers={DETAIL_WORKERS})...")

    out: dict[int, dict] = {}
    rate_limited = False
    deadline_hit = False
    t0 = time.time()
    progress_step = max(50, len(new_ids) // 20)

    def _worker(rid: int):
        try:
            return rid, fetch_detail(rid), None
        except RateLimited as rl:
            return rid, None, rl
        except Exception as e:
            return rid, None, e

    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as ex:
        futures = {ex.submit(_worker, rid): rid for rid in new_ids}
        for fut in as_completed(futures):
            if deadline and time.monotonic() > deadline:
                print(f"  [DEADLINE] hit after {len(out)} fetches — stopping detail-fetch phase")
                deadline_hit = True
                for f in futures:
                    if not f.done(): f.cancel()
                break
            rid, rep, err = fut.result()
            if isinstance(err, RateLimited):
                print(f"  [RATE-LIMITED] id={rid}: {err} — stopping detail-fetch phase")
                rate_limited = True
                for f in futures:
                    if not f.done(): f.cancel()
                break
            if err is not None:
                # Non-fatal individual failure (404, parse error, etc.)
                continue
            out[rid] = rep.to_dict()
            n_done = len(out)
            if n_done % progress_step == 0:
                rate = n_done / max(0.1, time.time() - t0)
                eta_s = (len(new_ids) - n_done) / max(0.1, rate)
                print(f"  ...{n_done}/{len(new_ids)} ({time.time()-t0:.1f}s · {rate:.1f}/s · ETA {eta_s:.0f}s)")

    print(f"  fetched {len(out)} metadata entries in {time.time()-t0:.1f}s · "
          f"rate_limited={rate_limited} · deadline_hit={deadline_hit}")
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


def extract_missing_texts(reports: dict[int, dict], *, deadline: float) -> dict:
    """Extract up to MAX_EXTRACTIONS_PER_RUN missing PDFs. Stops on
    RateLimited or when `deadline` (monotonic seconds) is exceeded.

    `deadline` is the *unified* deadline shared with the metadata-fetch
    phase — so the whole script respects MAX_RUN_SECONDS as a wall-clock
    budget regardless of how much time was spent in earlier phases.
    """
    if _check_cooldown_and_skip():
        return {"extracted": [], "failed": [], "rate_limited": True,
                "budget_hit": False, "skipped_due_to_cooldown": True,
                "candidates_total": 0}

    # Candidate selection — read per-attempt status markers (see CONV.md
    # "Per-attempt status markers"). Skip anything already classified:
    #   .txt           → already extracted (any source)
    #   .pypdf-empty   → known scanned/encrypted, queued for OCR
    #   .ocr-failed    → OCR permanent tombstone
    # Retry .pypdf-error (transient pypdf failure, could re-succeed) and
    # never-attempted reports.
    #
    # Also skip records present in `texts-meta.json`'s record_to_shard
    # map — those are already bundled into the sharded text store and
    # the per-record .txt files may no longer exist on disk (the
    # 2026-05-14 cleanup removed all docs/cag/text/*.txt files;
    # subsequent scrape runs would otherwise re-extract those records
    # repeatedly). Graceful fallback if texts-meta.json is missing.
    bundled_ids: set = set()
    texts_meta_path = DOCS / "texts-meta.json"
    if texts_meta_path.exists():
        try:
            with open(texts_meta_path, "r", encoding="utf-8") as f:
                texts_meta = json.load(f)
            bundled_ids = load_bundled_ids(DOCS)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! couldn't read texts-meta.json — proceeding without shard skip ({e})")

    # Also skip records with a bundled marker — once we've recorded
    # pypdf-empty / ocr-failed in markers.json, we don't retry. We DO
    # retry pypdf-error (transient parsing issues).
    bundled_markers = load_markers(DOCS)
    candidates = []
    skipped_marked = 0
    skipped_bundled = 0
    for rid, meta in reports.items():
        if not meta.get("pdf_url"):
            continue
        cid = str(rid)
        if cid in bundled_ids:
            skipped_bundled += 1
            continue
        bm = bundled_markers.get(cid)
        if bm in ("pypdf-empty", "ocr-failed"):
            skipped_marked += 1
            continue
        text_path        = TEXT_DIR / f"{rid}.txt"
        pypdf_empty_path = TEXT_DIR / f"{rid}.pypdf-empty"
        ocr_failed_path  = TEXT_DIR / f"{rid}.ocr-failed"
        if text_path.exists() or pypdf_empty_path.exists() or ocr_failed_path.exists():
            skipped_marked += 1
            continue
        candidates.append((rid, meta["pdf_url"]))
    # Newest first (highest id wins — CAG IDs are monotonic).
    candidates.sort(key=lambda c: -c[0])
    print(f"  candidates: {len(candidates)} reports missing extracted text "
          f"({skipped_marked} skipped — already marked; "
          f"{skipped_bundled} skipped — already in shards)")

    if not candidates:
        return {"extracted": [], "failed": [], "rate_limited": False,
                "budget_hit": False, "candidates_total": 0}

    # Calculate remaining budget from the unified deadline rather than
    # creating our own. This is the fix that closes the runner-timeout
    # bug from cag-backfill runs 25611721332 / 25612861100 / 25614057565 /
    # 25615072064 — they walked + metadata-fetched for ~15-25 min, then
    # extract phase set its OWN 50-min deadline starting fresh, blowing
    # past the 60-min GH Actions runner cap.
    remaining = deadline - time.monotonic()
    if remaining <= 60:
        print(f"  only {remaining:.0f}s left in budget — skipping extract phase")
        return {"extracted": [], "failed": [], "rate_limited": False,
                "budget_hit": True, "candidates_total": len(candidates)}

    target = candidates[:MAX_EXTRACTIONS_PER_RUN]
    print(f"  budget: extracting up to {len(target)} this run "
          f"(MAX_EXTRACTIONS_PER_RUN={MAX_EXTRACTIONS_PER_RUN}, "
          f"remaining={remaining:.0f}s, EXTRACT_WORKERS={EXTRACT_WORKERS})")

    extracted, failed = [], []
    rate_limited = False
    budget_hit = False

    # In-flight checkpointing — see CHECKPOINT_EVERY_N / CHECKPOINT_EVERY_S.
    last_checkpoint_at = time.monotonic()
    extracted_since_checkpoint = 0

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
                extracted_since_checkpoint += 1
            else:
                failed.append(rid)
            now = time.monotonic()
            # Checkpoint trigger: enough extractions OR enough wallclock
            # since last checkpoint. Done outside the rate-limit / budget
            # paths because we want the residue committed even on early exit.
            if (extracted_since_checkpoint >= CHECKPOINT_EVERY_N or
                (extracted_since_checkpoint > 0 and now - last_checkpoint_at >= CHECKPOINT_EVERY_S)):
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
                checkpoint_commit(
                    f"Auto-checkpoint CAG primary data (extracted={len(extracted)} this run) [{ts}]",
                    ["docs/cag/reports.json", "docs/cag/text/"],
                )
                extracted_since_checkpoint = 0
                last_checkpoint_at = now
            if now > deadline:
                print(f"  [BUDGET] wall-clock budget hit after {len(extracted)} extractions")
                budget_hit = True
                for f in futures:
                    if not f.done(): f.cancel()
                break

    # Final checkpoint of any post-loop residue. The workflow's "Commit and
    # push primary changes" step would also catch this, but a final in-script
    # push means even rate-limited / budget-hit exits leave nothing behind.
    if extracted_since_checkpoint > 0:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        checkpoint_commit(
            f"Auto-checkpoint CAG primary data (final, extracted={len(extracted)} this run) [{ts}]",
            ["docs/cag/reports.json", "docs/cag/text/"],
        )

    return {
        "extracted": extracted,
        "failed":    failed,
        "rate_limited": rate_limited,
        "budget_hit":   budget_hit,
        "candidates_total": len(candidates),
    }


# ── Phase 4: build manifest ────────────────────────────────────────────────

def _load_bundled_ids() -> set[str]:
    """Read record_to_shard keys from texts-meta.json. Empty set if
    missing or unreadable. The keys are the canonical composite_ids
    used by write_text_shards. For CAG: just the report id as a string.
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
    """Per-record presence map. Reads from texts-meta.json's
    record_to_shard first (post-bundling) — records are listed with
    `{"bundled": True}` (no per-file URL, since the content lives in
    a shard). Falls back to scanning TEXT_DIR for legacy per-file
    text/<id>.txt (transient between extract + derive).

    The app uses this purely as a "does record X have text?" lookup;
    actual content retrieval goes through the sharded text store.
    """
    manifest: dict[str, dict] = {}
    # Primary: bundled records.
    for rid in _load_bundled_ids():
        manifest[rid] = {"bundled": True}
    # Fallback / additive: any text/<id>.txt files on disk (transient).
    if TEXT_DIR.exists():
        for text_file in sorted(TEXT_DIR.glob("*.txt")):
            rid = text_file.stem
            # Don't overwrite a bundled entry; on-disk + bundled means
            # the disk copy will be cleaned by next derive's bundling.
            if rid not in manifest:
                manifest[rid] = {
                    "size": text_file.stat().st_size,
                    "url":  f"text/{text_file.name}",
                }
    return {"texts": manifest}


def compute_audit(reports: dict[int, dict]) -> dict:
    """Walk reports + text/ to produce a per-report status breakdown.

    Pure function of disk state. Owned by the derive phase. Used by ops
    to answer "what's the shape of the gap?" without grovelling through
    workflow logs.

    Output goes to docs/cag/audit.json. See CONV.md "Per-attempt status
    markers" and "Periodic audit JSON".
    """
    counts = {
        "reports":                    len(reports),
        "with_text":                  0,
        "pypdf_empty_awaiting_ocr":   0,
        "pypdf_error_retryable":      0,
        "ocr_failed_permanent":       0,
        "never_attempted":            0,
        "no_pdf_url":                 0,
    }
    bundled_ids = _load_bundled_ids()
    markers = load_markers(DOCS)
    for rid, meta in reports.items():
        if not meta.get("pdf_url"):
            counts["no_pdf_url"] += 1
            continue
        # Order: bundled text → bundled marker → fall-through to legacy
        # on-disk checks (transient between extract and derive).
        cid = str(rid)
        if cid in bundled_ids:
            counts["with_text"] += 1
        elif TEXT_DIR.exists() and (TEXT_DIR / f"{rid}.txt").exists():
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
        elif (TEXT_DIR / f"{rid}.ocr-failed").exists():
            counts["ocr_failed_permanent"] += 1
        elif (TEXT_DIR / f"{rid}.pypdf-empty").exists():
            counts["pypdf_empty_awaiting_ocr"] += 1
        elif (TEXT_DIR / f"{rid}.pypdf-error").exists():
            counts["pypdf_error_retryable"] += 1
        else:
            counts["never_attempted"] += 1
    return {
        "audited_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totals":     counts,
    }


# ── Phase 5+6: search bundle + index, BOTH SHARDED (v1.0c) ────────────────
#
# Same architecture as DRSC's build_static.py: split outputs into N shards
# by sorted reportKey range so no single file exceeds the 25 MiB per-asset
# cap on the host. Bundle shards each carry a slice of entries; index shards
# carry full vocab + slice of report_keys + slice of postings (postings
# indices are local to the shard, app applies offsets when merging).

DOCS_PER_SHARD = 2500


def _delete_legacy(path: Path) -> None:
    if path.exists():
        path.unlink()


def build_search_bundle(reports: dict[int, dict],
                        docs_per_shard: int = DOCS_PER_SHARD) -> dict | None:
    """Title + first N chars per report, sharded."""
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
        entries.append({"key": f"cag|{rid}", "title": meta.get("title", ""), "head": head})

    if not entries:
        return None

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

    _delete_legacy(BUNDLE_JSON)

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


def build_search_index(docs_per_shard: int = DOCS_PER_SHARD) -> dict | None:
    """Inverted token index over full body, sharded. Each shard carries the
    full vocab; postings are doc-local within the shard."""
    if not TEXT_DIR.exists():
        return None

    docs: list[tuple[str, set[str]]] = []
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
        docs.append((f"cag|{rid}", tokens))
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

    n_shards = max(1, (n_docs + docs_per_shard - 1) // docs_per_shard)
    shard_size = (n_docs + n_shards - 1) // n_shards
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _delta(lst: list[int]) -> list[int]:
        if not lst: return lst
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

    _delete_legacy(INDEX_JSON)

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


# ── Phase 7: meta ──────────────────────────────────────────────────────────

def write_meta(*, total_reports: int, total_with_text: int,
               bundle_stats: dict | None,
               index_stats: dict | None) -> dict:
    """Write meta.json — purely a function of disk state (no extract-time
    counters). The split-phase architecture separates extraction (writers)
    from derivation (this function's caller); per-run extract counts live
    in workflow logs, not in meta.
    """
    meta = {
        "version":      "1.0",
        "corpus":       "cag",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_reports":   total_reports,
        "total_with_text": total_with_text,
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
    """Phase 1 — produce primary files (reports.json + text/<id>.txt).
    No derived files. Safe to run concurrently with other extract-phase
    workflows; auto-merge handles different text/<id>.txt additions and
    reports.json is deterministic given upstream content.
    """
    print("\n[Extract 1/3] Walking cag.gov.in listing pages...")
    all_ids = enumerate_ids(max_pages=MAX_LISTING_PAGES)

    print("\n[Extract 2/3] Loading existing metadata + fetching new detail pages...")
    reports = load_existing_reports()
    cleaned = cleanup_html_entities(reports)
    if cleaned:
        print(f"  cleaned {cleaned} entries with HTML-entity-encoded fields (one-shot fix)")
    overall_deadline = time.monotonic() + MAX_RUN_SECONDS
    new_meta = fetch_new_metadata(set(reports.keys()), all_ids, deadline=overall_deadline)
    reports.update(new_meta)
    save_reports(reports)
    print(f"  reports.json: {len(reports)} total ({len(new_meta)} new this run)")

    print("\n[Extract 3/3] Extracting missing texts (priority: newest first)...")
    extract_stats = extract_missing_texts(reports, deadline=overall_deadline)
    print(f"  extracted={len(extract_stats['extracted'])} "
          f"failed={len(extract_stats['failed'])} "
          f"rate_limited={extract_stats.get('rate_limited', False)} "
          f"budget_hit={extract_stats.get('budget_hit', False)} "
          f"remaining_after={max(0, extract_stats.get('candidates_total', 0) - len(extract_stats.get('extracted', [])))}")


def phase_derive() -> None:
    """Phase 2 — regenerate derived files (manifest.json, search-bundle-*.json,
    search-index-*.json, meta.json) from the on-disk primary state.
    Pure function of reports.json + text/. Owned by cag-derive.yml; one
    run at a time via `cancel-in-progress: true` on the cag-derive group.
    """
    reports = load_existing_reports()

    print("\n[Derive 1/4] Building manifest.json...")
    manifest = build_manifest()
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    n_with_text = len(manifest["texts"])
    print(f"  manifest: {n_with_text} reports with extracted text")

    # Bundle per-record text files into 4-5 MB shards so the static-asset
    # file count stays well under Cloudflare's 20k-files-per-deployment cap.
    # CAG's manifest is flat: `texts[<report_id>] = {size, url}`. Composite
    # key passed to the shard helper is just the report_id, which is also
    # what the app uses to look up texts.
    print("\n[Derive] Building text shards...")
    # Only pass entries with a fresh on-disk text file. Bundled-only
    # entries (post-49538544) have no `url` field — write_text_shards's
    # preservation logic carries them forward without needing them here.
    items = sorted(
        (key, DOCS / entry["url"])
        for key, entry in manifest["texts"].items()
        if "url" in entry
    )
    text_meta = write_text_shards(DOCS, items)
    t = text_meta["totals"]
    print(f"  text-shards: {t['shards']} shard(s), {t['records_with_text']} records, "
          f"{t['total_text_bytes'] / 1024 / 1024:.1f} MB, "
          f"{t['r2_fallback']} via R2 sentinel, {t['skipped_oversize_no_r2']} skipped")

    # Consolidate per-record marker sidecars (.pypdf-empty / .pypdf-error
    # / .ocr-failed) into markers.json. Without this, markers accumulate
    # alongside text/ and eat into CF Pages's 20K-files-per-deploy cap.
    # CAG composite_id == path.stem (flat layout).
    bundled_ids = load_bundled_ids(DOCS)
    marker_stats = consolidate_markers(DOCS, TEXT_DIR, drop_record_ids=bundled_ids)
    print(f"  markers: {marker_stats['totals']} consolidated "
          f"(removed {marker_stats.get('removed_sidecar_count', 0)} sidecars)")

    print("\n[Derive 2/4] Building search-bundle (title + first 5K chars per report)...")
    bundle_stats = build_search_bundle(reports)
    if bundle_stats:
        mb = bundle_stats["size_bytes"] / (1024 * 1024)
        print(f"  search-bundle: {bundle_stats['total']} entries across {bundle_stats['shard_count']} shards · {mb:.1f} MB raw")
    else:
        print("  no extracted texts yet — skipping bundle build")

    print("\n[Derive 3/4] Building search-index (inverted token index, full body)...")
    index_stats = build_search_index()
    if index_stats:
        mb = index_stats["size_bytes"] / (1024 * 1024)
        print(f"  search-index: {index_stats['report_count']} docs across {index_stats['shard_count']} shards, "
              f"vocab={index_stats['vocab_size']}, postings={index_stats['total_postings']} · {mb:.1f} MB raw")
    else:
        print("  no extracted texts yet — skipping index build")

    print("\n[Derive 4/4] Writing meta.json + audit.json...")
    meta = write_meta(
        total_reports=len(reports),
        total_with_text=n_with_text,
        bundle_stats=bundle_stats,
        index_stats=index_stats,
    )
    print(json.dumps(meta, indent=2))

    audit = compute_audit(reports)
    if not write_json_idempotent(DOCS / "audit.json", audit):
        print("  [skip] audit.json unchanged (besides timestamp)")
    t = audit["totals"]
    print(f"\n  audit: with_text={t['with_text']} "
          f"pypdf_empty={t['pypdf_empty_awaiting_ocr']} "
          f"pypdf_error={t['pypdf_error_retryable']} "
          f"ocr_failed={t['ocr_failed_permanent']} "
          f"never_attempted={t['never_attempted']} "
          f"no_pdf_url={t['no_pdf_url']} "
          f"(total={t['reports']})")


def main():
    """Dispatch to extract / derive / both based on BUILD_PHASE.

    BUILD_PHASE=extract — for cag.yml, cag-backfill.yml writers. Produces only
                          reports.json + text/<id>.txt; never touches derived
                          files. Multiple writers can race-safely commit because
                          their changes are on different filenames.
    BUILD_PHASE=derive  — for cag-derive.yml. Reads disk state, regenerates
                          manifest.json + search-bundle-*.json +
                          search-index-*.json + meta.json. Single owner of
                          derived files (concurrency cancel-in-progress: true).
    BUILD_PHASE=all     — legacy / local-dev convenience. Runs both phases in
                          one process; same output as a writer + derive run
                          back-to-back. Don't use in CI — the split-phase
                          workflows exist precisely to avoid this race.
    """
    phase = os.environ.get("BUILD_PHASE", "all").lower()
    if phase not in ("extract", "derive", "all"):
        print(f"BUILD_PHASE={phase!r} is not one of extract|derive|all — aborting.", file=sys.stderr)
        sys.exit(2)

    print("=== ParliamentWatch CAG static builder ===")
    print(f"BUILD_PHASE             : {phase}")
    print(f"DOCS                    : {DOCS}")
    if phase in ("extract", "all"):
        print(f"MAX_LISTING_PAGES       : {MAX_LISTING_PAGES or '(walk to empty)'}")
        print(f"MAX_EXTRACTIONS_PER_RUN : {MAX_EXTRACTIONS_PER_RUN}")
        print(f"MAX_RUN_SECONDS         : {MAX_RUN_SECONDS}")
        print(f"EXTRACT_WORKERS         : {EXTRACT_WORKERS}")

    if phase in ("extract", "all"):
        phase_extract()

    if phase in ("derive", "all"):
        phase_derive()

    print("\nDone.")


if __name__ == "__main__":
    import requests as _rq
    try:
        main()
    except (_rq.exceptions.ConnectionError, _rq.exceptions.Timeout) as _e:
        print(f"::warning::Source unreachable ({type(_e).__name__}: {_e}); skipping this run \u2014 next scheduled run will retry.", flush=True)
        raise SystemExit(0)
