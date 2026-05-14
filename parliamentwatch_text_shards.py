"""Shared text-sharding helper for the parliamentwatch corpora.

Bundles per-record text files (`docs/<corpus>/text/<name>.txt`) into
size-targeted shard files (`docs/<corpus>/texts-NN.json`) plus a
manifest (`docs/<corpus>/texts-meta.json`). Reduces the static-assets
file count by 50-200x per corpus while keeping each shard well under
Cloudflare's 25 MiB per-file cap.

See `plan/cloudflare-strategy-002.md` for the design rationale (bundling
primary, R2 fallback for oversize records).

Each corpus's build script generates the input list
`[(composite_id, Path), ...]` in its own canonical sort order
(newest-first by convention) and passes it here. This module doesn't
know per-corpus naming or sort logic — it just packs greedily.

R2 fallback: any single record whose text exceeds `fallback_threshold`
gets a `{"r2": true}` sentinel in the shard. App fetches that record's
body from the R2 origin (already populated by the .github/workflows/
r2-sync.yml workflow). Threshold defaults to 1 MB — at the target 4.5 MB
shard size, allowing larger inline texts means one big record could
dominate a shard, defeating the cache-locality win.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Iterable, Optional


# Defaults. Per-corpus build scripts can override via kwargs.
DEFAULT_TARGET_BYTES        = 4_500_000   # ~4.5 MB per shard
# A single record's text > this gets a `{"r2":true}` sentinel instead of
# being inlined. Set well above the largest observed record (~3 MB across
# all corpora) so by default everything bundles inline. Lower the threshold
# once an R2 public URL is configured in r2_origin — that's when sentinels
# actually become useful as a fallback path.
DEFAULT_FALLBACK_THRESHOLD  = 10_000_000  # 10 MB

# Sentinel embedded in a shard when the record's text was too big to inline.
# App side checks `typeof value === 'object'` then reads the R2 origin.
R2_SENTINEL = {"r2": True}
# Byte cost of `{"r2":true}` plus the key wrapping in JSON output (commas,
# quotes, etc). Used to budget shard size so a run with many fallbacks still
# fits the target.
SENTINEL_BUDGET_BYTES = 50


def _shard_filename(idx: int) -> str:
    return f"texts-{idx:02d}.json"


def _meta_filename() -> str:
    return "texts-meta.json"


def write_text_shards(
    corpus_docs_dir: Path,
    items: Iterable[tuple[str, Path]],
    *,
    target_bytes: int = DEFAULT_TARGET_BYTES,
    fallback_threshold: int = DEFAULT_FALLBACK_THRESHOLD,
    r2_origin: Optional[str] = None,
) -> dict:
    """Pack texts into shards under `<corpus_docs_dir>/texts-NN.json`.

    Returns the meta dict (also written to `texts-meta.json`).
    Cleans up any pre-existing `texts-*.json` orphans first so a run
    with fewer records doesn't leave a stale higher-index shard
    behind.

    `items` is an iterable of (composite_id, text_file_path). The order
    determines pack order — pass records in canonical sort order so
    shard contents are stable across re-runs. Items whose file is
    missing are silently skipped (the manifest records this in
    `totals.missing_files` for observability).

    `r2_origin` is recorded in the meta for app-side fallback fetches;
    pass None if no R2 fallback is configured yet (in which case
    >threshold records are skipped entirely — caller should keep the
    threshold high enough that no records hit it).
    """
    corpus_docs_dir = Path(corpus_docs_dir)
    corpus_docs_dir.mkdir(parents=True, exist_ok=True)
    items = list(items)  # materialize so we can iterate twice

    # SHARD PRESERVATION: load existing shards' record contents so that
    # records bundled in a previous run survive the next run even when
    # text/<id>.txt is no longer on disk (post-2026-05-14 cleanup leaves
    # text/ empty between scrape extractions; build_manifest's `texts`
    # map reflects only newly-extracted records). Without this read,
    # write_text_shards would wipe the existing bundle every run.
    #
    # Semantics: shards are append-only with respect to each record key.
    # Once a record's content is in a shard, it stays until either:
    #   (a) a fresh disk read of items[<key>] overwrites it this run, or
    #   (b) the caller explicitly removes the key from items AND wipes
    #       texts-meta.json (which we'd treat as a hard reset).
    existing_records: dict = {}
    existing_meta_path = corpus_docs_dir / _meta_filename()
    if existing_meta_path.exists():
        try:
            with open(existing_meta_path, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
            for shard_entry in existing_meta.get("shards", []):
                sp = corpus_docs_dir / shard_entry["file"]
                if not sp.exists():
                    continue
                with open(sp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                for k, v in (payload.get("records") or {}).items():
                    existing_records[k] = v
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! couldn't preload existing shards ({e}) — starting fresh")
            existing_records = {}

    # Clean up previous shards. Preserve texts-meta.json (we overwrite
    # it below) — wildcard matches both shards and meta otherwise.
    # Safe to remove now because we've cached existing_records above.
    for path in glob.glob(str(corpus_docs_dir / "texts-*.json")):
        if Path(path).name == _meta_filename():
            continue
        try:
            os.remove(path)
        except OSError:
            pass

    shards: list[dict] = []                # per-shard manifest entries
    record_to_shard: dict[str, int] = {}   # composite_id → shard index

    cur_records: dict = {}                 # building shard's payload
    cur_bytes = 0                          # rolling byte count

    totals = {
        "records_with_text": 0,
        "shards": 0,
        "r2_fallback": 0,
        "total_text_bytes": 0,
        "missing_files": 0,
        "skipped_oversize_no_r2": 0,
        "from_existing_shards": 0,
    }

    def flush_shard() -> None:
        nonlocal cur_records, cur_bytes
        if not cur_records:
            return
        idx = len(shards)
        fname = _shard_filename(idx)
        path = corpus_docs_dir / fname
        payload = {
            "shard_index": idx,
            "records": cur_records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        on_disk_bytes = path.stat().st_size
        shards.append({
            "file": fname,
            "count": len(cur_records),
            "bytes": on_disk_bytes,
        })
        cur_records = {}
        cur_bytes = 0

    # Build the full set of (key → value) we want in the new shards.
    # Strategy: start from existing-shard contents (preserves prior bundle),
    # then overlay any items whose disk file exists (refreshes content).
    # Records present in items but with no disk file fall back to existing
    # content; records only in existing-shards (not in items at all) stay.
    item_paths = {key: path for key, path in items}
    final_records: dict = {}

    # Pass 1: bring existing-shard records forward as-is.
    for key, value in existing_records.items():
        final_records[key] = value

    # Pass 2: for items that have a fresh disk file, overwrite with the
    # newly-extracted content. For items whose path doesn't exist,
    # preserve the existing content (do nothing — already in final_records
    # from pass 1).
    for key, path in items:
        if not path.exists():
            if key not in final_records:
                # Not in any source — count as missing for the audit log.
                totals["missing_files"] += 1
            continue
        try:
            data = path.read_bytes()
        except OSError:
            if key not in final_records:
                totals["missing_files"] += 1
            continue
        size = len(data)
        if size > fallback_threshold:
            if r2_origin is None:
                totals["skipped_oversize_no_r2"] += 1
                if key not in final_records:
                    continue
                # If existing has content, leave it as-is; don't replace with
                # an oversize-skipped record.
                continue
            final_records[key] = R2_SENTINEL
            totals["r2_fallback"] += 1
        else:
            try:
                value = data.decode("utf-8")
            except UnicodeDecodeError:
                if key not in final_records:
                    totals["missing_files"] += 1
                continue
            final_records[key] = value
        totals["total_text_bytes"] += size

    # Sort the merged keys for deterministic shard packing. Stable order
    # ⇒ minimal shard diffs across runs (helps git delta + CF cache).
    sorted_keys = sorted(final_records.keys())

    # Pack into shards.
    for key in sorted_keys:
        value = final_records[key]
        if isinstance(value, dict):  # R2 sentinel
            value_cost = SENTINEL_BUDGET_BYTES
        else:
            value_cost = len(value.encode("utf-8")) if isinstance(value, str) else SENTINEL_BUDGET_BYTES

        if cur_records and cur_bytes + value_cost > target_bytes:
            flush_shard()
        cur_records[key] = value
        cur_bytes += value_cost
        record_to_shard[key] = len(shards)
        totals["records_with_text"] += 1

    flush_shard()
    totals["shards"] = len(shards)
    totals["from_existing_shards"] = len(existing_records) - sum(
        1 for k in existing_records if k in item_paths and item_paths[k].exists()
    )

    # Clean up per-record .txt files now that they're bundled. Leaves
    # the marker files (.pypdf-empty / .ocr-failed / .pypdf-error /
    # .empty / .error) intact — those are the "we tried, this is the
    # result" memory the extract phase needs to avoid re-attempting
    # records that produced no text. Recursive glob so we catch both
    # flat layouts (text/<id>.txt) and nested ones (text/<committee>/
    # <fid>.txt, text/<house>/<fid>.txt).
    text_root = corpus_docs_dir / "text"
    removed_txt = 0
    if text_root.exists():
        for path in text_root.rglob("*.txt"):
            try:
                path.unlink()
                removed_txt += 1
            except OSError:
                pass
    totals["txt_files_removed"] = removed_txt

    meta = {
        "shard_size_target_bytes": target_bytes,
        "fallback_threshold_bytes": fallback_threshold,
        "r2_origin": r2_origin,
        "totals": totals,
        "shards": shards,
        "record_to_shard": record_to_shard,
    }
    meta_path = corpus_docs_dir / _meta_filename()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta
