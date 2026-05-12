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

    # Clean up previous shards. Preserve texts-meta.json (we overwrite
    # it below) — wildcard matches both shards and meta otherwise.
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

    for key, path in items:
        if not path.exists():
            totals["missing_files"] += 1
            continue
        try:
            data = path.read_bytes()
        except OSError:
            totals["missing_files"] += 1
            continue

        size = len(data)
        if size > fallback_threshold:
            if r2_origin is None:
                # No fallback configured — skip rather than overwhelm a shard.
                totals["skipped_oversize_no_r2"] += 1
                continue
            value = R2_SENTINEL
            value_cost = SENTINEL_BUDGET_BYTES
            totals["r2_fallback"] += 1
        else:
            try:
                value = data.decode("utf-8")
            except UnicodeDecodeError:
                # Shouldn't happen for our text/ files, but failing soft is
                # better than aborting the whole derive.
                totals["missing_files"] += 1
                continue
            value_cost = size

        # If adding this would push the shard past the target, flush first.
        # (Exception: empty shard — even an oversize-but-not-fallback record
        # gets its own shard. Shouldn't happen given fallback_threshold but
        # keeps the loop honest.)
        if cur_records and cur_bytes + value_cost > target_bytes:
            flush_shard()

        cur_records[key] = value
        cur_bytes += value_cost
        record_to_shard[key] = len(shards)
        totals["records_with_text"] += 1
        totals["total_text_bytes"] += size

    flush_shard()
    totals["shards"] = len(shards)

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
