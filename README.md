# parliamentwatch-data

Static data mirror for [SansadSaar](https://github.com/NakliTechie/SansadSaar) — the browser app for India's parliamentary record. **Three corpora live** (DRSC + CAG + Bills); more in the pipeline. Each corpus's outputs live under `docs/<corpus>/`.

Scheduled GitHub Actions workflows scrape the upstream sources, extract text from new PDFs, and publish JSON + text files. The browser app fetches them with no server, no API key, no auth.

## Why a mirror?

The upstream sources don't permit cross-origin browser requests. This repo runs each corpus's scraper from GitHub Actions on a schedule and re-publishes via a CORS-friendly endpoint, so the single-file app can read it without a server.

The DRSC scraper (`scraper.py`, `pdf_utils.py`, `config.py`) is vendored from [pranaykotas/parliamentwatch](https://github.com/pranaykotas/parliamentwatch); credit + upstream maintenance: Pranay Kotasthane. CAG (`cag/scraper.py` + `build_cag.py`) and Bills (`bills/sansad/scraper.py` + `build_bills_sansad.py`) are independent scrapers added for v1.0b and v1.1.a respectively.

## Hosting

Served at `sansadsaar-data.naklitechie.com`.

## What's published

Each corpus's outputs land in `docs/<corpus>/` parallel to its siblings. **All three corpora share the same shape** for the deep-search artefacts (sharded record index + sharded `search-bundle-NN.json` + sharded `search-index-NN.json`) so v2's cross-corpus search can merge cleanly. See `Browser/CONV.md` § "Shard the index + the search bundle + the body-token index from day 1" for the full pattern.

### DRSC

```
https://sansadsaar-data.naklitechie.com/drsc/
├── meta.json                # version + last-updated + counts + bundle/index stats
├── committees.json          # 24 DRSCs (key → name + house)
├── reports.json             # all metadata (committee → list of reports)
├── manifest.json            # which texts are extracted (committee → report-id → size)
├── search-bundle-<NN>.json  # title + first 5K chars per report, sharded
├── search-index-<NN>.json   # inverted token index over full body, sharded
└── text/<committee>/<report-id>.txt
```

### CAG

```
https://sansadsaar-data.naklitechie.com/cag/
├── meta.json                # version + last-updated + counts + bundle/index stats
├── reports.json             # all metadata, flat (id → record)
├── manifest.json            # which texts are extracted (id → size)
├── search-bundle-<NN>.json
├── search-index-<NN>.json
└── text/<id>.txt
```

### Bills (sansad-side)

```
https://sansadsaar-data.naklitechie.com/bills/
├── meta.json                # version + last-updated + counts + bundle/index stats
├── index-meta.json          # shard manifest for the record index
├── index-<NN>.json          # ~1 MB shards, ~1000 records each, newest-first
├── manifest.json            # which texts are extracted (compositeId → size)
├── search-bundle-<NN>.json  # title + first 5K chars per bill with text
├── search-index-<NN>.json   # inverted token index over full body
└── text/<compositeId>.txt   # compositeId = `<billNumber>_<billYear>_<L|R>`
```

Bills' record index is sharded (`index-meta.json` + `index-NN.json`); DRSC and CAG indices are still single-file (`reports.json`) — within current cap. Both will shard at the same break-point if needed.

PDFs (`docs/<corpus>/pdfs/...`) are gitignored — they're cached locally for re-extraction but not committed; the extracted text in `docs/<corpus>/text/` is the canonical artefact.

## Schedules

| Corpus | Cadence |
|---|---|
| DRSC | every 4 hours |
| CAG | daily, with hourly backfill and a weekly OCR slow-lane |
| Bills | daily, with a 4-hourly backfill |

Each corpus's workflows can also be triggered manually via `workflow_dispatch`. Runs that are rate-limited by upstream skip the extraction phase for 6h before retrying — fire-and-forget safe.

## Build flow

Each corpus has its own orchestrator at the repo root, calling into the corpus-specific scraper module:

```
build_static.py        → docs/drsc/   (uses scraper.py + pdf_utils.py + config.py)
build_cag.py           → docs/cag/    (uses cag/scraper.py)
build_bills_sansad.py  → docs/bills/  (uses bills/sansad/scraper.py)
build_cag_ocr.py       → docs/cag/    (post-pypdf OCR for scanned PDFs)
```

Each orchestrator's phases:

1. Scrape / refresh metadata from upstream API or listing pages.
2. Extract missing texts (priority: newest first; respects budget + rate limits).
3. Write manifest + meta.
4. Build sharded `search-bundle-<NN>.json` (title + first 5K chars per record).
5. Build sharded `search-index-<NN>.json` (inverted token index over full body, delta-encoded postings).

Same constants across all three corpora (`_HEAD_CHARS=5000`, `_DOCS_PER_SHARD=2500`, `_FREQ_CUTOFF_HIGH=0.9`, `_FREQ_CUTOFF_LOW=2`, `_MAX_TOKEN_LEN=25`, same `_STOPWORDS` list, same `_TOKEN_RE`) so v2 cross-corpus search has uniform vocab + shard granularity to merge against. Per the Independence Principle, the constants are duplicated across `build_*.py` files, not imported.

## Local dev

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pick a corpus orchestrator:
python build_static.py            # DRSC
python build_cag.py               # CAG
python build_bills_sansad.py      # Bills (sansad-side)
```

Output goes into `docs/<corpus>/`. PDFs are downloaded to `docs/<corpus>/pdfs/` (gitignored via `docs/**/pdfs/`) and text to `docs/<corpus>/text/`.

For a fast dry-run that only refreshes the index + regenerates derived files without downloading new PDFs, set `MAX_EXTRACTIONS_PER_RUN=0`.
