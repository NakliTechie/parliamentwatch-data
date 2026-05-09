# parliamentwatch-data

Static data mirror for [SansadSaar](https://github.com/NakliTechie/SansadSaar) — the browser app for Indian Parliamentary documentary output. DRSC (Departmentally Related Standing Committees) is the first registered corpus; future corpora (CAG, Bills, Hansard, etc.) will land in their own sibling subfolders under `docs/`.

A scheduled GitHub Actions workflow scrapes [sansad.in](https://sansad.in), extracts text from new PDFs, and publishes JSON/text files via Cloudflare Workers + Static Assets. The browser app fetches them with no server, no API key, no auth.

## Why a mirror?

`sansad.in` blocks cross-origin browser requests (no `Access-Control-Allow-Origin`). A pure browser app cannot fetch its API directly. This repo runs the scraper on a schedule from GitHub Actions, then re-publishes the data via Cloudflare Workers — which serves with proper CORS — so the single-file app can read it from anywhere.

The scraper code (`scraper.py`, `pdf_utils.py`, `config.py`) is vendored from [pranaykotas/parliamentwatch](https://github.com/pranaykotas/parliamentwatch). Credit and upstream maintenance: Pranay Kotasthane.

## Hosting

- **Cloudflare Workers + Static Assets** serves `docs/` directly from CF's global edge.
- Two custom domains point at the same Worker:
  - `sansadsaar-data.naklitechie.com` (canonical)
  - `sansad-files.naklitechie.com` (legacy alias)
- `wrangler.toml` configures the deploy; CF Workers Builds runs `npx wrangler deploy` automatically on every push to main.
- `docs/_headers` sets cache + CORS rules.

## What's published (DRSC corpus)

```
https://sansadsaar-data.naklitechie.com/drsc/
├── meta.json              # version + last-updated + counts + bundle/index stats
├── committees.json        # 24 DRSCs (key → name + house)
├── reports.json           # all metadata (committee → list of reports)
├── manifest.json          # which texts are extracted (committee → report-id → size)
├── search-bundle.json     # title + first 5K chars per report (snippet preview + substring)
├── search-index.json      # inverted token index over the full body of every report
└── text/<committee>/<report-id>.txt
```

Each future corpus lands in `docs/<corpus>/` parallel to `docs/drsc/`.

## Schedule

- **Cron**: every 4 hours at minute :17 (`17 */4 * * *` UTC). Off-hour minute deliberate — top-of-hour is GH's busiest slot for scheduled jobs.
- **Manual trigger**: Actions → "Scrape and publish" → Run workflow. Inputs: `lok_sabhas`, `max_extractions`, `max_run_seconds`.
- **Per-run budget**: `MAX_EXTRACTIONS_PER_RUN=400` (configurable). At ~10s/PDF avg with 4 workers, that finishes in ~17 minutes wall clock. `MAX_RUN_SECONDS=1800` is a hard ceiling.
- **Rate-limit cooldown**: if the previous run was 429'd by sansad.in, the next run skips the extraction phase for 6h. Fire-and-forget safe.

## Build flow (`build_static.py`)

```
[1/7] Scrape committee metadata for configured Lok Sabhas
[2/7] Migrate any pre-v0.4 unprefixed text files (idempotent — usually no-op)
[3/7] Extract missing texts (priority: newest first; respects budget + rate limits)
[4/7] Build manifest.json + committees.json
[5/7] Build search-bundle.json (title + first 5K chars per report)
[6/7] Build search-index.json (inverted token index over full body)
[7/7] Write meta.json
```

All outputs land in `docs/drsc/`. The Action commits any diff and pushes; CF Workers redeploys automatically.

## Local dev

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_static.py
```

Output goes into `docs/drsc/`. PDFs are downloaded to `docs/drsc/pdfs/` (gitignored — re-extracted text is what matters) and text to `docs/drsc/text/<committee>/`.
