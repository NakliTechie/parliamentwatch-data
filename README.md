# parliamentwatch-data

Static data mirror for [SansadLocal](https://github.com/NakliTechie/SansadLocal) — the single-file browser app for Indian Parliamentary Committee reports.

A daily GitHub Actions workflow scrapes [sansad.in](https://sansad.in), extracts text from new PDFs, and publishes JSON/text files to GitHub Pages. The browser app fetches them with no server, no API key, no auth.

## Why a mirror?

`sansad.in` blocks cross-origin browser requests (no `Access-Control-Allow-Origin`). A pure browser app cannot fetch its API directly. This repo runs the scraper on a schedule from a server, then re-publishes the data on GitHub Pages — which **does** serve with proper CORS — so the single-file app can read it from anywhere.

The scraper code (`scraper.py`, `pdf_utils.py`, `config.py`) is vendored from [pranaykotas/parliamentwatch](https://github.com/pranaykotas/parliamentwatch). Credit and upstream maintenance: Pranay Kotasthane.

## What's published

```
https://naklitechie.github.io/parliamentwatch-data/
├── meta.json            # version + last-updated + counts
├── committees.json      # 24 DRSCs (key → name + house)
├── reports.json         # all metadata (committee → list of reports)
├── manifest.json        # which texts are extracted (committee → report-id → size)
└── text/<committee>/<report-id>.txt
```

## Schedule

- Daily at 04:30 UTC (10:00 IST)
- Manual trigger via `workflow_dispatch` (Actions → Scrape and publish → Run workflow)
- Each run scrapes metadata for all 24 committees and extracts text for the **20 most recent reports per committee** (configurable via the manual trigger).

## Local dev

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build_static.py
```

Output goes into `docs/`. PDFs are downloaded to `docs/pdfs/` (gitignored — re-extracted text is what matters) and text to `docs/text/<committee>/`.
