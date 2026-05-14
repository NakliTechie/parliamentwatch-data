# parliamentwatch-data

Static data mirror for [SansadSaar](https://github.com/NakliTechie/SansadSaar). Hosts the five document-centric corpora: **DRSC committee reports**, **CAG audits**, **Bills**, **Law Commission reports**, **Financial Committee reports**.

Served at `sansadsaar-data.naklitechie.com` via Cloudflare Pages.

## Why a separate repo?

The upstream sources don't permit cross-origin browser requests. This repo runs each corpus's scraper from GitHub Actions on a schedule and re-publishes JSON + text via a CORS-friendly endpoint, so the single-file SansadSaar app can read it without any backend.

The two high-volume person-centric corpora (debates + questions) and the Central Gazette corpus live in their own sibling repos to keep this one comfortably under Cloudflare Pages's per-deploy file count cap.

## Credits

DRSC scraper foundation (`scraper.py`, `pdf_utils.py`, `config.py`) is vendored from [pranaykotas/parliamentwatch](https://github.com/pranaykotas/parliamentwatch). Thanks to Pranay Kotasthane for the original work. CAG, Bills, LC, and FC scrapers are independent additions.
