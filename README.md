# parliamentwatch-data

Static data mirror for [SansadSaar](https://github.com/indiavotes/SansadSaar). Holds the document-centric corpora: **DRSC committee reports**, **CAG audits**, **Bills**, **Law Commission reports**, **Financial Committee reports**.

Served at `sansadsaar-data.naklitechie.com`.

## Why a separate repo?

Upstream sources don't permit cross-origin browser requests. GitHub Actions scrapes each corpus on a schedule and re-publishes JSON + text at a CORS-friendly endpoint the SansadSaar app reads directly — no backend.

## Family

- [SansadSaar](https://github.com/indiavotes/SansadSaar) — the app
- [sansadsaar-proceedings-data](https://github.com/NakliTechie/sansadsaar-proceedings-data) — Debates, Questions
- [sansadsaar-gazettes](https://github.com/NakliTechie/sansadsaar-gazettes) — Central Gazette
- [sansadsaar-lc](https://github.com/indiavotes/sansadsaar-lc) — Law Commission PDF archive

## Credits

DRSC scraper foundation (`scraper.py`, `pdf_utils.py`, `config.py`) is vendored from [pranaykotas/parliamentwatch](https://github.com/pranaykotas/parliamentwatch). Thanks to Pranay Kotasthane for the original work. CAG, Bills, LC, and FC scrapers are independent additions.
