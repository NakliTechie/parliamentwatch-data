# sansadsaar-lc

Archive of PDF reports from the **Law Commission of India**
(<https://lawcommissionofindia.nic.in/>).

This repo is auto-populated by the
[`parliamentwatch-data`](https://github.com/NakliTechie/parliamentwatch-data)
daily Action whenever a new Law Commission report PDF is fetched. It
exists as a durability hedge — the upstream site is a WordPress / S3WaaS
deployment with a track record of URL rotations and occasional content
loss across the Government of India CMS estate. If the source ever goes
offline, every PDF we've fetched is still here.

## Layout

```
pdfs/<report_number>.pdf       PDFs themselves (e.g. pdfs/278.pdf)
index.json                      manifest — keyed by report_number, lists
                                sha256, size_bytes, archived_at (ISO),
                                source_url
README.md                       this file
```

## Provenance

Each entry in `index.json` carries the upstream URL at the time of
archival plus an ISO timestamp. The PDFs are byte-identical copies of
the upstream — no re-encoding, no compression. Run `sha256sum
pdfs/<n>.pdf` against the recorded digest to verify.

## Live app

Reports here are searchable, summarisable, and queryable via AI at
<https://sansadsaar.naklitechie.com>. The app links to the live
upstream source first; this archive is the click-through fallback when
upstream 404s.

## Reuse

Reports are Government of India publications, in the public domain.
Layout is intentionally stable. You can `git clone` the whole archive
if you want a local copy:

```sh
git clone https://github.com/NakliTechie/sansadsaar-lc.git
```

## Companion repos

- [`parliamentwatch-data`](https://github.com/NakliTechie/parliamentwatch-data)
  — daily-mirrored extracted text + metadata for LC + other parliamentary
  corpora (DRSC standing committees, CAG audits, Bills).
- [`SansadSaar`](https://sansadsaar.naklitechie.com) — the browser app that
  consumes both this archive and the parliamentwatch-data mirror.
