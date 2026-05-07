#!/usr/bin/env python3
"""ParliamentWatch — static data builder for naklitechie/parliamentwatch-data.

Runs daily via GH Actions:
  1. Scrape metadata for all 24 DRSCs across configured Lok Sabhas
  2. Extract text for the N most recent reports per committee
  3. Write docs/{reports.json, manifest.json, committees.json, meta.json}

Output is served by GitHub Pages from /docs.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "docs"
DOCS.mkdir(exist_ok=True)

# Force scraper paths to docs/ before importing config-dependent modules
os.environ["DATA_DIR"] = str(DOCS)

from scraper import scrape_all_committees, load_existing_reports  # noqa: E402
from pdf_utils import get_report_text  # noqa: E402
from config import DRSC_COMMITTEES  # noqa: E402

# Cap text extraction per committee per run — full first-time extraction
# (~thousands of PDFs across all 24 committees) would exceed GH Actions
# 6h job limit and risk getting our IP rate-limited by sansad.in.
TEXT_LIMIT_PER_COMMITTEE = int(os.environ.get("TEXT_LIMIT_PER_COMMITTEE", "20"))

LOK_SABHAS = [int(x) for x in os.environ.get("LOK_SABHAS", "18").split(",")]


def _safe_filename(report_num):
    if report_num is None:
        return None
    return str(report_num).replace("/", "-").replace(" ", "_")


def extract_recent_texts():
    """Extract text for the latest N reports per committee."""
    reports = load_existing_reports()
    extracted, skipped, failed = [], [], []

    for committee_key, committee_reports in reports.items():
        # Reports are already sorted by (lok_sabha, report_number) desc
        for report in committee_reports[:TEXT_LIMIT_PER_COMMITTEE]:
            num = report.get("report_number")
            url = report.get("pdf_url")
            if not num or not url:
                continue
            safe = _safe_filename(num)
            text_path = DOCS / "text" / committee_key / f"{safe}.txt"
            if text_path.exists():
                skipped.append(f"{committee_key}/{safe}")
                continue
            try:
                text = get_report_text(url, committee_key, str(num))
                if text:
                    extracted.append(f"{committee_key}/{safe}")
                else:
                    failed.append(f"{committee_key}/{safe}")
            except Exception as e:
                print(f"  [extract] {committee_key}/{safe}: {e}")
                failed.append(f"{committee_key}/{safe}")

    return {"extracted": extracted, "skipped": skipped, "failed": failed}


def build_manifest():
    """List every extracted text file with its size."""
    manifest = {"texts": {}}
    text_root = DOCS / "text"
    if not text_root.exists():
        return manifest
    for committee_dir in sorted(text_root.iterdir()):
        if not committee_dir.is_dir():
            continue
        committee_key = committee_dir.name
        manifest["texts"][committee_key] = {}
        for text_file in sorted(committee_dir.glob("*.txt")):
            manifest["texts"][committee_key][text_file.stem] = {
                "size": text_file.stat().st_size,
                "url": f"text/{committee_key}/{text_file.name}",
            }
    return manifest


def build_committees_index():
    out = {}
    for key, c in DRSC_COMMITTEES.items():
        out[key] = {"name": c["name"], "house": c.get("house", "L")}
    with open(DOCS / "committees.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def write_meta(extract_stats, total_reports):
    meta = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lok_sabhas": LOK_SABHAS,
        "text_limit_per_committee": TEXT_LIMIT_PER_COMMITTEE,
        "total_reports": total_reports,
        "extract_stats": {
            "extracted": len(extract_stats["extracted"]),
            "skipped": len(extract_stats["skipped"]),
            "failed": len(extract_stats["failed"]),
        },
    }
    with open(DOCS / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    print("=== ParliamentWatch static builder ===")
    print(f"DATA_DIR              : {DOCS}")
    print(f"LOK_SABHAS            : {LOK_SABHAS}")
    print(f"TEXT_LIMIT_PER_COMMITTEE: {TEXT_LIMIT_PER_COMMITTEE}")

    print("\n[1/4] Scraping committee metadata...")
    for ls in LOK_SABHAS:
        print(f"  Lok Sabha {ls}:")
        scrape_all_committees(lok_sabha=ls)

    print("\n[2/4] Extracting text for recent reports...")
    extract_stats = extract_recent_texts()
    print(f"  extracted={len(extract_stats['extracted'])} "
          f"skipped={len(extract_stats['skipped'])} "
          f"failed={len(extract_stats['failed'])}")

    print("\n[3/4] Building manifest + committees index...")
    manifest = build_manifest()
    with open(DOCS / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    build_committees_index()

    print("\n[4/4] Writing meta.json...")
    reports = load_existing_reports()
    total = sum(len(v) for v in reports.values())
    meta = write_meta(extract_stats, total)
    print(json.dumps(meta, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
