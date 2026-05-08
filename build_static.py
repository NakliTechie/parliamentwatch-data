#!/usr/bin/env python3
"""ParliamentWatch — static data builder for naklitechie/parliamentwatch-data.

Per scheduled (or workflow_dispatch) run:
  1. Scrape metadata for all 24 DRSCs across configured Lok Sabhas
  2. Extract text for as many missing PDFs as fit in the per-run budget
     (MAX_EXTRACTIONS_PER_RUN, MAX_RUN_SECONDS) — newest reports first
  3. Write docs/{reports.json, manifest.json, committees.json, meta.json}

Politeness for the historical backfill (~14k missing PDFs as of 2026-05):
  - Concurrency capped at 4 (down from 10)
  - Random 250-500ms jitter between fetches (in pdf_utils._jitter)
  - Stops the moment sansad.in returns 429/403; next run resumes
  - Per-run budget caps how aggressive any single dispatch can be

Output is served by GitHub Pages from /docs.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "docs"
DOCS.mkdir(exist_ok=True)

# Force scraper paths to docs/ before importing config-dependent modules
os.environ["DATA_DIR"] = str(DOCS)

from scraper import scrape_all_committees, load_existing_reports  # noqa: E402
from pdf_utils import get_report_text, RateLimited  # noqa: E402
from config import DRSC_COMMITTEES  # noqa: E402

# Per-run caps. Tuned for the historical backfill — at ~10s/PDF avg with 4
# workers, MAX_EXTRACTIONS_PER_RUN=400 finishes in ~17 minutes wall clock.
# MAX_RUN_SECONDS provides a hard ceiling so a slow round doesn't spill into
# the GH Actions 6h job limit.
MAX_EXTRACTIONS_PER_RUN = int(os.environ.get("MAX_EXTRACTIONS_PER_RUN", "400"))
MAX_RUN_SECONDS = int(os.environ.get("MAX_RUN_SECONDS", "1800"))  # 30 min
EXTRACT_WORKERS = int(os.environ.get("EXTRACT_WORKERS", "4"))

LOK_SABHAS = [int(x) for x in os.environ.get("LOK_SABHAS", "18").split(",")]


def _safe_filename(report_num):
    if report_num is None:
        return None
    return str(report_num).replace("/", "-").replace(" ", "_")


def _missing_reports_priority_order():
    """Walk every report we have metadata for and yield those without an
    extracted text file, ordered (lok_sabha desc, report_number desc) so the
    most recent reports go first.

    This replaces the old per-committee-cap loop, which left everything older
    than the top-N invisible to the action. With this version, the per-run
    cap is global (MAX_EXTRACTIONS_PER_RUN) and any single run picks up where
    the last run left off — the on-disk text/<committee>/<id>.txt cache is
    the source of truth for "already done".
    """
    reports = load_existing_reports()
    candidates = []
    for committee_key, committee_reports in reports.items():
        for report in committee_reports:
            num = report.get("report_number")
            url = report.get("pdf_url")
            if not num or not url:
                continue
            safe = _safe_filename(num)
            text_path = DOCS / "text" / committee_key / f"{safe}.txt"
            if text_path.exists():
                continue
            candidates.append({
                "committee_key": committee_key,
                "report_number": num,
                "pdf_url": url,
                "lok_sabha": report.get("lok_sabha", 0) or 0,
                "safe": safe,
            })
    # Most recent first (LS18 before LS17, higher report number first within an LS)
    candidates.sort(key=lambda c: (c["lok_sabha"], c["report_number"]), reverse=True)
    return candidates


def extract_missing_texts():
    """Extract up to MAX_EXTRACTIONS_PER_RUN missing PDFs. Stops early on
    rate-limit (429/403) or wall-clock budget exhaustion."""
    candidates = _missing_reports_priority_order()
    print(f"  candidates: {len(candidates)} reports missing extracted text")
    if not candidates:
        return {"extracted": [], "failed": [], "rate_limited": False, "budget_hit": False}

    target = candidates[:MAX_EXTRACTIONS_PER_RUN]
    print(f"  budget: extracting up to {len(target)} this run "
          f"(MAX_EXTRACTIONS_PER_RUN={MAX_EXTRACTIONS_PER_RUN}, MAX_RUN_SECONDS={MAX_RUN_SECONDS}, EXTRACT_WORKERS={EXTRACT_WORKERS})")

    deadline = time.monotonic() + MAX_RUN_SECONDS
    extracted, failed = [], []
    rate_limited = False
    budget_hit = False

    def _do(c):
        try:
            text = get_report_text(c["pdf_url"], c["committee_key"], str(c["report_number"]))
            return c, text, None
        except RateLimited as rl:
            return c, None, rl
        except Exception as e:
            return c, None, e

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
        futures = {ex.submit(_do, c): c for c in target}
        for fut in as_completed(futures):
            c, text, err = fut.result()
            label = f"{c['committee_key']}/{c['safe']}"
            if isinstance(err, RateLimited):
                print(f"  [RATE-LIMITED] {label}: {err}")
                rate_limited = True
                # Cancel any not-yet-started futures and stop
                for f in futures:
                    if not f.done():
                        f.cancel()
                break
            elif text:
                extracted.append(label)
            else:
                failed.append(label)
            if time.monotonic() > deadline:
                print(f"  [BUDGET] wall-clock budget hit after {len(extracted)} extractions")
                budget_hit = True
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

    return {
        "extracted": extracted,
        "failed": failed,
        "rate_limited": rate_limited,
        "budget_hit": budget_hit,
        "candidates_total": len(candidates),
    }


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


def write_meta(extract_stats, total_reports, total_with_text):
    meta = {
        "version": "1.1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lok_sabhas": LOK_SABHAS,
        "max_extractions_per_run": MAX_EXTRACTIONS_PER_RUN,
        "extract_workers": EXTRACT_WORKERS,
        "total_reports": total_reports,
        "total_with_text": total_with_text,
        "extract_stats": {
            "extracted": len(extract_stats["extracted"]),
            "failed": len(extract_stats["failed"]),
            "rate_limited": extract_stats.get("rate_limited", False),
            "budget_hit": extract_stats.get("budget_hit", False),
            "candidates_remaining_after_run": max(
                0, extract_stats.get("candidates_total", 0) - len(extract_stats["extracted"])
            ),
        },
    }
    with open(DOCS / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    print("=== ParliamentWatch static builder ===")
    print(f"DATA_DIR                : {DOCS}")
    print(f"LOK_SABHAS              : {LOK_SABHAS}")
    print(f"MAX_EXTRACTIONS_PER_RUN : {MAX_EXTRACTIONS_PER_RUN}")
    print(f"MAX_RUN_SECONDS         : {MAX_RUN_SECONDS}")
    print(f"EXTRACT_WORKERS         : {EXTRACT_WORKERS}")

    print("\n[1/4] Scraping committee metadata...")
    for ls in LOK_SABHAS:
        print(f"  Lok Sabha {ls}:")
        scrape_all_committees(lok_sabha=ls)

    print("\n[2/4] Extracting missing texts (priority: newest first)...")
    extract_stats = extract_missing_texts()
    print(f"  extracted={len(extract_stats['extracted'])} "
          f"failed={len(extract_stats['failed'])} "
          f"rate_limited={extract_stats.get('rate_limited', False)} "
          f"budget_hit={extract_stats.get('budget_hit', False)}")

    print("\n[3/4] Building manifest + committees index...")
    manifest = build_manifest()
    with open(DOCS / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    build_committees_index()

    print("\n[4/4] Writing meta.json...")
    reports = load_existing_reports()
    total = sum(len(v) for v in reports.values())
    total_with_text = sum(len(v) for v in manifest.get("texts", {}).values())
    meta = write_meta(extract_stats, total, total_with_text)
    print(json.dumps(meta, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
