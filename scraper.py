"""Scraper for sansad.in parliamentary committee reports (vendored from upstream).

Source: https://github.com/pranaykotas/parliamentwatch/blob/main/scraper.py
"""

import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import REPORTS_API, RS_REPORTS_API, CURRENT_LOK_SABHA, DRSC_COMMITTEES, REPORTS_JSON, DATA_DIR

_MAX_WORKERS = 10


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_existing_reports():
    if os.path.exists(REPORTS_JSON):
        with open(REPORTS_JSON, "r") as f:
            return json.load(f)
    return {}


def save_reports(reports):
    ensure_data_dir()
    with open(REPORTS_JSON, "w") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)


def sanitize_url(url):
    if url:
        return url.replace("\\", "/")
    return url


_RS_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://sansad.in/rs/committees",
}


def _fetch_ls_committee_reports(committee_key, lok_sabha):
    committee = DRSC_COMMITTEES[committee_key]
    print(f"  Fetching {committee['name']} (LS {lok_sabha})...")
    params = {
        "house": "L",
        "committeeCode": committee["api_code"],
        "lsNo": lok_sabha,
        "page": 1,
        "size": 200,
        "sortOn": "reportNo",
        "sortBy": "desc",
    }
    try:
        resp = requests.get(REPORTS_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Error fetching {committee['name']}: {e}")
        return []

    records = data.get("records", [])
    reports = []
    for record in records:
        reports.append({
            "committee": committee_key,
            "committee_name": record.get("CommitteeName", committee["name"]).strip(),
            "report_number": record.get("reportNo"),
            "title": record.get("SubjectOfTheReport", ""),
            "presented_in_ls": record.get("PresentedInLS"),
            "laid_in_rs": record.get("LaidInRS"),
            "presented_to_speaker": record.get("PresentedToSpeaker"),
            "date_of_presentation": record.get("dateOfPresentation"),
            "date_of_adoption": record.get("dateOfAdoption"),
            "pdf_url": sanitize_url(record.get("url")),
            "pdf_url_hindi": sanitize_url(record.get("urlH")),
            "lok_sabha": record.get("Loksabha", lok_sabha),
            "house": "L",
        })
    print(f"  {len(reports)} reports for {committee['name']}")
    return reports


def _fetch_rs_committee_reports(committee_key, lok_sabha):
    committee = DRSC_COMMITTEES[committee_key]
    mst_comm_id = committee.get("mst_comm_id")
    if not mst_comm_id:
        print(f"  Skipping {committee['name']}: no mst_comm_id")
        return []

    print(f"  Fetching {committee['name']} (RS, all terms)...")
    all_records = []
    page = 1
    while True:
        params = {
            "mstCommId": mst_comm_id,
            "departmentId": "",
            "presentationYear": "",
            "search": "",
            "page": page,
            "size": 200,
            "sortOn": "reportNo",
            "sortBy": "desc",
            "locale": "en",
        }
        try:
            resp = requests.get(RS_REPORTS_API, params=params, headers=_RS_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error fetching {committee['name']}: {e}")
            break
        records = data.get("records", [])
        all_records.extend(records)
        total = data.get("_metadata", {}).get("totalElements", 0)
        if len(all_records) >= total or not records:
            break
        page += 1

    reports = []
    for record in all_records:
        reports.append({
            "committee": committee_key,
            "committee_name": record.get("committeeName") or committee["name"],
            "report_number": record.get("reportNo"),
            "title": record.get("subjectOfTheReport", ""),
            "presented_in_ls": None,
            "laid_in_rs": None,
            "presented_to_speaker": None,
            "date_of_presentation": record.get("dateOfPresentation"),
            "date_of_adoption": record.get("dateOfAdoption"),
            "pdf_url": sanitize_url(record.get("url")),
            "pdf_url_hindi": sanitize_url(record.get("urlHindi")),
            "lok_sabha": lok_sabha,
            "house": "R",
        })
    print(f"  {len(reports)} reports for {committee['name']}")
    return reports


def fetch_committee_reports(committee_key, lok_sabha=None):
    if lok_sabha is None:
        lok_sabha = CURRENT_LOK_SABHA
    house = DRSC_COMMITTEES[committee_key].get("house", "L")
    if house == "R":
        return _fetch_rs_committee_reports(committee_key, lok_sabha)
    return _fetch_ls_committee_reports(committee_key, lok_sabha)


def scrape_all_committees(committee_keys=None, lok_sabha=None):
    if committee_keys is None:
        committee_keys = list(DRSC_COMMITTEES.keys())

    all_reports = load_existing_reports()
    fetched = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        future_to_key = {
            ex.submit(fetch_committee_reports, k, lok_sabha): k
            for k in committee_keys if k in DRSC_COMMITTEES
        }
        for fut in as_completed(future_to_key):
            k = future_to_key[fut]
            try:
                fetched[k] = fut.result()
            except Exception as e:
                print(f"  Error fetching {k}: {e}")
                fetched[k] = []

    for key, fresh_reports in fetched.items():
        committee_house = DRSC_COMMITTEES[key].get("house", "L")
        existing_reports = {
            (r.get("report_number"), r.get("lok_sabha")): r
            for r in all_reports.get(key, [])
            if r.get("house") == committee_house
        }
        for r in fresh_reports:
            rid = (r.get("report_number"), r.get("lok_sabha"))
            existing_reports[rid] = r
        all_reports[key] = sorted(
            existing_reports.values(),
            key=lambda x: (x.get("lok_sabha") or 0, x.get("report_number") or 0),
            reverse=True,
        )

    save_reports(all_reports)
    return all_reports
