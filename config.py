"""Committee config + sansad.in API map.

Vendored from pranaykotas/parliamentwatch (LLM and email vars stripped — this
mirror only scrapes and publishes static data).
"""

import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

BASE_URL = "https://sansad.in"

REPORTS_API = f"{BASE_URL}/api_ls/committee/lsRSAllReports"      # LS committees
RS_REPORTS_API = f"{BASE_URL}/api_rs/committee/committee-reports"  # RS committees

CURRENT_LOK_SABHA = int(os.getenv("LOK_SABHA_NUMBER", "18"))

# All 24 Departmentally Related Standing Committees (DRSCs)
# 'house': "L" = Lok Sabha chaired, "R" = Rajya Sabha chaired
# Same api_code can mean different committees depending on house — never fetch
# a committee with the wrong house.
DRSC_COMMITTEES = {
    # --- Lok Sabha chaired (16) ---
    "agriculture":      {"name": "Agriculture, Animal Husbandry and Food Processing", "api_code": 5,  "house": "L"},
    "chemicals":        {"name": "Chemicals & Fertilizers",                            "api_code": 45, "house": "L"},
    "coal":             {"name": "Coal, Mines and Steel",                              "api_code": 46, "house": "L"},
    "defence":          {"name": "Defence",                                            "api_code": 7,  "house": "L"},
    "energy":           {"name": "Energy",                                             "api_code": 9,  "house": "L"},
    "external_affairs": {"name": "External Affairs",                                   "api_code": 11, "house": "L"},
    "finance":          {"name": "Finance",                                            "api_code": 12, "house": "L"},
    "consumer_affairs": {"name": "Consumer Affairs, Food and Public Distribution",     "api_code": 13, "house": "L"},
    "communications":   {"name": "Communications and Information Technology",         "api_code": 18, "house": "L"},
    "labour":           {"name": "Labour, Textiles and Skill Development",             "api_code": 19, "house": "L"},
    "petroleum":        {"name": "Petroleum & Natural Gas",                            "api_code": 23, "house": "L"},
    "railways":         {"name": "Railways",                                           "api_code": 28, "house": "L"},
    "rural_development":{"name": "Rural Development and Panchayati Raj",               "api_code": 32, "house": "L"},
    "social_justice":   {"name": "Social Justice & Empowerment",                       "api_code": 47, "house": "L"},
    "housing":          {"name": "Housing and Urban Affairs",                          "api_code": 41, "house": "L"},
    "water_resources":  {"name": "Water Resources",                                    "api_code": 44, "house": "L"},
    # --- Rajya Sabha chaired (8) ---
    # mst_comm_id is the integer in /rs/committees/{id} URLs — primary key for api_rs.
    "commerce":     {"name": "Commerce",                                       "api_code": 13, "mst_comm_id": 12, "house": "R"},
    "health":       {"name": "Health and Family Welfare",                      "api_code": 14, "mst_comm_id": 14, "house": "R"},
    "home_affairs": {"name": "Home Affairs",                                   "api_code": 15, "mst_comm_id": 15, "house": "R"},
    "education":    {"name": "Education, Women, Children, Youth and Sports",   "api_code": 16, "mst_comm_id": 16, "house": "R"},
    "industry":     {"name": "Industry",                                       "api_code": 17, "mst_comm_id": 17, "house": "R"},
    "personnel":    {"name": "Personnel, Public Grievances, Law and Justice",  "api_code": 18, "mst_comm_id": 18, "house": "R"},
    "science":      {"name": "Science and Technology, Environment and Forests","api_code": 24, "mst_comm_id": 19, "house": "R"},
    "transport":    {"name": "Transport, Tourism and Culture",                 "api_code": 31, "mst_comm_id": 20, "house": "R"},
}

# Data paths — overridable via env so build_static.py can redirect to /docs.
DATA_DIR = os.getenv("DATA_DIR", str(_SCRIPT_DIR / "data"))
REPORTS_JSON = os.path.join(DATA_DIR, "reports.json")
PDFS_DIR = os.path.join(DATA_DIR, "pdfs")
TEXT_DIR = os.path.join(DATA_DIR, "text")
