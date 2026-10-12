"""Build the self-contained Fed hike event-study HTML (Python port of build.mjs).

Reads ../tmp/fed_hike_target_bbg.csv, computes event-window series for each
hike cycle, and injects the payload into index.template.html -> dist/index.html.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent / "tmp" / "fed_hike_target_bbg.csv"
TEMPLATE_PATH = HERE / "index.template.html"
OUTPUT_PATH = HERE / "dist" / "index.html"

EVENT_DEFS = [
    ("1999-06-30", "06/30/99", "1999"),
    ("2004-06-30", "06/30/04", "2004"),
    ("2016-12-14", "12/14/16", "2016"),
    ("2022-03-16", "03/16/22", "2022"),
    ("2026-09-16", "09/16/26", "2026"),
]

WINDOW_START, WINDOW_END = -60, 500


def _num(value: str) -> float:
    return float(value) if value not in ("", None) else math.nan


def _round(value, places: int = 3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(value, places)


def load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        rows = []
        for rec in csv.DictReader(fh):
            rows.append(
                {
                    "date": rec["date"],
                    "fnccBasis": _num(rec["fncc_basis"]),
                    "fed": _num(rec["target_rate"]),
                    "ten": _num(rec["ust_10y"]),
                    "twosTens": _num(rec["twos_tens"]),
                    "fivesThirties": _num(rec["fives_thirties"]),
                    "nmcmfr30": _num(rec["nmcmfr30"]),
                    "creditOas": _num(rec["credit_oas"]),
                    "spx": _num(rec["spx"]),
                    "oil": _num(rec["oil"]),
                    "gold": _num(rec["gold"]),
                    "cpi": _num(rec["cpi_yoy"]),
                    "urate": _num(rec["urate"]),
                    "fedObserved": _num(rec["target_rate_observed"]),
                    "tenObserved": _num(rec["ust_10y_observed"]),
                    "twosTensObserved": _num(rec["twos_tens_observed"]),
                    "fivesThirtiesObserved": _num(rec["fives_thirties_observed"]),
                    "fnccObserved": _num(rec["fncc_basis_observed"]),
                    "nmcmfr30Observed": _num(rec["nmcmfr30_observed"]),
                }
            )
    return rows


def build_payload():
    rows = load_rows()
    by_date = {row["date"]: row for row in rows}
    latest = rows[-1]["date"]

    events = []
    for event_date, label, short in EVENT_DEFS:
        start = date.fromisoformat(event_date)
        day0 = by_date[event_date]
        series = []
        for day in range(WINDOW_START, WINDOW_END + 1):
            d = start + timedelta(days=day)
            d_iso = d.isoformat()
            row = by_date.get(d_iso)
            series.append(
                {
                    "day": day,
                    "date": d_iso,
                    "fed": _round(row["fed"], 2) if row else None,
                    "ten": _round(row["ten"], 3) if row else None,
                    "twosTens": _round(row["twosTens"], 1) if row else None,
                    "fivesThirties": _round(row["fivesThirties"], 1) if row else None,
                    "fnccBasis": _round(row["fnccBasis"], 1) if row else None,
                    "mort": _round(row["nmcmfr30"], 3) if row else None,
                    "creditOas": _round(row["creditOas"], 1) if row else None,
                    "spx": _round(row["spx"], 2) if row else None,
                    "oil": _round(row["oil"], 2) if row else None,
                    "gold": _round(row["gold"], 2) if row else None,
                    "cpi": _round(row["cpi"], 2) if row else None,
                    "urate": _round(row["urate"], 2) if row else None,
                    "twosTensChange": _round(row["twosTens"] - day0["twosTens"], 1) if row else None,
                    "fivesThirtiesChange": _round(row["fivesThirties"] - day0["fivesThirties"], 1) if row else None,
                    "fnccChange": _round(row["fnccBasis"] - day0["fnccBasis"], 1) if row else None,
                    "fedChange": _round((row["fed"] - day0["fed"]) * 100, 1) if row else None,
                    "tenChange": _round((row["ten"] - day0["ten"]) * 100, 1) if row else None,
                    "mortChange": _round((row["nmcmfr30"] - day0["nmcmfr30"]) * 100, 1) if row else None,
                    "creditOasChange": _round(row["creditOas"] - day0["creditOas"], 1) if row else None,
                    "spxChange": _round((row["spx"] / day0["spx"] - 1) * 100, 1) if row else None,
                    "oilChange": _round((row["oil"] / day0["oil"] - 1) * 100, 1) if row else None,
                    "goldChange": _round((row["gold"] / day0["gold"] - 1) * 100, 1) if row else None,
                    "cpiChange": _round((row["cpi"] - day0["cpi"]) * 100, 1) if row else None,
                    "urateChange": _round((row["urate"] - day0["urate"]) * 100, 1) if row else None,
                }
            )

        def at(day, key):
            point = next((p for p in series if p["day"] == day), None)
            return point[key] if point else None

        latest_day = min(WINDOW_END, (date.fromisoformat(latest) - start).days)
        partial = latest_day < WINDOW_END
        events.append(
            {
                "eventDate": event_date,
                "label": label,
                "short": short,
                "partial": partial,
                "availableThrough": latest if partial else (start + timedelta(days=WINDOW_END)).isoformat(),
                "day0Fed": day0["fed"],
                "day0Ten": day0["ten"],
                "pre60Ten": _round(at(0, "tenChange") - at(-60, "tenChange"), 1),
                "day30Ten": at(30, "tenChange"),
                "day60Ten": at(60, "tenChange"),
                "day120Ten": at(120, "tenChange"),
                "day250Ten": at(250, "tenChange"),
                "day500Ten": at(500, "tenChange"),
                "latestTen": at(latest_day, "tenChange"),
                "series": series,
            }
        )

    return {"latest": latest, "events": events}


def main() -> None:
    payload = build_payload()
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    OUTPUT_PATH.write_text(
        template.replace("/*__DATA__*/", json.dumps(payload)), encoding="utf-8"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
