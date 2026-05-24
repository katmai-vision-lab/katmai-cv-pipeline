"""
Fetch hydrological data (water level, stream flow, water temperature) from the
nearest USGS stream gauge to Brooks Falls.

The closest active USGS gauge with historical data is the Kvichak River at
Igiugig (site 15300500), ~106 km northwest of Brooks Falls. No USGS gauge
exists on Brooks River itself; this is the best available proxy in the same
Bristol Bay drainage basin. Data is available from 2023 onward at 15-minute
resolution.

Parameters fetched:
  00065 — Gage height (water level), feet          → requirement 7.1
  00060 — Discharge (stream flow), ft³/s           → requirement 7.2
  00010 — Water temperature, °C                    → supplementary

Data source: USGS National Water Information System (NWIS) Instantaneous
Values REST API. No API key required.

Usage:
    python -m src.environment.usgs_hydro --date 2023-07-15
    python -m src.environment.usgs_hydro --date 2023-07-15 --format csv
    python -m src.environment.usgs_hydro --date 2023-07-15 --output my_data.json
    python -m src.environment.usgs_hydro --start 2023-07-01 --end 2023-07-31
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_DIR

# ── Site registry ─────────────────────────────────────────────────────────────

SITE = {
    "site_no":   "15300500",
    "name":      "Kvichak River at Igiugig",
    "lat":       59.328889,
    "lon":       -155.899167,
    "dist_km":   106,
    "data_from": "2023-01-01",
    "note":      "Closest active USGS gauge to Brooks Falls; same Bristol Bay drainage basin.",
}

# USGS parameter codes
PARAMS = {
    "00065": {"name": "gage_height_ft",    "description": "Gage height (water level), feet"},
    "00060": {"name": "discharge_cfs",     "description": "Discharge (stream flow), cubic feet per second"},
    "00010": {"name": "water_temp_c",      "description": "Water temperature, degrees Celsius"},
}

_NWIS_URL = "https://waterservices.usgs.gov/nwis/iv/"

# Qualifier codes returned alongside each value
_QUALIFIERS = {"A": "approved", "P": "provisional", "e": "estimated"}


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_range(start_date: str, end_date: str) -> dict:
    """
    Fetch 15-minute hydrological readings for a date range (inclusive).

    Args:
        start_date: ISO date string, e.g. "2023-07-15"
        end_date:   ISO date string, e.g. "2023-07-15" (same as start for single day)

    Returns dict with keys:
        site         — site metadata
        start_date   — requested start
        end_date     — requested end
        readings     — list of dicts, one per 15-minute interval
        errors       — list of error strings (empty on full success)
    """
    param_str = ",".join(PARAMS.keys())
    resp = requests.get(
        _NWIS_URL,
        params={
            "sites":       SITE["site_no"],
            "format":      "rdb",
            "parameterCd": param_str,
            "startDT":     start_date,
            "endDT":       end_date,
        },
        timeout=30,
    )
    resp.raise_for_status()

    # Parse RDB format: skip # comment lines, first two non-comment lines are
    # column headers and format descriptors
    lines = [l for l in resp.text.split("\n") if not l.startswith("#") and l.strip()]
    if len(lines) < 3:
        return {
            "site": SITE, "start_date": start_date, "end_date": end_date,
            "readings": [],
            "errors": [f"No data returned for {start_date} – {end_date}. "
                       f"Note: data available from {SITE['data_from']} onward."],
        }

    headers = lines[0].split("\t")
    # Dynamically locate columns for each parameter code
    col_map = {}
    for i, h in enumerate(headers):
        for code in PARAMS:
            if h.endswith(code):
                col_map[code] = i
                col_map[code + "_cd"] = i + 1  # qualifier is always the next column

    readings = []
    for line in lines[2:]:  # skip header + format descriptor
        parts = line.split("\t")
        if len(parts) < 4 or parts[0] != "USGS":
            continue
        row = {
            "datetime":  parts[2],
            "timezone":  parts[3],
        }
        for code, idx in col_map.items():
            if code.endswith("_cd"):
                continue
            param_name = PARAMS[code]["name"]
            raw = parts[idx].strip() if idx < len(parts) else ""
            qual_raw = parts[col_map[code + "_cd"]].strip() if col_map.get(code + "_cd", 0) < len(parts) else ""
            row[param_name] = float(raw) if raw else None
            row[param_name + "_qualifier"] = _QUALIFIERS.get(qual_raw, qual_raw)
        readings.append(row)

    return {
        "site":       SITE,
        "start_date": start_date,
        "end_date":   end_date,
        "readings":   readings,
        "errors":     [],
    }


def fetch_day(date_str: str) -> dict:
    """Convenience wrapper for a single calendar day."""
    return fetch_range(date_str, date_str)


# ── Output helpers ────────────────────────────────────────────────────────────

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readings = data.get("readings", [])
    if not readings:
        print("No readings to write.", file=sys.stderr)
        return
    site = data["site"]
    fieldnames = ["site_no", "site_name", "date_range", "datetime", "timezone",
                  "gage_height_ft", "gage_height_ft_qualifier",
                  "discharge_cfs", "discharge_cfs_qualifier",
                  "water_temp_c", "water_temp_c_qualifier"]
    date_range = f"{data['start_date']}_{data['end_date']}"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in readings:
            writer.writerow({
                "site_no":    site["site_no"],
                "site_name":  site["name"],
                "date_range": date_range,
                **r,
            })


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch USGS hydrological data (water level, stream flow, water temp) "
                    "for the closest gauge to Brooks Falls"
    )

    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--date",
        help="Single date to fetch in YYYY-MM-DD format (e.g. 2023-07-15)"
    )
    date_group.add_argument(
        "--start",
        help="Start date for a range fetch (YYYY-MM-DD). Requires --end."
    )
    parser.add_argument(
        "--end",
        help="End date for a range fetch (YYYY-MM-DD). Required with --start."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path. Extension determines format (.json or .csv). "
             "Defaults to predictions/usgs_hydro/<date>.json"
    )
    parser.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="Output format when --output is not specified (default: json)"
    )

    args = parser.parse_args()

    if args.start and not args.end:
        parser.error("--end is required when --start is used.")

    if args.date:
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: invalid date '{args.date}'. Expected YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)
        start_date = end_date = args.date
        date_label = args.date
    else:
        for d, label in [(args.start, "--start"), (args.end, "--end")]:
            try:
                datetime.strptime(d, "%Y-%m-%d")
            except ValueError:
                print(f"Error: invalid date for {label}: '{d}'.", file=sys.stderr)
                sys.exit(1)
        start_date, end_date = args.start, args.end
        date_label = f"{start_date}_{end_date}"

    print("=" * 60)
    print("USGS Hydrological Data Fetch")
    print("=" * 60)
    print(f"Site     : {SITE['site_no']} — {SITE['name']}")
    print(f"Distance : ~{SITE['dist_km']} km from Brooks Falls")
    print(f"Date(s)  : {start_date}" + (f" → {end_date}" if end_date != start_date else ""))
    print(f"Params   : {', '.join(p['name'] for p in PARAMS.values())}")
    print()

    data = fetch_range(start_date, end_date)

    if data["errors"]:
        for e in data["errors"]:
            print(f"  ✗ {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  ✓ {SITE['name']}: {len(data['readings'])} readings "
          f"({len(data['readings']) // 96:.0f}d {len(data['readings']) % 96 * 15:.0f}min)"
          if len(data['readings']) >= 96
          else f"  ✓ {SITE['name']}: {len(data['readings'])} readings")

    if args.output:
        out_path = Path(args.output)
        fmt = out_path.suffix.lstrip(".") or args.format
    else:
        out_dir = PREDICTIONS_DIR / "usgs_hydro"
        out_path = out_dir / f"{date_label}.{args.format}"
        fmt = args.format

    if fmt == "csv":
        save_csv(data, out_path)
    else:
        save_json(data, out_path)

    print()
    print(f"Output   : {out_path}")


if __name__ == "__main__":
    main()
