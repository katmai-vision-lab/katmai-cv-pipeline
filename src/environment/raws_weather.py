"""
Fetch hourly weather data from NPS RAWS stations near Brooks Falls.

Three stations are used (all NPS-operated, all within 75 km of Brooks Falls):
  ATHF — Three Forks    ~22 km   (closest)
  ACOV — Coville        ~46 km
  APFA — Pfaff Mine     ~71 km

Data is sourced from the WRCC RAWS daily summary endpoint, which returns one
row per hour (24 rows/day) with wind, temperature, humidity, and precipitation.

Usage:
    python -m src.environment.raws_weather --date 2023-07-15
    python -m src.environment.raws_weather --date 2023-07-15 --station ATHF
    python -m src.environment.raws_weather --date 2023-07-15 --all-stations
    python -m src.environment.raws_weather --date 2023-07-15 --output my_data.json
"""

import argparse
import csv
import json
import sys
from datetime import datetime, date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_DIR

# ── Station registry ──────────────────────────────────────────────────────────

STATIONS = {
    "ATHF": {"name": "Three Forks",  "lat": 58 + 22/60 + 4/3600,  "lon": 155 + 23/60 + 2/3600},
    "ACOV": {"name": "Coville",      "lat": 58 + 48/60 + 9/3600,  "lon": 155 + 33/60 + 46/3600},
    "APFA": {"name": "Pfaff Mine",   "lat": 59 + 6/60  + 37/3600, "lon": 154 + 50/60 + 24/3600},
}

# Column names returned by the RAWS daily summary endpoint.
# The site returns 13 or 14 columns depending on station; we map positionally
# and drop any trailing columns we don't define here.
_COL_NAMES = [
    "time",
    "solar_rad",
    "wind_avg_mph",
    "wind_dir_deg",
    "wind_max_mph",
    "air_temp_mean_f",
    "air_temp_max_f",
    "air_temp_min_f",
    "soil_temp_mean_f",
    "humidity_pct",
    "dew_point_f",
    "wet_bulb_f",
    "precip_in",
    "precip_accum_in",   # present on some stations (14th col)
]

_RAWS_URL = "https://raws.dri.edu/cgi-bin/wea_daysum2.pl"


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_station_day(station_id: str, year: int, month: int, day: int) -> list[dict]:
    """
    Fetch hourly RAWS data for one station and one calendar day.

    Returns a list of dicts (one per hour, 24 entries) keyed by _COL_NAMES.
    Raises RuntimeError if the request succeeds but returns no data rows.
    """
    if station_id not in STATIONS:
        raise ValueError(f"Unknown station '{station_id}'. Valid: {list(STATIONS)}")

    payload = {
        "stn":  station_id,
        "mon":  f"{month:02d}",
        "day":  f"{day:02d}",
        "yea":  str(year)[-2:],
        "unit": "E",
        "typ":  "reg",
    }

    resp = requests.post(_RAWS_URL, data=payload, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        cols = [c for c in cols if c]
        if cols and ("am" in cols[0] or "pm" in cols[0]):
            row = {_COL_NAMES[i]: cols[i] for i in range(min(len(_COL_NAMES), len(cols)))}
            row["station_id"]   = station_id
            row["station_name"] = STATIONS[station_id]["name"]
            row["date"]         = f"{year:04d}-{month:02d}-{day:02d}"
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No data returned for station {station_id} on "
            f"{year:04d}-{month:02d}-{day:02d}. "
            "Check that the date is within the station's data range."
        )

    return rows


def fetch_day(year: int, month: int, day: int,
              station_ids: list[str] | None = None) -> dict:
    """
    Fetch one day's data for one or more stations.

    Args:
        year, month, day: Calendar date.
        station_ids: List of station IDs to query. Defaults to all three stations.

    Returns dict with keys:
        date         — ISO date string
        stations     — dict of {station_id: [hourly rows]}
        errors       — dict of {station_id: error message} for any failed fetches
    """
    station_ids = station_ids or list(STATIONS)
    result = {"date": f"{year:04d}-{month:02d}-{day:02d}", "stations": {}, "errors": {}}

    for sid in station_ids:
        try:
            result["stations"][sid] = fetch_station_day(sid, year, month, day)
            print(f"  ✓ {sid} ({STATIONS[sid]['name']}): {len(result['stations'][sid])} hourly rows")
        except Exception as e:
            result["errors"][sid] = str(e)
            print(f"  ✗ {sid} ({STATIONS[sid]['name']}): {e}", file=sys.stderr)

    return result


# ── Output helpers ────────────────────────────────────────────────────────────

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [row for rows in data["stations"].values() for row in rows]
    if not all_rows:
        print("No data to write to CSV.", file=sys.stderr)
        return
    fieldnames = ["date", "station_id", "station_name"] + _COL_NAMES
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch hourly NPS RAWS weather data for Brooks Falls area stations"
    )
    parser.add_argument(
        "--date", required=True,
        help="Date to fetch in YYYY-MM-DD format (e.g. 2023-07-15)"
    )

    station_group = parser.add_mutually_exclusive_group()
    station_group.add_argument(
        "--station", choices=list(STATIONS), default="ATHF",
        help="Single station to fetch (default: ATHF — Three Forks, closest to Brooks Falls)"
    )
    station_group.add_argument(
        "--all-stations", action="store_true",
        help="Fetch from all three stations (ATHF, ACOV, APFA)"
    )

    parser.add_argument(
        "--output", default=None,
        help="Output file path. Extension determines format: .json or .csv. "
             "Defaults to predictions/raws_weather/<date>_<station>.json"
    )
    parser.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="Output format when --output is not specified (default: json)"
    )

    args = parser.parse_args()

    try:
        d = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Error: invalid date '{args.date}'. Expected YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    station_ids = list(STATIONS) if args.all_stations else [args.station]
    station_label = "all-stations" if args.all_stations else args.station

    print("=" * 60)
    print("RAWS Weather Data Fetch")
    print("=" * 60)
    print(f"Date     : {args.date}")
    station_desc = ", ".join(f"{sid} ({STATIONS[sid]['name']})" for sid in station_ids)
    print(f"Stations : {station_desc}")
    print()

    data = fetch_day(d.year, d.month, d.day, station_ids)

    if args.output:
        out_path = Path(args.output)
        fmt = out_path.suffix.lstrip(".") or args.format
    else:
        out_dir = PREDICTIONS_DIR / "raws_weather"
        out_path = out_dir / f"{args.date}_{station_label}.{args.format}"
        fmt = args.format

    if fmt == "csv":
        save_csv(data, out_path)
    else:
        save_json(data, out_path)

    total_rows = sum(len(v) for v in data["stations"].values())
    print()
    print(f"Output   : {out_path}")
    print(f"Rows     : {total_rows}")
    if data["errors"]:
        print(f"Errors   : {data['errors']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
