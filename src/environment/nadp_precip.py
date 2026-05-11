"""
Fetch daily precipitation data from NADP NTN station AK97 near Brooks Falls.

AK97 is the sponsor-confirmed NADP National Trends Network station for the
Brooks Falls area. It provides daily precipitation totals and collector
diagnostics from the NADP precipitation chemistry monitoring network.

Coordinates for AK97 are not exposed by the NADP API; the station is located
in southwest Alaska (sponsor-confirmed as the appropriate proxy station).

Parameters returned per day:
  ppt                  — Daily precipitation, inches
  actDepth             — Gauge bucket depth, inches
  minVolt              — Minimum datalogger voltage
  minTemp              — Minimum panel temperature, °C
  maxTemp              — Maximum panel temperature, °C
  opdCounts            — Optical sensor detection counts
  coll1Cycles          — Collector 1 open/close cycles
  exp1                 — Collector 1 exposure time, hours
  coll2Cycles          — Collector 2 open/close cycles
  exp2                 — Collector 2 exposure time, hours
  coll3Cycles          — Collector 3 open/close cycles
  exp3                 — Collector 3 exposure time, hours
  pptDataCompleteness  — Data completeness fraction (e.g. "96/96")

Usage:
    python -m src.environment.nadp_precip --date 2023-07-15
    python -m src.environment.nadp_precip --date 2023-07-15 --output my_data.csv --format csv
    python -m src.environment.nadp_precip --start 2023-07-01 --end 2023-07-31
"""

import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_DIR

# ── Station registry ──────────────────────────────────────────────────────────

STATIONS = {
    "AK97": {
        "name": "AK97 (Southwest Alaska)",
        "lat": None,   # not exposed by NADP API; sponsor-confirmed station
        "lon": None,
        "tz_offset": -9,
    },
}

_COL_NAMES = [
    "dt",
    "day",
    "ppt",
    "actDepth",
    "minVolt",
    "minTemp",
    "maxTemp",
    "opdCounts",
    "coll1Cycles",
    "exp1",
    "coll2Cycles",
    "exp2",
    "coll3Cycles",
    "exp3",
    "pptDataCompleteness",
]

_API_URL = "https://api.slh.wisc.edu/nadp/nadpApiMain.php"
_API_PASSWORD = ")j\\M+w!W%x4o"   # public NADP payload credential


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_station_day(site_id: str, year: int, month: int, day: int) -> list[dict]:
    """
    Fetch NADP precipitation data for one station and one calendar day.

    Returns a list with one dict (one record per day).
    Raises RuntimeError if the request succeeds but returns no data rows.
    """
    if site_id not in STATIONS:
        raise ValueError(f"Unknown site '{site_id}'. Valid: {list(STATIONS)}")

    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    tz = STATIONS[site_id]["tz_offset"]

    payload = {
        "pageId": "PRECIPTABDATA",
        "siteId": site_id,
        "startDateTime": f"{date_str}T00:00:00.000Z",
        "endDateTime": f"{date_str}T23:00:00.000Z",
        "tzOffset": tz,
        "PWD": _API_PASSWORD,
    }

    resp = requests.post(_API_URL, data=payload, timeout=30)
    resp.raise_for_status()

    try:
        json_data = resp.json()
    except Exception as e:
        raise RuntimeError(
            f"NADP API returned non-JSON for {site_id} on {date_str}: {e}\n"
            f"Response: {resp.text[:300]}"
        )

    records = json_data.get("data", [])
    if not records:
        raise RuntimeError(
            f"No data returned for site {site_id} on {date_str}. "
            "Check that the date is within the station's data range."
        )

    for rec in records:
        rec["site_id"]   = site_id
        rec["site_name"] = STATIONS[site_id]["name"]
        rec["date"]      = date_str

    return records


def fetch_day(year: int, month: int, day: int,
              site_ids: list[str] | None = None) -> dict:
    """
    Fetch one day's data for one or more NADP sites.

    Returns dict with keys:
        date     — ISO date string
        sites    — dict of {site_id: [daily records]}
        errors   — dict of {site_id: error message} for any failed fetches
    """
    site_ids = site_ids or list(STATIONS)
    result = {"date": f"{year:04d}-{month:02d}-{day:02d}", "sites": {}, "errors": {}}

    for sid in site_ids:
        try:
            result["sites"][sid] = fetch_station_day(sid, year, month, day)
            print(f"  ✓ {sid} ({STATIONS[sid]['name']}): {len(result['sites'][sid])} record(s)")
        except Exception as e:
            result["errors"][sid] = str(e)
            print(f"  ✗ {sid} ({STATIONS[sid]['name']}): {e}", file=sys.stderr)

    return result


def fetch_range(start: date, end: date,
                site_ids: list[str] | None = None) -> dict:
    """
    Fetch data for a date range (inclusive) for one or more NADP sites.

    Returns dict with keys:
        start_date   — ISO date string
        end_date     — ISO date string
        sites        — dict of {site_id: [daily records across all days]}
        errors       — dict of {site_id: [error messages per failed day]}
    """
    site_ids = site_ids or list(STATIONS)
    result = {
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "sites":      {sid: [] for sid in site_ids},
        "errors":     {sid: [] for sid in site_ids},
    }

    current = start
    while current <= end:
        for sid in site_ids:
            try:
                records = fetch_station_day(sid, current.year, current.month, current.day)
                result["sites"][sid].extend(records)
            except Exception as e:
                result["errors"][sid].append(f"{current.isoformat()}: {e}")
        current += timedelta(days=1)

    for sid in site_ids:
        if not result["errors"][sid]:
            del result["errors"][sid]
        n = len(result["sites"][sid])
        status = "✓" if sid not in result["errors"] else "✗"
        print(f"  {status} {sid}: {n} record(s)")

    return result


# ── Output helpers ────────────────────────────────────────────────────────────

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    sites_key = "sites" if "sites" in data else "stations"
    all_rows = [row for rows in data[sites_key].values() for row in rows]
    if not all_rows:
        print("No data to write to CSV.", file=sys.stderr)
        return

    fieldnames = ["date", "site_id", "site_name"] + _COL_NAMES
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch daily NADP NTN precipitation data for Brooks Falls area"
    )

    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--date",
        help="Single date to fetch in YYYY-MM-DD format (e.g. 2023-07-15)"
    )
    date_group.add_argument(
        "--start",
        help="Start of date range in YYYY-MM-DD (use with --end)"
    )

    parser.add_argument(
        "--end",
        help="End of date range in YYYY-MM-DD (required with --start)"
    )
    parser.add_argument(
        "--site", choices=list(STATIONS), default="AK97",
        help="NADP site ID to fetch (default: AK97)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path. Extension determines format: .json or .csv. "
             "Defaults to predictions/nadp_precip/<date>_<site>.<format>"
    )
    parser.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="Output format when --output is not specified (default: json)"
    )

    args = parser.parse_args()

    if args.start and not args.end:
        parser.error("--end is required when --start is specified")

    print("=" * 60)
    print("NADP Precipitation Data Fetch")
    print("=" * 60)

    out_dir = PREDICTIONS_DIR / "nadp_precip"

    if args.date:
        try:
            d = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Error: invalid date '{args.date}'. Expected YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)

        print(f"Date : {args.date}")
        print(f"Site : {args.site} ({STATIONS[args.site]['name']})")
        print()

        data = fetch_day(d.year, d.month, d.day, [args.site])
        label = args.date

    else:
        try:
            start = datetime.strptime(args.start, "%Y-%m-%d").date()
            end   = datetime.strptime(args.end,   "%Y-%m-%d").date()
        except ValueError as e:
            print(f"Error: invalid date — {e}", file=sys.stderr)
            sys.exit(1)

        if start > end:
            print("Error: --start must be before --end.", file=sys.stderr)
            sys.exit(1)

        print(f"Range: {args.start} → {args.end}")
        print(f"Site : {args.site} ({STATIONS[args.site]['name']})")
        print()

        data = fetch_range(start, end, [args.site])
        label = f"{args.start}_to_{args.end}"

    if args.output:
        out_path = Path(args.output)
        fmt = out_path.suffix.lstrip(".") or args.format
    else:
        out_path = out_dir / f"{label}_{args.site}.{args.format}"
        fmt = args.format

    if fmt == "csv":
        save_csv(data, out_path)
    else:
        save_json(data, out_path)

    sites_key = "sites" if "sites" in data else "stations"
    total_rows = sum(len(v) for v in data[sites_key].values())
    print()
    print(f"Output : {out_path}")
    print(f"Records: {total_rows}")

    if data.get("errors"):
        print(f"Errors : {data['errors']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
