"""
Fetch NADP precipitation and collector status data for a monitoring site
using a ±2 day window around a target date.

This script retrieves precipitation chemistry monitoring metadata from the
National Atmospheric Deposition Program (NADP) precipitation API endpoint.

Parameters returned may include:
  ppt                  — Reported precipitation, inches
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

Data source:
    NADP API endpoint:
    https://api.slh.wisc.edu/nadp/nadpApiMain.php

Authentication:
    Uses the public NADP payload password required by the endpoint.

Usage:
    python nadp_precip.py
    df = fetch_precip_data("AK97", "2026-04-01")
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# ── API Configuration ─────────────────────────────────────────────────────────

_API_URL = "https://api.slh.wisc.edu/nadp/nadpApiMain.php"

_API_PASSWORD = ")j\\M+w!W%x4o"

# Default timezone offset used by Alaska NADP stations
_DEFAULT_TZ_OFFSET = -9

# ── Core Fetch ────────────────────────────────────────────────────────────────

def fetch_precip_data(site_id: str, date_str: str) -> dict:
    """
    Fetch precipitation monitoring data using a ±2 day window around a date.

    Args:
        site_id:
            NADP monitoring station identifier (e.g. "AK97")

        date_str:
            Target date in YYYY-MM-DD format

    Returns:
        Dict containing:
            site_id      — NADP site identifier
            start_date   — beginning of request window
            end_date     — end of request window
            records      — list of precipitation records
            errors       — list of errors (empty on success)
    """

    # Parse center date
    center_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Build ±2 day range
    start_date = center_date - timedelta(days=2)
    end_date = center_date + timedelta(days=2)

    # API datetime formatting
    start_datetime = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
    end_datetime = end_date.strftime("%Y-%m-%dT23:00:00.000Z")

    payload = {
        "pageId": "PRECIPTABDATA",
        "siteId": site_id,
        "startDateTime": start_datetime,
        "endDateTime": end_datetime,
        "tzOffset": _DEFAULT_TZ_OFFSET,
        "PWD": _API_PASSWORD,
    }

    response = requests.post(_API_URL, data=payload, timeout=30)

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        return {
            "site_id": site_id,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "records": [],
            "errors": [f"HTTP request failed: {e}"],
        }

    # Parse JSON safely
    try:
        json_data = response.json()
    except json.JSONDecodeError:
        return {
            "site_id": site_id,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "records": [],
            "errors": [
                "API returned non-JSON response.",
                response.text[:500],
            ],
        }

    records = json_data.get("data", [])

    if not records:
        return {
            "site_id": site_id,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "records": [],
            "errors": ["No data returned for requested date window."],
        }

    return {
        "site_id": site_id,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "records": records,
        "errors": [],
    }


# ── Output Helpers ────────────────────────────────────────────────────────────

def save_csv(data: dict, output_dir: str = ".") -> Path:
    """
    Save fetched precipitation data as CSV.

    Args:
        data:
            Output dict returned from fetch_precip_data()

        output_dir:
            Directory to save CSV into

    Returns:
        Path to generated CSV file
    """

    records = data.get("records", [])

    if not records:
        raise ValueError("No records available to save.")

    df = pd.DataFrame(records)

    output_path = (
        Path(output_dir)
        / f"{data['site_id']}_precip_{data['start_date']}_to_{data['end_date']}.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    return output_path


# ── Convenience Wrapper ───────────────────────────────────────────────────────

def fetch_and_save(site_id: str, date_str: str, output_dir: str = ".") -> dict:
    """
    Fetch precipitation data and save directly to CSV.

    Returns:
        Dict with:
            data        — fetched precipitation payload
            output_file — CSV file path
    """

    data = fetch_precip_data(site_id, date_str)

    if data["errors"]:
        return {
            "data": data,
            "output_file": None,
        }

    output_file = save_csv(data, output_dir)

    return {
        "data": data,
        "output_file": str(output_file),
    }


# ── Example CLI Usage ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    SITE_ID = "AK97"
    DATE = "2026-04-01"

    print("=" * 60)
    print("NADP Precipitation Data Fetch")
    print("=" * 60)
    print(f"Site      : {SITE_ID}")
    print(f"Target Day: {DATE}")
    print("Window    : ±2 days")
    print()

    result = fetch_and_save(SITE_ID, DATE)

    data = result["data"]

    if data["errors"]:
        print("Errors:")
        for err in data["errors"]:
            print(f"  ✗ {err}")
    else:
        print(f"  ✓ Retrieved {len(data['records'])} records")
        print(f"Output    : {result['output_file']}")