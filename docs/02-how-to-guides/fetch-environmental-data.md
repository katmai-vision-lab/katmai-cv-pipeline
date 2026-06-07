# Fetch Environmental Data

The environment module fetches publicly available data from three sources — USGS river hydrology, RAWS weather stations, and NADP precipitation — and aligns it to a video's recording date. No API keys required.

---

## Resolve a video's date and location

Before fetching data, the pipeline needs to know when a video was recorded. Camera recordings embed this in metadata; screen recordings (Explore.org downloads) require manual input.

```bash
python -m src.environment.video_context --video path/to/clip.mp4
```

The script auto-detects whether the file is a direct camera recording or a screen capture and either extracts the datetime from EXIF/media metadata or prompts you to enter it manually. Output is a `VideoContext` object used by the fetch commands below.

---

## USGS hydrological data

Fetches river water level, stream flow, and temperature from the Kvichak River gauge at Igiugig (USGS site 15300500 — the most relevant gauge for Brooks Falls salmon run timing).

```bash
# Single date
python -m src.environment.usgs_hydro --date 2023-07-15

# Date range
python -m src.environment.usgs_hydro \
    --start 2023-07-01 \
    --end 2023-07-31 \
    --output data/environment/kvichak_july2023.json
```

Output fields: `datetime`, `discharge_cfs` (stream flow), `gauge_height_ft`, `water_temp_c`.

---

## RAWS weather data

Fetches temperature, humidity, wind speed/direction, and precipitation from Remote Automated Weather Stations near Katmai. Three stations are available:

| Station | Distance to Brooks Falls | Best for |
|---|---|---|
| Three Forks | ~22 km | Default (closest) |
| Coville | ~46 km | Cross-check |
| Pfaff Mine | ~71 km | Fallback if Three Forks is down |

```bash
# Nearest station only (default)
python -m src.environment.raws_weather --date 2023-07-15

# All stations
python -m src.environment.raws_weather --date 2023-07-15 --all-stations
```

---

## NADP precipitation data

Fetches daily precipitation totals from the NADP network, station AK97 (King Salmon, AK — closest to Brooks Falls).

```bash
python -m src.environment.nadp_precip --date 2023-07-15

python -m src.environment.nadp_precip \
    --start 2023-07-01 \
    --end 2023-07-31
```

---

## Correlate environmental data with a video analysis

After running feeding behavior analysis, attach environmental context to the output:

```bash
python -m src.environment.video_context \
    --video path/to/clip.mp4 \
    --analysis predictions/<stem>_feeding_analysis/analysis.json \
    --enrich
```

This appends an `environmental_context` block to `analysis.json` with the weather, flow, and precipitation values matching the video's recording datetime.

---

## Output format

All environment fetches return JSON:

```json
{
  "date": "2023-07-15",
  "source": "USGS-15300500",
  "measurements": {
    "discharge_cfs": 4820,
    "gauge_height_ft": 3.41,
    "water_temp_c": 8.2
  }
}
```

---

## Data availability notes

- **USGS**: data goes back to the 1940s for the Kvichak gauge. Gaps exist during sensor outages.
- **RAWS**: availability varies by station; Three Forks has the most complete record.
- **NADP**: daily resolution only (no hourly). Gaps occur during instrument maintenance periods.
- All fetches are cached locally after the first request to avoid redundant API calls.
