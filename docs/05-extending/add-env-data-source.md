# Add a New Environmental Data Source

The environment module (`src/environment/`) fetches data from public APIs and returns structured dicts. Adding a new data source — a different river gauge, a new weather network, a tide table — follows the same three-step pattern as the existing modules.

---

## How the existing modules are structured

Look at `src/environment/usgs_hydro.py` as the reference:

1. **A fetch function** that calls an external API with a date or date range and returns a dict
2. **A CLI entry point** (`if __name__ == "__main__"`) for standalone use
3. **Caching** — results are saved to `data/environment/` on first fetch and reloaded on subsequent calls

The `video_context.py` module resolves the video's datetime and passes it to each fetch function.

---

## Step 1 — Create the module file

Create `src/environment/my_source.py`:

```python
"""
Fetch [description] from [source name].
"""
import argparse
import json
from datetime import date, datetime
from pathlib import Path

import requests

CACHE_DIR = Path("data/environment/my_source")
API_BASE = "https://api.example.com/data"


def fetch(target_date: date, use_cache: bool = True) -> dict:
    """
    Fetch data for target_date. Returns a dict with a 'measurements' key.
    """
    cache_file = CACHE_DIR / f"{target_date.isoformat()}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    resp = requests.get(API_BASE, params={
        "date": target_date.isoformat(),
        # add API-specific params here
    }, timeout=30)
    resp.raise_for_status()

    result = {
        "date": target_date.isoformat(),
        "source": "my_source",
        "measurements": _parse(resp.json()),
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, indent=2))
    return result


def _parse(raw: dict) -> dict:
    # extract only what you need from the raw API response
    return {
        "my_metric": raw["path"]["to"]["value"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    result = fetch(date.fromisoformat(args.date))
    print(json.dumps(result, indent=2))
```

---

## Step 2 — Wire it into video context enrichment

Open `src/environment/video_context.py` and find the `enrich()` function. Add your source alongside the existing ones:

```python
from .my_source import fetch as fetch_my_source

def enrich(context: VideoContext, analysis_json: dict) -> dict:
    d = context.recorded_at.date()
    analysis_json["environmental_context"] = {
        "usgs": fetch_usgs(d),
        "raws": fetch_raws(d),
        "nadp": fetch_nadp(d),
        "my_source": fetch_my_source(d),   # ADD THIS
    }
    return analysis_json
```

---

## Step 3 — Add a CLI entry in the TUI (optional)

If you want the new source accessible from the interactive menu, open `src/cli.py` and find the environmental data menu section. Add a case for your source following the same pattern as the USGS / RAWS entries.

---

## Design guidelines

- **Always cache** — the APIs are slow; repeated fetches from the same date should hit disk
- **Always handle gaps** — return `None` for measurements that are missing; don't raise exceptions mid-pipeline
- **Date alignment** — all modules align to UTC date; normalize your API's timestamps accordingly
- **No API keys** — the existing modules use entirely public, keyless APIs; prefer sources that don't require registration
