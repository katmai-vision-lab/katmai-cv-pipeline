import requests
import csv
from bs4 import BeautifulSoup

url = "https://raws.dri.edu/cgi-bin/wea_daysum2.pl"

header = [
    "Time",
    "SolarRad",
    "WindAve",
    "WindDir",
    "WindMax",
    "AirMean",
    "AirMax",
    "AirMin",
    "SoilMean",
    "Humidity",
    "DewPoint",
    "WetBulb",
    "Precip"
]

def scraper(station, year, month, day):
    url = "https://raws.dri.edu/cgi-bin/wea_daysum2.pl"

    payload = {
        "stn": station,
        "mon": f"{month:02d}",
        "day": f"{day:02d}",
        "yea": str(year)[-2:],  # last 2 digits
        "unit": "E",
        "typ": "reg"
    }

    r = requests.post(url, data=payload)
    soup = BeautifulSoup(r.text, "html.parser")

    rows = soup.find_all("tr")

    clean_rows = []

    for row in rows:
        cols = [c.get_text(strip=True) for c in row.find_all("td")]

        if not cols:
            continue

        cols = [c for c in cols if c != ""]

        if len(cols) > 0 and ("am" in cols[0] or "pm" in cols[0]):
            clean_rows.append(cols)

    return clean_rows

def get_weather_data(station, year, month, day):
    print(f"Fetching {station} Weather Data...")

    data = scraper(station, year, month, day)

    filename = f"{station}_{year}_{month:02d}_{day:02d}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    print(f"Saved -> {filename}")
    return