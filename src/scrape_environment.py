"""Scrape #1 and #2: hourly weather (rain) and PM2.5 -> bronze.weather_hourly, bronze.air_quality_hourly.

Both come from Open-Meteo public APIs (no key). They are two separate sources/endpoints:
  weather      https://archive-api.open-meteo.com/v1/archive      (reanalysis; precipitation, temperature, ...)
  air quality  https://air-quality-api.open-meteo.com/v1/air-quality (CAMS model; PM2.5 / PM10 - modelled, not station readings)
Why not Air4Thai station data: its history API returns no rows before 2026-09-01, i.e. nothing for our 25-31 Aug window.

Fetched on a 3x3 grid over Bangkok so rain/PM can differ between areas; gold assigns each camera to its nearest
grid point (cameras without coordinates get the grid mean).
The raw JSON responses are saved under data/raw_scrape/. Idempotent: a table that already has rows is skipped.
"""
from __future__ import annotations

import json
from datetime import datetime

import pyarrow as pa
import requests
from pyducklake import DoubleType, IntegerType, Schema, StringType, TimestampTZType, optional, required

from common import BKK, ROOT, open_catalog

UA = "bkk-cctv-course-project/0.1 (university data science coursework)"
START, END = "2026-08-25", "2026-08-31"
LATS = [13.68, 13.75, 13.82]
LONS = [100.42, 100.52, 100.62]
GRID = [(f"g{i}{j}", la, lo) for i, la in enumerate(LATS) for j, lo in enumerate(LONS)]
RAW_SCRAPE = ROOT / "data" / "raw_scrape"

SOURCES = {
    "weather": {
        "table": ("bronze", "weather_hourly"),
        "url": "https://archive-api.open-meteo.com/v1/archive",
        "vars": ["precipitation", "rain", "temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m"],
        "cols": ["precipitation_mm", "rain_mm", "temperature_c", "humidity_pct", "cloud_cover_pct", "wind_speed_kmh"],
    },
    "air_quality": {
        "table": ("bronze", "air_quality_hourly"),
        "url": "https://air-quality-api.open-meteo.com/v1/air-quality",
        "vars": ["pm2_5", "pm10", "us_aqi"],
        "cols": ["pm2_5", "pm10", "us_aqi"],
    },
}


def _schema(cols: list[str]) -> Schema:
    fields = [
        required("grid_id", StringType()),
        required("grid_lat", DoubleType()),
        required("grid_lng", DoubleType()),
        required("ts", TimestampTZType()),
    ]
    fields += [optional(c, DoubleType()) for c in cols]
    fields += [required("source_url", StringType()), required("scraped_at", TimestampTZType())]
    return Schema.of(*fields)


def _fetch(src: dict, name: str) -> list[dict]:
    r = requests.get(
        src["url"],
        params={
            "latitude": ",".join(str(g[1]) for g in GRID),
            "longitude": ",".join(str(g[2]) for g in GRID),
            "hourly": ",".join(src["vars"]),
            "start_date": START,
            "end_date": END,
            "timezone": "Asia/Bangkok",
        },
        headers={"User-Agent": UA},
        timeout=60,
    )
    r.raise_for_status()
    RAW_SCRAPE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(BKK).strftime("%Y%m%d_%H%M%S")
    (RAW_SCRAPE / f"{name}_{stamp}.json").write_text(r.text, encoding="utf-8")
    data = r.json()
    return data if isinstance(data, list) else [data]


def _load(catalog, name: str) -> int:
    src = SOURCES[name]
    ident = src["table"]
    if ident[0] not in catalog.list_namespaces():
        catalog.create_namespace(ident[0])
    try:
        table = catalog.load_table(ident)
    except Exception:
        table = catalog.create_table(ident, _schema(src["cols"]))
    if table.scan().count() > 0:
        print(f"{name}: already loaded ({table.scan().count()} rows), skipping")
        return 0

    payloads = _fetch(src, name)
    assert len(payloads) == len(GRID), f"expected {len(GRID)} locations, got {len(payloads)}"
    now = datetime.now(BKK)
    tz = pa.timestamp("us", tz="Asia/Bangkok")
    cols: dict[str, list] = {"grid_id": [], "grid_lat": [], "grid_lng": [], "ts": []}
    for c in src["cols"]:
        cols[c] = []
    for (gid, la, lo), p in zip(GRID, payloads):
        h = p["hourly"]
        for i, t in enumerate(h["time"]):
            cols["grid_id"].append(gid)
            cols["grid_lat"].append(p["latitude"])  # snapped model-grid coordinate returned by the API
            cols["grid_lng"].append(p["longitude"])
            cols["ts"].append(datetime.fromisoformat(t).replace(tzinfo=BKK))
            for var, c in zip(src["vars"], src["cols"]):
                v = h[var][i]
                cols[c].append(None if v is None else float(v))
    n = len(cols["ts"])
    arrays = {
        "grid_id": pa.array(cols["grid_id"], pa.string()),
        "grid_lat": pa.array(cols["grid_lat"], pa.float64()),
        "grid_lng": pa.array(cols["grid_lng"], pa.float64()),
        "ts": pa.array(cols["ts"], tz),
        **{c: pa.array(cols[c], pa.float64()) for c in src["cols"]},
        "source_url": pa.array([src["url"]] * n, pa.string()),
        "scraped_at": pa.array([now] * n, tz),
    }
    with catalog.begin_transaction() as txn:
        catalog.set_commit_message(f"{name}: {n} rows", author="scrape_environment.py")
        txn.load_table(ident).append(pa.table(arrays))
    print(f"{name}: loaded {n} rows")
    return n


def run() -> dict:
    catalog = open_catalog()
    out = {name: _load(catalog, name) for name in SOURCES}
    print(out)
    return out


if __name__ == "__main__":
    run()
