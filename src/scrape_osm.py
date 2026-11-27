"""Scrape #3: OpenStreetMap points of interest around the cameras -> bronze.osm_poi.

One Overpass query for the whole camera bounding box (kind to the public server). Distances / counts per camera
within a radius are computed later in gold, so bronze stays a raw copy of what OSM returned.

Categories: school, university, hospital, mall (mall/department store/supermarket), market, bus_stop,
            rail_station, traffic_signal.
The raw Overpass JSON is saved under data/raw_scrape/. Idempotent: skipped if the table already has rows.
"""
from __future__ import annotations

import json
from datetime import datetime

import pyarrow as pa
import requests
from pyducklake import DoubleType, Schema, StringType, TimestampTZType, optional, required

from common import BKK, ROOT, open_catalog

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
UA = "bkk-cctv-course-project/0.1 (university data science coursework)"
TABLE = ("bronze", "osm_poi")
RAW_SCRAPE = ROOT / "data" / "raw_scrape"
BBOX = (13.62, 100.33, 13.90, 100.70)  # south, west, north, east

QUERY = """
[out:json][timeout:180];
(
  nwr["amenity"~"^(school|university|college|hospital|marketplace)$"]({b});
  nwr["shop"~"^(mall|department_store|supermarket)$"]({b});
  nwr["railway"="station"]({b});
  node["highway"="bus_stop"]({b});
  node["highway"="traffic_signals"]({b});
);
out center tags;
"""

SCHEMA = Schema.of(
    required("osm_type", StringType()),
    required("osm_id", StringType()),
    required("category", StringType()),
    optional("name", StringType()),
    required("lat", DoubleType()),
    required("lng", DoubleType()),
    required("scraped_at", TimestampTZType()),
)


def categorize(tags: dict) -> str | None:
    a, s = tags.get("amenity"), tags.get("shop")
    if a in ("school",):
        return "school"
    if a in ("university", "college"):
        return "university"
    if a == "hospital":
        return "hospital"
    if a == "marketplace":
        return "market"
    if s in ("mall", "department_store", "supermarket"):
        return "mall"
    if tags.get("railway") == "station":
        return "rail_station"
    if tags.get("highway") == "bus_stop":
        return "bus_stop"
    if tags.get("highway") == "traffic_signals":
        return "traffic_signal"
    return None


def run() -> dict:
    catalog = open_catalog()
    if "bronze" not in catalog.list_namespaces():
        catalog.create_namespace("bronze")
    try:
        table = catalog.load_table(TABLE)
    except Exception:
        table = catalog.create_table(TABLE, SCHEMA)
    if table.scan().count() > 0:
        summary = {"loaded": 0, "table_rows": table.scan().count()}
        print(summary)
        return summary

    # Fixed bounding box (central Bangkok), NOT derived from the cameras that happen to have coordinates: that made the
    # result depend on how many coordinates the BMA page returned on the day. Covers every camera we have seen + margin.
    s, w, n, e = BBOX
    b = f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}"
    resp = requests.post(OVERPASS_URL, data={"data": QUERY.format(b=b)}, headers={"User-Agent": UA}, timeout=240)
    resp.raise_for_status()
    RAW_SCRAPE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(BKK).strftime("%Y%m%d_%H%M%S")
    (RAW_SCRAPE / f"osm_poi_{stamp}.json").write_text(resp.text, encoding="utf-8")

    rows = []
    for el in resp.json()["elements"]:
        tags = el.get("tags", {})
        cat = categorize(tags)
        pos = el if "lat" in el else el.get("center")
        if cat is None or not pos:
            continue
        rows.append((el["type"], str(el["id"]), cat, tags.get("name"), float(pos["lat"]), float(pos["lon"])))

    now = datetime.now(BKK)
    tz = pa.timestamp("us", tz="Asia/Bangkok")
    batch = pa.table(
        {
            "osm_type": pa.array([r[0] for r in rows], pa.string()),
            "osm_id": pa.array([r[1] for r in rows], pa.string()),
            "category": pa.array([r[2] for r in rows], pa.string()),
            "name": pa.array([r[3] for r in rows], pa.string()),
            "lat": pa.array([r[4] for r in rows], pa.float64()),
            "lng": pa.array([r[5] for r in rows], pa.float64()),
            "scraped_at": pa.array([now] * len(rows), tz),
        }
    )
    with catalog.begin_transaction() as txn:
        catalog.set_commit_message(f"osm poi: {len(rows)}", author="scrape_osm.py")
        txn.load_table(TABLE).append(batch)
    summary = {"loaded": len(rows), "table_rows": table.scan().count(), "bbox": b}
    print(summary)
    return summary


if __name__ == "__main__":
    run()
