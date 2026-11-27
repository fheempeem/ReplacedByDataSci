"""Scrape #0: camera coordinates -> bronze.camera_locations.

Source 1 (preferred, coord_source='bma'): BMA traffic platform public page http://www.bmatraffic.com/index.aspx
  The page embeds one row per camera: [id, code+name, name_en, intersection, direction, lat, lng, <internal ip>, icon].
  The internal IP column is deliberately NOT stored.
  !! The list the page returns is NOT stable: the same URL gave 372 rows at one moment and 291 minutes later.
  So we (a) fetch several times and union the rows, and (b) keep every camera ever seen in
  reference/bma_camera_coordinates.csv (checked in), which is merged in first. A fresh clone therefore reproduces the
  same coordinates even when the live page happens to omit cameras.
Source 2 (fallback, coord_source='nominatim'): OSM Nominatim geocoding of the camera name, only for cameras absent
  from every BMA source, and accepted only inside a Bangkok bounding box.
Cameras neither source resolves keep NULL lat/lng and are excluded from the map only.

Re-runs: cameras already resolved from BMA are kept. Cameras that are NULL or only Nominatim-resolved are checked against
BMA again (upgraded to 'bma' if BMA lists them now) but are NOT sent to Nominatim a second time.
The raw HTML of every fetch is saved under data/raw_scrape/.
"""
from __future__ import annotations

import csv
import re
import time
from datetime import datetime

import pyarrow as pa
import requests
from pyducklake import DoubleType, Schema, StringType, TimestampTZType, optional, required

from common import BKK, ROOT, open_catalog

BMA_URL = "http://www.bmatraffic.com/index.aspx"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
UA = "bkk-cctv-course-project/0.1 (university data science coursework)"
BBOX = (13.45, 14.05, 100.30, 100.95)  # lat_min, lat_max, lon_min, lon_max
TABLE = ("bronze", "camera_locations")
RAW_SCRAPE = ROOT / "data" / "raw_scrape"
REFERENCE_CSV = ROOT / "reference" / "bma_camera_coordinates.csv"
BMA_FETCHES = 3
BMA_FETCH_GAP_S = 2

SCHEMA = Schema.of(
    required("camera_id", StringType()),
    optional("camera_code", StringType()),
    optional("name_th", StringType()),
    optional("name_en", StringType()),
    optional("direction_en", StringType()),
    optional("lat", DoubleType()),
    optional("lng", DoubleType()),
    optional("coord_source", StringType()),
    required("source_url", StringType()),
    required("scraped_at", TimestampTZType()),
)

ROW_RE = re.compile(
    r"^\s*\['(?P<id>\d+)'\s*,\s*'(?P<code>.*?)'\s*,\s*'(?P<en>.*?)'\s*,\s*'(?P<th>.*?)'\s*,\s*'(?P<dir>.*?)'\s*,"
    r"\s*(?P<lat>-?\d+\.\d+)\s*,\s*(?P<lng>-?\d+\.\d+)\s*,",
    re.M,
)
REF_FIELDS = ["camera_id", "code", "name_en", "name_th", "direction", "lat", "lng"]


def parse_bma(html: str) -> dict[str, dict]:
    out = {}
    for m in ROW_RE.finditer(html):
        out[m["id"]] = {
            "code": m["code"].strip(), "name_en": m["en"].replace("\xa0", " ").strip(),
            "name_th": m["th"].strip(), "direction": m["dir"].strip(),
            "lat": float(m["lat"]), "lng": float(m["lng"]),
        }
    return out


def load_reference() -> dict[str, dict]:
    if not REFERENCE_CSV.exists():
        return {}
    with open(REFERENCE_CSV, encoding="utf-8", newline="") as f:
        return {
            r["camera_id"]: {"code": r["code"], "name_en": r["name_en"], "name_th": r["name_th"],
                             "direction": r["direction"], "lat": float(r["lat"]), "lng": float(r["lng"])}
            for r in csv.DictReader(f)
        }


def save_reference(rows: dict[str, dict]) -> None:
    REFERENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(REF_FIELDS)
        for cid in sorted(rows, key=int):
            r = rows[cid]
            w.writerow([cid, r["code"], r["name_en"], r["name_th"], r["direction"], r["lat"], r["lng"]])


def fetch_bma() -> tuple[dict[str, dict], dict]:
    """Union of the reference file and several live fetches. Returns (rows, stats)."""
    rows = load_reference()
    stats = {"reference_rows": len(rows), "live_fetch_sizes": []}
    RAW_SCRAPE.mkdir(parents=True, exist_ok=True)
    for i in range(BMA_FETCHES):
        if i:
            time.sleep(BMA_FETCH_GAP_S)
        try:
            resp = requests.get(BMA_URL, headers={"User-Agent": UA}, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            stats["live_fetch_sizes"].append(f"error: {type(e).__name__}")
            continue
        stamp = datetime.now(BKK).strftime("%Y%m%d_%H%M%S")
        (RAW_SCRAPE / f"bma_index_{stamp}_{i}.html").write_text(resp.text, encoding="utf-8")
        live = parse_bma(resp.text)
        stats["live_fetch_sizes"].append(len(live))
        rows.update(live)  # live wins over the reference
    stats["union_rows"] = len(rows)
    if rows:
        save_reference(rows)
    return rows, stats


def geocode(name_th: str) -> tuple[float, float] | None:
    q = f"{name_th.replace('-', ' ')}, กรุงเทพมหานคร"
    r = requests.get(
        NOMINATIM_URL,
        params={"q": q, "format": "jsonv2", "limit": 1, "countrycodes": "th", "accept-language": "th"},
        headers={"User-Agent": UA},
        timeout=30,
    )
    r.raise_for_status()
    hits = r.json()
    if not hits:
        return None
    lat, lon = float(hits[0]["lat"]), float(hits[0]["lon"])
    if BBOX[0] <= lat <= BBOX[1] and BBOX[2] <= lon <= BBOX[3]:
        return lat, lon
    return None


def run() -> dict:
    catalog = open_catalog()
    if "bronze" not in catalog.list_namespaces():
        catalog.create_namespace("bronze")
    try:
        table = catalog.load_table(TABLE)
    except Exception:
        table = catalog.create_table(TABLE, SCHEMA)
    con = catalog.connection

    # cameras still to resolve: never seen, or seen but not (yet) resolved from BMA
    cams = con.sql(
        """
        SELECT i.camera_id, i.camera_name
        FROM (SELECT camera_id, any_value(camera_name) AS camera_name FROM bkk.bronze.cctv_images GROUP BY 1) i
        LEFT JOIN bkk.bronze.camera_locations l USING (camera_id)
        WHERE l.camera_id IS NULL OR l.coord_source IS DISTINCT FROM 'bma'
        ORDER BY 1
        """
    ).df()
    if cams.empty:
        summary = {"changed": 0, "table_rows": table.scan().count()}
        print(summary)
        return summary

    bma, stats = fetch_bma()
    print({"bma": stats}, flush=True)
    existing = {r[0]: r[1] for r in con.sql("SELECT camera_id, coord_source FROM bkk.bronze.camera_locations").fetchall()}
    now = datetime.now(BKK)
    rows, upgraded = [], 0
    for cid, name in zip(cams["camera_id"], cams["camera_name"]):
        if cid in bma:
            b = bma[cid]
            rows.append((cid, b["code"], b["name_th"], b["name_en"], b["direction"], b["lat"], b["lng"], "bma", BMA_URL))
            upgraded += cid in existing
            continue
        if cid in existing:  # already tried (resolved or not): do not ask Nominatim again on every run
            continue
        time.sleep(1.1)  # Nominatim usage policy: max 1 request/second
        hit = geocode(name)
        if hit:
            rows.append((cid, None, name, None, None, hit[0], hit[1], "nominatim", NOMINATIM_URL))
        else:
            rows.append((cid, None, name, None, None, None, None, None, NOMINATIM_URL))

    if rows:
        ids = [r[0] for r in rows]
        tz = pa.timestamp("us", tz="Asia/Bangkok")
        batch = pa.table(
            {
                "camera_id": pa.array(ids, pa.string()),
                "camera_code": pa.array([r[1] for r in rows], pa.string()),
                "name_th": pa.array([r[2] for r in rows], pa.string()),
                "name_en": pa.array([r[3] for r in rows], pa.string()),
                "direction_en": pa.array([r[4] for r in rows], pa.string()),
                "lat": pa.array([r[5] for r in rows], pa.float64()),
                "lng": pa.array([r[6] for r in rows], pa.float64()),
                "coord_source": pa.array([r[7] for r in rows], pa.string()),
                "source_url": pa.array([r[8] for r in rows], pa.string()),
                "scraped_at": pa.array([now] * len(rows), tz),
            }
        )
        with catalog.begin_transaction() as txn:
            catalog.set_commit_message(f"camera locations: {len(rows)} changed", author="scrape_cameras.py")
            t = txn.load_table(TABLE)
            stale = [i for i in ids if i in existing]
            if stale:
                t.delete("camera_id IN (" + ", ".join(f"'{i}'" for i in stale) + ")")
            t.append(batch)

    final = dict(con.sql("SELECT coalesce(coord_source, 'none'), count(*) FROM bkk.bronze.camera_locations GROUP BY 1").fetchall())
    summary = {"changed": len(rows), "upgraded_to_bma": upgraded, "by_source": final, "table_rows": table.scan().count()}
    print(summary)
    return summary


if __name__ == "__main__":
    run()
