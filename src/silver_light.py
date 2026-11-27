"""Silver stage 3: light regime from solar elevation -> silver.light_regime.

Why: YOLO under-counts vehicles after dusk (small vehicles show only as headlights), and the
brightness flag in silver.image_features does not separate day from night (street lights keep the
mean brightness high). Sun position is an objective, data-source-free proxy.

Regime (solar elevation at Bangkok city centre, NOAA formulas, no refraction correction beyond -0.833):
  day       elevation >  -0.833 deg   (sun above horizon)
  twilight  -6 < elevation <= -0.833  (civil twilight)
  night     elevation <= -6
`is_dark` = regime != 'day'. Gold computes the congestion index separately per regime group.

Idempotent: only images missing from silver.light_regime are computed.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pyarrow as pa
from pyducklake import BooleanType, FloatType, Schema, StringType, required

from common import open_catalog

LAT, LON = 13.7563, 100.5018  # Bangkok
SUNSET_DEG = -0.833
CIVIL_DEG = -6.0
TABLE = ("silver", "light_regime")

SCHEMA = Schema.of(
    required("image_id", StringType()),
    required("solar_elevation_deg", FloatType()),
    required("light_regime", StringType()),
    required("is_dark", BooleanType()),
)


def solar_elevation(dt: datetime, lat: float = LAT, lon: float = LON) -> float:
    """Solar elevation in degrees (NOAA general solar position calculations)."""
    dt = dt.astimezone(timezone.utc)
    jd = dt.timestamp() / 86400.0 + 2440587.5
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (
        (1.914602 - t * (0.004817 + 0.000014 * t)) * math.sin(mr)
        + (0.019993 - 0.000101 * t) * math.sin(2 * mr)
        + 0.000289 * math.sin(3 * mr)
    )
    omega = 125.04 - 1934.136 * t
    lam = math.radians(l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(omega)))
    eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
    eps = math.radians(eps0 + 0.00256 * math.cos(math.radians(omega)))
    decl = math.asin(math.sin(eps) * math.sin(lam))
    y = math.tan(eps / 2) ** 2
    l0r = math.radians(l0)
    eqtime = 4 * math.degrees(
        y * math.sin(2 * l0r)
        - 2 * e * math.sin(mr)
        + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * y * y * math.sin(4 * l0r)
        - 1.25 * e * e * math.sin(2 * mr)
    )
    minutes = dt.hour * 60 + dt.minute + dt.second / 60
    tst = (minutes + eqtime + 4 * lon) % 1440
    ha = math.radians(tst / 4 - 180)
    latr = math.radians(lat)
    cos_zen = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(ha)
    return 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zen))))


def regime(elev: float) -> str:
    if elev > SUNSET_DEG:
        return "day"
    if elev > CIVIL_DEG:
        return "twilight"
    return "night"


def run() -> dict:
    catalog = open_catalog()
    if "silver" not in catalog.list_namespaces():
        catalog.create_namespace("silver")
    try:
        table = catalog.load_table(TABLE)
    except Exception:
        table = catalog.create_table(TABLE, SCHEMA)

    todo = catalog.connection.sql(
        """
        SELECT b.image_id, b.captured_at
        FROM bkk.bronze.cctv_images b
        WHERE NOT EXISTS (SELECT 1 FROM bkk.silver.light_regime l WHERE l.image_id = b.image_id)
        """
    ).df()
    if todo.empty:
        summary = {"computed": 0, "table_rows": table.scan().count()}
        print(summary)
        return summary

    elev = [solar_elevation(ts.to_pydatetime()) for ts in todo["captured_at"]]
    reg = [regime(e) for e in elev]
    batch = pa.table(
        {
            "image_id": pa.array(todo["image_id"].tolist(), pa.string()),
            "solar_elevation_deg": pa.array(elev, pa.float32()),
            "light_regime": pa.array(reg, pa.string()),
            "is_dark": pa.array([r != "day" for r in reg], pa.bool_()),
        }
    )
    with catalog.begin_transaction() as txn:
        catalog.set_commit_message(f"light regime: {len(todo)} images", author="silver_light.py")
        txn.load_table(TABLE).append(batch)
    summary = {"computed": len(todo), "table_rows": table.scan().count()}
    print(summary)
    return summary


if __name__ == "__main__":
    run()
