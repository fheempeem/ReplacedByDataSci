"""Final quality gate: checks that every stage's output is consistent with the stage before it.

Exit code 0 = all checks passed, 1 = at least one failed (run_pipeline.py then marks the run FAILED).
These are the same invariants shown on the dashboard's data-quality tab:
  every received image is accounted for (rejected OR checked), every checked image is detected exactly once,
  and gold re-aggregates exactly the images that were detected and not frozen.
"""
from __future__ import annotations

import sys

import pandas as pd

from common import EXPORT_DIR, RAW_DIR, open_catalog


def main() -> int:
    con = open_catalog().connection
    q = lambda sql: con.sql(sql).fetchone()[0]  # noqa: E731

    on_disk = sum(1 for _ in RAW_DIR.rglob("*.jpg"))
    n_bronze = q("SELECT count(*) FROM bkk.bronze.cctv_images")
    n_uniq = q("SELECT count(DISTINCT image_id) FROM bkk.bronze.cctv_images")
    n_rej = q("SELECT count(*) FROM bkk.silver.rejected_images")
    n_feat = q("SELECT count(*) FROM bkk.silver.image_features")
    n_overlap = q("SELECT count(*) FROM bkk.silver.rejected_images r JOIN bkk.silver.image_features f USING (image_id)")
    n_det = q("SELECT count(*) FROM bkk.silver.detections")
    n_det_u = q("SELECT count(DISTINCT image_id) FROM bkk.silver.detections")
    n_det_orphan = q("SELECT count(*) FROM bkk.silver.detections d WHERE NOT EXISTS (SELECT 1 FROM bkk.silver.image_features f WHERE f.image_id = d.image_id)")
    n_light = q("SELECT count(*) FROM bkk.silver.light_regime")
    n_wet = q("SELECT count(*) FROM bkk.silver.wetness")
    n_wet_u = q("SELECT count(DISTINCT image_id) FROM bkk.silver.wetness")
    g_null_wet = q("SELECT count(*) FROM bkk.gold.camera_hourly WHERE wet_score IS NULL")
    n_frozen = q("SELECT count(*) FROM bkk.silver.image_features WHERE is_frozen")
    n_scan = q("SELECT count(*) FROM bkk.silver.image_scan")
    n_scan_u = q("SELECT count(DISTINCT image_id) FROM bkk.silver.image_scan")
    n_usable = q("SELECT count(*) FROM bkk.silver.image_scan WHERE reason IS NULL")
    n_usable_missing_det = q(
        "SELECT count(*) FROM bkk.silver.image_scan s WHERE s.reason IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM bkk.silver.detections d WHERE d.image_id = s.image_id)")
    n_gold_expected = q(
        "SELECT count(*) FROM bkk.silver.detections d JOIN bkk.silver.image_features f USING (image_id) "
        "JOIN bkk.silver.image_scan s USING (image_id) WHERE NOT f.is_frozen AND s.reason IS NULL")
    g_rows = q("SELECT count(*) FROM bkk.gold.camera_hourly")
    g_keys = q("SELECT count(DISTINCT (camera_id, hour_start)) FROM bkk.gold.camera_hourly")
    g_imgs = q("SELECT sum(n_images) FROM bkk.gold.camera_hourly")
    g_null_wx = q("SELECT count(*) FROM bkk.gold.camera_hourly WHERE precipitation_mm IS NULL OR pm2_5 IS NULL OR congestion_index IS NULL")
    g_prof = q("SELECT count(*) FROM bkk.gold.camera_profile")
    n_cams = q("SELECT count(DISTINCT camera_id) FROM bkk.bronze.cctv_images")
    n_loc = q("SELECT count(*) FROM bkk.bronze.camera_locations")
    p_rows = len(pd.read_parquet(EXPORT_DIR / "camera_hourly.parquet"))

    checks = [
        ("bronze rows == jpg files on disk", n_bronze == on_disk, f"{n_bronze} vs {on_disk}"),
        ("bronze image_id unique", n_bronze == n_uniq, f"{n_bronze} vs {n_uniq}"),
        ("rejected + checked == bronze", n_rej + n_feat == n_bronze, f"{n_rej}+{n_feat} vs {n_bronze}"),
        ("no image both rejected and checked", n_overlap == 0, str(n_overlap)),
        ("image scan covers every checked image", n_scan == n_feat == n_scan_u, f"{n_scan} vs {n_feat}"),
        ("every usable image has a detection", n_usable_missing_det == 0, f"{n_usable_missing_det} missing of {n_usable}"),
        ("each image detected once", n_det == n_det_u, f"{n_det} vs {n_det_u}"),
        ("no detection without quality row", n_det_orphan == 0, str(n_det_orphan)),
        ("light regime for every image", n_light == n_bronze, f"{n_light} vs {n_bronze}"),
        ("wetness score for every checked image", n_wet == n_feat == n_wet_u, f"{n_wet} vs {n_feat}"),
        ("gold has no NULL wetness score", g_null_wet == 0, str(g_null_wet)),
        ("camera locations for every camera", n_loc == n_cams, f"{n_loc} vs {n_cams}"),
        ("gold keys unique (camera x hour)", g_rows == g_keys, f"{g_rows} vs {g_keys}"),
        ("gold images == detected, not frozen, usable", g_imgs == n_gold_expected, f"{g_imgs} vs {n_gold_expected}"),
        ("gold has no NULL rain/PM2.5/index", g_null_wx == 0, str(g_null_wx)),
        ("camera_profile one row per camera", g_prof == n_cams, f"{g_prof} vs {n_cams}"),
        ("dashboard export matches gold", p_rows == g_rows, f"{p_rows} vs {g_rows}"),
    ]
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<42} {detail}")
        failed += not ok
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
