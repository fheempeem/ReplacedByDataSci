"""Export gold tables (+ a coverage table for the quality page) to parquet for the dashboard.

The dashboard reads these files with pandas only: it never opens the lakehouse, so it starts fast, cannot
collide with a running pipeline, and can be handed to someone else as `data/export/` + `src/app.py`.
Run after gold.py. Overwrites the files every time (derived data).
"""
from __future__ import annotations

from common import ROOT, open_catalog

EXPORT_DIR = ROOT / "data" / "export"

QUERIES = {
    "camera_hourly": """
        SELECT * EXCLUDE (hour_start, service_date),
               CAST(strftime(hour_start, '%Y-%m-%d %H:%M:%S') AS TIMESTAMP) AS hour_local,
               strftime(service_date, '%Y-%m-%d') AS service_date
        FROM bkk.gold.camera_hourly ORDER BY camera_id, hour_start""",
    "camera_profile": "SELECT * FROM bkk.gold.camera_profile ORDER BY camera_id",
    "pipeline_audit": "SELECT * FROM bkk.gold.pipeline_audit ORDER BY step",
    "image_coverage": """
        SELECT b.camera_id,
               strftime(b.captured_at, '%Y-%m-%d') AS service_date,
               count(*) AS n_received,
               count(*) FILTER (WHERE r.image_id IS NOT NULL) AS n_rejected,
               count(*) FILTER (WHERE f.is_frozen) AS n_frozen,
               count(*) FILTER (WHERE l.is_dark) AS n_dark,
               count(*) FILTER (WHERE sc.reason IS NOT NULL) AS n_problem
        FROM bkk.bronze.cctv_images b
        LEFT JOIN bkk.silver.rejected_images r USING (image_id)
        LEFT JOIN bkk.silver.image_features f USING (image_id)
        LEFT JOIN bkk.silver.light_regime l USING (image_id)
        LEFT JOIN bkk.silver.image_scan sc USING (image_id)
        GROUP BY 1, 2 ORDER BY 1, 2""",
}


def run() -> dict:
    catalog = open_catalog()
    con = catalog.connection
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, sql in QUERIES.items():
        path = (EXPORT_DIR / f"{name}.parquet").as_posix()
        con.execute(f"COPY ({sql}) TO '{path}' (FORMAT PARQUET)")
        out[name] = con.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    print(out)
    return out


if __name__ == "__main__":
    run()
