"""Silver stage: find images that are unusable (glitch, test card, unreadable) -> silver.image_scan.

One row per checked image (silver.image_features), with the measurements and a `reason` that is NULL for a usable image:
  test_card     colour-bar "no signal" test pattern (> 5% saturated magenta/purple)
  glitch_green  video-decoder failure, large solid-green areas (> 5% saturated green)
  glitch_gray   video-decoder failure, grey smear (parts of the frame are filled with the decoder's grey 128). Any of:
                  (a) low contrast (std < 25), > 25% flat mid-tone 16x16 blocks and a noisy texture (Laplacian variance > 850);
                  (b) > 8% of the blocks are flat AND exactly grey 128 (+-3) and the texture is noisy (Laplacian variance > 1500);
                  (c) > 20% of the blocks are flat grey 128.
                Calibrated on contact sheets: real dim / hazy / empty-road scenes have flat blocks too but a calm texture
                (Laplacian variance < 1000) and rarely grey-128 blocks (up to 8%, all real). A first rule (std < 22) wrongly removed
                20 real scenes and missed 5 glitches; (b) and (c) catch the glitches that keep more picture (e.g. 13% grey-128 blocks).
  unreadable    so blurred / foggy / blinded that nothing can be made out (variance of the Laplacian < 60)
Images with a reason are excluded from the sampling for hand labelling, from YOLO (new runs), and from gold / the dashboard.
Thresholds were set by looking at contact sheets of the extreme and borderline images (hazy images with Laplacian variance 60-100
are kept: roads and vehicles are still visible). Idempotent: only images not yet scanned are processed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pyarrow as pa
from pyducklake import FloatType, Schema, StringType, TimestampTZType, optional, required

from common import BKK, open_catalog
from image_problems import metrics

TABLE = ("silver", "image_scan")
SCHEMA = Schema.of(
    required("image_id", StringType()),
    optional("reason", StringType()),
    optional("std", FloatType()),
    optional("lap_var", FloatType()),
    optional("green", FloatType()),
    optional("purple", FloatType()),
    optional("flat_mid_blocks", FloatType()),
    optional("gray_fill_blocks", FloatType()),
    required("scanned_at", TimestampTZType()),
)


def classify(m: dict) -> str | None:
    if not m.get("readable", False):
        return "unreadable"
    if m["purple"] > 0.05:
        return "test_card"
    if m["green"] > 0.05:
        return "glitch_green"
    gf = m.get("gray_fill_blocks") or 0.0
    if (m["std"] < 25 and m["flat_mid_blocks"] > 0.25 and m["lap_var"] > 850) or (gf > 0.08 and m["lap_var"] > 1500) or gf > 0.2:
        return "glitch_gray"  # a real grey glitch is full of noisy fragments (Laplacian variance 900+); a dim or hazy REAL scene is not
    if m["lap_var"] < 60:
        return "unreadable"
    return None


def reclassify(catalog, table) -> int:
    """Re-apply classify() to the stored measurements (no images are read), so the rules can be refined without a rescan."""
    df = catalog.connection.sql(
        "SELECT image_id, reason, std, lap_var, green, purple, flat_mid_blocks, gray_fill_blocks, scanned_at FROM bkk.silver.image_scan"
    ).df()
    if df.empty:
        return 0
    new = [classify({"readable": r.std == r.std, "std": r.std, "lap_var": r.lap_var, "green": r.green, "purple": r.purple,
                     "flat_mid_blocks": r.flat_mid_blocks, "gray_fill_blocks": r.gray_fill_blocks}) for r in df.itertuples()]
    df["new_reason"] = new
    ch = df[df["new_reason"].fillna("ok") != df["reason"].fillna("ok")]
    if ch.empty:
        return 0
    ids = ", ".join("'" + i.replace("'", "''") + "'" for i in ch["image_id"])
    tz = pa.timestamp("us", tz="Asia/Bangkok")
    batch = pa.table({
        "image_id": pa.array(ch["image_id"].tolist(), pa.string()), "reason": pa.array([None if (x is None or x != x) else x for x in ch["new_reason"].tolist()], pa.string()),
        "std": pa.array(ch["std"].tolist(), pa.float32()), "lap_var": pa.array(ch["lap_var"].tolist(), pa.float32()),
        "green": pa.array(ch["green"].tolist(), pa.float32()), "purple": pa.array(ch["purple"].tolist(), pa.float32()),
        "flat_mid_blocks": pa.array(ch["flat_mid_blocks"].tolist(), pa.float32()),
        "gray_fill_blocks": pa.array(ch["gray_fill_blocks"].tolist(), pa.float32()),
        "scanned_at": pa.array([t.to_pydatetime() for t in ch["scanned_at"]], tz),
    })
    with catalog.begin_transaction() as txn:
        catalog.set_commit_message(f"image scan: reclassified {len(ch)}", author="silver_problems.py")
        t = txn.load_table(TABLE)
        t.delete(f"image_id IN ({ids})")
        t.append(batch)
    return len(ch)


def run() -> dict:
    catalog = open_catalog()
    if "silver" not in catalog.list_namespaces():
        catalog.create_namespace("silver")
    try:
        table = catalog.load_table(TABLE)
        if "gray_fill_blocks" not in [f.name for f in table.schema.fields]:  # older table without the newest measurement: rebuild
            print("image_scan has an old schema: rebuilding it (all images are scanned again)", flush=True)
            catalog.drop_table(TABLE)
            raise KeyError
    except Exception:
        table = catalog.create_table(TABLE, SCHEMA)
    todo = catalog.connection.sql(
        """
        SELECT f.image_id FROM bkk.silver.image_features f
        WHERE NOT EXISTS (SELECT 1 FROM bkk.silver.image_scan s WHERE s.image_id = f.image_id)
        ORDER BY f.image_id
        """
    ).df()["image_id"].tolist()
    if not todo:
        n_re = reclassify(catalog, table)
        by = dict(catalog.connection.sql("SELECT coalesce(reason, 'ok'), count(*) FROM bkk.silver.image_scan GROUP BY 1").fetchall())
        summary = {"scanned": 0, "reclassified": n_re, "table_rows": table.scan().count(), "by_reason": by}
        print(summary)
        return summary
    with ThreadPoolExecutor(max_workers=8) as pool:
        ms = list(pool.map(metrics, todo, chunksize=256))
    now = datetime.now(BKK)
    tz = pa.timestamp("us", tz="Asia/Bangkok")
    get = lambda k: pa.array([m.get(k) for m in ms], pa.float32())  # noqa: E731
    batch = pa.table(
        {
            "image_id": pa.array(todo, pa.string()),
            "reason": pa.array([classify(m) for m in ms], pa.string()),
            "std": get("std"), "lap_var": get("lap_var"), "green": get("green"), "purple": get("purple"),
            "flat_mid_blocks": get("flat_mid_blocks"), "gray_fill_blocks": get("gray_fill_blocks"),
            "scanned_at": pa.array([now] * len(todo), tz),
        }
    )
    with catalog.begin_transaction() as txn:
        catalog.set_commit_message(f"image scan: {len(todo)}", author="silver_problems.py")
        txn.load_table(TABLE).append(batch)
    by = dict(catalog.connection.sql("SELECT coalesce(reason, 'ok'), count(*) FROM bkk.silver.image_scan GROUP BY 1").fetchall())
    summary = {"scanned": len(todo), "table_rows": table.scan().count(), "by_reason": by}
    print(summary)
    return summary


if __name__ == "__main__":
    run()
