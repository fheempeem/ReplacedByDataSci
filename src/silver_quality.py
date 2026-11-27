"""Silver stage 1: image quality checks (bronze -> silver.image_features + silver.rejected_images).

Rules (documented so they can be shown in the video):
  REJECT  unreadable_image   Pillow cannot decode the file
  REJECT  blank_image        pixel std-dev < 3 on gray thumbnail (white / black / no-signal frame)
  FLAG    is_night           mean brightness < 60 (kept: YOLO may under-count, dashboard can filter)
  FLAG    is_frozen          mean abs diff to the previous frame of the same camera < 0.3 (camera stuck)
Rejected images are NOT sent to YOLO. Bronze rows are never modified.
Idempotent: only images not yet in silver.image_features or silver.rejected_images are processed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pyarrow as pa
from PIL import Image
from pyducklake import (
    BooleanType,
    FloatType,
    IntegerType,
    Schema,
    StringType,
    TimestampTZType,
    optional,
    required,
)

from common import BKK, RAW_DIR, open_catalog

THUMB = (64, 36)
BLANK_STD = 3.0
NIGHT_BRIGHTNESS = 60.0
FROZEN_DIFF = 0.3

FEATURES = ("silver", "image_features")
REJECTED = ("silver", "rejected_images")

FEATURES_SCHEMA = Schema.of(
    required("image_id", StringType()),
    required("camera_id", StringType()),
    required("captured_at", TimestampTZType()),
    required("width", IntegerType()),
    required("height", IntegerType()),
    required("brightness", FloatType()),
    required("contrast", FloatType()),
    required("is_night", BooleanType()),
    optional("diff_to_prev", FloatType()),
    required("is_frozen", BooleanType()),
    required("checked_at", TimestampTZType()),
)
REJECTED_SCHEMA = Schema.of(
    required("image_id", StringType()),
    required("camera_id", StringType()),
    required("captured_at", TimestampTZType()),
    required("reason", StringType()),
    required("rejected_at", TimestampTZType()),
)


def _measure(source_file: str):
    """Return (width, height, brightness, contrast, thumb_array) or None if unreadable."""
    try:
        with Image.open(RAW_DIR / source_file) as im:
            im.load()
            w, h = im.size
            g = np.asarray(im.convert("L").resize(THUMB), dtype=np.float32)
        return w, h, float(g.mean()), float(g.std()), g
    except Exception:
        return None


def _ensure(catalog, ident, schema):
    if ident[0] not in catalog.list_namespaces():
        catalog.create_namespace(ident[0])
    try:
        return catalog.load_table(ident)
    except Exception:
        return catalog.create_table(ident, schema)


def run() -> dict:
    catalog = open_catalog()
    feats = _ensure(catalog, FEATURES, FEATURES_SCHEMA)
    rejs = _ensure(catalog, REJECTED, REJECTED_SCHEMA)
    con = catalog.connection

    todo = con.sql(
        """
        SELECT b.image_id, b.source_file, b.camera_id, b.captured_at
        FROM bkk.bronze.cctv_images b
        WHERE NOT EXISTS (SELECT 1 FROM bkk.silver.image_features f WHERE f.image_id = b.image_id)
          AND NOT EXISTS (SELECT 1 FROM bkk.silver.rejected_images r WHERE r.image_id = b.image_id)
        ORDER BY b.camera_id, b.captured_at
        """
    ).df()
    if todo.empty:
        summary = {"processed": 0, "features": feats.scan().count(), "rejected": rejs.scan().count()}
        print(summary)
        return summary

    with ThreadPoolExecutor(max_workers=8) as pool:
        measured = list(pool.map(_measure, todo["source_file"]))

    now = datetime.now(BKK)
    tz = pa.timestamp("us", tz="Asia/Bangkok")
    rej_rows, ok_rows = [], []
    prev_thumb: dict[str, np.ndarray] = {}  # last good frame per camera (todo is sorted by camera, time)
    for row, m in zip(todo.itertuples(index=False), measured):
        if m is None:
            rej_rows.append((row, "unreadable_image"))
            continue
        w, h, bright, contrast, thumb = m
        if contrast < BLANK_STD:
            rej_rows.append((row, "blank_image"))
            continue
        prev = prev_thumb.get(row.camera_id)
        diff = float(np.abs(thumb - prev).mean()) if prev is not None else None
        prev_thumb[row.camera_id] = thumb
        ok_rows.append((row, w, h, bright, contrast, diff))

    with catalog.begin_transaction() as txn:
        catalog.set_commit_message(
            f"silver quality: {len(ok_rows)} ok, {len(rej_rows)} rejected", author="silver_quality.py"
        )
        if ok_rows:
            txn.load_table(FEATURES).append(
                pa.table(
                    {
                        "image_id": pa.array([r.image_id for r, *_ in ok_rows], pa.string()),
                        "camera_id": pa.array([r.camera_id for r, *_ in ok_rows], pa.string()),
                        "captured_at": pa.array([r.captured_at.to_pydatetime() for r, *_ in ok_rows], tz),
                        "width": pa.array([w for _, w, *_ in ok_rows], pa.int32()),
                        "height": pa.array([h for _, _, h, *_ in ok_rows], pa.int32()),
                        "brightness": pa.array([b for _, _, _, b, *_ in ok_rows], pa.float32()),
                        "contrast": pa.array([c for _, _, _, _, c, _ in ok_rows], pa.float32()),
                        "is_night": pa.array([b < NIGHT_BRIGHTNESS for _, _, _, b, *_ in ok_rows], pa.bool_()),
                        "diff_to_prev": pa.array([d for *_, d in ok_rows], pa.float32()),
                        "is_frozen": pa.array([d is not None and d < FROZEN_DIFF for *_, d in ok_rows], pa.bool_()),
                        "checked_at": pa.array([now] * len(ok_rows), tz),
                    }
                )
            )
        if rej_rows:
            txn.load_table(REJECTED).append(
                pa.table(
                    {
                        "image_id": pa.array([r.image_id for r, _ in rej_rows], pa.string()),
                        "camera_id": pa.array([r.camera_id for r, _ in rej_rows], pa.string()),
                        "captured_at": pa.array([r.captured_at.to_pydatetime() for r, _ in rej_rows], tz),
                        "reason": pa.array([why for _, why in rej_rows], pa.string()),
                        "rejected_at": pa.array([now] * len(rej_rows), tz),
                    }
                )
            )

    summary = {"processed": len(todo), "features": feats.scan().count(), "rejected": rejs.scan().count()}
    print(summary)
    return summary


if __name__ == "__main__":
    run()
