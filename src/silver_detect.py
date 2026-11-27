"""Silver stage 2: vehicle detection with pretrained YOLOv8 (COCO) -> silver.detections.

Counts car / motorcycle / bus / truck per image. Only images that passed silver_quality
(i.e. exist in silver.image_features) are processed. Resumable and idempotent: commits every
CHUNK images and skips image_ids already in silver.detections.

Usage: python src/silver_detect.py [--limit N] [--model yolov8m.pt]
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pyarrow as pa
from PIL import Image
from pyducklake import FloatType, IntegerType, Schema, StringType, TimestampTZType, optional, required

from common import BKK, LAKE_DIR, RAW_DIR, open_catalog

TABLE = ("silver", "detections")
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF = 0.15  # lowered from 0.25 after people checked 508 boxes (docs/verification_round2.md): recall 0.62 -> 0.77, precision 0.97 -> 0.95
IMGSZ = 640
BATCH = 64
CHUNK = 2048

SCHEMA = Schema.of(
    required("image_id", StringType()),
    required("n_car", IntegerType()),
    required("n_motorcycle", IntegerType()),
    required("n_bus", IntegerType()),
    required("n_truck", IntegerType()),
    required("n_vehicles", IntegerType()),
    optional("mean_conf", FloatType()),
    required("box_area_frac", FloatType()),
    required("model", StringType()),
    required("conf_threshold", FloatType()),
    required("detected_at", TimestampTZType()),
)


def _load_bgr(source_file: str) -> np.ndarray:
    # PIL handles Thai (non-ASCII) paths on Windows; ultralytics expects BGR arrays.
    with Image.open(RAW_DIR / source_file) as im:
        return np.asarray(im.convert("RGB"))[:, :, ::-1].copy()


def run(limit: int | None = None, model_name: str = "yolov8m.pt") -> dict:
    from ultralytics import YOLO, settings

    models_dir = LAKE_DIR.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    settings.update({"weights_dir": str(models_dir)})
    model = YOLO(str(models_dir / model_name))
    device = 0 if __import__("torch").cuda.is_available() else "cpu"

    catalog = open_catalog()
    if "silver" not in catalog.list_namespaces():
        catalog.create_namespace("silver")
    try:
        table = catalog.load_table(TABLE)
        old = catalog.connection.sql("SELECT DISTINCT conf_threshold FROM bkk.silver.detections").fetchall()
        if any(abs(float(r[0]) - CONF) > 1e-6 for r in old):  # threshold changed: every image must be detected again
            print(f"detections were made at conf {[round(float(r[0]), 2) for r in old]}, now {CONF}: rebuilding the table", flush=True)
            catalog.drop_table(TABLE)
            raise KeyError
    except Exception:
        table = catalog.create_table(TABLE, SCHEMA)

    skip_problems = ""
    if "image_scan" in [t[-1] if isinstance(t, tuple) else t for t in catalog.list_tables("silver")]:
        # glitch / test-card / unreadable images (silver_problems.py) are not worth a detection
        skip_problems = "AND NOT EXISTS (SELECT 1 FROM bkk.silver.image_scan s WHERE s.image_id = f.image_id AND s.reason IS NOT NULL)"
    q = f"""
        SELECT f.image_id, b.source_file
        FROM bkk.silver.image_features f
        JOIN bkk.bronze.cctv_images b USING (image_id)
        WHERE NOT EXISTS (SELECT 1 FROM bkk.silver.detections d WHERE d.image_id = f.image_id) {skip_problems}
        ORDER BY f.captured_at, f.camera_id
    """
    if limit:
        q += f" LIMIT {int(limit)}"
    todo = catalog.connection.sql(q).df()
    total = len(todo)
    print(f"to detect: {total} images on device={device}", flush=True)

    tz = pa.timestamp("us", tz="Asia/Bangkok")
    done = 0
    for c0 in range(0, total, CHUNK):
        chunk = todo.iloc[c0 : c0 + CHUNK]
        rows = []
        for b0 in range(0, len(chunk), BATCH):
            part = chunk.iloc[b0 : b0 + BATCH]
            imgs = [_load_bgr(p) for p in part["source_file"]]
            results = model.predict(
                imgs, imgsz=IMGSZ, conf=CONF, classes=list(COCO_VEHICLES), device=device,
                half=device != "cpu", verbose=False,
            )
            for image_id, r in zip(part["image_id"], results):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                wh = r.boxes.xywhn.cpu().numpy()[:, 2:4] if len(cls) else np.zeros((0, 2))
                counts = {k: int((cls == k).sum()) for k in COCO_VEHICLES}
                rows.append(
                    (
                        image_id, counts[2], counts[3], counts[5], counts[7], int(len(cls)),
                        float(conf.mean()) if len(conf) else None,
                        float(min(1.0, (wh[:, 0] * wh[:, 1]).sum())) if len(cls) else 0.0,
                    )
                )
        now = datetime.now(BKK)
        batch = pa.table(
            {
                "image_id": pa.array([r[0] for r in rows], pa.string()),
                "n_car": pa.array([r[1] for r in rows], pa.int32()),
                "n_motorcycle": pa.array([r[2] for r in rows], pa.int32()),
                "n_bus": pa.array([r[3] for r in rows], pa.int32()),
                "n_truck": pa.array([r[4] for r in rows], pa.int32()),
                "n_vehicles": pa.array([r[5] for r in rows], pa.int32()),
                "mean_conf": pa.array([r[6] for r in rows], pa.float32()),
                "box_area_frac": pa.array([r[7] for r in rows], pa.float32()),
                "model": pa.array([model_name] * len(rows), pa.string()),
                "conf_threshold": pa.array([CONF] * len(rows), pa.float32()),
                "detected_at": pa.array([now] * len(rows), tz),
            }
        )
        with catalog.begin_transaction() as txn:
            catalog.set_commit_message(f"yolo detections: {len(rows)} images", author="silver_detect.py")
            txn.load_table(TABLE).append(batch)
        done += len(rows)
        print(f"  {done}/{total}", flush=True)

    summary = {"detected_now": done, "table_rows": table.scan().count()}
    print(summary, flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="yolov8m.pt")
    a = ap.parse_args()
    run(a.limit, a.model)
