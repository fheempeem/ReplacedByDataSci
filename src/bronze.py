"""Bronze stage: one row per received CCTV image, raw fields kept as parsed from the file name.

bronze.cctv_images
  image_id          camera_id/file name (unique key)
  camera_id         from file name
  camera_name       from file name
  view_desc         from file name (camera view description)
  captured_at       from file name, Bangkok time (+07:00)
  source_file       path relative to the raw root
  file_size_bytes
  sha256            content hash (used in silver to find frozen/duplicate frames)
  ingested_at       when this row was loaded

Idempotent: re-running only appends images whose image_id is not yet in the table.
Image content is NOT validated here; that is silver's job.
"""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pyarrow as pa
from pyducklake import (
    IntegerType,
    Schema,
    StringType,
    TimestampTZType,
    optional,
    required,
)

from common import BKK, RAW_DIR, open_catalog

NAME_RE = re.compile(r"^(?P<cam>\d+)_(?P<name>[^_]*)_(?P<desc>.*)_(?P<d>\d{8})_(?P<t>\d{6})\.jpg$")
TABLE = ("bronze", "cctv_images")

SCHEMA = Schema.of(
    required("image_id", StringType()),
    required("camera_id", StringType()),
    optional("camera_name", StringType()),
    optional("view_desc", StringType()),
    required("captured_at", TimestampTZType()),
    required("source_file", StringType()),
    required("file_size_bytes", IntegerType()),
    required("sha256", StringType()),
    required("ingested_at", TimestampTZType()),
)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_files() -> tuple[list[dict], list[str]]:
    rows, unparsed = [], []
    for path in sorted(RAW_DIR.rglob("*.jpg")):
        m = NAME_RE.match(path.name)
        if not m:
            unparsed.append(str(path))
            continue
        rel = path.relative_to(RAW_DIR).as_posix()
        rows.append(
            {
                "image_id": rel,
                "camera_id": m["cam"],
                "camera_name": m["name"],
                "view_desc": m["desc"],
                "captured_at": datetime.strptime(m["d"] + m["t"], "%Y%m%d%H%M%S").replace(tzinfo=BKK),
                "source_file": rel,
                "file_size_bytes": path.stat().st_size,
                "_path": path,
            }
        )
    return rows, unparsed


def run() -> dict:
    catalog = open_catalog()
    if "bronze" not in catalog.list_namespaces():
        catalog.create_namespace("bronze")
    try:
        table = catalog.load_table(TABLE)
    except Exception:
        table = catalog.create_table(TABLE, SCHEMA)

    scanned, unparsed = _scan_files()
    existing = set(table.scan().select("image_id").to_pandas()["image_id"])
    new_rows = [r for r in scanned if r["image_id"] not in existing]

    if new_rows:
        with ThreadPoolExecutor(max_workers=8) as pool:
            hashes = list(pool.map(_sha256, (r["_path"] for r in new_rows)))
        now = datetime.now(BKK)
        tz = pa.timestamp("us", tz="Asia/Bangkok")
        batch = pa.table(
            {
                "image_id": pa.array([r["image_id"] for r in new_rows], pa.string()),
                "camera_id": pa.array([r["camera_id"] for r in new_rows], pa.string()),
                "camera_name": pa.array([r["camera_name"] for r in new_rows], pa.string()),
                "view_desc": pa.array([r["view_desc"] for r in new_rows], pa.string()),
                "captured_at": pa.array([r["captured_at"] for r in new_rows], tz),
                "source_file": pa.array([r["source_file"] for r in new_rows], pa.string()),
                "file_size_bytes": pa.array([r["file_size_bytes"] for r in new_rows], pa.int32()),
                "sha256": pa.array(hashes, pa.string()),
                "ingested_at": pa.array([now] * len(new_rows), tz),
            }
        )
        with catalog.begin_transaction() as txn:
            catalog.set_commit_message(f"bronze ingest: {len(new_rows)} images", author="bronze.py")
            txn.load_table(TABLE).append(batch)

    summary = {
        "files_found": len(scanned),
        "unparsed_names": len(unparsed),
        "already_loaded": len(existing),
        "newly_loaded": len(new_rows),
        "table_rows": table.scan().count(),
    }
    print(summary)
    return summary


if __name__ == "__main__":
    run()
