"""Shared paths and lakehouse helpers."""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from pyducklake import Catalog

BKK = ZoneInfo("Asia/Bangkok")
ROOT = Path(__file__).resolve().parents[1]

# Raw images are read-only inputs; override with CCTV_RAW_DIR on other machines.
RAW_DIR = Path(
    os.environ.get(
        "CCTV_RAW_DIR",
        r"C:\Users\peemp\Downloads\cctv_images\cctv_images",
    )
)
LAKE_DIR = Path(os.environ.get("CCTV_LAKE_DIR", ROOT / "data" / "lakehouse"))
EXPORT_DIR = ROOT / "data" / "export"


def open_catalog() -> Catalog:
    """Open (creating if needed) the DuckLake catalog used by every pipeline stage."""
    LAKE_DIR.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(
        "bkk",
        str(LAKE_DIR / "metadata.ducklake"),
        data_path=str(LAKE_DIR / "data"),
    )
    catalog.set_option("data_inlining_row_limit", "0")
    catalog.connection.execute("SET TimeZone = 'Asia/Bangkok'")
    return catalog
