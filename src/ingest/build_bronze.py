"""Bronze layer: one row per image file, parsed from the filename.

Filename pattern: <camID>_<district>_<direction>_<YYYYMMDD>_<HHMMSS>.jpg
Nothing is deleted. Blank "camera offline" placeholders (tiny files) are flagged.
"""
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import BRONZE, IMAGES_DIR, OFFLINE_MAX_BYTES  # noqa: E402

PATTERN = r"^(?P<camera_id>[^_]+)_(?P<name1>.+?)_(?P<name2>.+)_(?P<date>\d{8})_(?P<time>\d{6})\.jpe?g$"


def scan(images_dir: Path = IMAGES_DIR) -> pd.DataFrame:
    rows = []
    for cam_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        with os.scandir(cam_dir) as it:
            for e in it:
                if e.is_file():
                    rows.append((cam_dir.name, e.name, str(Path(e.path)), e.stat().st_size))
    df = pd.DataFrame(rows, columns=["camera_dir", "filename", "path", "file_size"])

    parts = df["filename"].str.extract(PATTERN, flags=2)
    df = pd.concat([df, parts], axis=1)
    df["parsed_ok"] = df["camera_id"].notna()
    df["timestamp"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S", errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = df["timestamp"].dt.day_of_week >= 5
    df["is_offline_stub"] = df["file_size"] < OFFLINE_MAX_BYTES
    df["camera_matches_folder"] = df["camera_id"] == df["camera_dir"]
    return df.sort_values(["camera_id", "timestamp"]).reset_index(drop=True)


def add_duplicate_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag byte-identical frames (hash only files that share a file size, to stay fast)."""
    dup_candidates = df[df.duplicated(["camera_id", "file_size"], keep=False) & ~df["is_offline_stub"]]
    hashes = {}
    for idx, p in dup_candidates["path"].items():
        with open(p, "rb") as f:
            hashes[idx] = hashlib.md5(f.read()).hexdigest()
    df["md5"] = pd.Series(hashes)
    df["is_duplicate"] = df.duplicated(["camera_id", "md5"], keep="first") & df["md5"].notna()
    return df


def main() -> Path:
    BRONZE.mkdir(parents=True, exist_ok=True)
    df = add_duplicate_flag(scan())
    out = BRONZE / "images.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df):,} images, {df['camera_id'].nunique()} cameras -> {out}")
    return out


if __name__ == "__main__":
    main()
