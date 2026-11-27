"""Data-quality report on the Bronze table: coverage per day/hour/camera, offline stubs, duplicates."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import BRONZE, REPORTS  # noqa: E402


def main() -> None:
    df = pd.read_parquet(BRONZE / "images.parquet")
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = []
    p = lines.append

    p("# Bronze coverage report\n")
    p(f"- images: {len(df):,}; cameras: {df['camera_id'].nunique()}")
    p(f"- unparsed filenames: {(~df['parsed_ok']).sum()}")
    p(f"- camera id differs from folder: {(~df['camera_matches_folder']).sum()}")
    p(f"- offline stubs (<5 KB): {df['is_offline_stub'].sum()}")
    p(f"- byte-identical duplicates: {df['is_duplicate'].sum()}\n")

    p("## Images and offline stubs per day\n")
    per_day = df.groupby(["date", "weekday"]).agg(
        images=("path", "size"), offline=("is_offline_stub", "sum")).reset_index()
    p(per_day.to_markdown(index=False) + "\n")

    p("## Images per hour (expected ~6 per camera-hour = 600 per hour per day for 100 cameras)\n")
    per_hour = df.groupby("hour").agg(images=("path", "size"), offline=("is_offline_stub", "sum")).reset_index()
    p(per_hour.to_markdown(index=False) + "\n")

    p("## Cameras by number of offline stubs (top 15)\n")
    cam = df.groupby(["camera_id", "name1"]).agg(
        images=("path", "size"), offline=("is_offline_stub", "sum")).reset_index()
    cam["offline_pct"] = (100 * cam["offline"] / cam["images"]).round(1)
    p(cam.sort_values("offline", ascending=False).head(15).to_markdown(index=False) + "\n")

    p("## Day x hour coverage heatmap (images over all cameras)\n")
    p(df.pivot_table(index="date", columns="hour", values="path", aggfunc="size", fill_value=0)
        .to_markdown() + "\n")

    p("## Thursday 27 Aug: which cameras are short? (images per camera, lowest 15)\n")
    thu = df[df["date"].astype(str) == "2026-08-27"].groupby("camera_id").size().sort_values().head(15)
    p(thu.to_frame("images").to_markdown() + "\n")

    out = REPORTS / "bronze_coverage.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
