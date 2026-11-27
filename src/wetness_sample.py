"""Draw a blind sample of images to label as dry / wet road / raining, to test rain data and the image-based detector.

60 images, 20 from each Open-Meteo rain band of the image's camera-hour (>=1 mm, 0.1-1 mm, <0.1 mm), shuffled.
Outputs in data/wetness/:  images/NN.png (2x, number only), labels_blank.csv, hidden_rain.csv (rain mm: do NOT open
before labelling). Uses only file names + the dashboard export, so it can run while the lakehouse is busy.
Label scale (labels_<who>.csv, column `label`): 0 dry, 1 wet surface but no visible rain, 2 raining (streaks, drops,
umbrellas, rain on the lens), blank/-1 = cannot tell.
"""
from __future__ import annotations

import random
import re

import pandas as pd
from PIL import Image, ImageDraw

from common import EXPORT_DIR, RAW_DIR, ROOT

SEED = 77
PER_BAND = 20
OUT = ROOT / "data" / "wetness"
NAME_RE = re.compile(r"^(?P<cam>\d+)_.*_(?P<d>\d{8})_(?P<t>\d{6})\.jpg$")


def main() -> None:
    hourly = pd.read_parquet(EXPORT_DIR / "camera_hourly.parquet", columns=["camera_id", "hour_local", "rain_mm", "light_group"])
    rows = []
    for p in RAW_DIR.rglob("*.jpg"):
        m = NAME_RE.match(p.name)
        if m and p.stat().st_size > 5000:
            rows.append((p.relative_to(RAW_DIR).as_posix(), m["cam"], pd.Timestamp(m["d"] + m["t"]).floor("h")))
    files = pd.DataFrame(rows, columns=["image_id", "camera_id", "hour_local"])
    df = files.merge(hourly, on=["camera_id", "hour_local"])  # inner join: drops frozen/rejected hours
    bands = {
        "rain>=1mm": df["rain_mm"] >= 1.0,
        "0.1-1mm": (df["rain_mm"] >= 0.1) & (df["rain_mm"] < 1.0),
        "<0.1mm": df["rain_mm"] < 0.1,
    }
    rng = random.Random(SEED)
    parts = []
    for name, mask in bands.items():
        pool = df[mask].reset_index(drop=True)
        pick = pool.iloc[rng.sample(range(len(pool)), PER_BAND)].assign(band=name)
        parts.append(pick)
        print(f"{name}: pool {len(pool)}")
    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "idx", [f"{i + 1:02d}" for i in range(len(sample))])

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for r in sample.itertuples():
        with Image.open(RAW_DIR / r.image_id) as im:
            big = im.convert("RGB").resize((im.width * 2, im.height * 2), Image.LANCZOS)
        d = ImageDraw.Draw(big)
        d.rectangle([0, 0, 44, 22], fill=(0, 0, 0))
        d.text((6, 5), r.idx, fill=(255, 255, 255))
        big.save(OUT / "images" / f"{r.idx}.png")
    sample[["idx", "image_id", "camera_id", "hour_local", "rain_mm", "band", "light_group"]].to_csv(
        OUT / "hidden_rain.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"idx": sample["idx"], "label": "", "note": ""}).to_csv(OUT / "labels_blank.csv", index=False, encoding="utf-8-sig")
    print("sample written:", len(sample), "->", OUT)


if __name__ == "__main__":
    main()
