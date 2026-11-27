"""Draw a stratified random sample of images for a manual (hand) count, to validate YOLO.

Outputs under data/validation/:
  images/NN.png              the images, upscaled 2x, labelled only with the sample number NN (no YOLO boxes!)
  sample_blind.csv           NN, file_name, regime, camera, time  + EMPTY count columns for counters to fill in
  sample_yolo_hidden.csv     NN and YOLO's counts. Do NOT open before counting: it would bias the counter.

Stratified by light regime because night/dusk is where YOLO is suspected to under-count:
24 day, 8 twilight, 8 night. Fixed seed, frozen-frame images excluded. Re-running reproduces the same sample.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from common import RAW_DIR, ROOT, open_catalog

SEED = 2026
PER_REGIME = {"day": 24, "twilight": 8, "night": 8}
OUT = ROOT / "data" / "validation"


def main() -> None:
    con = open_catalog().connection
    df = con.sql(
        """
        SELECT d.image_id, b.source_file, b.camera_id, strftime(b.captured_at, '%Y-%m-%d %H:%M') AS captured,
               l.light_regime, d.n_car, d.n_motorcycle, d.n_bus, d.n_truck, d.n_vehicles
        FROM bkk.silver.detections d
        JOIN bkk.bronze.cctv_images b USING (image_id)
        JOIN bkk.silver.image_features f USING (image_id)
        JOIN bkk.silver.light_regime l USING (image_id)
        JOIN bkk.silver.image_scan sc USING (image_id)
        WHERE NOT f.is_frozen AND sc.reason IS NULL   -- never sample glitch / test-card / unreadable images
        ORDER BY d.image_id
        """
    ).df()

    rng = random.Random(SEED)
    parts = []
    for regime, n in PER_REGIME.items():
        pool = df[df["light_regime"] == regime]
        pick = rng.sample(range(len(pool)), n)
        parts.append(pool.iloc[pick])
    sample = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle: regimes interleaved
    sample.insert(0, "idx", [f"{i + 1:02d}" for i in range(len(sample))])

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for r in sample.itertuples():
        with Image.open(RAW_DIR / r.source_file) as im:
            big = im.convert("RGB").resize((im.width * 2, im.height * 2), Image.LANCZOS)
        d = ImageDraw.Draw(big)
        d.rectangle([0, 0, 44, 22], fill=(0, 0, 0))
        d.text((6, 5), r.idx, fill=(255, 255, 255))
        big.save(OUT / "images" / f"{r.idx}.png")

    blind = sample[["idx", "light_regime", "camera_id", "captured"]].copy()
    blind.insert(1, "file_name", sample["source_file"].map(lambda p: Path(p).name))  # original image file name
    for c in ("car", "motorcycle", "bus", "truck", "notes"):
        blind[c] = ""
    blind.to_csv(OUT / "sample_blind.csv", index=False, encoding="utf-8-sig")
    sample[["idx", "image_id", "n_car", "n_motorcycle", "n_bus", "n_truck", "n_vehicles"]].to_csv(
        OUT / "sample_yolo_hidden.csv", index=False, encoding="utf-8-sig"
    )
    print(sample["light_regime"].value_counts().to_dict(), "->", OUT)


if __name__ == "__main__":
    main()
