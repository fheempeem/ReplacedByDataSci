"""Round 2 verification: draw fresh samples that do NOT overlap round 1 (data/wetness, data/validation).

  rain    150 DAYTIME images, stratified 3 x 3 by Open-Meteo rain band x CLIP wetness band. Each stratum is sampled at random, and the
          stratum sizes are stored as weights so the AUC of BOTH detectors can be computed without bias from how we chose images
          (choosing images by a detector's own score would otherwise flatter / hurt that detector).
  boxes   60 images (30 day, 15 twilight, 15 night), YOLOv8m at conf 0.15 (the lowest threshold under discussion). People will
          judge every box, so precision and recall can be computed for any threshold >= 0.15. Images with 2-40 boxes only.

Outputs under data/round2/ : samples/rain/R###.jpg, samples/boxes/B##.jpg (2x, number only),
                             hidden/rain_hidden.csv and hidden/boxes_hidden.csv  (answers: do NOT show to labellers).
Fixed seed: re-running gives the same sample.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import EXPORT_DIR, RAW_DIR, ROOT, open_catalog

SEED = 2027
OUT = ROOT / "data" / "round2"
COCO = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF = 0.15
N_RAIN = 150
N_BOX = {"day": 30, "twilight": 15, "night": 15}
OM_BANDS = [("<0.1mm", -1, 0.1), ("0.1-1mm", 0.1, 1.0), (">=1mm", 1.0, 1e9)]
CLIP_BANDS = [("clip<0.25", -1, 0.25), ("clip0.25-0.5", 0.25, 0.5), ("clip>=0.5", 0.5, 1.01)]


def used_ids() -> set[str]:
    ids = set()
    for f in (ROOT / "data" / "wetness" / "hidden_rain.csv", ROOT / "data" / "validation" / "sample_yolo_hidden.csv"):
        if f.exists():
            ids |= set(pd.read_csv(f, usecols=["image_id"])["image_id"])
    return ids


def save2x(src_rel: str, dst: Path, label: str) -> None:
    with Image.open(RAW_DIR / src_rel) as im:
        big = im.convert("RGB").resize((im.width * 2, im.height * 2), Image.LANCZOS)
    d = ImageDraw.Draw(big)
    d.rectangle([0, 0, 50, 22], fill=(0, 0, 0))
    d.text((6, 5), label, fill=(255, 255, 255))
    dst.parent.mkdir(parents=True, exist_ok=True)
    big.save(dst, quality=92)


def load_pool(con) -> pd.DataFrame:
    """Every usable image (not frozen, not glitch / test-card / unreadable) with its light regime, CLIP score and Open-Meteo rain."""
    pool = con.sql(
        """
        SELECT b.image_id, b.camera_id, strftime(b.captured_at, '%Y-%m-%d %H') AS hour_key,
               l.light_regime, w.p_wet
        FROM bkk.bronze.cctv_images b
        JOIN bkk.silver.image_features f USING (image_id)
        JOIN bkk.silver.light_regime l USING (image_id)
        JOIN bkk.silver.wetness w USING (image_id)
        JOIN bkk.silver.image_scan sc USING (image_id)
        WHERE NOT f.is_frozen AND sc.reason IS NULL   -- never sample glitch / test-card / unreadable images
        """
    ).df()
    hourly = pd.read_parquet(EXPORT_DIR / "camera_hourly.parquet", columns=["camera_id", "hour_local", "rain_mm"])
    hourly["hour_key"] = hourly["hour_local"].dt.strftime("%Y-%m-%d %H")
    return pool.merge(hourly[["camera_id", "hour_key", "rain_mm"]], on=["camera_id", "hour_key"], how="inner")


def main() -> None:
    con = open_catalog().connection
    pool = load_pool(con)
    pool = pool[~pool["image_id"].isin(used_ids())].reset_index(drop=True)
    print("pool after removing round-1 images:", len(pool))
    rng = random.Random(SEED)

    # ------------------------------------------------------------------ rain sample (daytime, 3 x 3 strata)
    day = pool[pool["light_regime"] == "day"].copy()
    day["om_band"] = pd.cut(day["rain_mm"], [b[1] for b in OM_BANDS] + [OM_BANDS[-1][2]], labels=[b[0] for b in OM_BANDS], right=False).astype(str)
    day["clip_band"] = pd.cut(day["p_wet"], [b[1] for b in CLIP_BANDS] + [CLIP_BANDS[-1][2]], labels=[b[0] for b in CLIP_BANDS], right=False).astype(str)
    strata = day.groupby(["om_band", "clip_band"]).size().rename("pool").reset_index()
    base = N_RAIN // len(strata)
    strata["n"] = np.minimum(base, strata["pool"])
    left = N_RAIN - int(strata["n"].sum())
    while left > 0:  # hand out the remainder to strata that still have images, largest pools first
        for i in strata.sort_values("pool", ascending=False).index:
            if left > 0 and strata.loc[i, "n"] < strata.loc[i, "pool"]:
                strata.loc[i, "n"] += 1
                left -= 1
    print(strata.to_string(index=False))
    picks = []
    for r in strata.itertuples():
        sub = day[(day["om_band"] == r.om_band) & (day["clip_band"] == r.clip_band)]
        take = sub.iloc[rng.sample(range(len(sub)), int(r.n))].copy()
        take["weight"] = r.pool / r.n
        picks.append(take)
    rain = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
    rain.insert(0, "idx", [f"R{i + 1:03d}" for i in range(len(rain))])
    for r in rain.itertuples():
        save2x(r.image_id, OUT / "samples" / "rain" / f"{r.idx}.jpg", r.idx)
    (OUT / "hidden").mkdir(parents=True, exist_ok=True)
    rain[["idx", "image_id", "camera_id", "hour_key", "rain_mm", "p_wet", "om_band", "clip_band", "weight"]].to_csv(
        OUT / "hidden" / "rain_hidden.csv", index=False, encoding="utf-8-sig")
    print("rain sample:", len(rain))

    # ------------------------------------------------------------------ box-review sample + YOLO at conf 0.15
    from ultralytics import YOLO, settings

    models = ROOT / "data" / "models"
    settings.update({"weights_dir": str(models)})
    model = YOLO(str(models / "yolov8m.pt"))
    rest = pool[~pool["image_id"].isin(set(rain["image_id"]))]
    chosen, rows = [], []
    for regime, n_want in N_BOX.items():
        cand = rest[rest["light_regime"] == regime]
        order = rng.sample(range(len(cand)), min(len(cand), n_want * 6))
        got = 0
        for k in order:
            if got == n_want:
                break
            iid = cand.iloc[k]["image_id"]
            with Image.open(RAW_DIR / iid) as im:
                arr = np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
                w, h = im.size
            res = model.predict(arr, imgsz=640, conf=CONF, classes=list(COCO), device=0, half=True, verbose=False)[0]
            if not (2 <= len(res.boxes) <= 40):
                continue
            chosen.append((iid, cand.iloc[k]["camera_id"], regime, w, h, res))
            got += 1
        print(f"boxes sample {regime}: {got}/{n_want}")
    random.Random(SEED).shuffle(chosen)
    box_rows, img_rows = [], []
    for n, (iid, cam, regime, w, h, res) in enumerate(chosen, start=1):
        idx = f"B{n:02d}"
        save2x(iid, OUT / "samples" / "boxes" / f"{idx}.jpg", idx)
        xyxy, cls, conf = res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()
        order = np.lexsort((xyxy[:, 0], (xyxy[:, 1] + xyxy[:, 3]) / 2 // 25))  # reading order; does not reveal confidence
        for bid, j in enumerate(order, start=1):
            box_rows.append({"idx": idx, "box_id": bid, "cls": COCO[int(cls[j])], "conf": float(conf[j]),
                             "x1": float(xyxy[j, 0]), "y1": float(xyxy[j, 1]), "x2": float(xyxy[j, 2]), "y2": float(xyxy[j, 3])})
        img_rows.append({"idx": idx, "image_id": iid, "camera_id": cam, "light_regime": regime, "width": w, "height": h})
    pd.DataFrame(box_rows).to_csv(OUT / "hidden" / "boxes_hidden.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(img_rows).to_csv(OUT / "hidden" / "box_images_hidden.csv", index=False, encoding="utf-8-sig")
    print(f"box images: {len(img_rows)}  boxes to review: {len(box_rows)}  (>= 0.25: {sum(b['conf'] >= 0.25 for b in box_rows)})")


if __name__ == "__main__":
    main()
