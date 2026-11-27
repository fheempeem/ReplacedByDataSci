"""Replace round-2 sample images that the image scan now flags (glitch / test card / unreadable), KEEPING their ids,
so that already-built kits and the person assignments stay valid.    python src/round2_fix_samples.py

  rain   the flagged image is replaced by a random usable DAYTIME image from the SAME stratum (Open-Meteo band x CLIP band);
         the weight of every stratum is then recomputed from the clean pool (weight = usable images in the stratum / images sampled).
  boxes  the flagged image is replaced by a random usable image of the same light regime that has 2-40 boxes at conf >= 0.15;
         its boxes (YOLOv8m) replace the old ones in boxes_hidden.csv.
Run round2_kit.py afterwards to rebuild the kits.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
from PIL import Image

from common import RAW_DIR, ROOT, open_catalog
from round2_samples import CLIP_BANDS, COCO, CONF, OM_BANDS, OUT, load_pool, save2x, used_ids

SEED = 31


def band(series: pd.Series, bands) -> pd.Series:
    return pd.cut(series, [b[1] for b in bands] + [bands[-1][2]], labels=[b[0] for b in bands], right=False).astype(str)


def main() -> None:
    con = open_catalog().connection
    scan = con.sql("SELECT image_id, reason FROM bkk.silver.image_scan WHERE reason IS NOT NULL").df().set_index("image_id")["reason"]
    pool_all = load_pool(con)
    rain = pd.read_csv(OUT / "hidden" / "rain_hidden.csv", dtype={"idx": str, "camera_id": str})
    boxes_img = pd.read_csv(OUT / "hidden" / "box_images_hidden.csv", dtype={"idx": str, "camera_id": str})
    boxes = pd.read_csv(OUT / "hidden" / "boxes_hidden.csv", dtype={"idx": str, "camera_id": str})
    used = used_ids() | set(rain["image_id"]) | set(boxes_img["image_id"])

    # ---------------------------------------------------------------- rain
    bad_rain = rain[rain["image_id"].isin(scan.index)]
    print("rain images to replace:", [(r.idx, scan[r.image_id]) for r in bad_rain.itertuples()])
    day = pool_all[pool_all["light_regime"] == "day"].copy()
    day["om_band"], day["clip_band"] = band(day["rain_mm"], OM_BANDS), band(day["p_wet"], CLIP_BANDS)
    for r in bad_rain.itertuples():
        cand = day[(day["om_band"] == r.om_band) & (day["clip_band"] == r.clip_band) & ~day["image_id"].isin(used)]
        pick = cand.iloc[random.Random(SEED + int(r.idx[1:])).randrange(len(cand))]
        i = rain.index[rain["idx"] == r.idx][0]
        for col in ("image_id", "camera_id", "hour_key", "rain_mm", "p_wet"):
            rain.at[i, col] = str(pick[col]) if col == "camera_id" else pick[col]
        used.add(pick["image_id"])
        save2x(pick["image_id"], OUT / "samples" / "rain" / f"{r.idx}.jpg", r.idx)
        print(f"  {r.idx} -> camera {pick.camera_id}, {pick.hour_key} h ({r.om_band}, {r.clip_band})")
    # weights from the clean pool (round-1 images stay excluded, as in the original draw)
    clean = day[~day["image_id"].isin(used_ids())]
    size = clean.groupby(["om_band", "clip_band"]).size().rename("pool").reset_index()
    n = rain.groupby(["om_band", "clip_band"]).size().rename("n").reset_index()
    w = size.merge(n, on=["om_band", "clip_band"])
    w["weight"] = w["pool"] / w["n"]
    rain = rain.drop(columns="weight").merge(w[["om_band", "clip_band", "weight"]], on=["om_band", "clip_band"], how="left")
    rain.to_csv(OUT / "hidden" / "rain_hidden.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- boxes
    bad_box = boxes_img[boxes_img["image_id"].isin(scan.index)]
    print("box images to replace:", [(r.idx, scan[r.image_id]) for r in bad_box.itertuples()])
    if len(bad_box):
        from ultralytics import YOLO, settings

        settings.update({"weights_dir": str(ROOT / "data" / "models")})
        model = YOLO(str(ROOT / "data" / "models" / "yolov8m.pt"))
    for r in bad_box.itertuples():
        cand = pool_all[(pool_all["light_regime"] == r.light_regime) & ~pool_all["image_id"].isin(used)].reset_index(drop=True)
        order = random.Random(SEED + int(r.idx[1:])).sample(range(len(cand)), min(60, len(cand)))
        for k in order:
            iid = cand.iloc[k]["image_id"]
            with Image.open(RAW_DIR / iid) as im:
                arr = np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
                width, height = im.size
            res = model.predict(arr, imgsz=640, conf=CONF, classes=list(COCO), device=0, half=True, verbose=False)[0]
            if 2 <= len(res.boxes) <= 40:
                break
        else:
            raise SystemExit("no replacement found")
        used.add(iid)
        save2x(iid, OUT / "samples" / "boxes" / f"{r.idx}.jpg", r.idx)
        xyxy, cls, conf = res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()
        order_b = np.lexsort((xyxy[:, 0], (xyxy[:, 1] + xyxy[:, 3]) / 2 // 25))
        new_rows = [{"idx": r.idx, "box_id": b, "cls": COCO[int(cls[j])], "conf": float(conf[j]),
                     "x1": float(xyxy[j, 0]), "y1": float(xyxy[j, 1]), "x2": float(xyxy[j, 2]), "y2": float(xyxy[j, 3])}
                    for b, j in enumerate(order_b, start=1)]
        boxes = pd.concat([boxes[boxes["idx"] != r.idx], pd.DataFrame(new_rows)], ignore_index=True)
        m = boxes_img["idx"] == r.idx
        for col, val in (("image_id", iid), ("camera_id", str(cand.iloc[k]["camera_id"])), ("width", width), ("height", height)):
            boxes_img[col] = boxes_img[col].astype(object)
            boxes_img.loc[m, col] = val
        print(f"  {r.idx} -> camera {cand.iloc[k]['camera_id']}, {r.light_regime}, {len(new_rows)} boxes")
    boxes = boxes.sort_values(["idx", "box_id"])
    boxes.to_csv(OUT / "hidden" / "boxes_hidden.csv", index=False, encoding="utf-8-sig")
    boxes_img.to_csv(OUT / "hidden" / "box_images_hidden.csv", index=False, encoding="utf-8-sig")
    print("samples fixed. Rebuild the kits with: python src/round2_kit.py --people 5")


if __name__ == "__main__":
    main()
