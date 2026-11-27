"""Does a different YOLO setting close the gap to the manual counts? (40-image check, GPU, ~1 minute)

Re-runs the same pretrained model with other image sizes / confidence thresholds on the validation sample and compares
with the manual counts. Caution: the settings are judged on the same 40 images the manual counts came from, so pick
a setting from here only as a hypothesis and confirm on fresh images before re-running the whole dataset.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from PIL import Image

from common import RAW_DIR, ROOT
from validation_compare import CLASSES, VAL, load

COCO = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
SETTINGS = [(640, 0.25), (960, 0.25), (1280, 0.25), (1280, 0.15), (1280, 0.10), (640, 0.15)]


def main() -> None:
    from ultralytics import YOLO, settings

    models = ROOT / "data" / "models"
    settings.update({"weights_dir": str(models)})
    name = sys.argv[1] if len(sys.argv) > 1 else "yolov8m.pt"
    model = YOLO(str(models / name))
    df = load("claude")
    hidden = pd.read_csv(VAL / "sample_yolo_hidden.csv", dtype={"idx": str})
    # find source files through the lakehouse-free route: image_id is the relative path
    imgs = [np.asarray(Image.open(RAW_DIR / iid).convert("RGB"))[:, :, ::-1].copy() for iid in hidden["image_id"]]

    rows = []
    for imgsz, conf in SETTINGS:
        res = model.predict(imgs, imgsz=imgsz, conf=conf, classes=list(COCO), device=0, half=True, verbose=False)
        counts = []
        for r in res:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            counts.append({c: int((cls == k).sum()) for k, c in COCO.items()})
        c = pd.DataFrame(counts)
        tot = c.sum(axis=1).to_numpy()
        for grp, mask in [("day", (df["group"] == "day").to_numpy()), ("dusk/night", (df["group"] != "day").to_numpy())]:
            man = df["manual_total"].to_numpy()[mask]
            rows.append({
                "setting": f"{imgsz}/{conf}", "group": grp, "ratio": tot[mask].sum() / man.sum(),
                "mae": np.abs(tot[mask] - man).mean(),
                "moto_ratio": c["motorcycle"].to_numpy()[mask].sum() / max(1, df["man_motorcycle"].to_numpy()[mask].sum()),
                "car_ratio": c["car"].to_numpy()[mask].sum() / max(1, df["man_car"].to_numpy()[mask].sum()),
                "over": int((tot[mask] > man * 1.5).sum()),
            })
    out = pd.DataFrame(rows).pivot(index="setting", columns="group")
    out = out.reindex([f"{a}/{b}" for a, b in SETTINGS])
    print(f"model {name}  (ratio = yolo/manual total; 'over' = images where yolo > 1.5x manual)")
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
