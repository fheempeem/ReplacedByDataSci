"""Draw YOLO detections (boxes + class + confidence) on images, so a person can check what was counted.

    uv run python src/annotate.py                      the 40 validation images -> data/validation/annotated/
    uv run python src/annotate.py --image 11/xxx.jpg   any image(s) by id (path relative to the raw folder)

Solid thick box   = detection kept by the pipeline (conf >= 0.25)
Thin box with '?' = extra detections that appear only if the threshold is lowered to 0.15 (NOT in the dataset)
Colours (categorical palette, colour-blind checked): car blue, motorcycle orange, bus aqua, truck yellow.
The annotated validation images show YOLO's answers: do not give them to someone who is about to hand-count.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from common import RAW_DIR, ROOT

COCO = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {"car": (42, 120, 214), "motorcycle": (235, 104, 52), "bus": (27, 175, 122), "truck": (237, 161, 0)}
CONF_USED, CONF_LOW = 0.25, 0.15
SCALE = 2
VAL = ROOT / "data" / "validation"


def font(size: int) -> ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_image(im: Image.Image, boxes: list[tuple], title: str) -> Image.Image:
    big = im.convert("RGB").resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS)
    d = ImageDraw.Draw(big, "RGBA")
    f_small = font(13)
    for cls, conf, (x1, y1, x2, y2) in sorted(boxes, key=lambda b: b[1]):  # strong boxes drawn last (on top)
        x1, y1, x2, y2 = (v * SCALE for v in (x1, y1, x2, y2))
        col = COLORS[cls]
        strong = conf >= CONF_USED
        d.rectangle([x1, y1, x2, y2], outline=col + (255 if strong else 200,), width=3 if strong else 1)
        text = f"{cls} {conf:.2f}" if strong else f"? {cls} {conf:.2f}"
        w = d.textlength(text, font=f_small)
        ty = max(0, y1 - 16)
        d.rectangle([x1, ty, x1 + w + 6, ty + 16], fill=col + (235 if strong else 150,))
        d.text((x1 + 3, ty + 1), text, fill=(255, 255, 255) if cls != "truck" else (11, 11, 11), font=f_small)
    n_used = sum(c >= CONF_USED for _, c, _ in boxes)
    n_extra = len(boxes) - n_used
    banner = f"{title}   |   counted: {n_used}   extra if conf 0.15: +{n_extra}"
    d.rectangle([0, 0, big.width, 24], fill=(0, 0, 0, 200))
    d.text((8, 4), banner, fill=(255, 255, 255), font=font(14))
    return big


def detect(model, imgs: list[np.ndarray]) -> list[list[tuple]]:
    res = model.predict(imgs, imgsz=640, conf=CONF_LOW, classes=list(COCO), device=0, half=True, verbose=False)
    out = []
    for r in res:
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        out.append([(COCO[int(k)], float(c), tuple(float(v) for v in b)) for k, c, b in zip(cls, conf, xyxy)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="*", help="image ids (relative to the raw folder); default: validation sample")
    ap.add_argument("--out", help="output folder")
    args = ap.parse_args()

    from ultralytics import YOLO, settings

    models = ROOT / "data" / "models"
    settings.update({"weights_dir": str(models)})
    model = YOLO(str(models / "yolov8m.pt"))

    if args.image:
        ids = args.image
        names = [Path(i).stem[:60] for i in ids]
        out_dir = Path(args.out) if args.out else ROOT / "data" / "annotated"
        meta = None
    else:
        hidden = pd.read_csv(VAL / "sample_yolo_hidden.csv", dtype={"idx": str})
        ids, names = hidden["image_id"].tolist(), hidden["idx"].tolist()
        out_dir = Path(args.out) if args.out else VAL / "annotated"
        meta = hidden
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [Image.open(RAW_DIR / i) for i in ids]
    dets = detect(model, [np.asarray(im.convert("RGB"))[:, :, ::-1].copy() for im in imgs])
    for name, im, boxes in zip(names, imgs, dets):
        draw_image(im, boxes, f"#{name}").save(out_dir / f"{name}.png")
    print(f"{len(ids)} annotated images -> {out_dir}")

    if meta is not None:
        write_gallery(out_dir, meta, dets)


def write_gallery(out_dir: Path, meta: pd.DataFrame, dets: list[list[tuple]]) -> None:
    """index.html: every annotated image with manual vs YOLO counts, for quick review in a browser."""
    manual = None
    mp = VAL / "manual_counts_claude.csv"
    if mp.exists():
        manual = pd.read_csv(mp, dtype={"idx": str}).set_index("idx")
    blind = pd.read_csv(VAL / "sample_blind.csv", dtype={"idx": str}).set_index("idx")
    rows = []
    for (_, r), boxes in zip(meta.iterrows(), dets):
        idx = r["idx"]
        used = [b for b in boxes if b[1] >= CONF_USED]
        by = {c: sum(1 for b in used if b[0] == c) for c in COLORS}
        extra = len(boxes) - len(used)
        man = ""
        if manual is not None and idx in manual.index:
            m = manual.loc[idx]
            man = f"{int(m['car'] + m['motorcycle'] + m['bus'] + m['truck'])} (car {int(m['car'])}, moto {int(m['motorcycle'])}, bus {int(m['bus'])}, truck {int(m['truck'])})"
        rows.append(
            f"<figure><img src='{idx}.png' loading='lazy'><figcaption><b>#{idx}</b> {html.escape(str(blind.loc[idx, 'light_regime']))}, "
            f"camera {html.escape(str(blind.loc[idx, 'camera_id']))}, {html.escape(str(blind.loc[idx, 'captured']))}<br>"
            f"YOLO counted <b>{len(used)}</b> (car {by['car']}, moto {by['motorcycle']}, bus {by['bus']}, truck {by['truck']}); "
            f"+{extra} more at conf 0.15<br>Manual count: <b>{html.escape(man) or 'n/a'}</b></figcaption></figure>"
        )
    page = (
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>YOLO annotated validation images</title>"
        "<style>:root{color-scheme:light dark}body{font:15px system-ui,'Segoe UI',sans-serif;margin:16px;max-width:1500px}"
        "h1{font-size:20px}.legend span{display:inline-block;padding:2px 8px;margin-right:6px;border-radius:3px;color:#fff;font-size:13px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}"
        "figure{margin:0}img{width:100%;height:auto;border-radius:4px}figcaption{font-size:13px;line-height:1.5;padding-top:4px}</style>"
        "<h1>YOLOv8m detections on the 40 hand-counted images</h1>"
        "<p class=legend><span style='background:#2a78d6'>car</span><span style='background:#eb6834'>motorcycle</span>"
        "<span style='background:#1baf7a'>bus</span><span style='background:#eda100;color:#0b0b0b'>truck</span>"
        "Thick box = counted by the pipeline (conf &ge; 0.25). Thin box with ? = would appear only at conf 0.15.</p>"
        "<p>The manual counts were made by an AI model, not a person; use these images to check them yourself.</p>"
        f"<div class=grid>{''.join(rows)}</div>"
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print("gallery ->", out_dir / "index.html")


if __name__ == "__main__":
    main()
