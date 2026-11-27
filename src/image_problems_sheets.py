"""Contact sheets of the most extreme images per problem metric, for calibrating by eye.  data/quality_scan/sheet_<metric>.jpg"""
import sys

import pandas as pd
from PIL import Image, ImageDraw

from common import RAW_DIR, ROOT

OUT = ROOT / "data" / "quality_scan"
df = pd.read_parquet(OUT / "image_metrics.parquet")
groups = {
    "lowest_std": df.nsmallest(12, "std"),
    "flat_midtone": df.nlargest(12, "flat_mid_blocks"),
    "lowest_sharpness": df.nsmallest(12, "lap_var"),
    "most_black": df.nlargest(12, "black"),
    "most_white": df.nlargest(12, "white"),
}
only = sys.argv[1:] or list(groups)
T = (320, 264)
for name in only:
    g = groups[name].reset_index(drop=True)
    sheet = Image.new("RGB", (T[0] * 4, T[1] * 3), (30, 30, 30))
    for i, r in g.iterrows():
        with Image.open(RAW_DIR / r["image_id"]) as im:
            t = im.convert("RGB").resize(T)
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 150, 16], fill=(0, 0, 0))
        d.text((3, 2), f"{i + 1} std{r['std']:.0f} lap{r['lap_var']:.0f}", fill=(255, 255, 0))
        sheet.paste(t, ((i % 4) * T[0], (i // 4) * T[1]))
    sheet.save(OUT / f"sheet_{name}.jpg", quality=85)
    print("saved", OUT / f"sheet_{name}.jpg")
