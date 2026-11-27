"""Contact sheets of BORDERLINE images (random 12 per band) to decide where 'too bad to label' starts.  data/quality_scan/band_<name>.jpg"""
import pandas as pd
from PIL import Image, ImageDraw

from common import RAW_DIR, ROOT

OUT = ROOT / "data" / "quality_scan"
d = pd.read_parquet(OUT / "image_metrics.parquet")
bands = {
    "purple_gt5": d[d.purple > 0.05],
    "green_5_15": d[(d.green > 0.05) & (d.green <= 0.15)],
    "lap_60_100": d[(d.lap_var >= 60) & (d.lap_var < 100)],
    "lap_100_200": d[(d.lap_var >= 100) & (d.lap_var < 200)],
    "lap_200_400": d[(d.lap_var >= 200) & (d.lap_var < 400)],
}
T = (320, 264)
for name, g in bands.items():
    g = g.sample(min(12, len(g)), random_state=3).reset_index(drop=True)
    sheet = Image.new("RGB", (T[0] * 4, T[1] * 3), (30, 30, 30))
    for i, r in g.iterrows():
        with Image.open(RAW_DIR / r["image_id"]) as im:
            t = im.convert("RGB").resize(T)
        dr = ImageDraw.Draw(t)
        dr.rectangle([0, 0, 190, 16], fill=(0, 0, 0))
        dr.text((3, 2), f"{i + 1} lap{r['lap_var']:.0f} g{r['green']:.2f} p{r['purple']:.2f}", fill=(255, 255, 0))
        sheet.paste(t, ((i % 4) * T[0], (i // 4) * T[1]))
    sheet.save(OUT / f"band_{name}.jpg", quality=85)
    print("saved", name, len(g))
