"""Scan every checked image for problems that make it useless for labelling (and suspicious for analysis).

    uv run python src/image_problems.py

Per image (grayscale): mean, std, share of near-white / near-black pixels, share of FLAT mid-tone 16x16 blocks (typical of
decoder glitches and grey fill-ins), share of rows identical to the row above (smeared / frozen strips), sharpness
(variance of the Laplacian) and a normalised sharpness (against the image's own contrast).
Writes data/quality_scan/image_metrics.parquet. Judging what is "too bad" is done separately, by looking at examples.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from common import RAW_DIR, ROOT, open_catalog

OUT = ROOT / "data" / "quality_scan"


def metrics(image_id: str) -> dict:
    try:
        with Image.open(RAW_DIR / image_id) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
            g = np.asarray(im.convert("L"), dtype=np.float32)
    except Exception:
        return {"image_id": image_id, "readable": False}
    r, gg, bb = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green = float(((gg > 90) & (r < 70) & (bb < 70) & (gg - r > 60)).mean())      # solid green fill (video decoder failure)
    purple = float(((r > 90) & (bb > 90) & (gg < 60) & (r - gg > 50)).mean())     # magenta/purple fill
    h, w = g.shape
    hb, wb = h // 16 * 16, w // 16 * 16
    blocks = g[:hb, :wb].reshape(hb // 16, 16, wb // 16, 16).transpose(0, 2, 1, 3).reshape(-1, 256)
    b_std, b_mean = blocks.std(axis=1), blocks.mean(axis=1)
    lap = (g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    gi = g.astype(np.uint8)
    return {
        "image_id": image_id, "readable": True, "width": w, "height": h,
        "mean": float(g.mean()), "std": float(g.std()), "green": green, "purple": purple,
        "white": float((g >= 245).mean()), "black": float((g <= 10).mean()),
        "flat_mid_blocks": float(((b_std < 1.5) & (b_mean > 35) & (b_mean < 215)).mean()),
        # blocks that are flat AND exactly the decoder's grey fill (128 +-3): the signature of a grey-smear glitch
        "gray_fill_blocks": float(((b_std < 1.5) & (np.abs(b_mean - 128) <= 3)).mean()),
        "flat_any_blocks": float((b_std < 1.5).mean()),
        "same_row": float(np.all(gi[1:] == gi[:-1], axis=1).mean()),
        "lap_var": float(lap.var()),
    }


def main() -> None:
    con = open_catalog().connection
    ids = con.sql("SELECT image_id FROM bkk.silver.image_features ORDER BY image_id").df()["image_id"].tolist()
    print(f"scanning {len(ids)} images ...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(metrics, ids, chunksize=256))
    df = pd.DataFrame(rows)
    df["sharp_norm"] = df["lap_var"] / (df["std"] ** 2 + 1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "image_metrics.parquet", index=False)
    print(f"unreadable: {int((~df['readable']).sum())}")
    for col in ("std", "white", "black", "flat_mid_blocks", "same_row", "lap_var", "sharp_norm"):
        q = df[col].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(3).tolist()
        print(f"{col:<16} p0.1 {q[0]:>9}  p1 {q[1]:>9}  median {q[2]:>9}  p99 {q[3]:>9}  p99.9 {q[4]:>9}")
    print("saved", OUT / "image_metrics.parquet")


if __name__ == "__main__":
    main()
