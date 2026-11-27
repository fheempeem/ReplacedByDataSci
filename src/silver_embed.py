"""Silver stage: CLIP image embeddings for every quality-checked image (input for the wet-road / rain detector).

Model: openai/clip-vit-base-patch16 (image tower, fp16 on GPU). Each image is squashed to 224x224 (not centre-cropped, so
the whole camera view is kept). Embeddings are L2-normalised, stored as float16 in
data/embeddings/clip_b16/part_NNNN.parquet (image_id + 512 numbers). Resumable and idempotent: images already embedded
are skipped, and a part file is written every CHUNK images.

Usage: python src/silver_embed.py [--limit N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from common import RAW_DIR, ROOT, open_catalog

MODEL_ID = "openai/clip-vit-base-patch16"
OUT = ROOT / "data" / "embeddings" / "clip_b16"
CHUNK = 4096
BATCH = 128
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class Imgs(torch.utils.data.Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(RAW_DIR / self.paths[i]) as im:
            a = np.asarray(im.convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
        return torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1))


def embedded_ids() -> set[str]:
    ids: set[str] = set()
    for p in OUT.glob("part_*.parquet"):
        ids.update(pq.read_table(p, columns=["image_id"]).column("image_id").to_pylist())
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    con = open_catalog().connection
    todo = con.sql("SELECT image_id FROM bkk.silver.image_features ORDER BY captured_at, camera_id").df()["image_id"].tolist()
    done = embedded_ids()
    todo = [i for i in todo if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"to embed: {len(todo)} (already {len(done)})", flush=True)
    if not todo:
        return

    from transformers import CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    n_parts = len(list(OUT.glob("part_*.parquet")))
    for c0 in range(0, len(todo), CHUNK):
        ids = todo[c0 : c0 + CHUNK]
        loader = torch.utils.data.DataLoader(Imgs(ids), batch_size=BATCH, num_workers=4, shuffle=False)
        feats = []
        with torch.no_grad():
            for x in loader:
                x = x.to(device, dtype=model.dtype)
                out = model.get_image_features(pixel_values=x)
                out = out.pooler_output if hasattr(out, "pooler_output") else out
                f = torch.nn.functional.normalize(out.float(), dim=-1)
                feats.append(f.cpu().numpy().astype(np.float16))
        emb = np.concatenate(feats)
        table = pa.table({"image_id": ids, "emb": pa.FixedSizeListArray.from_arrays(pa.array(emb.ravel(), pa.float16()), emb.shape[1])})
        pq.write_table(table, OUT / f"part_{n_parts:04d}.parquet")
        n_parts += 1
        print(f"  {min(c0 + CHUNK, len(todo))}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
