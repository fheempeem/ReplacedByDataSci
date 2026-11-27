"""Evaluate image-based wetness detectors against the blind hand labels (data/wetness/labels_*.csv).

Detectors compared (all scored for "road looks wet or it is raining"):
  open_meteo   rain_mm of the image's camera-hour (what the project used so far)
  zero_shot    CLIP: P(wet-type prompt) among wet + dry prompts, no training on our labels
  probe_loo    logistic regression on the CLIP embedding, leave-one-out cross-validated on the labels
Metrics: ROC AUC (+ bootstrap 95% interval). Reported for ALL labelled images and for DAYTIME only: night labels are less
reliable (street lights make dry asphalt look glossy).

Also exposes load_embeddings() and zero_shot_scores() for src/silver_wetness.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from common import ROOT

EMB_DIR = ROOT / "data" / "embeddings" / "clip_b16"
WET = ROOT / "data" / "wetness"
MODEL_ID = "openai/clip-vit-base-patch16"

PROMPTS_WET = [
    "a photo of a street in the rain",
    "a photo of a wet road with puddles and reflections",
    "a rainy day in the city, wet asphalt",
    "a photo of a wet road at night with reflections of lights in the rain",
]
PROMPTS_DRY = [
    "a photo of a dry road on a sunny day",
    "a photo of a dry street on a cloudy day",
    "a photo of a dry asphalt road, no rain",
    "a photo of a dry road at night with street lights",
]


def load_embeddings(ids: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    all_ids, mats = [], []
    for p in sorted(EMB_DIR.glob("part_*.parquet")):
        t = pq.read_table(p)
        all_ids += t.column("image_id").to_pylist()
        mats.append(np.array(t.column("emb").to_pylist(), dtype=np.float32))
    emb = np.concatenate(mats)
    if ids is None:
        return all_ids, emb
    pos = {i: k for k, i in enumerate(all_ids)}
    keep = [i for i in ids if i in pos]
    return keep, emb[[pos[i] for i in keep]]


def text_embeddings(prompts: list[str]) -> np.ndarray:
    from transformers import CLIPModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    with torch.no_grad():
        out = model.get_text_features(**tok(prompts, padding=True, return_tensors="pt"))
        out = out.pooler_output if hasattr(out, "pooler_output") else out
        return torch.nn.functional.normalize(out, dim=-1).numpy()


def zero_shot_scores(emb: np.ndarray, text: np.ndarray | None = None) -> np.ndarray:
    """P(wet-type prompt): softmax over all prompts (temperature 100, as CLIP), summed over the wet ones."""
    if text is None:
        text = text_embeddings(PROMPTS_WET + PROMPTS_DRY)
    logits = 100.0 * emb @ text.T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p[:, : len(PROMPTS_WET)].sum(axis=1)


def auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 2000, seed: int = 3) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    est = roc_auc_score(y, s)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def labelled(counter: str = "claude") -> pd.DataFrame:
    h = pd.read_csv(WET / "hidden_rain.csv", dtype={"idx": str})
    lab = pd.read_csv(WET / f"labels_{counter}.csv", dtype={"idx": str})
    d = h.merge(lab, on="idx")
    d = d[d["label"] >= 0].copy()
    d["wet"] = (d["label"] >= 1).astype(int)
    ids, emb = load_embeddings(d["image_id"].tolist())
    d = d[d["image_id"].isin(ids)].set_index("image_id").loc[ids].reset_index()
    d["_emb_idx"] = range(len(d))
    return d, emb


def main() -> None:
    import sys

    counter = sys.argv[1] if len(sys.argv) > 1 else "claude"
    d, emb = labelled(counter)
    d["zero_shot"] = zero_shot_scores(emb)
    probe = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), emb, d["wet"], cv=LeaveOneOut(), method="predict_proba")[:, 1]
    d["probe_loo"] = probe
    d["open_meteo"] = d["rain_mm"]
    print(f"labelled images: {len(d)} (wet/raining {int(d['wet'].sum())}); embeddings available for them: {len(emb)}\n")
    print(f"{'subset':<10}{'n':>4}{'wet':>5}   {'detector':<12}{'AUC':>6}   95% interval")
    out_rows = []
    for name, sub in [("all", d), ("day", d[d["light_group"] == "day"]), ("dark", d[d["light_group"] != "day"])]:
        if sub["wet"].nunique() < 2:
            print(f"{name:<10}{len(sub):>4}{int(sub['wet'].sum()):>5}   (only one class)")
            continue
        for det in ["open_meteo", "zero_shot", "probe_loo"]:
            a, lo, hi = auc_ci(sub["wet"].to_numpy(), sub[det].to_numpy())
            print(f"{name:<10}{len(sub):>4}{int(sub['wet'].sum()):>5}   {det:<12}{a:>6.2f}   {lo:.2f}-{hi:.2f}")
            out_rows.append({"subset": name, "detector": det, "auc": a, "lo": lo, "hi": hi, "n": len(sub), "n_wet": int(sub["wet"].sum())})
        print()
    exp = ROOT / "data" / "export"
    exp.mkdir(parents=True, exist_ok=True)
    # the dashboard reads wetness_validation.csv (the AI labeller); other labellers get their own file
    fname = "wetness_validation.csv" if counter == "claude" else f"wetness_validation_{counter}.csv"
    pd.DataFrame(out_rows).to_csv(exp / fname, index=False)
    d[["idx", "image_id", "light_group", "rain_mm", "label", "wet", "zero_shot", "probe_loo", "note"]].to_csv(
        WET / f"eval_scores_{counter}.csv", index=False, encoding="utf-8-sig")
    day = d[d["light_group"] == "day"].sort_values("zero_shot", ascending=False)
    print("daytime images ranked by zero-shot score (label: 0 dry, 1 wet, 2 raining):")
    print(day[["idx", "zero_shot", "label", "rain_mm", "note"]].head(14).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
