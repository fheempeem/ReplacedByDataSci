"""Summarise labels/counts returned by teammates: how much do people agree, and how do the machines compare with people?

    1. Put the returned files in place (or point --inbox at a folder holding them; they are COPIED):
         labels_<name>.csv         -> data/wetness/
         manual_counts_<name>.csv  -> data/validation/
    2. uv run python src/agreement.py [--inbox data/returned] [--exclude-ai]

Wet-road task:   pairwise agreement (% and Cohen's kappa), then AUC of Open-Meteo rain and CLIP wetness against the consensus
                 of the human labellers (daytime only; night labels are unreliable).
Counting task:   pairwise agreement of the totals, then the YOLO/human ratio for each counter and for the average human.
The AI labeller is called `claude`; it is shown for comparison but left out of the human consensus.
"""
from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from common import ROOT
from validation_compare import CLASSES

WET = ROOT / "data" / "wetness"
VAL = ROOT / "data" / "validation"
AI = "claude"


def collect(inbox: Path) -> None:
    for p in inbox.glob("*.csv"):
        if p.name.startswith("labels_"):
            shutil.copy(p, WET / p.name)
        elif p.name.startswith("manual_counts_"):
            shutil.copy(p, VAL / p.name)
        else:
            print(f"skipped (unknown name): {p.name}")


def wetness_labels() -> pd.DataFrame:
    cols = {}
    for p in sorted(WET.glob("labels_*.csv")):
        name = p.stem[len("labels_"):]
        if name == "blank":
            continue
        d = pd.read_csv(p, dtype={"idx": str})
        d["label"] = pd.to_numeric(d["label"], errors="coerce")
        d.loc[d["label"] < 0, "label"] = np.nan  # -1 = can't tell
        s = d.set_index("idx")["label"]
        if s.notna().sum() == 0:
            print(f"  (no labels yet in {p.name})")
            continue
        cols[name] = s
    return pd.DataFrame(cols)


def wetness_report(exclude_ai: bool) -> None:
    L = wetness_labels()
    print("=== WET-ROAD LABELS ===")
    if L.empty:
        print("no label files found")
        return
    humans = [c for c in L.columns if c != AI]
    print(f"labellers: {', '.join(L.columns)} (humans: {len(humans)})  images: {len(L)}")
    for a, b in itertools.combinations(L.columns, 2):
        both = L[[a, b]].dropna()
        if len(both) < 5:
            continue
        agree = (both[a] == both[b]).mean()
        k3 = cohen_kappa_score(both[a].astype(int), both[b].astype(int), weights="linear") if both.stack().nunique() > 1 else float("nan")
        kb = cohen_kappa_score((both[a] >= 1).astype(int), (both[b] >= 1).astype(int)) if ((both[a] >= 1).nunique() > 1 and (both[b] >= 1).nunique() > 1) else float("nan")
        print(f"  {a} vs {b}: n={len(both)}  same 0/1/2 label {agree:.0%}  weighted kappa {k3:.2f}  wet-vs-dry kappa {kb:.2f}")
    print("  (kappa: <0.4 poor, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 near perfect)")

    if not humans:
        print("\n  NO HUMAN LABELS YET. Showing the AI labeller alone (same numbers as docs/wetness_validation.md).")
    use = humans if humans else list(L.columns)
    basis = f"{len(use)} human labeller(s)" if humans else "the AI labeller only"
    wet = (L[use] >= 1).where(L[use].notna())
    share = wet.mean(axis=1)  # share of humans calling it wet
    cons = (share >= 0.5).astype(int).where(share.notna() & (share != 0.5))  # ties dropped
    h = pd.read_csv(WET / "hidden_rain.csv", dtype={"idx": str}).set_index("idx")
    from wetness_eval import load_embeddings, zero_shot_scores

    ids, emb = load_embeddings(h["image_id"].tolist())
    zs = pd.Series(zero_shot_scores(emb), index=h.reset_index().set_index("image_id").loc[ids, "idx"].values)
    d = pd.DataFrame({"cons": cons, "rain_mm": h["rain_mm"], "zero_shot": zs, "light": h["light_group"]}).dropna(subset=["cons"])
    print(f"\n  consensus of {basis}: {int(d['cons'].sum())} wet of {len(d)} images")
    for name, sub in [("daytime", d[d["light"] == "day"]), ("all (night labels less reliable)", d)]:
        if sub["cons"].nunique() < 2:
            print(f"  {name}: only one class, AUC undefined")
            continue
        print(f"  {name:<34} n={len(sub):>2}  AUC Open-Meteo {roc_auc_score(sub['cons'], sub['rain_mm']):.2f}   AUC CLIP {roc_auc_score(sub['cons'], sub['zero_shot']):.2f}")


def count_report() -> None:
    print("\n=== VEHICLE COUNTS ===")
    C = {}
    for p in sorted(VAL.glob("manual_counts_*.csv")):
        name = p.stem[len("manual_counts_"):]
        d = pd.read_csv(p, dtype={"idx": str}).set_index("idx")
        d[CLASSES] = d[CLASSES].apply(pd.to_numeric, errors="coerce")
        d["total"] = d[CLASSES].sum(axis=1, min_count=len(CLASSES))
        if d["total"].notna().sum() == 0:
            continue
        C[name] = d
    if not C:
        print("no count files found")
        return
    print(f"counters: {', '.join(C)}")
    for a, b in itertools.combinations(C, 2):
        both = pd.concat([C[a]["total"], C[b]["total"]], axis=1, keys=[a, b]).dropna()
        if len(both) < 5:
            continue
        diff = (both[a] - both[b]).abs().mean()
        print(f"  {a} vs {b}: n={len(both)}  mean totals {both[a].mean():.1f} / {both[b].mean():.1f}  mean |difference| {diff:.1f}  Spearman {spearmanr(both[a], both[b]).statistic:.2f}")
    yolo = pd.read_csv(VAL / "sample_yolo_hidden.csv", dtype={"idx": str}).set_index("idx")
    yolo["yolo_total"] = yolo[[f"n_{c}" for c in CLASSES]].sum(axis=1)
    blind = pd.read_csv(VAL / "sample_blind.csv", dtype={"idx": str}).set_index("idx")
    grp = np.where(blind["light_regime"] == "day", "day", "dusk/night")
    humans = [c for c in C if c != AI]
    avg = pd.concat([C[c]["total"] for c in humans], axis=1).mean(axis=1) if humans else None
    print("\n  YOLO found this share of the vehicles people counted (sum over images):")
    for name in list(C) + (["average of humans"] if len(humans) > 1 else []):
        man = avg if name == "average of humans" else C[name]["total"]
        df = pd.DataFrame({"man": man, "yolo": yolo["yolo_total"], "g": pd.Series(grp, index=blind.index)}).dropna(subset=["man"])
        parts = [f"{g}: {sub['yolo'].sum() / sub['man'].sum():.2f} (n={len(sub)})" for g, sub in df.groupby("g")]
        print(f"   {name:<20} all {df['yolo'].sum() / df['man'].sum():.2f}   " + "   ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", type=Path, help="folder with returned CSV files to copy into place first")
    ap.add_argument("--exclude-ai", action="store_true")
    a = ap.parse_args()
    if a.inbox:
        collect(a.inbox)
    wetness_report(a.exclude_ai)
    count_report()


if __name__ == "__main__":
    main()
