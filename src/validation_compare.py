"""Compare manual (hand) counts with YOLO counts on the validation sample.

Reads data/validation/{sample_blind.csv, sample_yolo_hidden.csv, manual_counts_*.csv}.
Writes data/validation/yolo_vs_manual.csv and prints the summary used in the report/dashboard.

Main number: the YOLO/manual ratio of total vehicles (sum over images) per light group; 1.0 = no bias,
0.6 = YOLO finds 60% of the vehicles a person counts. Rank agreement (Spearman) matters separately: the congestion
index only needs counts to be proportional to traffic, not exact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

VAL = Path(__file__).resolve().parents[1] / "data" / "validation"
CLASSES = ["car", "motorcycle", "bus", "truck"]


def load(counter: str = "claude") -> pd.DataFrame:
    blind = pd.read_csv(VAL / "sample_blind.csv", dtype={"idx": str})[["idx", "light_regime", "camera_id", "captured"]]
    yolo = pd.read_csv(VAL / "sample_yolo_hidden.csv", dtype={"idx": str})
    man = pd.read_csv(VAL / f"manual_counts_{counter}.csv", dtype={"idx": str}).fillna({"notes": ""})
    df = blind.merge(man, on="idx").merge(yolo, on="idx")
    for c in CLASSES:
        df[f"man_{c}"] = df[c]
        df[f"yolo_{c}"] = df[f"n_{c}"]
    df["manual_total"] = df[[f"man_{c}" for c in CLASSES]].sum(axis=1)
    df["yolo_total"] = df[[f"yolo_{c}" for c in CLASSES]].sum(axis=1)
    df["group"] = np.where(df["light_regime"] == "day", "day", "dusk/night")
    return df[["idx", "light_regime", "group", "camera_id", "captured", "manual_total", "yolo_total",
               *[f"man_{c}" for c in CLASSES], *[f"yolo_{c}" for c in CLASSES], "notes"]]


def ratio_ci(d: pd.DataFrame, n_boot: int = 4000, seed: int = 5) -> tuple[float, float, float]:
    """sum(yolo)/sum(manual) with a 95% bootstrap interval over images."""
    rng = np.random.default_rng(seed)
    y, m = d["yolo_total"].to_numpy(float), d["manual_total"].to_numpy(float)
    est = y.sum() / m.sum()
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        boots.append(y[i].sum() / m[i].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def summarise(df: pd.DataFrame) -> dict:
    out = {"groups": {}, "classes": {}}
    for name, d in [("all", df), ("day", df[df["group"] == "day"]), ("dusk/night", df[df["group"] != "day"])]:
        est, lo, hi = ratio_ci(d)
        rho = spearmanr(d["manual_total"], d["yolo_total"])
        out["groups"][name] = {
            "n": len(d), "manual_mean": d["manual_total"].mean(), "yolo_mean": d["yolo_total"].mean(),
            "ratio": est, "ratio_lo": lo, "ratio_hi": hi,
            "mean_error": (d["yolo_total"] - d["manual_total"]).mean(),
            "mae": (d["yolo_total"] - d["manual_total"]).abs().mean(),
            "spearman": float(rho.statistic), "pearson": float(pearsonr(d["manual_total"], d["yolo_total"]).statistic),
            "yolo_zero_when_manual_positive": int(((d["yolo_total"] == 0) & (d["manual_total"] > 0)).sum()),
        }
    for grp, d in [("day", df[df["group"] == "day"]), ("dusk/night", df[df["group"] != "day"])]:
        out["classes"][grp] = {
            c: {"manual": int(d[f"man_{c}"].sum()), "yolo": int(d[f"yolo_{c}"].sum()),
                "ratio": float(d[f"yolo_{c}"].sum() / d[f"man_{c}"].sum()) if d[f"man_{c}"].sum() else float("nan")}
            for c in CLASSES
        }
    return out


def main() -> None:
    counter = sys.argv[1] if len(sys.argv) > 1 else "claude"
    df = load(counter)
    df.to_csv(VAL / f"yolo_vs_manual_{counter}.csv", index=False, encoding="utf-8-sig")
    s = summarise(df)
    print(f"counter: {counter}   images: {len(df)}\n")
    print(f"{'group':<11}{'n':>3} {'manual':>7} {'yolo':>6} {'ratio':>6} {'95% interval':>14} {'err':>6} {'MAE':>5} {'spearman':>9} {'yolo=0 while people>0':>22}")
    for name, g in s["groups"].items():
        print(f"{name:<11}{g['n']:>3} {g['manual_mean']:>7.1f} {g['yolo_mean']:>6.1f} {g['ratio']:>6.2f} "
              f"{g['ratio_lo']:>6.2f}-{g['ratio_hi']:<6.2f}{g['mean_error']:>+6.1f} {g['mae']:>5.1f} {g['spearman']:>9.2f} {g['yolo_zero_when_manual_positive']:>12}")
    print("\nper class (sum over images)   manual  yolo  ratio")
    for grp, cl in s["classes"].items():
        for c, v in cl.items():
            print(f"  {grp:<11}{c:<12}{v['manual']:>6}{v['yolo']:>6}{v['ratio']:>7.2f}")
    worst = df.assign(err=df["yolo_total"] - df["manual_total"]).sort_values("err").head(5)
    print("\nlargest under-counts (yolo - manual):")
    print(worst[["idx", "light_regime", "manual_total", "yolo_total", "err", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
