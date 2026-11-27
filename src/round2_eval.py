"""Evaluate round-2 returns:  python src/round2_eval.py [--inbox <folder with the CSVs people sent back>]

Reads data/round2/returns/{rain,boxes,counts}_p<k>_<name>.csv (the inbox files are COPIED there) and joins them with the hidden
answers in data/round2/hidden/.

BOXES  every box was judged by 2 people. A box is "useful" if judged ok or wrong_class (it marks a real vehicle), "extra" if
       not_vehicle or duplicate. For each confidence threshold t >= 0.15 (what the pipeline would keep if it used t):
         precision_t  = useful boxes / boxes with conf >= t
         recall_t     = useful boxes with conf >= t / (all useful boxes + vehicles people said had no box)
         count ratio  = boxes kept / true vehicles    (1.0 = the pipeline count would be right)
       95% intervals: bootstrap over IMAGES. Caveat: vehicles detected below 0.15 are not in the denominator, so recall is
       slightly optimistic, and boxes show the predicted class, which can nudge the judge.
RAIN   weighted AUC of Open-Meteo rain and CLIP wetness against the human consensus, on the 150 fresh daytime images
       (weights = stratum size / sample size, so choosing images by a detector's score does not favour that detector);
       also the 60 round-1 images; agreement between the two raters (kappa).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from common import ROOT
from validation_compare import CLASSES, VAL

R2 = ROOT / "data" / "round2"
RET = R2 / "returns"
HID = R2 / "hidden"
THRESH = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


def collect(inbox: Path) -> None:
    RET.mkdir(parents=True, exist_ok=True)
    for p in inbox.glob("*.csv"):
        if p.name.split("_")[0] in ("rain", "boxes", "counts"):
            shutil.copy(p, RET / p.name)


def read_all(task: str) -> pd.DataFrame:
    frames = []
    for p in sorted(RET.glob(f"{task}_p*_*.csv")):
        d = pd.read_csv(p, dtype=str, keep_default_na=False)
        d["rater"] = p.stem.split("_", 1)[1]  # "p3_Mint"
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------------------------------------------------------------------------- boxes
def boxes_report(n_boot: int = 1000, seed: int = 1) -> pd.DataFrame | None:
    raw = read_all("boxes")
    if raw.empty:
        print("BOXES: no files in data/round2/returns/")
        return None
    if "bad" in raw.columns:  # images a judge marked as unusable (glitch, colour bars, unreadable) are left out entirely
        bad_imgs = set(raw.loc[(raw["kind"] == "image") & (raw["bad"] == "1"), "idx"])
        raw = raw[~raw["idx"].isin(bad_imgs)]
        print(f"BOXES: images marked BAD by at least one judge and excluded: {len(bad_imgs)} {sorted(bad_imgs)[:12]}")
    hid = pd.read_csv(HID / "boxes_hidden.csv", dtype={"idx": str})
    img = pd.read_csv(HID / "box_images_hidden.csv", dtype={"idx": str})[["idx", "light_regime"]]
    b = raw[raw["kind"] == "box"].copy()
    b["box_id"] = b["box_id"].astype(int)
    b = b[b["verdict"] != ""].merge(hid, on=["idx", "box_id"]).merge(img, on="idx")
    b["useful"] = b["verdict"].isin(["ok", "wrong_class"])
    b["class_ok"] = b["verdict"] == "ok"
    m = raw[raw["kind"] == "image"].copy()
    for c in ("missed_car", "missed_motorcycle", "missed_other"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["missed"] = m[["missed_car", "missed_motorcycle", "missed_other"]].sum(axis=1, min_count=3)
    m = m.dropna(subset=["missed"])
    n_raters = b["rater"].nunique()
    print(f"BOXES: {n_raters} raters, {b[['idx', 'box_id']].drop_duplicates().shape[0]} distinct boxes judged, {b['idx'].nunique()} images")

    # agreement between raters on "is this a real vehicle"
    wide = b.assign(useful_i=b["useful"].astype(int)).pivot_table(index=["idx", "box_id"], columns="rater", values="useful_i", aggfunc="first")
    ks = []
    for r1 in wide.columns:
        for r2 in wide.columns:
            if r1 < r2:
                both = wide[[r1, r2]].dropna().astype(int)
                if len(both) >= 10 and both[r1].nunique() > 1 and both[r2].nunique() > 1:
                    ks.append((r1, r2, len(both), (both[r1] == both[r2]).mean(), cohen_kappa_score(both[r1], both[r2])))
    if ks:
        print("  agreement on 'real vehicle?':", "; ".join(f"{a} vs {c}: {n} boxes, {p:.0%} same, kappa {k:.2f}" for a, c, n, p, k in ks[:6]))
    print(f"  class correct among real vehicles: {b.loc[b['useful'], 'class_ok'].mean():.0%}; "
          f"duplicates {b['verdict'].eq('duplicate').mean():.1%}; not a vehicle {b['verdict'].eq('not_vehicle').mean():.1%}")

    images = sorted(b["idx"].unique())
    miss_by_img = m.groupby("idx")["missed"].mean()  # mean over the raters of that image

    def metrics(sel_imgs: list[str]) -> pd.DataFrame:
        bb = b[b["idx"].isin(sel_imgs)]
        useful_all = bb.groupby("idx")["useful"].sum() / bb.groupby("idx")["rater"].nunique()  # per-rater average
        true_total = float((useful_all + miss_by_img.reindex(useful_all.index).fillna(0)).sum())
        rows = []
        for t in THRESH:
            k = bb[bb["conf"] >= t]
            n_r = bb.groupby("idx")["rater"].nunique()
            kept = float((k.groupby("idx").size().reindex(n_r.index).fillna(0) / n_r).sum())
            useful = float((k[k["useful"]].groupby("idx").size().reindex(n_r.index).fillna(0) / n_r).sum())
            rows.append({"threshold": t, "boxes_kept": kept, "precision": useful / kept if kept else np.nan,
                         "recall": useful / true_total, "count_ratio": kept / true_total})
        return pd.DataFrame(rows)

    point = metrics(images)
    rng = np.random.default_rng(seed)
    boots = [metrics(list(rng.choice(images, len(images), replace=True))) for _ in range(n_boot)]
    for col in ("precision", "recall", "count_ratio"):
        arr = np.stack([x[col].to_numpy() for x in boots])
        point[col + "_lo"], point[col + "_hi"] = np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0)
    point["f1"] = 2 * point["precision"] * point["recall"] / (point["precision"] + point["recall"])
    print("\n  threshold | boxes kept/img-set | precision (95%)      | recall (95%)         | count ratio (95%)   | F1")
    for r in point.itertuples():
        print(f"  {r.threshold:>8.2f} | {r.boxes_kept:>17.1f} | {r.precision:.2f} ({r.precision_lo:.2f}-{r.precision_hi:.2f})  | "
              f"{r.recall:.2f} ({r.recall_lo:.2f}-{r.recall_hi:.2f})  | {r.count_ratio:.2f} ({r.count_ratio_lo:.2f}-{r.count_ratio_hi:.2f}) | {r.f1:.2f}")
    # the band that the 0.15 vs 0.25 decision is about
    band = b[(b["conf"] >= 0.15) & (b["conf"] < 0.25)]
    if len(band):
        print(f"\n  boxes with conf 0.15-0.25 (what lowering the threshold adds): {len(band)} ratings, "
              f"{band['useful'].mean():.0%} marked as real vehicles (vs {b.loc[b['conf'] >= 0.25, 'useful'].mean():.0%} at >= 0.25)")
    for reg, sub in b.groupby("light_regime"):
        print(f"  {reg:<9} precision overall {sub['useful'].mean():.2f}  at conf 0.15-0.25: "
              f"{sub.loc[(sub['conf'] < 0.25), 'useful'].mean():.2f}  at >= 0.25: {sub.loc[sub['conf'] >= 0.25, 'useful'].mean():.2f}")
    point.to_csv(R2 / "results_boxes.csv", index=False)
    return point


# ------------------------------------------------------------------------------------------------ rain
def weighted_auc(y, s, w) -> float:
    return roc_auc_score(y, s, sample_weight=w) if len(set(y)) == 2 else float("nan")


def rain_report(n_boot: int = 1000, seed: int = 2) -> None:
    raw = read_all("rain")
    if raw.empty:
        print("RAIN: no files in data/round2/returns/")
        return
    bad_imgs = set(raw.loc[raw["label"].str.strip().str.lower() == "bad", "idx"])
    raw = raw[~raw["idx"].isin(bad_imgs)].copy()  # marked "Bad image" by at least one rater: excluded entirely
    print(f"\nRAIN: images marked BAD by at least one rater and excluded: {len(bad_imgs)} {sorted(bad_imgs)[:12]}")
    raw["label"] = pd.to_numeric(raw["label"], errors="coerce")
    raw.loc[raw["label"] < 0, "label"] = np.nan
    raw["wet"] = (raw["label"] >= 1).astype(float).where(raw["label"].notna())
    per = raw.groupby("idx").agg(n=("wet", "count"), wet_mean=("wet", "mean"))
    print(f"RAIN: {raw['rater'].nunique()} raters, {len(per)} images, {int((per['n'] >= 2).sum())} judged by 2+")
    wide = raw.pivot_table(index="idx", columns="rater", values="label", aggfunc="first")
    ks = []
    for r1 in wide.columns:
        for r2 in wide.columns:
            if r1 < r2:
                both = wide[[r1, r2]].dropna()
                if len(both) >= 10 and both.stack().nunique() > 1:
                    ks.append((r1, r2, len(both), (both[r1] == both[r2]).mean(), cohen_kappa_score(both[r1].astype(int), both[r2].astype(int), weights="linear")))
    if ks:
        print("  agreement (0/1/2 label):", "; ".join(f"{a} vs {c}: {n} img, {p:.0%} same, weighted kappa {k:.2f}" for a, c, n, p, k in ks[:6]))

    hid = pd.read_csv(HID / "rain_hidden.csv", dtype={"idx": str}).set_index("idx")
    fresh = per[per.index.str.startswith("R")].join(hid[["rain_mm", "p_wet", "weight"]]).dropna(subset=["wet_mean"])
    both_agree = fresh[(fresh["n"] >= 2) & (fresh["wet_mean"].isin([0.0, 1.0]))]
    print(f"  fresh daytime images with a clear consensus: {len(both_agree)} of {len(fresh)} (2 raters agree)")
    if both_agree["wet_mean"].nunique() < 2:
        print("  only one class in the consensus: AUC undefined")
        return
    y = both_agree["wet_mean"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    print("  weighted AUC on the fresh sample (consensus labels):")
    export_rows = []
    for name, col in (("Open-Meteo rain (mm/h)", "rain_mm"), ("CLIP wetness", "p_wet")):
        a = weighted_auc(y, both_agree[col].to_numpy(), both_agree["weight"].to_numpy())
        vals = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                vals.append(weighted_auc(y[i], both_agree[col].to_numpy()[i], both_agree["weight"].to_numpy()[i]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"    {name:<24} {a:.2f}  ({lo:.2f}-{hi:.2f})")
        export_rows.append({"subset": "day", "detector": "open_meteo" if col == "rain_mm" else "zero_shot", "auc": a, "lo": lo, "hi": hi,
                            "n": len(y), "n_wet": int(y.sum())})
    pd.DataFrame(export_rows).to_csv(ROOT / "data" / "export" / "wetness_validation_people.csv", index=False)  # read by the dashboard
    thr = both_agree.assign(wet=y)
    w = thr["weight"]
    for name, pred in (("Open-Meteo >= 1 mm/h", thr["rain_mm"] >= 1.0), ("CLIP >= 0.5", thr["p_wet"] >= 0.5)):
        tp, fn = (w[pred & (thr["wet"] == 1)]).sum(), (w[~pred & (thr["wet"] == 1)]).sum()
        tn, fp = (w[~pred & (thr["wet"] == 0)]).sum(), (w[pred & (thr["wet"] == 0)]).sum()
        print(f"    {name:<22} sensitivity {tp / (tp + fn):.2f}  specificity {tn / (tn + fp):.2f}  (weighted)")
    print(f"  share of daytime images that look wet (weighted estimate for the whole daytime set): {np.average(y, weights=both_agree['weight']):.0%}")


# ------------------------------------------------------------------------------------------------ counts
def counts_report() -> None:
    raw = read_all("counts")
    if raw.empty:
        print("\nCOUNTS: no files in data/round2/returns/")
        return
    for c in CLASSES:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if "bad" in raw.columns:
        bad_imgs = set(raw.loc[raw["bad"] == "1", "idx"])
        raw = raw[~raw["idx"].isin(bad_imgs)].copy()
        print(f"\nCOUNTS: images marked BAD by at least one counter and excluded: {len(bad_imgs)} {sorted(bad_imgs)[:12]}")
    raw["total"] = raw[CLASSES].sum(axis=1, min_count=len(CLASSES))
    raw["orig"] = raw["idx"].str[1:]  # C07 -> the round-1 counting image 07
    wide = raw.pivot_table(index="orig", columns="rater", values="total")
    print(f"COUNTS: {raw['rater'].nunique()} counters, {len(wide)} images counted")
    for r1 in wide.columns:
        for r2 in wide.columns:
            if r1 < r2:
                both = wide[[r1, r2]].dropna()
                if len(both) >= 5:
                    print(f"  {r1} vs {r2}: {len(both)} images, mean totals {both[r1].mean():.1f} / {both[r2].mean():.1f}, "
                          f"mean |difference| {(both[r1] - both[r2]).abs().mean():.1f}, Spearman {spearmanr(both[r1], both[r2]).statistic:.2f}")
    yolo = pd.read_csv(VAL / "sample_yolo_hidden.csv", dtype={"idx": str}).set_index("idx")
    yolo_total = yolo[[f"n_{c}" for c in CLASSES]].sum(axis=1)
    regime = pd.read_csv(VAL / "sample_blind.csv", dtype={"idx": str}).set_index("idx")["light_regime"].map(lambda r: "day" if r == "day" else "dusk/night")
    avg = wide.mean(axis=1)
    print("  YOLO found this share of the vehicles people counted (sum over images; 1.00 = all):")
    for name, man in [*((c, wide[c]) for c in wide.columns), ("average of counters", avg)]:
        df = pd.DataFrame({"man": man, "yolo": yolo_total, "g": regime}).dropna(subset=["man"])
        parts = "   ".join(f"{g}: {s['yolo'].sum() / s['man'].sum():.2f} (n={len(s)})" for g, s in df.groupby("g"))
        print(f"   {name:<22} all {df['yolo'].sum() / df['man'].sum():.2f}   {parts}")
    ai = VAL / "manual_counts_claude.csv"
    if ai.exists():
        a = pd.read_csv(ai, dtype={"idx": str}).set_index("idx")
        a_total = a[CLASSES].sum(axis=1)
        both = pd.DataFrame({"ai": a_total, "people": avg}).dropna()
        print(f"  AI counts vs the average of the counters: {len(both)} images, mean {both['ai'].mean():.1f} vs {both['people'].mean():.1f}, "
              f"Spearman {spearmanr(both['ai'], both['people']).statistic:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", type=Path)
    a = ap.parse_args()
    if a.inbox:
        collect(a.inbox)
    boxes_report()
    rain_report()
    counts_report()


if __name__ == "__main__":
    main()
