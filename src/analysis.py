"""Data-science layer used by the dashboard (pure pandas/numpy/sklearn, no lakehouse access).

1. cluster_cameras   K-Means on each camera's weekday day-time congestion *shape* + weekend ratio (k <= 3).
2. rain_effect       within-(camera, hour-of-day, day-type) effect of rain on congestion, cluster-bootstrap CI over cameras.
3. pm_effect         same design for PM2.5 (per +10 ug/m3).
4. anomalies         residual z-scores of camera-hours vs the camera's own typical hour.

Everything works on DAYTIME rows only (light_group == 'day'): after dusk YOLO under-counts, so day and dark
rows are not comparable. Congestion index = vehicles / the camera's own p95 (see gold.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROFILE_HOURS = list(range(7, 19))  # 07:00-18:59 daytime hours present for every camera
RAIN_MM = 1.0
MAX_K = 3  # map / scatter colour rule: only the first three categorical slots are safe for all pairs


def day_rows(hourly: pd.DataFrame) -> pd.DataFrame:
    return hourly[hourly["light_group"] == "day"].copy()


# ---------------------------------------------------------------- clustering
def cluster_cameras(hourly: pd.DataFrame, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (assignments, cluster_profiles, k_scores)."""
    d = day_rows(hourly)
    wk = d[~d["is_weekend"] & d["hour_of_day"].isin(PROFILE_HOURS)]
    prof = wk.pivot_table(index="camera_id", columns="hour_of_day", values="congestion_index", aggfunc="mean")
    prof = prof.reindex(columns=PROFILE_HOURS)
    prof = prof.apply(lambda r: r.fillna(r.mean()), axis=1).dropna()
    shape = prof.div(prof.mean(axis=1), axis=0)  # shape only: level differences between cameras removed

    we = d[d["is_weekend"]].groupby("camera_id")["avg_vehicles"].mean()
    wd = d[~d["is_weekend"]].groupby("camera_id")["avg_vehicles"].mean()
    ratio = (we / wd).reindex(shape.index)
    X = pd.concat([shape, ratio.rename("weekend_ratio")], axis=1).dropna()
    Z = StandardScaler().fit_transform(X.to_numpy())

    scores = []
    models = {}
    for k in range(2, MAX_K + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
        scores.append({"k": k, "silhouette": silhouette_score(Z, km.labels_)})
        models[k] = km
    scores = pd.DataFrame(scores)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    labels = models[best_k].labels_

    assign = pd.DataFrame({"camera_id": X.index, "cluster": labels})
    sh = shape.loc[X.index].copy()
    sh["cluster"] = labels
    # order clusters by the hour at which their mean shape peaks, so names/colours are stable across runs
    peaks = sh.groupby("cluster").mean().idxmax(axis=1)
    order = {c: i for i, c in enumerate(peaks.sort_values().index)}
    assign["cluster"] = assign["cluster"].map(order)
    sh["cluster"] = sh["cluster"].map(order)
    cp = sh.groupby("cluster").mean().T
    cp.index.name = "hour_of_day"
    cp = cp.reset_index().melt(id_vars="hour_of_day", var_name="cluster", value_name="relative_congestion")
    counts = assign["cluster"].value_counts().to_dict()
    cp["n_cameras"] = cp["cluster"].map(counts)
    return assign, cp, scores


def name_clusters(cp: pd.DataFrame) -> dict[int, str]:
    """Plain-language names from each cluster's own mean shape."""
    names = {}
    for c, g in cp.groupby("cluster"):
        g = g.sort_values("hour_of_day")
        peak = int(g.loc[g["relative_congestion"].idxmax(), "hour_of_day"])
        spread = float(g["relative_congestion"].max() - g["relative_congestion"].min())
        if spread < 0.35:
            names[c] = "Steady all day"
        elif peak <= 10:
            names[c] = f"Morning peak ({peak:02d}:00)"
        elif peak >= 15:
            names[c] = f"Evening peak ({peak:02d}:00)"
        else:
            names[c] = f"Midday peak ({peak:02d}:00)"
    # keep names unique
    seen: dict[str, int] = {}
    for c in sorted(names):
        seen[names[c]] = seen.get(names[c], 0) + 1
        if seen[names[c]] > 1:
            names[c] = f"{names[c]} #{seen[names[c]]}"
    return names


# ------------------------------------------------------------ effect of rain / PM2.5
def _demean(d: pd.DataFrame, cols: list[str], by: list[str]) -> pd.DataFrame:
    out = d.copy()
    for c in cols:
        out[c + "_dm"] = out[c] - out.groupby(by)[c].transform("mean")
    return out


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    den = float((x * x).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def _bootstrap_by_camera(d: pd.DataFrame, ycol: str, xcol: str, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = {c: g for c, g in d.groupby("camera_id")}
    ids = np.array(list(groups))
    ys = {c: g[ycol].to_numpy() for c, g in groups.items()}
    xs = {c: g[xcol].to_numpy() for c, g in groups.items()}
    res = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        res.append(_slope(np.concatenate([ys[c] for c in pick]), np.concatenate([xs[c] for c in pick])))
    return np.array(res)


def rain_effect(hourly: pd.DataFrame, n_boot: int = 300, seed: int = 11) -> dict:
    """Effect of a rainy hour (>= 1 mm) on congestion, holding camera x hour-of-day x day-type fixed."""
    d = day_rows(hourly).dropna(subset=["congestion_index", "rain_mm"])
    d["rainy"] = (d["rain_mm"] >= RAIN_MM).astype(float)
    key = ["camera_id", "hour_of_day", "is_weekend"]
    d = _demean(d, ["congestion_index", "rainy", "avg_vehicles"], key)
    est = _slope(d["congestion_index_dm"].to_numpy(), d["rainy_dm"].to_numpy())
    boots = _bootstrap_by_camera(d, "congestion_index_dm", "rainy_dm", n_boot, seed)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    base = float(d.loc[d["rainy"] == 0, "congestion_index"].mean())

    bins = pd.cut(d["rain_mm"], [-0.01, 0.1, 1, 5, 1e9], labels=["dry (<0.1)", "drizzle (0.1-1)", "rain (1-5)", "heavy (>5)"])
    by_bin = (
        d.assign(bin=bins).groupby("bin", observed=True)
        .agg(n_camera_hours=("congestion_index", "size"), effect=("congestion_index_dm", "mean"))
        .reset_index()
    )
    by_bin["effect_pct_of_dry_mean"] = 100 * by_bin["effect"] / base
    return {
        "effect": est, "ci_low": float(lo), "ci_high": float(hi), "baseline_ci": base,
        "effect_pct": 100 * est / base, "ci_low_pct": 100 * float(lo) / base, "ci_high_pct": 100 * float(hi) / base,
        "n_camera_hours": int(len(d)), "n_rainy": int(d["rainy"].sum()), "by_bin": by_bin,
        "significant": not (lo <= 0 <= hi),
    }


def pm_effect(hourly: pd.DataFrame, n_boot: int = 300, seed: int = 13) -> dict:
    """Effect of +10 ug/m3 PM2.5 on congestion, same fixed-effects design."""
    d = day_rows(hourly).dropna(subset=["congestion_index", "pm2_5"])
    d["pm10u"] = d["pm2_5"] / 10.0
    key = ["camera_id", "hour_of_day", "is_weekend"]
    d = _demean(d, ["congestion_index", "pm10u"], key)
    est = _slope(d["congestion_index_dm"].to_numpy(), d["pm10u_dm"].to_numpy())
    boots = _bootstrap_by_camera(d, "congestion_index_dm", "pm10u_dm", n_boot, seed)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    base = float(d["congestion_index"].mean())
    return {
        "effect": est, "ci_low": float(lo), "ci_high": float(hi), "baseline_ci": base,
        "effect_pct": 100 * est / base, "ci_low_pct": 100 * float(lo) / base, "ci_high_pct": 100 * float(hi) / base,
        "n_camera_hours": int(len(d)), "pm_sd": float(d["pm2_5"].std()), "pm_mean": float(d["pm2_5"].mean()),
        "significant": not (lo <= 0 <= hi),
    }


# --------------------------------------------- weather measures side by side
MEASURES = {
    "Open-Meteo rain (mm/h)": "rain_mm",
    "Image wetness (CLIP score)": "wet_score",
    "Image wet share (images >= 0.5)": "wet_share",
}


def weather_effects(hourly: pd.DataFrame, n_boot: int = 300, seed: int = 21) -> pd.DataFrame:
    """Effect of +1 SD of each weather measure on daytime congestion, same fixed-effects design as rain_effect.

    x and the congestion index are demeaned within camera x hour-of-day x day-type; the SD is the SD of the DEMEANED x, so the
    different measures are on a comparable scale (a typical hour-to-hour-and-day-to-day fluctuation). CI: bootstrap over cameras.
    Subsets: all days, and weekdays only (avoids mixing the weekend with the wettest day).
    """
    rows = []
    d0 = day_rows(hourly).dropna(subset=["congestion_index", "rain_mm", "wet_score"])
    key = ["camera_id", "hour_of_day", "is_weekend"]
    for subset, d in [("all days", d0), ("weekdays only", d0[~d0["is_weekend"]])]:
        base = float(d["congestion_index"].mean())
        for name, col in MEASURES.items():
            e = _demean(d, ["congestion_index", col], key)
            x = e[col + "_dm"].to_numpy()
            sd = float(np.sqrt((x**2).mean()))
            e["x_sd"] = e[col + "_dm"] / sd
            est = _slope(e["congestion_index_dm"].to_numpy(), e["x_sd"].to_numpy())
            boots = _bootstrap_by_camera(e, "congestion_index_dm", "x_sd", n_boot, seed)
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            rows.append({"measure": name, "subset": subset, "effect_pct_per_sd": 100 * est / base,
                         "lo": 100 * float(lo) / base, "hi": 100 * float(hi) / base, "sd_x": sd, "n": int(len(e)),
                         "significant": not (lo <= 0 <= hi)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- anomalies
def anomalies(hourly: pd.DataFrame, z_cut: float = 3.0) -> pd.DataFrame:
    """Camera-hours far from the camera's own typical value for that hour and day-type.

    resid = congestion_index - mean(camera, hour_of_day, is_weekend);  z = resid / robust sd of that camera's residuals.
    A camera-level scale is used because weekend cells have only two days (a per-cell sd would be meaningless).
    """
    d = day_rows(hourly).dropna(subset=["congestion_index"]).copy()
    key = ["camera_id", "hour_of_day", "is_weekend"]
    d["resid"] = d["congestion_index"] - d.groupby(key)["congestion_index"].transform("mean")
    mad = d.groupby("camera_id")["resid"].transform(lambda s: 1.4826 * (s - s.median()).abs().median())
    sd = d.groupby("camera_id")["resid"].transform("std")
    scale = mad.where(mad > 0.02, sd)
    d["z"] = d["resid"] / scale
    d["anomaly"] = d["z"].abs() >= z_cut
    d["direction"] = np.where(d["z"] > 0, "busier than usual", "emptier than usual")
    return d


def anomaly_summary(d: pd.DataFrame) -> dict:
    a = d[d["anomaly"]]
    rain_share_anom = float(a["is_rainy"].mean()) if len(a) else float("nan")
    rain_share_all = float(d["is_rainy"].mean())
    return {"n_anomalies": int(len(a)), "n_total": int(len(d)), "rate_pct": 100 * len(a) / len(d),
            "rain_share_anomalies": rain_share_anom, "rain_share_all": rain_share_all}
