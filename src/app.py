"""Bangkok CCTV traffic dashboard (Streamlit + pydeck + plotly).

Run:  uv run streamlit run src/app.py
Reads only data/export/*.parquet (made by src/export_gold.py). Four tabs, one per project component:
  1 Map            where / when is traffic heavy
  2 Profiles       hourly shape per camera + K-Means groups
  3 Rain & air     does weather move traffic?  + anomalies
  4 Data quality   what the pipeline kept, rejected and assumed
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from plotly.subplots import make_subplots

import analysis as A

EXPORT = Path(__file__).resolve().parents[1] / "data" / "export"

# Reference palette (validated, see dataviz skill): categorical slots 1-3 and the blue sequential ramp.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CLUSTER_COLORS = [BLUE, ORANGE, AQUA]  # k <= 3: the first three slots validate for map/scatter (all-pairs)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MUTED = "#898781"

st.set_page_config(page_title="Bangkok CCTV Traffic", layout="wide", page_icon="🚦")


def hex_rgb(h: str) -> list[int]:
    return [int(h[i : i + 2], 16) for i in (1, 3, 5)]


def seq_color(v: float, lo: float, hi: float) -> list[int]:
    """Map a value onto the one-hue blue ramp (light = low, dark = high)."""
    if v is None or np.isnan(v):
        return hex_rgb("#c3c2b7") + [180]
    t = 0.0 if hi <= lo else min(1.0, max(0.0, (v - lo) / (hi - lo)))
    pos = t * (len(SEQ) - 1)
    i = min(int(pos), len(SEQ) - 2)
    a, b = np.array(hex_rgb(SEQ[i])), np.array(hex_rgb(SEQ[i + 1]))
    return [int(x) for x in a + (b - a) * (pos - i)] + [225]


def load_wet_validation():
    people = EXPORT / "wetness_validation_people.csv"  # written by round2_eval.py from the 5 friends' labels
    if people.exists():
        return pd.read_csv(people).assign(by_people=True)
    p = EXPORT / "wetness_validation.csv"
    return pd.read_csv(p).assign(by_people=False) if p.exists() else None


@st.cache_data(show_spinner=False)
def load(stamp: float):  # `stamp` (files' mtime) is only a cache key, so re-exported data is picked up
    hourly = pd.read_parquet(EXPORT / "camera_hourly.parquet")
    profile = pd.read_parquet(EXPORT / "camera_profile.parquet")
    audit = pd.read_parquet(EXPORT / "pipeline_audit.parquet")
    cover = pd.read_parquet(EXPORT / "image_coverage.parquet")
    return hourly, profile, audit, cover


@st.cache_data(show_spinner="Running analysis...")
def analyse(hourly: pd.DataFrame):
    assign, cprof, scores = A.cluster_cameras(hourly)
    return {
        "assign": assign, "cprof": cprof, "scores": scores, "names": A.name_clusters(cprof),
        "wx": A.weather_effects(hourly), "pm": A.pm_effect(hourly), "anom": A.anomalies(hourly),
    }


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    """Recessive chrome: hairline grid, no chart border, transparent so it follows the Streamlit theme."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12, x=0),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
    )
    fig.update_xaxes(showgrid=False, linecolor=MUTED)
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.25)", zeroline=False)
    return fig


def label(c: str, names: dict) -> str:
    return f"{c} · {names.get(c, '')}"


hourly, profile, audit, cover = load(max(p.stat().st_mtime for p in EXPORT.glob("*.parquet")))
names = profile.set_index("camera_id")["name_th"].to_dict()
res = analyse(hourly)
days = sorted(hourly["service_date"].unique())
day = hourly[hourly["light_group"] == "day"]

st.title("Bangkok CCTV traffic, 25–31 Aug 2026")
st.caption(
    "100 BMA cameras · vehicles counted by YOLOv8 (confidence 0.15) on 60,125 usable images · congestion index = vehicles ÷ the camera's own "
    "95th percentile (so cameras are comparable). Counts are relative: people checked 508 boxes and found "
    "95% real vehicles, but only about 77% of vehicles get a box (fewer motorcycles and after dusk)."
)

tab_map, tab_prof, tab_env, tab_dq = st.tabs(["① Map", "② Profiles & groups", "③ Rain, air & anomalies", "④ Data quality"])

# =============================================================== 1. MAP
with tab_map:
    wk = day[~day["is_weekend"]]["avg_vehicles"].mean()
    we = day[day["is_weekend"]]["avg_vehicles"].mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cameras (mapped / total)", f"{profile['lat'].notna().sum()} / {len(profile)}")
    k2.metric("Vehicles per image, weekday day", f"{wk:.1f}")
    k3.metric("Weekend vs weekday", f"{100 * (we / wk - 1):+.0f}%", help="Daytime, vehicles per image, all cameras")
    k4.metric("Cameras quieter on weekends", f"{int((profile['avg_vehicles_weekend'] < profile['avg_vehicles_weekday']).sum())} / {len(profile)}")

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    sel_day = c1.selectbox("Day", ["All days"] + days)
    hour = c2.slider("Hour of day", 6, 21, 17)
    metric = c3.radio("Colour by", ["Congestion index", "Vehicles per image"], horizontal=True)
    col = "congestion_index" if metric == "Congestion index" else "avg_vehicles"

    sub = hourly[hourly["hour_of_day"] == hour]
    if sel_day != "All days":
        sub = sub[sub["service_date"] == sel_day]
    m = sub.groupby(["camera_id", "name_th", "lat", "lng"], dropna=False)[col].mean().reset_index()
    m = m.dropna(subset=["lat", "lng"])
    lo, hi = (0.0, 1.0) if col == "congestion_index" else (0.0, float(m[col].quantile(0.95)) if len(m) else 1.0)
    m["color"] = m[col].apply(lambda v: seq_color(v, lo, hi))
    m["value"] = m[col].round(2)

    if hour >= 19 or hour == 6:
        st.warning("Dusk/night hours: the detector misses small vehicles seen only as headlights, so values here read low. "
                   "Compare against other cameras at the same hour, not against daytime.", icon="🌙")
    left, right = st.columns([2.2, 1])
    with left:
        deck = pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", data=m, get_position="[lng, lat]", get_fill_color="color",
                              get_radius=140, radius_min_pixels=8, radius_max_pixels=18, pickable=True, stroked=True,
                              get_line_color=[137, 135, 129, 200], line_width_min_pixels=1)],
            initial_view_state=pdk.ViewState(latitude=13.755, longitude=100.525, zoom=11.2),
            tooltip={"html": "<b>{camera_id}</b> {name_th}<br/>" + metric + ": <b>{value}</b>"},
        )
        st.pydeck_chart(deck, use_container_width=True)
        bar = "".join(
            f'<span style="display:inline-block;width:{100 / len(SEQ)}%;height:10px;background:{c}"></span>' for c in SEQ
        )
        st.markdown(
            f'<div style="max-width:320px">{bar}<div style="display:flex;justify-content:space-between;'
            f'font-size:12px;color:{MUTED}"><span>{lo:g} (lighter)</span><span>{hi:g}+ (darker)</span></div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{len(m)} cameras shown. {int(profile['lat'].isna().sum())} cameras have no coordinates "
                   "(BMA list does not publish them) and are excluded from the map only.")
    with right:
        st.subheader("Busiest at this time")
        top = m.sort_values(col, ascending=False).head(10)[["camera_id", "name_th", "value"]]
        top.columns = ["Camera", "Name", metric]
        st.dataframe(top, hide_index=True, use_container_width=True)

# ====================================================== 2. PROFILES + GROUPS
with tab_prof:
    st.subheader("When is traffic heaviest?")
    hh = day[day["hour_of_day"] <= 18].assign(
        Day_type=lambda d: np.where(d["is_weekend"], "Weekend (Sat–Sun)", "Weekday (Tue–Fri, Mon)")
    )
    g = hh.groupby(["hour_of_day", "Day_type"])["congestion_index"].mean().reset_index()
    fig = px.line(g, x="hour_of_day", y="congestion_index", color="Day_type", markers=True,
                  color_discrete_map={"Weekday (Tue–Fri, Mon)": BLUE, "Weekend (Sat–Sun)": ORANGE},
                  labels={"hour_of_day": "Hour of day", "congestion_index": "Mean congestion index", "Day_type": ""})
    fig.update_traces(line=dict(width=2.5), marker=dict(size=8, line=dict(width=2, color="white")))
    fig.update_xaxes(dtick=1)
    st.plotly_chart(style(fig), use_container_width=True, theme="streamlit")
    st.caption("Daytime hours only (06:00–18:59), where counts are comparable. Index 1.0 = the camera's own busy-hour level.")
    with st.expander("View as table"):
        st.dataframe(g.pivot(index="hour_of_day", columns="Day_type", values="congestion_index").round(3))

    st.divider()
    st.subheader("Camera groups (K-Means)")
    assign, cprof, scores, cl_names = res["assign"], res["cprof"], res["scores"], res["names"]
    k = len(cl_names)
    colors = {c: CLUSTER_COLORS[c] for c in cl_names}
    sil = float(scores.loc[scores["k"] == k, "silhouette"].iloc[0])
    st.write(
        f"Cameras are grouped by the **shape** of their weekday daytime profile (level removed) plus their weekend/weekday ratio. "
        f"k = {k} was chosen by silhouette score ({sil:.2f}); k is capped at 3 so groups stay distinguishable on the map. "
        + ("That score is modest: the groups are a useful simplification, not sharply separated clusters." if sil < 0.4 else "")
    )
    cp = cprof.assign(group=cprof["cluster"].map(cl_names))
    cp["group"] = cp.apply(lambda r: f"{r['group']} ({int(r['n_cameras'])} cameras)", axis=1)
    cmap = {f"{cl_names[c]} ({int(cprof.loc[cprof['cluster'] == c, 'n_cameras'].iloc[0])} cameras)": colors[c] for c in cl_names}
    gl, gr = st.columns([1.1, 1])
    with gl:
        f2 = px.line(cp, x="hour_of_day", y="relative_congestion", color="group", markers=True,
                     color_discrete_map=cmap,
                     labels={"hour_of_day": "Hour of day", "relative_congestion": "Congestion relative to own daily mean", "group": ""})
        f2.update_traces(line=dict(width=2.5), marker=dict(size=8, line=dict(width=2, color="white")))
        f2.update_xaxes(dtick=1)
        st.plotly_chart(style(f2), use_container_width=True, theme="streamlit")
    with gr:
        pm = profile.merge(assign, on="camera_id").dropna(subset=["lat", "lng"])
        pm["color"] = pm["cluster"].map(lambda c: hex_rgb(colors[c]) + [225])
        pm["group"] = pm["cluster"].map(cl_names)
        deck2 = pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", data=pm, get_position="[lng, lat]", get_fill_color="color", get_radius=140,
                              radius_min_pixels=8, radius_max_pixels=18, pickable=True, stroked=True,
                              get_line_color=[137, 135, 129, 200], line_width_min_pixels=1)],
            initial_view_state=pdk.ViewState(latitude=13.755, longitude=100.525, zoom=10.6),
            tooltip={"html": "<b>{camera_id}</b> {name_th}<br/>{group}"},
        )
        st.pydeck_chart(deck2, use_container_width=True)
    with st.expander("View group members / table"):
        tbl = profile.merge(assign, on="camera_id", how="left")
        tbl["group"] = tbl["cluster"].map(cl_names)
        st.dataframe(tbl[["camera_id", "name_th", "group", "avg_vehicles_day", "avg_vehicles_weekday", "avg_vehicles_weekend",
                          "peak_hour_day", "n_school", "n_mall", "n_rail_station"]].round(2), hide_index=True)

    st.divider()
    st.subheader("Compare one camera with the city")
    cam = st.selectbox("Camera", sorted(profile["camera_id"], key=int), format_func=lambda c: label(c, names))
    one = hh[hh["camera_id"] == cam].groupby("hour_of_day")["congestion_index"].mean().rename("This camera")
    city = hh.groupby("hour_of_day")["congestion_index"].mean().rename("All cameras")
    cmp_df = pd.concat([one, city], axis=1).reset_index().melt(id_vars="hour_of_day", var_name="series", value_name="ci")
    f3 = px.line(cmp_df, x="hour_of_day", y="ci", color="series", markers=True,
                 color_discrete_map={"This camera": BLUE, "All cameras": MUTED},
                 labels={"hour_of_day": "Hour of day", "ci": "Mean congestion index", "series": ""})
    f3.update_traces(line=dict(width=2.5))
    f3.update_xaxes(dtick=1)
    st.plotly_chart(style(f3, 300), use_container_width=True, theme="streamlit")

# ================================================ 3. RAIN, AIR, ANOMALIES
with tab_env:
    pmr, anom = res["pm"], res["anom"]
    st.subheader("Does weather move traffic?")
    wx = res["wx"]
    vt = load_wet_validation()

    st.markdown("##### How well does each rain measure match what the camera actually shows?")
    if vt is not None:
        vd = vt[(vt["subset"] == "day") & vt["detector"].isin(["open_meteo", "zero_shot"])].copy()
        vd["Measure"] = vd["detector"].map({"open_meteo": "Open-Meteo rain (model)", "zero_shot": "Image wetness (CLIP)"})
        fv = go.Figure()
        for r in vd.itertuples():
            col = ORANGE if r.detector == "open_meteo" else BLUE
            fv.add_trace(go.Scatter(x=[r.auc], y=[r.Measure], mode="markers", name=r.Measure,
                                    marker=dict(size=12, color=col, line=dict(width=2, color="white")),
                                    error_x=dict(type="data", symmetric=False, array=[r.hi - r.auc], arrayminus=[r.auc - r.lo],
                                                 thickness=2.5, color=col)))
        fv.add_vline(x=0.5, line_dash="dot", line_color=MUTED)
        fv.update_xaxes(range=[0.2, 1.0], title="AUC: 0.5 = no better than a coin flip, 1.0 = perfect")
        fv.update_layout(showlegend=False)
        st.plotly_chart(style(fv, 200), use_container_width=True, theme="streamlit")
        n_v, nw_v = int(vd["n"].iloc[0]), int(vd["n_wet"].iloc[0])
        if bool(vd["by_people"].iloc[0]):
            st.caption(
                f"Test: {n_v} fresh daytime images labelled blind by 5 people as dry / wet / raining (consensus of 2 raters; {nw_v} looked wet; "
                "sampled so that neither detector is favoured). Open-Meteo is a coarse weather model (~9 km grid) and often misses "
                "local showers; the image score is a CLIP zero-shot guess and is **not usable after dusk** (street lights make "
                "dry roads look wet). Details: docs/verification_round2.md."
            )
        else:
            st.caption(
                f"Test: {n_v} daytime images labelled blind as dry / wet / raining ({nw_v} looked wet). The labels were made by an AI model, "
                "not a person, so treat this as a rough check. Details: docs/wetness_validation.md."
            )

    st.markdown("##### Effect of +1 SD of each weather measure on congestion (daytime)")
    st.write(
        "Each effect compares a camera-hour with the **same camera at the same hour of day and day-type** on other days, "
        "so differences between cameras and the daily rhythm are removed. One SD is a typical fluctuation of that measure "
        "after that adjustment. Intervals are 95% bootstrap intervals resampling cameras."
    )
    meas = list(dict.fromkeys(wx["measure"]))
    fw = go.Figure()
    for subset, color, off in [("all days", BLUE, 0.13), ("weekdays only", ORANGE, -0.13)]:
        s = wx[wx["subset"] == subset].set_index("measure").reindex(meas)
        ypos = [i + off for i in range(len(meas))]
        fw.add_trace(go.Scatter(
            x=s["effect_pct_per_sd"], y=ypos, mode="markers", name=subset,
            marker=dict(size=11, color=color, line=dict(width=2, color="white")),
            error_x=dict(type="data", symmetric=False, array=(s["hi"] - s["effect_pct_per_sd"]).tolist(),
                         arrayminus=(s["effect_pct_per_sd"] - s["lo"]).tolist(), thickness=2.5, color=color),
            hovertemplate="%{x:+.2f}%<extra>" + subset + "</extra>"))
    fw.add_vline(x=0, line_color=MUTED, line_width=1)
    fw.update_yaxes(tickmode="array", tickvals=list(range(len(meas))), ticktext=meas, autorange="reversed")
    fw.update_xaxes(title="Change in congestion index per +1 SD (%)")
    st.plotly_chart(style(fw, 300), use_container_width=True, theme="streamlit")
    widest = float(max(wx["hi"].abs().max(), wx["lo"].abs().max()))
    if not wx["significant"].any():
        st.info(
            f"**No detectable effect of rain/wetness on daytime congestion this week.** None of the six estimates is "
            f"distinguishable from zero, and every interval stays within ±{widest:.1f}% per SD, so a large effect is unlikely "
            "for the kind of wetness seen here. On the picture test the image-based measure matched "
            "what people saw clearly better than Open-Meteo (the intervals do not overlap), and it gave the same answer."
        )
    else:
        st.info("At least one interval excludes zero; see the chart.")
    st.caption(
        "Limits: one week; heavy rain is rare (few camera-hours); YOLO may also miss more vehicles in rain (lens drops, glare), "
        "which could hide a real drop in traffic; the image score is validated on daytime only."
    )
    with st.expander("View as table"):
        st.dataframe(wx.round(2), hide_index=True)

    st.markdown("##### PM2.5 +10 µg/m³")
    pcol1, pcol2 = st.columns([1, 2])
    with pcol1:
        st.metric("Change in congestion", f"{pmr['effect_pct']:+.1f}%",
                  help=f"95% interval {pmr['ci_low_pct']:+.1f}% to {pmr['ci_high_pct']:+.1f}%")
        st.write(f"95% interval: **{pmr['ci_low_pct']:+.1f}% to {pmr['ci_high_pct']:+.1f}%** · "
                 f"PM2.5 averaged {pmr['pm_mean']:.1f} µg/m³ (SD {pmr['pm_sd']:.1f})")
    with pcol2:
        st.warning(
            "Read as an association only. PM2.5 here is a **model estimate** on a coarse (~40 km) grid, so Bangkok gets only "
            "about four distinct values, and it varies mostly from day to day: it can stand in for anything else that "
            "differed between days. 10 µg/m³ is also about 3 SD, beyond what this week shows.",
            icon="⚠️",
        )

    st.markdown("##### Rain, wetness and traffic day by day")
    daily = day.groupby("service_date").agg(ci=("congestion_index", "mean"), rain=("rain_mm", "mean"), wet=("wet_score", "mean")).reset_index()
    fs = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                       subplot_titles=("Mean daytime congestion index", "Open-Meteo rain (mm per hour)", "Image wetness score (CLIP, 0–1)"))
    fs.add_trace(go.Scatter(x=daily["service_date"], y=daily["ci"], mode="lines+markers", line=dict(color=BLUE, width=2.5),
                            marker=dict(size=9, line=dict(width=2, color="white")), name="Congestion index"), row=1, col=1)
    fs.add_trace(go.Bar(x=daily["service_date"], y=daily["rain"], marker_color=ORANGE, marker_cornerradius=4, name="Rain (mm/h)"), row=2, col=1)
    fs.add_trace(go.Bar(x=daily["service_date"], y=daily["wet"], marker_color=AQUA, marker_cornerradius=4, name="Image wetness"), row=3, col=1)
    fs = style(fs, 560)
    fs.update_layout(showlegend=False)  # one series per panel: the panel titles name them
    st.plotly_chart(fs, use_container_width=True, theme="streamlit")
    st.caption("Saturday 29 Aug looks wettest in the images; Open-Meteo puts more rain on Thursday–Friday, where the pictures look dry.")

    st.divider()
    st.subheader("Unusual camera-hours")
    sm = A.anomaly_summary(anom)
    st.write(
        f"A camera-hour is flagged when it is ≥ 3 robust SD from that camera's typical value for the hour and day-type: "
        f"**{sm['n_anomalies']} of {sm['n_total']:,}** ({sm['rate_pct']:.1f}%). "
        f"{100 * sm['rain_share_anomalies']:.0f}% of flagged hours had ≥ 1 mm/h rain in the weather model vs "
        f"{100 * sm['rain_share_all']:.0f}% of all hours (that model is a weak rain proxy, see above)."
    )
    top = anom[anom["anomaly"]].copy()
    top["abs_z"] = top["z"].abs()
    top = top.sort_values("abs_z", ascending=False).head(15)
    top["Camera"] = top["camera_id"].map(lambda c: label(c, names))
    top["When"] = top["hour_local"].dt.strftime("%a %d %b %H:00")
    show = top[["Camera", "When", "congestion_index", "z", "direction", "is_rainy"]].rename(
        columns={"congestion_index": "Congestion index", "z": "z-score", "direction": "Direction", "is_rainy": "Rainy hour"})
    st.dataframe(show.round(2), hide_index=True, use_container_width=True)
    st.caption("Direction matters: very high values may be incidents or events; very low values can also mean a camera problem.")
    hist = px.histogram(anom, x="z", nbins=60, labels={"z": "z-score of camera-hour"})
    hist.update_traces(marker_color=BLUE, marker_cornerradius=2)
    for v in (-3, 3):
        hist.add_vline(x=v, line_dash="dot", line_color=ORANGE)
    st.plotly_chart(style(hist, 260), use_container_width=True, theme="streamlit")

# ============================================================ 4. DATA QUALITY
with tab_dq:
    st.subheader("What the pipeline kept")
    n_in = int(audit.loc[audit["stage"] == "bronze.cctv_images", "n_rows"].iloc[0])
    n_rej = int(audit.loc[audit["stage"] == "silver.rejected_images", "n_rows"].iloc[0])
    n_ok = int(audit.loc[audit["stage"] == "silver.image_features", "n_rows"].iloc[0])
    n_frozen = int(audit.loc[audit["stage"] == "silver.image_features (frozen)", "n_rows"].iloc[0])
    n_used = int(hourly["n_images"].sum())
    n_dark = int(audit.loc[audit["stage"] == "silver.light_regime (dark)", "n_rows"].iloc[0])
    fl = pd.DataFrame({
        "stage": ["Received (bronze)", "Passed quality checks (silver)", "Detected by YOLO (silver)", "Used in gold averages"],
        "images": [n_in, n_ok, n_ok, n_used],
    })
    ff = px.bar(fl, y="stage", x="images", orientation="h", text=fl["images"].map(lambda v: f"{v:,}"))
    ff.update_traces(marker_color=[SEQ[2], SEQ[3], SEQ[4], SEQ[5]], marker_cornerradius=4, textposition="outside")
    ff.update_yaxes(autorange="reversed", title="")
    ff.update_xaxes(title="Images", range=[0, n_in * 1.15])
    st.plotly_chart(style(ff, 260), use_container_width=True, theme="streamlit")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rejected (blank / unreadable)", f"{n_rej:,}", help="Camera returned a blank frame; kept in bronze with the reason in silver.rejected_images")
    q2.metric("Frozen-frame images excluded", f"{n_frozen:,}", help="Near-identical to the previous frame of the same camera")
    q3.metric("Dusk / night images", f"{n_dark:,}", help="Flagged by sun position; detector under-counts here")
    q4.metric("Cameras with map position", f"{int(profile['lat'].notna().sum())} / {len(profile)}")
    n_prob = int(cover["n_problem"].sum()) if "n_problem" in cover.columns else 0
    st.caption(f"Also excluded: {n_prob:,} glitch / test-card / unreadable images (silver.image_scan); they are not used for counts or labelling.")

    st.markdown("##### Images received per day")
    cd = cover.groupby("service_date")["n_received"].sum().reset_index()
    fd = px.bar(cd, x="service_date", y="n_received", labels={"service_date": "Day", "n_received": "Images"})
    fd.update_traces(marker_color=BLUE, marker_cornerradius=4)
    st.plotly_chart(style(fd, 260), use_container_width=True, theme="streamlit")
    low = cd.loc[cd["n_received"] < 0.9 * cd["n_received"].median(), "service_date"].tolist()
    if low:
        st.caption(f"Fewer images on {', '.join(low)}: a collection gap, not a traffic effect. Averages per image are used so gaps do not bias results.")

    st.markdown("##### Coverage: images per camera per day")
    heat = cover.pivot(index="camera_id", columns="service_date", values="n_received").fillna(0)
    hf = px.imshow(heat.values, x=list(heat.columns), y=[label(c, names)[:34] for c in heat.index], aspect="auto",
                   color_continuous_scale=SEQ, labels=dict(color="Images"))
    hf.update_yaxes(showticklabels=False)
    st.plotly_chart(style(hf, 380), use_container_width=True, theme="streamlit")

    st.markdown("##### Data sources & caveats")
    src = pd.DataFrame([
        ("CCTV images", "BMA traffic platform (course dataset)", "61,319 JPEGs, 100 cameras, ~10-min cadence, 06:00–21:00"),
        ("Camera coordinates", "bmatraffic.com page (66) + OSM Nominatim (8)", "26 cameras unresolved; excluded from maps only"),
        ("Rain / weather", "Open-Meteo archive (reanalysis), 3×3 grid", "Hourly; ~52% of hours show ≥0.1 mm so a 1 mm threshold is used"),
        ("Wetness from images", "CLIP ViT-B/16 zero-shot on every image", "Daytime only; checked by 5 people on 150 fresh images (AUC 0.95 vs 0.69 for Open-Meteo rain), unusable after dusk"),
        ("PM2.5", "Open-Meteo air quality (CAMS model)", "Modelled on a ~40 km grid (≈4 distinct values over the city), not station readings; Air4Thai has no data for these dates"),
        ("Places", "OpenStreetMap via Overpass", "3,862 points; counted within 300 m of each camera"),
    ], columns=["Source", "Where from", "Notes"])
    st.dataframe(src, hide_index=True, use_container_width=True)
    with st.expander("Full pipeline audit (row counts per table)"):
        st.dataframe(audit[["step", "stage", "n_rows", "note"]], hide_index=True, use_container_width=True)
    st.info(
        "**Known limitations.** (1) YOLO under-counts after dusk, especially motorcycles, so night values are not comparable with daytime; "
        "(2) congestion is normalised per camera against its own busy level; (3) seven days is enough to see the weekday/weekend rhythm "
        "but not seasonal or holiday effects; (4) vehicles are counted at YOLO confidence 0.15. Five people judged 508 boxes: 95% were real vehicles and about 77% of vehicles got a box (precision 0.95, recall 0.77), so counts understate by roughly a fifth, more at dusk/night and for motorcycles. Counts follow true traffic closely (rank correlation 0.89 with hand counts), so patterns hold but absolute numbers understate. See docs/verification_round2.md.",
        icon="ℹ️",
    )
