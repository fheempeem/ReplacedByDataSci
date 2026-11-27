"""Build a self-contained labelling kit for teammates: data/labeling_kit/ (and a .zip next to it).

    uv run python src/make_labeling_kit.py

Two tasks, each a single offline HTML page (no server, no internet) that saves progress in the browser and exports a CSV:
  1_wetness/  60 images -> label 0 dry / 1 wet / 2 raining / unsure         (checks the rain data)
  2_counting/ 40 images -> count cars, motorcycles, buses, trucks            (checks YOLO)
The kit contains ONLY numbered images + the page + instructions. It must never contain the AI's labels, YOLO's counts or
Open-Meteo's rain, because seeing them would bias the labeller. `check_no_leak()` enforces that.
"""
from __future__ import annotations

import html
import json
import shutil
import zipfile

import pandas as pd
from PIL import Image

from common import ROOT

KIT = ROOT / "data" / "labeling_kit"
SRC_WET = ROOT / "data" / "wetness"
SRC_CNT = ROOT / "data" / "validation"

CSS = """
:root{color-scheme:light dark;--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e1e0d9;--blue:#2a78d6;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#383835;--blue:#3987e5;--card:#222221}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:18px;margin:0 0 4px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
main{max-width:980px;margin:0 auto;padding:16px}
.guide{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:15px}
.guide li{margin:2px 0}
article{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:16px}
article.done{border-color:var(--blue)}
article img{width:100%;height:auto;border-radius:4px;display:block}
.ctl{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin-top:10px}
label.opt{display:inline-flex;gap:6px;align-items:center;cursor:pointer;padding:6px 10px;border:1px solid var(--line);border-radius:6px;min-height:24px}
label.opt:has(input:checked){border-color:var(--blue);outline:2px solid var(--blue)}
input[type=number]{width:70px;padding:6px;font-size:16px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
input[type=text]{flex:1;min-width:200px;padding:6px;font-size:15px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--blue);background:var(--blue);color:#fff;cursor:pointer}
button.ghost{background:transparent;color:var(--ink);border-color:var(--line)}
.num{font-weight:600}
.muted{color:var(--muted);font-size:14px}
"""

JS = r"""
const KEY = "kit_" + TASK;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function isDone(id) {
  const s = state[id] || {};
  return TASK === "wetness" ? (s.label !== undefined && s.label !== "") : ["car","motorcycle","bus","truck"].every(k => s[k] !== undefined && s[k] !== "");
}
function refresh() {
  let n = 0;
  IDS.forEach(id => { const d = isDone(id); if (d) n++; const el = document.getElementById("a" + id); if (el) el.classList.toggle("done", d); });
  document.getElementById("progress").textContent = n + " / " + IDS.length + " done";
}
function set(id, k, v) { (state[id] = state[id] || {})[k] = v; save(); refresh(); }
function init() {
  IDS.forEach(id => {
    const s = state[id] || {};
    document.querySelectorAll("[data-id='" + id + "']").forEach(inp => {
      const k = inp.dataset.k;
      if (inp.type === "radio") { inp.checked = String(s[k]) === inp.value; inp.addEventListener("change", () => set(id, k, inp.value)); }
      else { if (s[k] !== undefined) inp.value = s[k]; inp.addEventListener("input", () => set(id, k, inp.value)); }
    });
  });
  refresh();
}
function download() {
  const who = (document.getElementById("who").value || "").trim().replace(/[^A-Za-z0-9_-]/g, "");
  if (!who) { alert("Type your name first (letters/numbers only, e.g. Mint). It goes in the file name."); return; }
  let csv, name;
  if (TASK === "wetness") {
    csv = "idx,label,note\n" + IDS.map(id => { const s = state[id] || {}; const lab = (s.label === undefined || s.label === "") ? "" : s.label;
      return [id, lab, '"' + (s.note || "").replace(/"/g, '""') + '"'].join(","); }).join("\n");
    name = "labels_" + who + ".csv";
  } else {
    csv = "idx,car,motorcycle,bus,truck,notes\n" + IDS.map(id => { const s = state[id] || {};
      return [id, s.car ?? "", s.motorcycle ?? "", s.bus ?? "", s.truck ?? "", '"' + (s.note || "").replace(/"/g, '""') + '"'].join(","); }).join("\n");
    name = "manual_counts_" + who + ".csv";
  }
  const left = IDS.filter(id => !isDone(id)).length;
  if (left && !confirm(left + " images are not finished yet. Download anyway?")) return;
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove();
}
function resetAll() { if (confirm("Clear everything you entered on this page?")) { state = {}; save(); location.reload(); } }
init();
"""


def page(title: str, task: str, ids: list[str], guide_html: str, controls) -> str:
    cards = []
    for i in ids:
        cards.append(
            f"<article id='a{i}'><div class='num'>#{i}</div><img src='images/{i}.jpg' loading='lazy' alt='image {i}'>"
            f"<div class='ctl'>{controls(i)}<input type='text' data-id='{i}' data-k='note' placeholder='note (optional)'></div></article>"
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f"<header><h1>{html.escape(title)}</h1><div class='row'><span id='progress' class='muted'></span>"
        "<label>Your name <input id='who' type='text' style='min-width:120px;flex:none;width:140px' placeholder='e.g. Mint'></label>"
        "<button onclick='download()'>Download my CSV</button><button class='ghost' onclick='resetAll()'>Clear</button></div></header>"
        f"<main><div class='guide'>{guide_html}</div>{''.join(cards)}</main>"
        f"<script>const TASK={json.dumps(task)};const IDS={json.dumps(ids)};{JS}</script></body></html>"
    )


WET_GUIDE = """
<b>Task: is the road wet or is it raining? (60 images)</b>
<ul>
<li><b>0 Dry</b> &ndash; road looks dry.</li>
<li><b>1 Wet</b> &ndash; surface is wet or damp (dark, shiny, puddles, wet tyre tracks) but you cannot see rain falling.</li>
<li><b>2 Raining</b> &ndash; you can see rain: streaks or drops, drops on the camera lens, umbrellas, rain coats.</li>
<li><b>Can't tell</b> &ndash; haze, glare, too dark. Better to say can't tell than to guess.</li>
</ul>
<p class='muted'>Night images: street lights make a dry road look shiny. Only call it wet if you also see puddles, reflections of moving
cars on water, drops or rain coats. Work alone, do not discuss answers until everyone has finished. Your progress is saved in this
browser; at the end type your name and press <b>Download my CSV</b>, then send the file back.</p>
"""

CNT_GUIDE = """
<b>Task: count the vehicles in each image (40 images)</b>
<ul>
<li>Count every vehicle you can clearly recognise, including far-away ones and <b>parked</b> ones.</li>
<li>Count a vehicle if at least about half of it is visible. A motorcycle <b>with its rider = 1</b>.</li>
<li><b>Car</b>: cars, taxis, SUVs, vans, tuk-tuks. <b>Motorcycle</b>: motorbikes and scooters. <b>Bus</b>: buses and minibuses.
<b>Truck</b>: pick-ups, box trucks, large trucks.</li>
<li>If a scene is very crowded, give your best estimate and write "crowded" in the note. Enter 0 where there are none (do not leave blank).</li>
</ul>
<p class='muted'>Work alone. The images are shown 2&times; larger; zoom in with your browser (Ctrl +) if useful. Progress is saved in this browser;
at the end type your name and press <b>Download my CSV</b>, then send the file back.</p>
"""


def wet_controls(i: str) -> str:
    opts = [("0", "0 Dry"), ("1", "1 Wet"), ("2", "2 Raining"), ("-1", "Can't tell")]
    return "".join(
        f"<label class='opt'><input type='radio' name='r{i}' data-id='{i}' data-k='label' value='{v}'>{t}</label>" for v, t in opts
    )


def cnt_controls(i: str) -> str:
    return "".join(
        f"<label>{t} <input type='number' min='0' max='200' data-id='{i}' data-k='{k}'></label>"
        for k, t in [("car", "Cars"), ("motorcycle", "Motorcycles"), ("bus", "Buses"), ("truck", "Trucks")]
    )


def build() -> None:
    if KIT.exists():
        shutil.rmtree(KIT)
    for sub, src, ids_file in [("1_wetness", SRC_WET, "labels_blank.csv"), ("2_counting", SRC_CNT, "sample_blind.csv")]:
        (KIT / sub / "images").mkdir(parents=True)
        ids = pd.read_csv(src / ids_file, dtype={"idx": str})["idx"].tolist()
        for i in ids:  # the source images were JPEGs already; JPEG q95 keeps them visually identical at a fraction of the size
            with Image.open(src / "images" / f"{i}.png") as im:
                im.convert("RGB").save(KIT / sub / "images" / f"{i}.jpg", quality=95, subsampling=0)
        if sub == "1_wetness":
            (KIT / sub / "wetness_labeling.html").write_text(
                page("Wet road / rain labelling", "wetness", ids, WET_GUIDE, wet_controls), encoding="utf-8")
        else:
            (KIT / sub / "vehicle_counting.html").write_text(
                page("Vehicle counting", "counting", ids, CNT_GUIDE, cnt_controls), encoding="utf-8")
    (KIT / "README_TH.md").write_text(README_TH, encoding="utf-8")
    check_no_leak()
    zpath = KIT.parent / "labeling_kit.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(KIT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(KIT.parent))
    n_files = sum(1 for _ in KIT.rglob("*") if _.is_file())
    print(f"kit: {KIT}  ({n_files} files)  zip: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


def check_no_leak() -> None:
    """Fail loudly if anything that reveals an answer slipped into the kit."""
    banned_names = ("hidden", "yolo", "labels_claude", "manual_counts_claude", "eval_scores", "annotated", "rain")
    for p in KIT.rglob("*"):
        if p.is_file() and any(b in p.name.lower() for b in banned_names):
            raise SystemExit(f"LEAK: {p}")
    allowed = {".jpg", ".html", ".md"}
    bad = [p for p in KIT.rglob("*") if p.is_file() and p.suffix not in allowed]
    if bad:
        raise SystemExit(f"unexpected file types in kit: {bad[:3]}")
    for p in KIT.rglob("*.html"):
        text = p.read_text(encoding="utf-8").lower()
        for w in ("rain_mm", "open-meteo", "yolo", "claude"):
            if w in text:
                raise SystemExit(f"LEAK: '{w}' appears in {p.name}")


README_TH = """# ชุดติดป้าย (labeling kit)

ขอบคุณที่ช่วยครับ ใช้เวลาประมาณ 30–45 นาทีต่องาน ทำคนเดียว **ห้ามปรึกษากันก่อนส่ง** (เพื่อให้เทียบคำตอบของแต่ละคนได้จริง)

## วิธีทำ
1. แตกไฟล์ zip แล้วเปิดไฟล์ `.html` ด้วย Chrome / Edge ได้เลย (ไม่ต้องใช้อินเทอร์เน็ต ไม่ต้องติดตั้งอะไร)
   - `1_wetness/wetness_labeling.html` : ดูภาพ 60 ภาพ แล้วเลือกว่า **แห้ง / เปียก / ฝนตก / บอกไม่ได้**
   - `2_counting/vehicle_counting.html` : นับรถ 40 ภาพ แยก รถยนต์ / มอเตอร์ไซค์ / รถบัส / รถบรรทุก
2. ทำไปได้เรื่อยๆ ระบบจำคำตอบไว้ในเบราว์เซอร์ (ปิดแล้วเปิดใหม่ในเครื่องเดิมได้) ห้ามกด Clear ถ้ายังทำไม่เสร็จ
3. เสร็จแล้วพิมพ์ **ชื่อ** (ภาษาอังกฤษ) ที่ช่องด้านบน กด **Download my CSV** ได้ไฟล์ `labels_<ชื่อ>.csv` หรือ `manual_counts_<ชื่อ>.csv`
4. ส่งไฟล์ CSV กลับมา

## กติกาสั้นๆ
- **ติดป้ายฝน:** 0 = ถนนแห้ง, 1 = ถนนเปียก/ชื้น (มืด เงา มีรอยล้อเปียก แอ่งน้ำ) แต่ไม่เห็นฝนตก, 2 = เห็นฝนตกชัด (เส้นฝน เม็ดฝนบนเลนส์ ร่ม เสื้อกันฝน), ถ้าไม่แน่ใจเลือก "Can't tell" ดีกว่าเดา
  - ภาพกลางคืน: ไฟถนนทำให้ถนนแห้งดูมันวาว ให้ตอบว่าเปียกต่อเมื่อเห็นหลักฐานเพิ่ม เช่น แอ่งน้ำ ละอองฝนในแสงไฟ เสื้อกันฝน
- **นับรถ:** นับทุกคันที่เห็นชัดว่าเป็นรถ (เห็นอย่างน้อยครึ่งคัน) รวมคันที่อยู่ไกลและคันที่จอดอยู่ มอเตอร์ไซค์รวมคนขี่นับเป็น 1 คัน ถ้าไม่มีให้ใส่ 0 ถ้าแน่นมากให้ประมาณและเขียนโน้ตว่า crowded

ในโฟลเดอร์นี้มีแต่ภาพที่ติดเลขกับหน้าเว็บ ไม่มีเฉลยหรือผลของโมเดลใดๆ
"""


if __name__ == "__main__":
    build()
