"""Build the round-2 labelling kits, one self-contained zip per person:  python src/round2_kit.py [--people 5] [--test]

    data/labeling_kits/person_1.zip ... person_N.zip   each holds rain.html, boxes.html, counts.html, README_TH.md

Every page has its images EMBEDDED (works from anywhere, even straight from inside a zip), saves progress in the browser and
exports a CSV named  <task>_p<person>_<name>.csv.  Every image goes to TWO different people (so agreement can be measured).
  rain.html    dry / wet / raining for 210 images (60 round-1 images + 150 fresh daytime images)
  boxes.html   check the boxes a detector drew on 60 images. All boxes START as correct; click only the wrong ones (small menu),
               press "I checked this image" for every image; hide the boxes to see the original (button, or hold H); zoom 1x/2x/3x;
               missed vehicles with +/- buttons.
  counts.html  count vehicles in the 40 round-1 counting images
The kits hold NO answers: no rain values, no CLIP scores, no earlier labels, no confidence values (boxes show only a number and
the predicted type as colour).  --test builds a tiny version in data/_test/ for trying the pages in a small preview pane.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import shutil
import zipfile

import pandas as pd
from PIL import Image

from common import ROOT

R2 = ROOT / "data" / "round2"
OUT = ROOT / "data" / "labeling_kits"
TEST = ROOT / "data" / "_test"
SEED = 9


def pairs(ids: list[str], n_people: int) -> dict[str, list[int]]:
    """Image k goes to persons (k mod N) and (k+1 mod N): balanced, always two different people."""
    return {i: [k % n_people + 1, (k + 1) % n_people + 1] for k, i in enumerate(ids)}


def data_uri(img: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
:root{color-scheme:light dark;--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e1e0d9;--blue:#2a78d6;--card:#fff;--ok:#0c8a0c}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#383835;--blue:#3987e5;--card:#222221;--ok:#3fc23f}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px}
h1{font-size:17px;margin:0 0 2px}.row{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center}
main{max-width:1060px;margin:0 auto;padding:12px 16px}
.guide{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:15px}
.guide li{margin:2px 0}
article{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:14px}
article.done{border-color:var(--blue);box-shadow:0 0 0 1px var(--blue)}
.top{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;margin-bottom:6px}
.badge{font-size:13px;padding:1px 8px;border-radius:10px;border:1px solid var(--line);color:var(--muted)}
article.done .badge{border-color:var(--ok);color:var(--ok)}
.wrap{position:relative}.wrap img{width:100%;height:auto;border-radius:4px;display:block}
.wrap svg{position:absolute;left:0;top:0;width:100%;height:100%}
.scroller{overflow:auto;max-height:78vh;border-radius:4px}
.zoomer{position:relative;width:100%}.zoomer img{width:100%;height:auto;display:block}
.zoomer svg{position:absolute;left:0;top:0;width:100%;height:100%}
.bx{cursor:pointer}
.scroller.pan{cursor:grab}.scroller.dragging{cursor:grabbing}.scroller.dragging .bx{cursor:grabbing}
.zoomer img{user-select:none;-webkit-user-drag:none}
button.bad.on{border-color:#d03b3b;outline:2px solid #d03b3b;color:var(--ink);background:transparent}
article.isbad{border-color:#d03b3b;box-shadow:0 0 0 1px #d03b3b;opacity:.75}
body.hideboxes svg,svg.off{visibility:hidden}
.ctl{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin-top:8px}
label.opt{display:inline-flex;gap:6px;align-items:center;cursor:pointer;padding:6px 10px;border:1px solid var(--line);border-radius:6px}
label.opt:has(input:checked){border-color:var(--blue);outline:2px solid var(--blue)}
input[type=number]{width:64px;padding:6px;font-size:16px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
input[type=text]{padding:6px;font-size:15px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
input.note{flex:1;min-width:180px}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--blue);background:var(--blue);color:#fff;cursor:pointer}
button.ghost{background:transparent;color:var(--ink);border-color:var(--line)}
button.on{border-color:var(--blue);outline:2px solid var(--blue);background:transparent;color:var(--ink)}
button.big{padding:9px 16px;font-weight:600}
.step{display:inline-flex;align-items:center;gap:4px}.step b{min-width:22px;text-align:center}.step button{padding:2px 10px}
.num{font-weight:600}.muted{color:var(--muted);font-size:14px}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:4px}
#menu{display:none;position:fixed;z-index:20;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px;box-shadow:0 4px 18px rgba(0,0,0,.35);min-width:190px}
#menu .mh{font-weight:600;padding:2px 8px 6px;color:var(--muted)}
#menu button{display:block;width:100%;text-align:left;margin:2px 0;background:transparent;color:var(--ink);border:1px solid var(--line)}
#menu button:hover{border-color:var(--blue)}
"""

JS_COMMON = r"""
const esc = t => '"' + (t || "").replace(/"/g, '""') + '"';
function whoName() { const who = (document.getElementById("who").value || "").trim().replace(/[^A-Za-z0-9_-]/g, "");
  if (!who) { alert("Type your name first (letters/numbers only, e.g. Mint)."); return null; } return who; }
function saveFile(name, csv) { const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove(); }
"""

JS_SIMPLE = r"""
const KEY = "r2_" + TASK + "_p" + PERSON; let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
function isBad(id) { const s = state[id] || {}; return TASK === "rain" ? s.label === "bad" : s.bad === "1"; }
function isDone(id) { const s = state[id] || {}; if (isBad(id)) return true;
  return TASK === "rain" ? (s.label !== undefined && s.label !== "") : ["car","motorcycle","bus","truck"].every(k => s[k] !== undefined && s[k] !== ""); }
function refresh() { let n = 0; IDS.forEach(id => { const d = isDone(id); if (d) n++; const a = document.getElementById("a" + id);
  a.classList.toggle("done", d && !isBad(id)); a.classList.toggle("isbad", isBad(id)); });
  document.getElementById("progress").textContent = n + " / " + IDS.length + " done"; }
function set(id, k, v) { (state[id] = state[id] || {})[k] = v; save(); refresh(); }
IDS.forEach(id => { const s = state[id] || {};
  document.querySelectorAll("[data-id='" + id + "']").forEach(inp => { const k = inp.dataset.k;
    if (inp.type === "radio") { inp.checked = String(s[k]) === inp.value; inp.onchange = () => set(id, k, inp.value); }
    else if (inp.type === "checkbox") { inp.checked = s[k] === "1"; inp.onchange = () => set(id, k, inp.checked ? "1" : ""); }
    else { inp.value = s[k] !== undefined ? s[k] : ""; inp.oninput = () => set(id, k, inp.value); } }); });
function download() { const who = whoName(); if (!who) return; let csv;
  if (TASK === "rain") csv = "idx,label,note\n" + IDS.map(id => { const s = state[id] || {}; return [id, s.label === undefined ? "" : s.label, esc(s.note)].join(","); }).join("\n");
  else csv = "idx,car,motorcycle,bus,truck,bad,notes\n" + IDS.map(id => { const s = state[id] || {}; return [id, s.car ?? "", s.motorcycle ?? "", s.bus ?? "", s.truck ?? "", s.bad === "1" ? 1 : "", esc(s.note)].join(","); }).join("\n");
  const left = IDS.filter(id => !isDone(id)).length;
  if (left && !confirm(left + " of " + IDS.length + " images are not finished. Download anyway?")) return;
  saveFile(TASK + "_p" + PERSON + "_" + who + ".csv", csv); }
function resetAll() { if (confirm("Clear everything you entered on this page?")) { state = {}; save(); location.reload(); } }
refresh();
"""

JS_BOXES = r"""
const KEY = "r2_boxes_p" + PERSON; let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const S = id => (state[id] = state[id] || { v: {} });
const COL = { car: "#2a78d6", motorcycle: "#eb6834", bus: "#1baf7a", truck: "#eda100" };
const SYM = { ok: "", wrong_class: "≠", not_vehicle: "✗", duplicate: "dup" };
const zoom = {}; let menuFor = null;
const area = b => (b.x2 - b.x1) * (b.y2 - b.y1);
const verdict = (id, bid) => S(id).v[bid] || "ok";

function draw(id) {
  const svg = document.getElementById("s" + id), u = 1 / (zoom[id] || 1); let h = "";
  [...BOXES[id]].sort((a, b) => area(b) - area(a)).forEach(b => {
    const v = verdict(id, b.id), col = COL[b.cls], w = b.x2 - b.x1, ht = b.y2 - b.y1;
    const stroke = v === "duplicate" ? "#c3c2b7" : v === "not_vehicle" ? "#d03b3b" : col;
    const dash = v === "wrong_class" ? `stroke-dasharray="${5 * u} ${3 * u}"` : v === "duplicate" ? `stroke-dasharray="${3 * u} ${3 * u}"` : "";
    const label = b.id + (SYM[v] ? " " + SYM[v] : "");
    const tw = (label.length * 6.3 + 6) * u, th = 12.5 * u, ty = b.y1 - th >= 0 ? b.y1 - th : b.y1;
    h += `<g class="bx" onclick="openMenu(event,'${id}',${b.id})"><title>#${b.id} ${b.cls}${v !== "ok" ? " - " + v.replace("_", " ") : ""}</title>` +
      `<rect x="${b.x1}" y="${b.y1}" width="${w}" height="${ht}" fill="rgba(0,0,0,0.001)" stroke="#000" stroke-opacity="0.55" stroke-width="${3.2 * u}"/>` +
      `<rect x="${b.x1}" y="${b.y1}" width="${w}" height="${ht}" fill="${v === "not_vehicle" ? "rgba(208,59,59,0.12)" : "rgba(0,0,0,0.001)"}" stroke="${stroke}" stroke-width="${1.5 * u}" ${dash}/>` +
      (v === "not_vehicle" ? `<line x1="${b.x1}" y1="${b.y1}" x2="${b.x2}" y2="${b.y2}" stroke="#d03b3b" stroke-width="${1.4 * u}"/><line x1="${b.x2}" y1="${b.y1}" x2="${b.x1}" y2="${b.y2}" stroke="#d03b3b" stroke-width="${1.4 * u}"/>` : "") +
      `<rect x="${b.x1}" y="${ty}" width="${tw}" height="${th}" fill="${col}" fill-opacity="0.92"/>` +
      `<text x="${b.x1 + 3 * u}" y="${ty + th - 3 * u}" font-size="${9.5 * u}" fill="${b.cls === "truck" ? "#0b0b0b" : "#fff"}" font-family="system-ui,sans-serif">${label}</text></g>`;
  });
  svg.innerHTML = h;
}
function setZoom(id, z) { zoom[id] = z; document.getElementById("zm" + id).style.width = (100 * z) + "%"; draw(id);
  document.querySelector("#a" + id + " .scroller").classList.toggle("pan", z > 1);
  [1, 2, 3].forEach(k => document.getElementById("z" + k + id).classList.toggle("on", k === z)); }
/* drag the picture to pan when zoomed (mouse); a click without moving still opens the box menu */
let drag = null, suppress = false;
document.querySelectorAll(".scroller").forEach(sc => sc.addEventListener("mousedown", e => {
  if (e.button !== 0) return; drag = { sc, x: e.clientX, y: e.clientY, l: sc.scrollLeft, t: sc.scrollTop, moved: false }; e.preventDefault(); }));
window.addEventListener("mousemove", e => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) > 4) { drag.moved = true; drag.sc.classList.add("dragging"); closeMenu(); }
  if (drag.moved) { drag.sc.scrollLeft = drag.l - dx; drag.sc.scrollTop = drag.t - dy; } });
window.addEventListener("mouseup", () => { if (!drag) return;
  if (drag.moved) { suppress = true; setTimeout(() => { suppress = false; }, 0); } drag.sc.classList.remove("dragging"); drag = null; });
function toggleHide(id) { const svg = document.getElementById("s" + id), off = svg.classList.toggle("off");
  document.getElementById("hb" + id).textContent = off ? "Show boxes" : "Hide boxes (original)"; }
function openMenu(ev, id, bid) { ev.stopPropagation(); if (suppress) return; menuFor = { id, bid }; const b = BOXES[id].find(x => x.id === bid), m = document.getElementById("menu");
  m.querySelector(".mh").textContent = "#" + bid + " " + b.cls; m.style.display = "block";
  m.style.left = Math.max(4, Math.min(ev.clientX + 6, innerWidth - 210)) + "px"; m.style.top = Math.max(4, Math.min(ev.clientY + 6, innerHeight - 200)) + "px"; }
function closeMenu() { document.getElementById("menu").style.display = "none"; menuFor = null; }
function pick(v) { if (!menuFor) return; const { id, bid } = menuFor; if (v === "ok") delete S(id).v[bid]; else S(id).v[bid] = v; save(); draw(id); closeMenu(); }
function step(id, k, d) { const s = S(id); s[k] = Math.max(0, (s[k] || 0) + d); document.getElementById("n" + k + id).textContent = s[k]; save(); }
function toggleChecked(id) { const s = S(id); s.checked = !s.checked; if (s.checked) s.bad = false; save(); refreshCard(id); refresh(); }
function toggleBad(id) { const s = S(id); s.bad = !s.bad; if (s.bad) s.checked = false; save(); refreshCard(id); refresh(); }
const done = id => !!(S(id).checked || S(id).bad);
function refreshCard(id) { const s = S(id), a = document.getElementById("a" + id);
  a.classList.toggle("done", !!s.checked); a.classList.toggle("isbad", !!s.bad);
  document.getElementById("bd" + id).textContent = s.bad ? "bad image (not counted)" : s.checked ? "checked" : "not checked yet";
  document.getElementById("ck" + id).textContent = s.checked ? "✔ Checked (click to undo)" : "✔ I checked this image";
  document.getElementById("ck" + id).classList.toggle("on", !!s.checked);
  document.getElementById("bad" + id).classList.toggle("on", !!s.bad);
  document.getElementById("bad" + id).textContent = s.bad ? "Bad image: marked (click to undo)" : "Bad image (unusable)"; }
function refresh() { const n = IDS.filter(done).length; document.getElementById("progress").textContent = n + " / " + IDS.length + " images done"; }
document.addEventListener("click", closeMenu);
document.getElementById("menu").addEventListener("click", e => e.stopPropagation());
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "Escape") closeMenu();
  else if (menuFor && "1234".includes(e.key)) pick(["ok", "wrong_class", "not_vehicle", "duplicate"][+e.key - 1]);
  else if ((e.key === "h" || e.key === "H") && !e.repeat) document.body.classList.add("hideboxes"); });
document.addEventListener("keyup", e => { if (e.key === "h" || e.key === "H") document.body.classList.remove("hideboxes"); });
window.addEventListener("blur", () => document.body.classList.remove("hideboxes"));
function download() { const who = whoName(); if (!who) return;
  const rows = ["kind,idx,box_id,verdict,missed_car,missed_motorcycle,missed_other,bad,note"];
  IDS.forEach(id => { const s = S(id), ck = !!s.checked && !s.bad;   // a bad image is never exported as checked
    BOXES[id].forEach(b => rows.push(["box", id, b.id, ck ? (s.v[b.id] || "ok") : "", "", "", "", "", ""].join(",")));
    rows.push(["image", id, "", "", ck ? (s.mcar || 0) : "", ck ? (s.mmoto || 0) : "", ck ? (s.mother || 0) : "", s.bad ? 1 : "", esc(s.note)].join(",")); });
  const left = IDS.filter(id => !done(id)).length;
  if (left && !confirm(left + " of " + IDS.length + " images are not finished (their boxes will be left blank). Download anyway?")) return;
  saveFile("boxes_p" + PERSON + "_" + who + ".csv", rows.join("\n")); }
function resetAll() { if (confirm("Clear everything you entered on this page?")) { state = {}; save(); location.reload(); } }
IDS.forEach(id => { const s = S(id);
  ["mcar", "mmoto", "mother"].forEach(k => { document.getElementById("n" + k + id).textContent = s[k] || 0; });
  const note = document.getElementById("note" + id); note.value = s.note || ""; note.oninput = () => { S(id).note = note.value; save(); };
  setZoom(id, 1); refreshCard(id); });
refresh();
"""


def page(title, task, ids, person, guide, cards, boxes=None, extra_header="") -> str:
    js = JS_COMMON + (JS_BOXES if task == "boxes" else JS_SIMPLE)
    menu = ("<div id='menu'><div class='mh'></div><button onclick=\"pick('ok')\">1 &nbsp;&#10003; correct</button>"
            "<button onclick=\"pick('wrong_class')\">2 &nbsp;&ne; vehicle, wrong type</button>"
            "<button onclick=\"pick('not_vehicle')\">3 &nbsp;&#10007; not a vehicle</button>"
            "<button onclick=\"pick('duplicate')\">4 &nbsp;dup (second box, same vehicle)</button></div>") if task == "boxes" else ""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{CSS}</style></head><body><header><h1>{title} <span class='muted'>(person {person})</span></h1>"
        f"<div class='row'><span id='progress' class='muted'></span>{extra_header}<label>Name <input id='who' type='text' style='width:120px' placeholder='e.g. Mint'></label>"
        "<button onclick='download()'>Download my CSV</button><button class='ghost' onclick='resetAll()'>Clear</button></div></header>"
        f"<main><div class='guide'>{guide}</div>{cards}</main>{menu}"
        f"<script>const TASK={json.dumps(task)};const PERSON={person};const IDS={json.dumps(ids)};const BOXES={json.dumps(boxes or {})};{js}</script></body></html>"
    )


RAIN_GUIDE = """<b>Task 1: is the road wet or is it raining?</b>
<ul><li><b>0 Dry</b> &ndash; road looks dry.</li>
<li><b>1 Wet</b> &ndash; wet or damp surface (dark, shiny, puddles, wet tyre tracks) but no rain visible.</li>
<li><b>2 Raining</b> &ndash; you can see rain: streaks or drops, drops on the camera lens, umbrellas, rain coats.</li>
<li><b>Can't tell</b> &ndash; haze, glare, too dark. Better than guessing.</li>
<li><b>Bad image</b> &ndash; the picture itself is broken: green or grey smear, colour bars, blank, or blurred beyond recognition. It will not be counted.</li></ul>
<p class='muted'>Night: street lights make a dry road look shiny; call it wet only with extra evidence (puddles, drops in the lights, rain coats).
Work alone, do not discuss answers until everyone has finished.</p>"""

COUNT_GUIDE = """<b>Task 3: count the vehicles in each image</b>
<ul><li>Count every vehicle you can clearly recognise, far-away and <b>parked</b> ones too, if at least about half of it is visible. A motorcycle <b>with its rider = 1</b>.</li>
<li><b>Car</b>: cars, taxis, SUVs, vans, tuk-tuks. <b>Motorcycle</b>. <b>Bus</b>: buses and minibuses. <b>Truck</b>: pick-ups, box trucks, large trucks.</li>
<li>Crowded scene: best estimate and write "crowded". Enter 0 where there are none (do not leave blank).</li>
<li>If the picture itself is broken (green or grey smear, colour bars, blank, blurred beyond recognition) tick <b>Bad image</b>; it will not be counted.</li></ul>
<p class='muted'>Work alone. Zoom the browser (Ctrl +) if it helps.</p>"""

BOX_GUIDE = """<b>Task 2: check the boxes a detection program drew</b>
<ul><li><b>Every box starts as correct.</b> Click a box <b>only if it is wrong</b>; a small menu opens (or press 1-4): </li></ul>
<div style='margin-left:24px'><b>&#10003; correct</b> a vehicle of that type (nothing to do) &middot;
<b>&ne; wrong type</b> a vehicle, but the colour/type is wrong &middot; <b>&#10007; not a vehicle</b> (shadow, sign, building, a person alone...) &middot;
<b>dup</b> a second box on a vehicle that already has a box.</div>
<ul><li>Box colour = type the program guessed:
<span class='sw' style='background:#2a78d6'></span>car <span class='sw' style='background:#eb6834'></span>motorcycle
<span class='sw' style='background:#1baf7a'></span>bus <span class='sw' style='background:#eda100'></span>truck.
A box around <b>several</b> vehicles together is <b>not a vehicle</b> unless it clearly marks one.</li>
<li><b>See the original picture:</b> press <b>Hide boxes</b> under the picture, or <b>hold the H key</b>. Use <b>1&times; 2&times; 3&times;</b> to zoom, then <b>drag the picture with the mouse</b> to move around (a click without moving still opens the box menu).</li>
<li><b>Broken picture</b> (green or grey smear, colour bars, blank, blurred beyond recognition): press <b>Bad image (unusable)</b> instead of checking it; it will not be counted.</li>
<li>Then enter how many vehicles you can see that have <b>no box at all</b> (+ / &minus;; motorcycle with rider = 1; 0 is fine).</li>
<li><b>Press &quot;I checked this image&quot; on every image.</b> Images you do not mark as checked are left blank, not counted as correct.</li></ul>
<p class='muted'>Work alone. Hover a box to see its type.</p>"""


def build(n_people: int, test: bool = False) -> None:
    rng = random.Random(SEED)
    # ----------------------------------------------------------------------------- source images and ids
    rain_src = {p.stem: p for p in sorted((R2 / "samples" / "rain").glob("R*.jpg"))}
    for i in pd.read_csv(ROOT / "data" / "wetness" / "labels_blank.csv", dtype={"idx": str})["idx"]:
        rain_src[f"W{i}"] = ROOT / "data" / "wetness" / "images" / f"{i}.png"
    rain_ids = sorted(rain_src)
    rng.shuffle(rain_ids)
    cnt_src = {f"C{i}": ROOT / "data" / "validation" / "images" / f"{i}.png"
               for i in pd.read_csv(ROOT / "data" / "validation" / "sample_blind.csv", dtype={"idx": str})["idx"]}
    cnt_ids = sorted(cnt_src)
    rng.shuffle(cnt_ids)
    bh = pd.read_csv(R2 / "hidden" / "boxes_hidden.csv", dtype={"idx": str})
    bimg = pd.read_csv(R2 / "hidden" / "box_images_hidden.csv", dtype={"idx": str})
    box_ids = list(bimg["idx"])
    rng.shuffle(box_ids)
    boxes = {  # the hidden table also holds the confidence: deliberately NOT copied
        i: [{"id": int(b.box_id), "cls": b.cls, "x1": round(b.x1 * 2, 1), "y1": round(b.y1 * 2, 1), "x2": round(b.x2 * 2, 1), "y2": round(b.y2 * 2, 1)}
            for b in bh[bh["idx"] == i].itertuples()] for i in box_ids}
    dims = {r.idx: (r.width * 2, r.height * 2) for r in bimg.itertuples()}
    A = {"rain": pairs(rain_ids, n_people), "boxes": pairs(box_ids, n_people), "counts": pairs(cnt_ids, n_people)}

    out_dir = TEST if test else OUT
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    q_rain, q_cnt, q_box = (45, 45, 50) if test else (82, 85, 90)

    def load(path):
        with Image.open(path) as im:
            return im.convert("RGB")

    def rain_card(i):
        opts = [("0", "0 Dry"), ("1", "1 Wet"), ("2", "2 Raining"), ("-1", "Can't tell"), ("bad", "Bad image")]
        r = "".join(f"<label class='opt'><input type='radio' name='r{i}' data-id='{i}' data-k='label' value='{v}'>{t}</label>" for v, t in opts)
        return (f"<article id='a{i}'><div class='num'>#{i}</div><div class='wrap'><img src='{data_uri(load(rain_src[i]), q_rain)}'></div>"
                f"<div class='ctl'>{r}<input class='note' type='text' data-id='{i}' data-k='note' placeholder='note (optional)'></div></article>")

    def count_card(i):
        f = "".join(f"<label>{t} <input type='number' min='0' max='200' data-id='{i}' data-k='{k}'></label>" for k, t in
                    [("car", "Cars"), ("motorcycle", "Motorcycles"), ("bus", "Buses"), ("truck", "Trucks")])
        return (f"<article id='a{i}'><div class='num'>#{i}</div><div class='wrap'><img src='{data_uri(load(cnt_src[i]), q_cnt)}'></div>"
                f"<div class='ctl'>{f}<label class='opt'><input type='checkbox' data-id='{i}' data-k='bad'>Bad image</label>"
                f"<input class='note' type='text' data-id='{i}' data-k='note' placeholder='note (optional)'></div></article>")

    def box_card(i):
        w, h = dims[i]
        zb = "".join(f"<button id='z{k}{i}' class='ghost' onclick='setZoom(\"{i}\",{k})'>{k}&times;</button>" for k in (1, 2, 3))
        st = "".join(f"<span class='step'>{t} <button class='ghost' onclick='step(\"{i}\",\"{k}\",-1)'>&minus;</button><b id='n{k}{i}'>0</b>"
                     f"<button class='ghost' onclick='step(\"{i}\",\"{k}\",1)'>+</button></span>"
                     for k, t in (("mcar", "Missed cars"), ("mmoto", "Missed motorcycles"), ("mother", "Missed other")))
        src = data_uri(load(R2 / "samples" / "boxes" / f"{i}.jpg"), q_box)
        return (f"<article id='a{i}'><div class='top'><span class='num'>#{i}</span><span class='muted'>{len(boxes[i])} boxes</span>"
                f"<span class='badge' id='bd{i}'>not checked yet</span><span class='muted'>Zoom</span>{zb}"
                f"<button class='ghost' id='hb{i}' onclick='toggleHide(\"{i}\")'>Hide boxes (original)</button>"
                f"<button class='ghost bad' id='bad{i}' onclick='toggleBad(\"{i}\")'>Bad image (unusable)</button></div>"
                f"<div class='scroller'><div class='zoomer' id='zm{i}'><img src='{src}'><svg id='s{i}' viewBox='0 0 {w} {h}'></svg></div></div>"
                f"<div class='ctl'>{st}<input class='note' id='note{i}' type='text' placeholder='note (optional)'>"
                f"<button class='big ghost' id='ck{i}' onclick='toggleChecked(\"{i}\")'>&#10004; I checked this image</button></div></article>")

    people = [1] if test else list(range(1, n_people + 1))
    stats = []
    for p in people:
        mine = {t: [i for i in ids if p in A[t][i]] for t, ids in (("rain", rain_ids), ("boxes", box_ids), ("counts", cnt_ids))}
        if test:  # a tiny version for the small preview pane
            mine = {"rain": mine["rain"][:2], "boxes": sorted(mine["boxes"], key=lambda i: len(boxes[i]))[:2], "counts": mine["counts"][:2]}
        pages = {
            "rain.html": page("Task 1: wet road / rain", "rain", mine["rain"], p, RAIN_GUIDE, "".join(rain_card(i) for i in mine["rain"])),
            "boxes.html": page("Task 2: check the boxes", "boxes", mine["boxes"], p, BOX_GUIDE, "".join(box_card(i) for i in mine["boxes"]),
                               {i: boxes[i] for i in mine["boxes"]},
                               "<button class='ghost' onclick=\"document.body.classList.toggle('hideboxes')\">Hide / show ALL boxes (or hold H)</button>"),
            "counts.html": page("Task 3: count vehicles", "counts", mine["counts"], p, COUNT_GUIDE, "".join(count_card(i) for i in mine["counts"])),
        }
        check_no_leak(pages)
        folder = out_dir / f"person_{p}"
        folder.mkdir(parents=True)
        for name, html in pages.items():
            (folder / name).write_text(html, encoding="utf-8")
        (folder / "README_TH.md").write_text(readme(p, len(mine["rain"]), len(mine["boxes"]), len(mine["counts"]),
                                                    sum(len(boxes[i]) for i in mine["boxes"])), encoding="utf-8")
        if not test:
            z = out_dir / f"person_{p}.zip"
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(folder.iterdir()):
                    zf.write(f, f"person_{p}/{f.name}")
            stats.append((p, z.stat().st_size / 1e6, {t: len(v) for t, v in mine.items()}, sum(len(boxes[i]) for i in mine["boxes"])))
    if not test:
        for p in people:
            shutil.rmtree(out_dir / f"person_{p}")
        rows = [{"task": t, "idx": i, "person_a": a[0], "person_b": a[1]} for t, M in A.items() for i, a in M.items()]
        pd.DataFrame(rows).to_csv(R2 / "hidden" / "assignments.csv", index=False)
        for p, mb, n, nb in stats:
            print(f"person_{p}.zip  {mb:5.1f} MB   images {n}   boxes to check {nb}")
    else:
        print("test pages in", out_dir / "person_1", {f.name: f.stat().st_size // 1024 for f in (out_dir / "person_1").iterdir()})


def check_no_leak(pages: dict[str, str]) -> None:
    for name, html in pages.items():
        t = re.sub(r"data:image/jpeg;base64,[A-Za-z0-9+/=]+", "", html).lower()  # ignore the embedded pictures
        for w in ("rain_mm", "p_wet", "clip", "open-meteo", "claude", "yolo", "rain_hidden", "boxes_hidden", "om_band", "clip_band"):
            if w in t:  # (the CSS word 'visibility:hidden' is fine, so 'hidden' alone is not banned)
                raise SystemExit(f"LEAK: '{w}' in {name}")
        if re.search(r"(?<!font-)weight", t):
            raise SystemExit(f"LEAK: sampling weight in {name}")
        if re.search(r"\bconf\b|confidence|\"conf\"", t):
            raise SystemExit(f"LEAK: confidence values in {name}")


def readme(p: int, n_rain: int, n_box: int, n_cnt: int, n_boxes: int) -> str:
    return f"""# ชุดตรวจสอบรอบที่ 2 : คนที่ {p}

ขอบคุณที่ช่วยครับ ใช้เวลาประมาณ **1.5–2 ชั่วโมง** ทำคนเดียว **ห้ามปรึกษากันก่อนส่ง** (เพื่อให้เทียบความเห็นของแต่ละคนได้จริง)

## วิธีทำ
1. เปิดไฟล์ `.html` ทั้ง 3 ไฟล์ด้วย Chrome หรือ Edge (ภาพฝังอยู่ในไฟล์แล้ว เปิดจากไหนก็ได้ ไม่ต้องต่ออินเทอร์เน็ต)
2. ระบบจำคำตอบไว้ในเบราว์เซอร์ ปิดแล้วเปิดใหม่ในเครื่องเดิมได้ **ห้ามกด Clear ถ้ายังทำไม่เสร็จ**
3. เสร็จแต่ละไฟล์ พิมพ์ **ชื่อ** (ภาษาอังกฤษ) แล้วกด **Download my CSV** ส่งไฟล์กลับมา (ได้ 3 ไฟล์)

## งานของคุณ ({n_rain} + {n_box} + {n_cnt} ภาพ)
- **rain.html ({n_rain} ภาพ):** แห้ง / เปียก / ฝนตก / บอกไม่ได้ (ประมาณ 30 นาที)
- **boxes.html ({n_box} ภาพ, {n_boxes} กรอบ):** ภาพมีกรอบที่โปรแกรมวาดไว้ **ทุกกรอบเริ่มต้นเป็น "ถูก"** ให้คลิกเฉพาะกรอบที่ผิด
  (เมนูเล็ก: ≠ ผิดชนิด / ✗ ไม่ใช่รถ / dup กรอบซ้ำ) อยากดูภาพต้นฉบับกด **Hide boxes** หรือ **กดค้าง H** ซูมได้ 1×/2×/3×
  แล้วบอกจำนวนรถที่ **ไม่มีกรอบ** ด้วยปุ่ม +/− และ **กด "I checked this image" ให้ครบทุกภาพ** (ภาพที่ไม่กดจะไม่ถูกนับ)
  ตอนซูมให้ **ลากภาพด้วยเมาส์** เพื่อเลื่อนดูส่วนอื่นได้ (คลิกเฉยๆ โดยไม่ลาก = เปิดเมนูของกรอบ)
- **ภาพเสีย** (เขียว/เทาเละ/แถบสี/ขาวล้วน/มัวจนดูไม่ออก) ในทั้ง 3 งานมีปุ่มหรือช่อง **Bad image** กดแล้วภาพนั้นจะไม่ถูกนับ
- **counts.html ({n_cnt} ภาพ):** นับรถแยกชนิดในภาพที่ไม่มีกรอบ

กติกาละเอียดอยู่ด้านบนของแต่ละหน้า ภาพกลางคืนยากทุกคน ตอบ "Can't tell" ได้ถ้าไม่แน่ใจ
ในโฟลเดอร์นี้ไม่มีเฉลยหรือผลของโมเดลใดๆ ยกเว้นกรอบที่ให้คุณตรวจ
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=5)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    build(a.people, a.test)
