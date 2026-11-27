# Bangkok CCTV traffic analysis (2110403 DSDE-CEDT 2026/1 project)

Turns 61,319 raw BMA CCTV images (100 cameras, 25–31 Aug 2026) into a tested lakehouse and a dashboard of traffic
patterns, with rain, PM2.5 and places (OpenStreetMap) as extra sources.

```
raw JPEGs ─► bronze ─► silver ─► gold ─► parquet export ─► Streamlit dashboard
              (what arrived)  (checked + YOLO + light)  (camera x hour)
public web ─► bronze (cameras, weather, PM2.5, OSM)
```

## Run it

Needs [uv](https://docs.astral.sh/uv/) and (for the YOLO step) an NVIDIA GPU; CPU works but takes ~1–2 hours.

```powershell
uv sync
$env:CCTV_RAW_DIR = "C:\path\to\cctv_images\cctv_images"   # folder holding the camera sub-folders
uv run python run_pipeline.py            # all stages; safe to re-run, finished work is skipped
uv run streamlit run src/app.py          # dashboard
```

Useful variants: `--list`, `--dry-run`, `--from gold`, `--only gold,export`, `--skip yolo`.
Each run writes a JSON report to `data/logs/`. The last stage (`validate`) must report 17/17 checks.

## Stages

| stage | script | layer | what it does |
|---|---|---|---|
| bronze | `src/bronze.py` | bronze | register every image: camera, capture time (from file name), size, SHA-256 |
| quality | `src/silver_quality.py` | silver | reject blank/unreadable images (505); flag frozen frames (585) |
| light | `src/silver_light.py` | silver | day / twilight / night from solar elevation |
| problems | `src/silver_problems.py` | silver | find unusable images (33 colour-bar test cards, 16 green glitches, 31 grey glitches, 55 unreadable) -> `silver.image_scan`; excluded from YOLO, gold and the labelling kits |
| yolo | `src/silver_detect.py` | silver | YOLOv8m vehicle counts (car, motorcycle, bus, truck) |
| embed | `src/silver_embed.py` | silver | CLIP image embeddings (needs a GPU for speed; downloads ~600 MB model once) |
| wetness | `src/silver_wetness.py` | silver | wet-road / rain score per image (CLIP zero-shot), daytime use only |
| cameras | `src/scrape_cameras.py` | bronze | camera coordinates: BMA page, OSM Nominatim fallback |
| weather_air | `src/scrape_environment.py` | bronze | hourly rain + PM2.5, Open-Meteo, 3×3 grid |
| osm | `src/scrape_osm.py` | bronze | schools, malls, stations, bus stops… via Overpass |
| gold | `src/gold.py` | gold | camera × hour table, camera profile, audit (full rebuild) |
| export | `src/export_gold.py` | serve | parquet files for the dashboard |
| validate | `src/validate.py` | check | 17 cross-stage consistency checks |

Analysis used by the dashboard lives in `src/analysis.py` (K-Means, rain/PM2.5 effect with camera × hour fixed
effects and bootstrap intervals, anomaly z-scores).

## Things to know before trusting a number

* **Congestion index** = vehicles ÷ the camera's own 95th percentile, computed separately for daytime and
  dusk/night, because YOLO under-counts after dusk (small vehicles appear only as headlights). Do not compare
  night with day. A 40-image hand-count check (`docs/yolo_validation.md`) found YOLO detects ~64% of vehicles by
  day and ~47% at dusk/night, motorcycles worst (~20%): patterns are trustworthy, absolute counts are too low.
* **Problem images**: 135 of 60,814 images (0.2%) are glitches, test cards or too blurred to read. Rules and calibration
  (contact sheets; hazy images with Laplacian variance 60-100 are kept) are in `src/silver_problems.py`; excluding them changes no
  conclusion. Labelling pages have a **Bad image** button; such images are not counted.
* **Detector threshold**: vehicles are counted at YOLO conf 0.15 (was 0.25 until 27 Nov 2026). Five people checked 508 boxes:
  precision 0.95, recall 0.77 at 0.15 versus 0.97 / 0.62 at 0.25 (`python src/round2_eval.py`). Mean vehicles per image rose from 7.1 to 9.0.
  Counts are still below the real number (about 19% missed); the old results are kept in `data/backup/`.
* **Coordinates**: 85 of 100 cameras have a map position (79 BMA, 6 Nominatim). BMA's public list is unstable
  (the same URL returned 372 rows, later 291), so the 79 project cameras that BMA lists are kept in
  `reference/bma_camera_coordinates.csv` (cameras with no images in this dataset were removed), which is merged into each run. Keep that file in version control.
* **Rain**: Open-Meteo rain is a weak match for what the cameras show (daytime AUC 0.62 on 46 AI-labelled images), so the
  project also uses an image-based wetness score (CLIP, AUC 0.78, daytime only). Both gave no detectable effect on daytime
  traffic. See `docs/wetness_validation.md`; the labels were made by an AI model and should be repeated by a person.
* **Verification by people (round 2)** is prepared: `data/labeling_kits/person_1.zip` ... `person_5.zip` (see `docs/verification_round2.md`) and
  `python src/round2_eval.py`. Until teammates return their files, the accuracy figures above come from AI-made labels.
* **PM2.5** is a model (CAMS, ~40 km grid, about four distinct values over Bangkok), not station data; Air4Thai has
  nothing for these dates. Treat PM2.5 results as associations.
* DuckLake note: `CAST(timestamptz AS DATE)` returned a wrong year here, so dates are built from `strftime` text.
* Live sources (OSM, weather) can change over time; re-scraping later may differ slightly from the saved lakehouse.
* `data/` (lakehouse, models, logs, exports) is not meant for version control; the raw images are never modified.
