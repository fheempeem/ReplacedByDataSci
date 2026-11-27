# YOLO validation by hand count

**Question:** how many of the vehicles a person can see does YOLOv8m (conf 0.25, 640 px) find?

## Method
* `src/validation_sample.py`: 40 images drawn at random (seed 2026), stratified by light regime
  (24 day, 8 twilight, 8 night), frozen frames excluded. Images are shown 2x larger, labelled only with a number.
* Counted **blind** (YOLO's counts were in a separate file that was not opened until counting finished).
  Rule: count every vehicle clearly recognisable as a vehicle (at least about half visible, distant ones if identifiable,
  parked ones too, motorcycle + rider = 1). Files: `data/validation/manual_counts_claude.csv`.
* `src/validation_compare.py` compares; `src/validation_settings.py` tests other YOLO settings.

## IMPORTANT limitation
The counter was **Claude (an AI model) looking at the images, not a person.** It is a reasonable independent check but
not ground truth: distant, dark and densely packed vehicles are hard to count for anyone (dense scenes were
estimated, +-5). The sample is small (40). **Teammates should repeat the count**: `python src/make_labeling_kit.py` builds
`data/labeling_kit.zip` (offline web page, 40 numbered images, no YOLO answers); put the returned `manual_counts_<name>.csv` in
`data/validation/` and run `python src/agreement.py` (agreement between counters and the YOLO/human ratio for each) or
`python src/validation_compare.py <name>` for one counter.

## Results (YOLO / manual, sum over images)
| group | n | manual mean | YOLO mean | ratio (95% interval) | rank corr. (Spearman) |
|---|---|---|---|---|---|
| all | 40 | 12.3 | 7.0 | 0.57 (0.50–0.66) | 0.79 |
| day | 24 | 12.4 | 7.9 | 0.64 (0.52–0.77) | 0.70 |
| dusk/night | 16 | 12.2 | 5.8 | 0.47 (0.37–0.56) | 0.93 |

| class | day: manual → YOLO | dusk/night: manual → YOLO |
|---|---|---|
| car | 191 → 142 (0.74) | 153 → 76 (0.50) |
| motorcycle | 77 → 15 (**0.19**) | 35 → 8 (0.23) |
| bus | 8 → 6 | 3 → 6 |
| truck | 21 → 27 | 4 → 2 |

* YOLO found **no** vehicle in 3 night/dusk images where a person saw vehicles; never in the daytime sample.
* Counts are close to proportional to traffic: log-log slope 0.92 (a scene with twice the vehicles shows as ~1.9x).
  Dense scenes are under-counted most (ratio 0.47 when more than 15 vehicles).
* Largest misses: dense dusk queue (17 vs 38), a motorcycle queue (9 vs 27), a distant flyover (4 vs 21).

## What it means for the project
* **Safe:** relative comparisons (weekday vs weekend, hour of day, camera vs camera after per-camera normalisation).
* **Not safe:** absolute vehicle counts, motorcycle counts, and day-vs-night comparisons.
* Image-level counts are noisy (day Pearson 0.55); averaging ~6 images per camera-hour helps.

## Other settings (same 40 images; hypotheses only, tuned on the same sample)
| setting (size/conf) | day ratio | dusk/night ratio | day MAE | images with YOLO > 1.5x manual |
|---|---|---|---|---|
| 640 / 0.25 (used) | 0.63 | 0.47 | 4.9 | 0 |
| 960 / 0.25 | 0.62 | 0.44 | 5.2 | 0 |
| 1280 / 0.25 | 0.61 | 0.36 | 5.7 | 0 |
| 1280 / 0.15 | 0.77 | 0.52 | 4.9 | 3 |
| 1280 / 0.10 | 0.93 | 0.67 | 4.1 | 4 |
| **640 / 0.15** | **0.84** | **0.60** | **3.9** | **0** |

Higher resolution did not help. Lowering the confidence threshold to 0.15 at 640 px recovers much of the gap with no
over-counting and the same speed (~50 min GPU for the full set). Motorcycle recall stays low (<= 0.29) in every setting, so
that would need a different model. **Before re-running the whole dataset, confirm on fresh hand-counted images.**
