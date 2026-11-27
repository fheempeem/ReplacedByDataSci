# Is the rain measure right? Checking weather data against what the cameras show

**Question:** the project joined hourly rain from Open-Meteo (a ~9 km weather model) to every image. Does "rain" in that
data match what the camera actually sees (wet road, rain)? Can we measure wetness from the images instead?

## Method
* `src/wetness_sample.py`: 60 images drawn at random (seed 77), 20 from each Open-Meteo rain band of their camera-hour
  (>= 1 mm, 0.1-1 mm, < 0.1 mm), shuffled, shown 2x larger with only a number on them. The rain values sit in
  `data/wetness/hidden_rain.csv`, which was not opened until labelling was finished.
* Labels (`data/wetness/labels_claude.csv`): 0 dry, 1 wet surface, 2 raining (streaks, drops, umbrellas, lens drops), -1 cannot tell.
  59 images were labelled (21 wet or raining, 1 unsure dropped).
* Image detector: CLIP ViT-B/16 (`src/silver_embed.py`, embeddings of every image) asked zero-shot whether the picture
  shows a wet road / rain (`src/wetness_eval.py`, `src/silver_wetness.py` -> `silver.wetness`, column `p_wet`).
* Compared by ROC AUC (0.5 = coin flip, 1 = perfect) with a bootstrap 95% interval.

## IMPORTANT limitations
* **The labeller was Claude (an AI model), not a person.** Night/dusk labels are the least reliable: street lights make dry
  asphalt look wet, and 12 of the 13 dark images were labelled wet. So **only daytime (46 images, 9 wet) is used to judge detectors.**
* Very small test: intervals are wide and overlap. **Teammates should label the same 60 images.** Build the kit with
  `python src/make_labeling_kit.py` (-> `data/labeling_kit.zip`, offline web pages, no answers inside), send it out, then put the
  returned `labels_<name>.csv` in `data/wetness/` and run `python src/agreement.py` (agreement between labellers, and AUC of
  Open-Meteo and CLIP against the human consensus) or `python src/wetness_eval.py <name>` for one labeller. Do not open
  `hidden_rain.csv` before labelling.

## Results (daytime, 46 images)
| detector | AUC | 95% interval |
|---|---|---|
| Open-Meteo rain (mm/h) | 0.62 | 0.47-0.78 |
| CLIP zero-shot wetness | **0.78** | 0.58-0.95 |
| logistic probe on CLIP embedding (leave-one-out) | 0.61 | 0.40-0.82 |

* Open-Meteo with the project's threshold (>= 1 mm/h) found 4 of 9 wet-looking daytime images and flagged 16 of 37 dry ones.
  Hours with < 0.1 mm were dry in all 12 images, but only about 26% (9 of 34) of hours with >= 0.1 mm looked wet.
* Evening rain is missed: two night images with clear umbrellas / raincoats (#31, #48) were 0.0 mm in Open-Meteo.
* The zero-shot CLIP score ranks the two clearly raining daytime images first. The trained probe is worse (too few labels).
* "All images" AUC (0.89 for CLIP) is **not meaningful**: dark images score high on `p_wet` almost regardless of rain
  (84% of dark images >= 0.5), a "dark = wet" shortcut. `p_wet` must not be used at dusk or night.

## Sanity checks on the whole dataset (daytime camera-hours)
* Cameras agree within the same date-hour: date-hour explains 72% of the variance of camera-demeaned `wet_score`
  (0.01 when shuffled), i.e. it behaves like a city-wide condition, not per-camera noise.
* Wetness is highest on Sat 29 Aug (0.45) and still high on Sun 30 (0.37); lowest Tue-Wed (~0.25). Open-Meteo puts more rain on
  Thu-Fri than the pictures show.
* It also drifts upward through the afternoon (0.22 at 10:00, 0.61 at 18:00) because the light falls: a light artefact, removed
  in the traffic analysis by comparing the same camera at the same hour on different days.
* Spearman correlation with Open-Meteo rain is only 0.19-0.21.

## Effect on traffic (daytime congestion index, per +1 SD of each measure)
Same fixed-effects design as before (camera x hour-of-day x weekday/weekend), 95% bootstrap interval over cameras:

| measure | all days | weekdays only |
|---|---|---|
| Open-Meteo rain | +0.1% (-0.4, +0.5) | -0.4% (-1.0, +0.1) |
| image wetness score | +0.6% (-0.2, +1.4) | +0.8% (-0.2, +1.8) |
| share of images with p_wet >= 0.5 | -0.4% (-1.2, +0.3) | -0.4% (-1.1, +0.4) |

**Conclusion:** no detectable effect of rain/wetness on daytime congestion in this week, and the better wetness measure does not
change that. The intervals bound the effect within roughly +-2% per SD. Caveats: one week, heavy rain is rare, and YOLO may also
miss more vehicles in rain, which could hide a real drop in traffic.

## What changed in the project
* New pipeline stages `embed` and `wetness`; `gold.camera_hourly` has `wet_score` and `wet_share`; the dashboard's rain section
  now shows this check and the three measures side by side; `validate` has two extra checks.
* The earlier statement "no detectable effect of rain (+0.3%)" rested only on the Open-Meteo measure, which turned out to be a
  weak proxy. It is now supported by an image-based measure as well, with the limits above.
