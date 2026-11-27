# Verification round 2: judge the detector's boxes and the rain measures with people

Round 1 (`docs/yolo_validation.md`, `docs/wetness_validation.md`) was done by an AI labeller on small samples. Round 2 asks teammates,
on fresh images, two questions that round 1 could not answer well:

1. **Is each box YOLO draws a real vehicle, and does lowering the confidence threshold (0.25 -> 0.15) add real vehicles or junk?**
2. **Which rain measure matches the pictures better, Open-Meteo or the CLIP wetness score?**

## What is in the kits (`data/labeling_kits/person_1.zip ... person_5.zip`, built by `python src/round2_kit.py --people 5`)
One self-contained zip per person (about 8.5 MB): three web pages with the pictures embedded (work from anywhere, even inside a zip,
no server, nothing to install), a Thai README, and a CSV export named `<task>_p<k>_<name>.csv`.

| task | images in total | what the person does |
|---|---|---|
| 1 Rain | 210 = 150 fresh daytime + the 60 round-1 images | dry / wet / raining / can't tell |
| 2 Boxes | 60 fresh images, 503 boxes at conf >= 0.15 | click only the WRONG boxes; count vehicles that have no box |
| 3 Counts | the 40 round-1 counting images | count cars, motorcycles, buses, trucks |

* Every image is assigned to **two different people** (verified from the zips themselves). With 5 people each person gets 84 rain
  images, 24 box images (184-223 boxes) and 16 count images, about 1.5-2 hours. For another number of people, rebuild with `--people N`.
* The kits hold no answers: no rain values, no CLIP scores, no earlier labels, no confidence values. A box shows only its number and,
  by colour, the type the detector guessed. `round2_kit.py` refuses to build if it finds an answer word in a page.
* This replaces the first round-2 design (one shared zip with a person-number picker), which was too slow to use.

## The box-checking page (design decisions)
* **Every box starts as correct.** The person clicks only wrong boxes (small menu, or keys 1-4): wrong type, not a vehicle, duplicate.
  This cuts the clicks by roughly 70-80% compared with judging every box.
* **"I checked this image" must be pressed on every image.** Images that are not marked are exported with blank verdicts, never as
  "correct", so a box is never counted as correct without the person saying they looked.
* To see the original picture: a Hide-boxes button per image, a hide-all button, or hold **H**. Zoom 1x/2x/3x per image; the box tags
  stay small at every zoom. Missed vehicles are entered with +/- buttons.
* Cost of the "start as correct" design: people tend to agree with what is shown, so **precision may come out a little too high**. Two
  independent judges per image and their agreement (reported by `round2_eval.py`) limit, but do not remove, this effect. No decoy
  boxes were inserted (decision: keep the data clean).

## How the samples were drawn (`src/round2_samples.py`, fixed seed, no overlap with round 1)
* **Rain:** daytime only, stratified 3 x 3 by Open-Meteo rain band (< 0.1, 0.1-1, >= 1 mm/h) x CLIP band (< 0.25, 0.25-0.5, >= 0.5),
  16-17 images per stratum. The stratum sizes are stored as weights; the evaluation uses **weighted AUC**, so choosing images by a
  detector's own score does not favour or hurt that detector.
* **Boxes:** 30 day, 15 twilight, 15 night images with 2-40 boxes each (very crowded scenes are left out to keep the work feasible);
  108 of the 503 boxes are in the 0.15-0.25 confidence band, the one the threshold decision is about.

## Evaluating the returns
```powershell
uv run python src/round2_eval.py --inbox <folder with the returned CSV files>
```
* **Boxes:** precision, recall and count ratio for thresholds 0.15 ... 0.60 with 95% bootstrap intervals over images, the share of real
  vehicles among boxes with conf 0.15-0.25 (what lowering the threshold would add), class accuracy, duplicates, agreement between judges.
* **Rain:** weighted AUC of Open-Meteo and CLIP against the human consensus (both raters agree) on the fresh sample, sensitivity /
  specificity at the thresholds used, rater agreement (weighted kappa).

## Limits to keep in mind
* Recall is a little optimistic: vehicles YOLO found only below 0.15 are not in the denominator.
* A judge sees the predicted type (as a colour), which can nudge the verdict. The counting task (Task 3, images without boxes) is still
  needed to measure total recall independently.
* Images that two raters label differently are left out of the consensus AUC; the script reports how many.
* Seven days of data and a daytime-only rain check: this validates the measures, it does not add data.
* `round2_eval.py` was tested only on synthetic returns (made up, deleted). Its numbers on real returns will be the first real ones.


## Update: problem images and page controls

* Images flagged by `silver.image_scan` (glitch, test card, unreadable) are never sampled. Two flagged images found after the kits were built (R066, B32) were replaced by `src/round2_fix_samples.py` (same ids, same stratum / light regime; stratum weights recomputed).
* Every page has a **Bad image** button; `round2_eval.py` leaves those images out.
* When zoomed (2x/3x) the image can be panned by dragging with the mouse.
* Counting images (16 per person) are evaluated by `counts_report` (rater agreement and YOLO ratio).
