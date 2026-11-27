"""End-to-end pipeline runner:  raw images + public sources  ->  bronze -> silver -> gold -> dashboard files.

    uv run python run_pipeline.py                 run every stage in order
    uv run python run_pipeline.py --list          show the stages and what each needs
    uv run python run_pipeline.py --from gold     run gold and everything after it
    uv run python run_pipeline.py --only gold,export
    uv run python run_pipeline.py --skip yolo     skip the slow GPU step (re-use existing detections)
    uv run python run_pipeline.py --dry-run       print the plan without running

Design (the points the course asks for in "workflow management"):
  * Ordered stages with declared dependencies (see --list). A partial run (--only/--from/--skip) assumes the earlier
    outputs already exist; it does not check that. If a skipped stage left a gap, the `validate` stage reports it.
  * Every stage is idempotent: re-running adds nothing (bronze/silver/scrape append only missing rows,
    gold/export rebuild derived tables from scratch), so a failed or interrupted run can simply be re-run.
  * Each stage runs in its own process (DuckLake allows one catalog handle per process) and a failing stage stops the
    run immediately (fail-fast) with a non-zero exit code.
  * The last stage `validate` re-checks cross-stage invariants (row counts, uniqueness, no orphans).
  * One JSON run report per run in data/logs/ (stage, status, seconds, output tail) for lineage / the video.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
LOG_DIR = ROOT / "data" / "logs"


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    layer: str
    purpose: str
    needs: tuple[str, ...] = field(default_factory=tuple)


STAGES: list[Stage] = [
    Stage("bronze", "bronze.py", "bronze", "register every CCTV image (camera, capture time, hash) from file names"),
    Stage("quality", "silver_quality.py", "silver", "reject blank/unreadable images, flag frozen frames", ("bronze",)),
    Stage("light", "silver_light.py", "silver", "day / twilight / night from solar elevation", ("bronze",)),
    Stage("problems", "silver_problems.py", "silver", "find glitch / colour-bar test card / unreadable images", ("quality",)),
    Stage("yolo", "silver_detect.py", "silver", "count cars/motorcycles/buses/trucks with YOLOv8 (GPU, slow)", ("problems",)),
    Stage("embed", "silver_embed.py", "silver", "CLIP image embeddings (GPU) for the wet-road score", ("quality",)),
    Stage("wetness", "silver_wetness.py", "silver", "image-based wet/rain score per image (CLIP zero-shot)", ("embed",)),
    Stage("cameras", "scrape_cameras.py", "bronze", "scrape camera coordinates (BMA page, OSM fallback)", ("bronze",)),
    Stage("weather_air", "scrape_environment.py", "bronze", "scrape hourly rain + PM2.5 (Open-Meteo)"),
    Stage("osm", "scrape_osm.py", "bronze", "scrape points of interest around cameras (Overpass)", ("cameras",)),
    Stage("gold", "gold.py", "gold", "camera x hour table + camera profile + audit (full rebuild)",
          ("problems", "yolo", "light", "wetness", "cameras", "weather_air", "osm")),
    Stage("export", "export_gold.py", "serve", "write parquet files the dashboard reads", ("gold",)),
    Stage("validate", "validate.py", "check", "cross-stage consistency checks", ("export",)),
]
BY_NAME = {s.name: s for s in STAGES}


def select(args) -> list[Stage]:
    names = [s.name for s in STAGES]
    if args.only:
        chosen = [n.strip() for n in args.only.split(",") if n.strip()]
    else:
        chosen = names
        if args.start:
            if args.start not in BY_NAME:
                sys.exit(f"unknown stage for --from: {args.start}. choose from {', '.join(names)}")
            chosen = names[names.index(args.start):]
    skip = {n.strip() for n in args.skip.split(",") if n.strip()} if args.skip else set()
    for n in list(chosen) + list(skip):
        if n not in BY_NAME:
            sys.exit(f"unknown stage: {n}. choose from {', '.join(names)}")
    return [BY_NAME[n] for n in names if n in chosen and n not in skip]


def run_stage(stage: Stage, env: dict) -> dict:
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(SRC / stage.script)], cwd=ROOT, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    # drop the library deprecation noise so the report stays readable
    lines = [l for l in out.splitlines() if l.strip() and "Warning" not in l and "pandas.pydata" not in l and ".df()" not in l]
    return {"stage": stage.name, "script": stage.script, "returncode": proc.returncode,
            "status": "ok" if proc.returncode == 0 else "FAILED", "seconds": round(time.time() - t0, 1),
            "output_tail": lines[-12:]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Bangkok CCTV data pipeline")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--only", help="comma-separated stage names to run")
    ap.add_argument("--from", dest="start", help="run this stage and every stage after it")
    ap.add_argument("--skip", help="comma-separated stage names to skip")
    ap.add_argument("--dry-run", action="store_true", help="show the plan only")
    args = ap.parse_args()

    if args.list:
        print(f"{'stage':<13}{'layer':<8}{'needs':<40}purpose")
        for s in STAGES:
            print(f"{s.name:<13}{s.layer:<8}{', '.join(s.needs) or '-':<40}{s.purpose}")
        return 0

    plan = select(args)
    if not plan:
        print("nothing to run")
        return 0
    print("plan: " + " -> ".join(s.name for s in plan))
    if args.dry_run:
        return 0

    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": str(SRC), "PYTHONWARNINGS": "ignore"}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    results, failed = [], None
    for i, stage in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {stage.name:<12} {stage.purpose} ...", flush=True)
        r = run_stage(stage, env)
        results.append(r)
        print(f"      -> {r['status']} in {r['seconds']}s", flush=True)
        for line in r["output_tail"][-6:]:
            print(f"         {line}")
        if r["returncode"] != 0:
            failed = stage.name
            print(f"\nSTOPPED: stage '{stage.name}' failed (exit {r['returncode']}). Later stages were not run.")
            break

    report = {
        "started": started.isoformat(timespec="seconds"), "finished": datetime.now().isoformat(timespec="seconds"),
        "plan": [s.name for s in plan], "failed_stage": failed, "results": results,
    }
    path = LOG_DIR / f"pipeline_{started:%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(r["seconds"] for r in results)
    print(f"\n{'FAILED' if failed else 'SUCCESS'}: {len(results)}/{len(plan)} stages run, {total:.0f}s total. report: {path.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
