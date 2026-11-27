"""Central paths. Big files live outside OneDrive.

Each teammate can point these at their own folders with environment variables:
  CCTV_IMAGES_DIR  folder that contains the 100 camera sub-folders
  DSDE_DATA_DIR    folder for bronze/silver/gold/models/labels
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

IMAGES_DIR = Path(os.environ.get("CCTV_IMAGES_DIR", r"C:\Users\peemp\Downloads\cctv_images\cctv_images"))
DATA_DIR = Path(os.environ.get("DSDE_DATA_DIR", r"C:\Users\peemp\dsde_data"))
BRONZE = DATA_DIR / "bronze"
SILVER = DATA_DIR / "silver"
GOLD = DATA_DIR / "gold"
MODELS = DATA_DIR / "models"
LABELS = DATA_DIR / "labels"
REPORTS = REPO / "reports"

# Small, reproducible files that ARE kept in the repo
CAMERA_SPLIT = REPO / "configs" / "camera_split.csv"
LABEL_SAMPLE = REPO / "labeling" / "sample.csv"

OFFLINE_MAX_BYTES = 5_000  # blank "camera offline" placeholders are ~1.4 KB
SEED = 42
CLASSES = ["light", "moderate", "heavy", "unusable"]
