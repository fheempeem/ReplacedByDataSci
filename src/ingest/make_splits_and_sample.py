"""Create (1) the camera-level train/val/test split and (2) the stratified labeling sample.

Split: 100 cameras -> 70 train / 15 val / 15 test, fixed seed. Splitting by camera avoids
leakage between near-identical frames of the same camera.

Sample: 15 non-offline images per camera (1,500 total), 3 from each of 5 time-of-day bands,
so rush hours, midday and dusk are all represented. 10% of the sample (150 images) is given to
two members so labeler agreement can be measured. Members are numbered 1-6.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import BRONZE, CAMERA_SPLIT, LABEL_SAMPLE, SEED  # noqa: E402

N_MEMBERS = 6
PER_BAND = 3
BANDS = {"06-08": (6, 8), "09-11": (9, 11), "12-14": (12, 14), "15-17": (15, 17), "18-21": (18, 21)}
OVERLAP_FRAC = 0.10


def make_camera_split(cameras: list[str], rng: np.random.Generator) -> pd.DataFrame:
    cams = sorted(cameras, key=lambda c: int(c))
    order = rng.permutation(len(cams))
    split = np.empty(len(cams), dtype=object)
    n_train, n_val = round(0.70 * len(cams)), round(0.15 * len(cams))
    split[order[:n_train]] = "train"
    split[order[n_train:n_train + n_val]] = "val"
    split[order[n_train + n_val:]] = "test"
    return pd.DataFrame({"camera_id": cams, "split": split})


def make_sample(df: pd.DataFrame, split: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    usable = df[~df["is_offline_stub"] & df["parsed_ok"]].copy()
    parts = []
    for cam, g in usable.groupby("camera_id"):
        for _, (lo, hi) in BANDS.items():
            band = g[(g["hour"] >= lo) & (g["hour"] <= hi)]
            if len(band) == 0:
                continue
            take = min(PER_BAND, len(band))
            parts.append(band.iloc[rng.choice(len(band), take, replace=False)])
    s = pd.concat(parts).merge(split, on="camera_id")
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
    s["sample_id"] = [f"s{i:04d}" for i in range(len(s))]
    s["rel_path"] = s["camera_dir"] + "/" + s["filename"]

    # Round-robin assignment: member 1..6
    s["member_a"] = (np.arange(len(s)) % N_MEMBERS) + 1
    s["member_b"] = 0  # 0 = nobody (not an overlap image)
    overlap_idx = rng.choice(len(s), int(OVERLAP_FRAC * len(s)), replace=False)
    s.loc[overlap_idx, "member_b"] = (s.loc[overlap_idx, "member_a"] % N_MEMBERS) + 1
    s["is_overlap"] = s["member_b"] > 0
    return s[["sample_id", "camera_id", "rel_path", "timestamp", "hour", "split",
              "member_a", "member_b", "is_overlap"]]


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(BRONZE / "images.parquet")
    split = make_camera_split(df["camera_id"].unique().tolist(), rng)
    CAMERA_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(CAMERA_SPLIT, index=False)
    print(split["split"].value_counts().to_dict(), "->", CAMERA_SPLIT)

    sample = make_sample(df, split, rng)
    LABEL_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(LABEL_SAMPLE, index=False)
    print(len(sample), "images; overlap:", int(sample["is_overlap"].sum()), "->", LABEL_SAMPLE)
    print("per split:", sample["split"].value_counts().to_dict())
    per_member = pd.concat([sample["member_a"], sample.loc[sample["is_overlap"], "member_b"]]).value_counts().sort_index()
    print("labels per member:", per_member.to_dict())


if __name__ == "__main__":
    main()
