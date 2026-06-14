"""
Temporal train/val/test split:
  train+val : 2022–2023 CVEs  (80/20 split within)
  test       : 2024 CVEs (held out entirely)

Saves splits as pickle files for downstream scripts.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from parse_data import load_all

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42


def make_splits(force: bool = False) -> dict[str, pd.DataFrame]:
    splits_path = DATA_DIR / "splits.pkl"
    if splits_path.exists() and not force:
        print(f"Loading cached splits from {splits_path.name}")
        with open(splits_path, "rb") as f:
            return pickle.load(f)

    df = load_all([2022, 2023, 2024])

    # Temporal split by CVE ID year
    df["year"] = df["cve_id"].str.extract(r"CVE-(\d{4})-").astype(int)
    trainval = df[df["year"] <= 2023].copy()
    test = df[df["year"] == 2024].copy()

    train, val = train_test_split(trainval, test_size=0.2, random_state=SEED, stratify=trainval["severity"])

    splits = {"train": train.reset_index(drop=True),
              "val":   val.reset_index(drop=True),
              "test":  test.reset_index(drop=True)}

    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)

    print(f"\nSplit sizes:")
    for name, s in splits.items():
        print(f"  {name:5s}: {len(s):,}  | severity: {dict(s['severity'].value_counts())}")

    return splits


if __name__ == "__main__":
    splits = make_splits(force=True)
    print(f"\nUnique CWEs — train: {splits['train']['cwe'].nunique()}, "
          f"test: {splits['test']['cwe'].nunique()}")
