"""Stratified, dedupe-aware train/val/test split with fixed seed (Alex).

Method: StratifiedGroupKFold with N_FOLDS=20 folds (each fold ~5% of rows,
stratified on product, whole groups only). Folds are then pooled:
round(test*20) folds -> test, round(val*20) folds -> val, the rest -> train.
With 70/15/15 that is 14/3/3 folds. Because folds never split a group_id, no
near-duplicate group can appear in two splits.

Small classes: a class needs >= 20 groups for every fold to get one of them.
Classes with fewer groups still get a best-effort split (a warning is printed);
with min_class_count=200 this should not happen in practice.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from cfpb_triage.data.load import resolve

N_FOLDS = 20
SPLIT_COLUMNS = ["complaint_id", "product", "group_id"]


def make_splits(df_clean: pd.DataFrame, cfg: dict
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split cleaned data into (train, val, test), stratified and group-aware."""
    s = cfg["split"]
    n_test, n_val = round(s["test"] * N_FOLDS), round(s["val"] * N_FOLDS)
    if n_test < 1 or n_val < 1 or n_test + n_val >= N_FOLDS:
        raise ValueError(f"split ratios {s} must be multiples of {1 / N_FOLDS:.2f}")

    df = df_clean.reset_index(drop=True)
    groups_per_class = df.groupby("product")["group_id"].nunique()
    small = groups_per_class[groups_per_class < N_FOLDS]
    if len(small):
        print(f"[split] WARNING: classes with < {N_FOLDS} groups get best-effort "
              f"placement and may be missing from val/test: {small.to_dict()}")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg["seed"])
    fold = np.empty(len(df), dtype=int)
    with warnings.catch_warnings():  # the small-class warning is reported above
        warnings.simplefilter("ignore", UserWarning)
        for k, (_, idx) in enumerate(sgkf.split(df, df["product"], df["group_id"])):
            fold[idx] = k

    test = df[fold < n_test]
    val = df[(fold >= n_test) & (fold < n_test + n_val)]
    train = df[fold >= n_test + n_val]
    _print_report(df, train, val, test)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def proportion_table(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame,
                     test: pd.DataFrame) -> pd.DataFrame:
    """Per-class proportions in full data and each split (plus max abs deviation)."""
    parts = {"full": df, "train": train, "val": val, "test": test}
    tab = pd.DataFrame({k: v["product"].value_counts(normalize=True) for k, v in parts.items()})
    tab = tab.fillna(0.0).sort_values("full", ascending=False)
    tab["max_abs_dev"] = tab[["train", "val", "test"]].sub(tab["full"], axis=0).abs().max(axis=1)
    return tab


def _print_report(df, train, val, test) -> None:
    n = len(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"[split] {name:<5} rows={len(part):>8,} ({len(part) / n:6.2%})  "
              f"groups={part['group_id'].nunique():>8,}")
    with pd.option_context("display.width", 160, "display.max_colwidth", 45,
                           "display.max_columns", None,
                           "display.float_format", "{:.4f}".format):
        print(proportion_table(df, train, val, test))


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                out_dir: str | Path = "data/processed/splits") -> None:
    """Write {train,val,test}.csv with complaint_id, product, group_id."""
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[SPLIT_COLUMNS].to_csv(out / f"{name}.csv", index=False)
    print(f"[split] saved splits -> {out}")


def load_splits(split_dir: str | Path = "data/processed/splits",
                cleaned_path: str | Path = "data/interim/cleaned.csv"
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load saved splits joined back to the cleaned data (all cleaned columns)."""
    clean = read_cleaned(cleaned_path)
    out = []
    for name in ("train", "val", "test"):
        ids = pd.read_csv(resolve(split_dir) / f"{name}.csv", dtype={"complaint_id": str})
        out.append(clean[clean["complaint_id"].isin(ids["complaint_id"])].reset_index(drop=True))
    return tuple(out)


def read_cleaned(path: str | Path = "data/interim/cleaned.csv") -> pd.DataFrame:
    """Read data/interim/cleaned.csv with the dtypes from the shared interface."""
    return pd.read_csv(resolve(path), dtype={"complaint_id": str, "zip": str, "group_id": "int64"})
