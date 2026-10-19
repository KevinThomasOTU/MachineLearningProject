"""Macro F1 mean/std over seeds or folds (Sayon)."""
from __future__ import annotations

import pandas as pd

METRICS = ("macro_f1", "weighted_f1", "accuracy")


def summarize(runs: pd.DataFrame, by: str = "model", metrics=METRICS) -> pd.DataFrame:
    """One row per model: mean and sample std (ddof=1) of each metric over runs.

    A deterministic model (NB) has std 0; a single run gets std 0 instead of NaN.
    Row order follows the first appearance of each model in `runs`.
    """
    g = runs.groupby(by, sort=False)
    out = g[list(metrics)].agg(["mean", "std"]).fillna(0.0)
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    out["n_runs"] = g.size()
    return out.reset_index()


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    """'0.512 ± 0.004' for the memo table."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"