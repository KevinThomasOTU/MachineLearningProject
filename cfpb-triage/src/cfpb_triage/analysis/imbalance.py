"""Class imbalance stats: ratio, entropy, per-class counts (Subangan).

Expected Macro F1 of a stratified-random predictor
--------------------------------------------------
Let p_k be the true class proportions in the evaluation set and q_k the
proportions the predictor samples from (the training proportions). The
predictor draws each label independently of the input, so for class k:

    precision_k = P(true = k | pred = k) = p_k
    recall_k    = P(pred = k | true = k) = q_k
    F1_k        = 2 p_k q_k / (p_k + q_k)

    E[Macro F1] ~= (1/K) * sum_k 2 p_k q_k / (p_k + q_k)

(ratio-of-expectations approximation, exact as n -> infinity). With stratified
splits q_k = p_k, so F1_k = p_k and

    E[Macro F1] ~= (1/K) * sum_k p_k = 1/K.

So the chance-level Macro F1 is simply 1/K regardless of imbalance, while
chance-level accuracy is sum_k p_k^2 (dominated by the majority class).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def class_table(labels) -> pd.DataFrame:
    """Per-class counts and proportions, sorted by count (desc)."""
    counts = pd.Series(labels).value_counts()
    return pd.DataFrame({"product": counts.index, "count": counts.to_numpy(),
                         "proportion": (counts / counts.sum()).to_numpy()})


def expected_macro_f1_stratified(p_true, q_pred=None) -> float:
    """E[Macro F1] of a stratified-random predictor (see module docstring)."""
    p = np.asarray(p_true, dtype=float)
    q = p if q_pred is None else np.asarray(q_pred, dtype=float)
    f1 = np.where(p + q > 0, 2 * p * q / np.where(p + q > 0, p + q, 1), 0.0)
    return float(f1.mean())


def imbalance_report(labels) -> tuple[dict, pd.DataFrame]:
    """Summary dict + per-class DataFrame (product, count, proportion).

    Dict keys: n, n_classes, majority_class, majority_share, minority_class,
    minority_share, imbalance_ratio (max/min count), entropy (bits),
    normalized_entropy (entropy / log2 K; 1 = perfectly balanced),
    expected_macro_f1_stratified, expected_accuracy_stratified.
    """
    tab = class_table(labels)
    p = tab["proportion"].to_numpy()
    k = len(tab)
    entropy = float(-(p * np.log2(p)).sum())
    summary = {
        "n": int(tab["count"].sum()),
        "n_classes": k,
        "majority_class": tab["product"].iloc[0],
        "majority_share": float(p[0]),
        "minority_class": tab["product"].iloc[-1],
        "minority_share": float(p[-1]),
        "imbalance_ratio": float(tab["count"].iloc[0] / tab["count"].iloc[-1]),
        "entropy": entropy,
        "normalized_entropy": entropy / np.log2(k) if k > 1 else 0.0,
        "expected_macro_f1_stratified": expected_macro_f1_stratified(p),
        "expected_accuracy_stratified": float((p ** 2).sum()),
    }
    return summary, tab
