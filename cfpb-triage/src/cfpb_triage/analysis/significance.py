"""Paired bootstrap / McNemar between models """
from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def _codes(y_true, *preds):
    """Map string labels to 0..K-1 using the union of all labels."""
    labels = np.unique(np.concatenate([np.asarray(y_true), *map(np.asarray, preds)]))
    idx = {c: i for i, c in enumerate(labels)}
    enc = lambda y: np.fromiter((idx[v] for v in y), dtype=np.int64, count=len(y))
    return len(labels), enc(y_true), *(enc(p) for p in preds)


def _macro_f1(t: np.ndarray, p: np.ndarray, k: int) -> float:
    """Fast Macro F1 on integer codes; matches sklearn (zero_division=0).

    F1_c = 2 TP / (n_true_c + n_pred_c), averaged over classes that appear in
    y_true or y_pred (same label set sklearn uses).
    """
    cm = np.bincount(t * k + p, minlength=k * k).reshape(k, k)
    tp, n_true, n_pred = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    denom = n_true + n_pred
    present = denom > 0
    return float((2 * tp[present] / denom[present]).mean())


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 42,
                 level: float = 0.95) -> dict:
    """Percentile bootstrap CI for Macro F1 (resample val rows with replacement)."""
    k, t, p = _codes(y_true, y_pred)
    rng = np.random.default_rng(seed)
    n = len(t)
    stats = np.array([_macro_f1(t[i], p[i], k)
                      for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    lo, hi = np.quantile(stats, [(1 - level) / 2, (1 + level) / 2])
    return {"macro_f1": _macro_f1(t, p, k), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": n_boot}


def paired_bootstrap(y_true, pred_a, pred_b, n_boot: int = 1000, seed: int = 42,
                     level: float = 0.95) -> dict:
    """Paired bootstrap of Macro F1(A) - Macro F1(B) on the SAME resampled rows.

    p_value is the two-sided share of bootstrap differences on the other side
    of 0 (with a +1 correction so it is never exactly 0).
    """
    k, t, a, b = _codes(y_true, pred_a, pred_b)
    rng = np.random.default_rng(seed)
    n = len(t)
    diffs = np.empty(n_boot)
    for j in range(n_boot):
        i = rng.integers(0, n, n)
        diffs[j] = _macro_f1(t[i], a[i], k) - _macro_f1(t[i], b[i], k)
    lo, hi = np.quantile(diffs, [(1 - level) / 2, (1 + level) / 2])
    tail = min((diffs <= 0).sum(), (diffs >= 0).sum())
    return {"diff": _macro_f1(t, a, k) - _macro_f1(t, b, k), "ci_low": float(lo),
            "ci_high": float(hi), "p_value": min(1.0, 2 * (tail + 1) / (n_boot + 1)),
            "n_boot": n_boot}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar test (chi-square, continuity-corrected) on per-row correctness.

    b = A right & B wrong, c = A wrong & B right.
    """
    y, a, b_ = map(np.asarray, (y_true, pred_a, pred_b))
    ra, rb = a == y, b_ == y
    b, c = int((ra & ~rb).sum()), int((~ra & rb).sum())
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "chi2": stat,
            "p_value": float(chi2.sf(stat, df=1))}
