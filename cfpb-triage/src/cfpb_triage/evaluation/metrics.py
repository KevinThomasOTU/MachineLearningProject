"""Macro F1, per-class P/R, confusion matrix, timing (Sayon)."""
from __future__ import annotations

import time

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def evaluate(y_true, y_pred, sample_weight=None) -> dict:
    """Headline metrics; Macro F1 is the primary one.

    zero_division=0: a class that is never predicted gets F1 = 0 (no warning).
    sample_weight=group_size gives the template-weighted sensitivity check.
    """
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0,
                             sample_weight=sample_weight),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0,
                                sample_weight=sample_weight),
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
    }


def per_class_report(y_true, y_pred, labels) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, largest class first."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels),
                                                 zero_division=0)
    out = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s},
                       index=pd.Index(list(labels), name="product"))
    return out.sort_values("support", ascending=False)


def confusion(y_true, y_pred, labels) -> pd.DataFrame:
    """Raw-count confusion matrix with labelled rows (true) and columns (pred)."""
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=list(labels)),
                        index=pd.Index(list(labels), name="true"),
                        columns=pd.Index(list(labels), name="pred"))


class Timer:
    """Wall-clock timer:  with Timer() as t: ...   then t.s is the elapsed seconds."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self._t0
