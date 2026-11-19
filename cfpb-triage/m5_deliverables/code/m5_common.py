"""Shared helpers for every M5 script (Alex). Import with:  from m5_common import ...

Rules (same as M2/M3/M4): TF-IDF is fit on TRAIN only, all tuning is scored on VAL with the
pre-registered primary metric (Macro F1), and the TEST split is touched only by
m5_final_test.py, once, after best_configs.json is frozen.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from cfpb_triage.data.load import load_config
from cfpb_triage.data.split import load_splits
from cfpb_triage.evaluation.metrics import evaluate, per_class_report
from cfpb_triage.features.tfidf import fit_transform
from cfpb_triage.models.knn import KNNCosine
from cfpb_triage.models.logreg import fit_logreg

M5 = Path(__file__).resolve().parents[1]
RESULTS = M5 / "results"
TABLES, FIGS, LOGS = RESULTS / "tables", RESULTS / "figures", RESULTS / "logs"
BEST_CONFIGS = TABLES / "best_configs.json"
VAL_PREDS = TABLES / "val_predictions_final.csv"
TEST_PREDS = TABLES / "test_predictions_final.csv"

PRIMARY = "macro_f1"            # fixed in M2; never changed after seeing results
TUNED = ["nb", "lr", "knn"]     # the three substantive models
NAIVE = ["majority", "random"]
LABELS = {"majority": "Majority class", "random": "Stratified Random",
          "nb": "Multinomial NB", "lr": "L2 Logistic Regression", "knn": "k-NN (cosine)"}

# Same tuning effort for every model: every TF-IDF setting x a small hyper-parameter grid.
SEARCH_SPACE = {
    "nb": {"alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], "fit_prior": [True, False]},
    "lr": {"C": [0.3, 1, 3, 10, 30], "class_weight": [None, "balanced"]},
    "knn": {"k": [5, 11, 15, 25, 31, 51], "weights": ["uniform", "distance"]},
}
QUICK_SPACE = {"nb": {"alpha": [0.03, 0.3], "fit_prior": [True]},
               "lr": {"C": [1, 10], "class_weight": [None]},
               "knn": {"k": [5, 15], "weights": ["distance"]}}


def ensure_dirs() -> None:
    for d in (TABLES, FIGS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def tfidf_grid(cfg: dict, quick: bool = False) -> list[dict]:
    """The 12 TF-IDF settings of config.yaml (2 stop-word x 2 n-gram x 3 vocabulary sizes)."""
    t = cfg["tfidf"]
    grid = [dict(stop_words=sw, ngram_range=tuple(ng), max_features=mf, sublinear_tf=True)
            for sw, ng, mf in itertools.product(t["stop_words"], t["ngram_range"],
                                                t["max_features"])]
    return grid[:: max(1, len(grid) // 2)] if quick else grid   # quick = 2 settings


def tfidf_id(p: dict) -> str:
    return f"sw={p['stop_words'] or 'none'}|ng={p['ngram_range'][0]}-{p['ngram_range'][1]}" \
           f"|mf={p['max_features']}"


def param_grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*space.values())]


def load_data():
    """(train, val). Test is deliberately not returned; see load_data_with_test()."""
    train, val, _ = load_splits()
    return train, val


def load_data_with_test():
    """ONLY m5_final_test.py may call this."""
    return load_splits()


def features(train, val_or_more, tfidf: dict):
    """TF-IDF fit on train; returns (vectorizer, X_train, X_other...)."""
    others = val_or_more if isinstance(val_or_more, (list, tuple)) else [val_or_more]
    return fit_transform(train["norm_text"], *[d["norm_text"] for d in others], **tfidf)


def fit_model(name: str, params: dict, X, y, seed: int = 42):
    """Fit one tuned model on (X, y) with the given hyper-parameters."""
    if name == "nb":
        return MultinomialNB(alpha=params["alpha"],
                             fit_prior=params.get("fit_prior", True)).fit(X, y)
    if name == "lr":
        return fit_logreg(X, y, C=params["C"], class_weight=params.get("class_weight"), seed=seed)
    if name == "knn":
        return KNNCosine(k=params["k"], weights=params["weights"]).fit(X, y)
    raise ValueError(name)


def macro_pr(y_true, y_pred, labels) -> dict:
    rep = per_class_report(y_true, y_pred, labels)
    return {"macro_precision": rep["precision"].mean(), "macro_recall": rep["recall"].mean()}


def full_metrics(y_true, y_pred, labels, sample_weight=None) -> dict:
    return {**evaluate(y_true, y_pred, sample_weight=sample_weight),
            **macro_pr(y_true, y_pred, labels)}


def default_best(cfg: dict) -> dict:
    """Fallback until the tuning run exists: the M4 winners on the single M3 TF-IDF setting."""
    t = dict(cfg["baseline"]["tfidf"])
    t["ngram_range"] = tuple(t["ngram_range"])
    return {"nb": {"tfidf": t, "params": {"alpha": 0.03, "fit_prior": True}},
            "lr": {"tfidf": t, "params": {"C": 10, "class_weight": None}},
            "knn": {"tfidf": t, "params": {"k": 15, "weights": "distance"}}}


def read_best(cfg: dict) -> dict:
    """best_configs.json written by m5_tune.py (falls back to the M4 defaults with a warning)."""
    if BEST_CONFIGS.exists():
        best = json.loads(BEST_CONFIGS.read_text())
        for m in best.values():
            m["tfidf"]["ngram_range"] = tuple(m["tfidf"]["ngram_range"])
        return best
    print("[m5] WARNING: best_configs.json not found, using the M4 defaults")
    return default_best(cfg)


def read_val_predictions() -> pd.DataFrame:
    if not VAL_PREDS.exists():
        raise FileNotFoundError(f"{VAL_PREDS} missing: run m5_tune.py first (Alex).")
    return pd.read_csv(VAL_PREDS, dtype={"complaint_id": str})


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Tiny markdown table writer (no tabulate dependency)."""
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
