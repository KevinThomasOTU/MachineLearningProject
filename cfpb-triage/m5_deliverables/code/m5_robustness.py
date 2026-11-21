"""M5 sensitivity / robustness analysis (Alex). Validation split only.

Four questions about WHY the models behave as they do:
  Q1 Training-size sensitivity: does more data help, and for which model?   (learning_curve_all)
  Q2 Near-copy sensitivity: how much of the score comes from val complaints that have a
     near-copy in train (cosine >= 0.8, same probe as M3)?                   (robust_nearcopy)
  Q3 Template weighting: do results change if a complaint counts group_size times,
     i.e. as it would in the raw, un-deduplicated data?                      (robust_weighted)
  Q4 Input truncation: how fast does each model degrade when only the first N tokens of
     the narrative are available (as in an early-triage setting)?            (robust_truncation)

    python m5_deliverables/code/m5_robustness.py [--parts curve nearcopy weighted truncation]

Needs best_configs.json + val_predictions_final.csv from m5_tune.py.
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cfpb_triage.analysis.distances import max_cosine_to_train
from cfpb_triage.data.load import load_config
from cfpb_triage.utils.seed import set_seed
from m5_common import (FIGS, LABELS, TABLES, TUNED, ensure_dirs, features, fit_model,
                       full_metrics, load_data, read_best, read_val_predictions)
from m5_runlog import logged_run

NEAR_COPY = 0.8
COLORS = {"nb": "#1f77b4", "lr": "#d62728", "knn": "#2ca02c"}


def stratified_subsample(y: pd.Series, frac: float, rng) -> np.ndarray:
    if frac >= 1.0:
        return np.arange(len(y))
    return np.sort(np.concatenate([
        rng.choice(g.index.to_numpy(), max(2, int(len(g) * frac)), replace=False)
        for _, g in y.groupby(y)]))


def learning_curve(train, val, best, labels, seed):
    rng, rows = np.random.RandomState(seed), []
    for frac in (0.1, 0.25, 0.5, 1.0):
        idx = stratified_subsample(train["product"], frac, rng)
        sub = train.iloc[idx]
        for m in TUNED:
            b = best[m]
            _, Xtr, Xva = features(sub, val, b["tfidf"])     # vocabulary re-fit on the subsample
            pred = fit_model(m, b["params"], Xtr, sub["product"], seed).predict(Xva)
            rows.append({"model": m, "fraction": frac, "n_train": len(idx),
                         **full_metrics(val["product"], pred, labels)})
            print(f"  curve {m} {frac:.2f} macroF1={rows[-1]['macro_f1']:.4f}", flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(TABLES / "learning_curve_all.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m, g in cv.groupby("model"):
        ax.plot(g["n_train"], g["macro_f1"], "-o", color=COLORS[m], label=LABELS[m])
    ax.set(xscale="log", xlabel="Training rows (stratified subsample)",
           ylabel="Validation Macro F1", title="Training-size sensitivity")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "m5_learning_curve.png", dpi=150)
    plt.close(fig)


def nearcopy_mask(train, val) -> pd.Series:
    """Same probe as M3 (scripts/07_discrepancy.py): TF-IDF fit on train, cosine >= 0.8."""
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000)
    S_tr = vec.fit_transform(train["norm_text"])
    mask = max_cosine_to_train(vec.transform(val["norm_text"]), S_tr) >= NEAR_COPY
    pd.DataFrame({"complaint_id": val["complaint_id"], "near_copy": mask}).to_csv(
        TABLES / "val_nearcopy_mask.csv", index=False)
    return pd.Series(mask, index=val.index)


def subset_tables(val, preds, labels, mask):
    rows = []
    for name, sel in [("all val", np.ones(len(val), bool)), ("has near-copy in train", mask.values),
                      ("no near-copy in train", ~mask.values)]:
        if sel.sum() == 0:      # e.g. synthetic data without near-copies
            continue
        for m in TUNED:
            rows.append({"subset": name, "n": int(sel.sum()), "model": m,
                         **full_metrics(preds.loc[sel, "product"], preds.loc[sel, m], labels)})
    pd.DataFrame(rows).to_csv(TABLES / "robust_nearcopy.csv", index=False)
    w = preds["group_size"].to_numpy()
    rows = [{"weighting": wt, "model": m,
             **full_metrics(preds["product"], preds[m], labels, sample_weight=sw)}
            for wt, sw in (("unweighted (one row per complaint group)", None),
                           ("weighted by group_size (raw-data view)", w)) for m in TUNED]
    pd.DataFrame(rows).to_csv(TABLES / "robust_weighted.csv", index=False)


def truncation(train, val, best, labels, seed):
    rows = []
    for m in TUNED:
        b = best[m]
        vec, Xtr = features(train, [], b["tfidf"])[:2]
        clf = fit_model(m, b["params"], Xtr, train["product"], seed)
        for n_tok in (10, 25, 50, 100, None):
            text = val["norm_text"] if n_tok is None else val["norm_text"].map(
                lambda s: " ".join(s.split()[:n_tok]))
            pred = clf.predict(vec.transform(text))
            rows.append({"model": m, "first_n_tokens": n_tok or "full",
                         **full_metrics(val["product"], pred, labels)})
            print(f"  trunc {m} {n_tok or 'full'} macroF1={rows[-1]['macro_f1']:.4f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "robust_truncation.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m, g in t.groupby("model"):
        x = [int(v) if v != "full" else 400 for v in g["first_n_tokens"]]
        ax.plot(x, g["macro_f1"], "-o", color=COLORS[m], label=LABELS[m])
    ax.set(xscale="log", xlabel="Tokens of narrative available (400 = full text)",
           ylabel="Validation Macro F1", title="Robustness to truncated narratives")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "m5_truncation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["curve", "nearcopy", "weighted", "truncation"])
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    ensure_dirs()
    train, val = load_data()
    labels = sorted(train["product"].unique())
    best = read_best(cfg)
    with logged_run("m5_robustness", vars(args), seed) as run:
        if "curve" in args.parts:
            learning_curve(train, val, best, labels, seed)
        if {"nearcopy", "weighted"} & set(args.parts):
            preds = read_val_predictions()
            assert (preds["complaint_id"].values == val["complaint_id"].values).all()
            mask = nearcopy_mask(train, val)
            subset_tables(val, preds, labels, mask)
        if "truncation" in args.parts:
            truncation(train, val, best, labels, seed)
        run["outputs"] = [str(TABLES / f) for f in
                          ("learning_curve_all.csv", "robust_nearcopy.csv", "robust_weighted.csv",
                           "robust_truncation.csv")]


if __name__ == "__main__":
    main()
