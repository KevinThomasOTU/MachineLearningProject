"""M5 uncertainty from repeated group-aware cross-validation (Sayon). TEST IS NOT USED.

The M3-M4 splits are fixed, and NB / LR / k-NN are deterministic, so "multiple seeds" on the
same split would show no variability. The uncertainty that matters is "how much would the
Macro F1 change with a different draw of the data?". We therefore pool TRAIN+VAL (never test)
and run R repeats of 5-fold StratifiedGroupKFold (different shuffle seed per repeat, whole
near-duplicate groups stay in one fold). Hyper-parameters and TF-IDF settings are the FROZEN
best_configs.json values (no re-tuning inside folds, stated as a limitation: the CV scores are
slightly optimistic because the configs were chosen on val, which is part of the pool).
TF-IDF is re-fit on each fold's training part.

    python m5_deliverables/code/m5_repeated_cv.py [--repeats 2] [--folds 5] [--models nb lr knn]

Writes tables/cv_runs.csv (one row per model x repeat x fold) and tables/cv_summary.csv
(mean, std over all folds, std of the per-repeat means, min/max).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from cfpb_triage.data.load import load_config
from cfpb_triage.evaluation.metrics import Timer
from cfpb_triage.models.baseline import majority_class, stratified_random
from cfpb_triage.utils.seed import set_seed
from m5_common import (LABELS, NAIVE, TABLES, TUNED, ensure_dirs, features, fit_model,
                       full_metrics, load_data, read_best)
from m5_runlog import logged_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=TUNED)
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    ensure_dirs()
    train, val = load_data()
    pool = pd.concat([train, val], ignore_index=True)   # TEST deliberately excluded
    labels = sorted(pool["product"].unique())
    best = read_best(cfg)
    runs = []
    with logged_run("m5_repeated_cv", vars(args), seed) as run:
        for rep in range(args.repeats):
            cv = StratifiedGroupKFold(args.folds, shuffle=True, random_state=seed + rep)
            for fold, (tr, te) in enumerate(cv.split(pool, pool["product"], pool["group_id"])):
                a, b = pool.iloc[tr], pool.iloc[te]
                y_tr, y_te = a["product"], b["product"]
                assert not set(a["group_id"]) & set(b["group_id"])    # leakage check
                rec = lambda m, pred, fit_s: runs.append(
                    {"model": m, "repeat": rep, "fold": fold, "n_train": len(a), "n_eval": len(b),
                     "fit_s": round(fit_s, 2), **full_metrics(y_te, pred, labels)})
                rec("majority", majority_class(y_tr).predict([[0]] * len(b)), 0)
                rec("random", stratified_random(y_tr, seed + rep).predict([[0]] * len(b)), 0)
                for m in args.models:
                    bc = best[m]
                    _, Xa, Xb = features(a, b, bc["tfidf"])
                    with Timer() as t:
                        clf = fit_model(m, bc["params"], Xa, y_tr, seed)
                    rec(m, clf.predict(Xb), t.s)
                    print(f"[cv] rep={rep} fold={fold} {m:5s} macroF1={runs[-1]['macro_f1']:.4f}",
                          flush=True)
        df = pd.DataFrame(runs)
        df.to_csv(TABLES / "cv_runs.csv", index=False)
        per_rep = df.groupby(["model", "repeat"])["macro_f1"].mean().groupby("model").std()
        g = df.groupby("model", sort=False)["macro_f1"]
        summ = pd.DataFrame({"mean": g.mean(), "std_over_folds": g.std(ddof=1),
                             "std_of_repeat_means": per_rep.reindex(g.mean().index),
                             "min": g.min(), "max": g.max(), "n_runs": g.size(),
                             "fit_s_mean": df.groupby("model")["fit_s"].mean()}).reset_index()
        order = {m: i for i, m in enumerate(NAIVE + TUNED)}
        summ = summ.sort_values("model", key=lambda s: s.map(order))
        summ.insert(1, "label", summ["model"].map(LABELS))
        summ.to_csv(TABLES / "cv_summary.csv", index=False)
        print(summ.round(4).to_string(index=False))
        run["outputs"] = [str(TABLES / "cv_summary.csv")]


if __name__ == "__main__":
    main()
