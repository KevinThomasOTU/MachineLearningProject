"""M4 follow-up: why Euclidean k-NN collapses on TF-IDF, and the fix.

On L2-normalised TF-IDF, ||a-b||^2 = 2 - 2cos(a,b), so cosine and Euclidean should rank
neighbours identically. That breaks when a vector is all zeros (no token of the narrative is in the
10,000-feature vocabulary): it sits at distance exactly 1 from every unit vector, so for any query
whose best real neighbour has cosine < 0.5 the empty rows become the "nearest" neighbours.
This script counts empty rows, then re-runs k-NN with them removed from the reference set.

    python m4_deliverables/code/m4_knn_empty_rows.py
Writes m4_deliverables/results/tables/knn_empty_rows.csv (val only; test untouched).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from cfpb_triage.data.load import load_config
from cfpb_triage.data.split import load_splits
from cfpb_triage.evaluation.metrics import evaluate
from cfpb_triage.features.tfidf import fit_transform
from cfpb_triage.utils.seed import set_seed

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"


def row_norms(X) -> np.ndarray:
    return np.sqrt(X.multiply(X).sum(axis=1)).A1


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    train, val, _ = load_splits()
    ytr, yva = train["product"], val["product"]
    _, Xtr, Xva = fit_transform(train["norm_text"], val["norm_text"], **cfg["baseline"]["tfidf"])

    empty_tr, empty_va = row_norms(Xtr) == 0, row_norms(Xva) == 0
    print(f"empty train rows: {empty_tr.sum()} ({empty_tr.mean():.2%}); "
          f"empty val rows: {empty_va.sum()} ({empty_va.mean():.2%})")
    keep = ~empty_tr
    rows = [{"model": "(count) empty train rows", "value": int(empty_tr.sum())},
            {"model": "(count) empty val rows", "value": int(empty_va.sum())}]
    preds = {}
    for metric in ("cosine", "euclidean"):
        for k in (15,):
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm="brute",
                                       weights="distance", n_jobs=-1)
            clf.fit(Xtr[keep], ytr[keep])
            p = clf.predict(Xva)
            preds[metric] = p
            r = {"model": f"kNN {metric} k={k} (empty train rows removed)", **evaluate(yva, p)}
            rows.append(r)
            print(f"  {r['model']:55s} macroF1={r['macro_f1']:.4f} acc={r['accuracy']:.4f}", flush=True)
    agree = float((preds["cosine"] == preds["euclidean"]).mean())
    rows.append({"model": "cosine vs euclidean prediction agreement (k=15, fixed)", "value": agree})
    print("agreement:", agree)
    pd.DataFrame(rows).to_csv(TABLES / "knn_empty_rows.csv", index=False)


if __name__ == "__main__":
    main()
