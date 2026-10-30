"""M4 initial validation results: L2 Logistic Regression and k-NN next to the M3 NB baseline.

Run from the cfpb-triage folder (package installed with `pip install -e .`):
    python m4_deliverables/code/m4_initial_models.py [--parts nb lr knn curve]

Rules kept from M2/M3: TF-IDF is fit on TRAIN only, everything is scored on VAL,
the test split is never scored, seed 42, one fixed TF-IDF setting (1-2-grams,
10,000 features, sublinear tf; the full 12-setting grid is M5).
Output goes to m4_deliverables/results/ so no existing result file is changed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from cfpb_triage.data.load import load_config
from cfpb_triage.data.split import load_splits
from cfpb_triage.evaluation.metrics import Timer, confusion, evaluate, per_class_report
from cfpb_triage.evaluation.plots import plot_confusion
from cfpb_triage.features.tfidf import fit_transform
from cfpb_triage.utils.seed import set_seed

OUT = Path(__file__).resolve().parents[1] / "results"
TABLES, FIGS = OUT / "tables", OUT / "figures"


def macro_pr(y_true, y_pred, labels) -> dict:
    rep = per_class_report(y_true, y_pred, labels)
    return {"macro_precision": rep["precision"].mean(), "macro_recall": rep["recall"].mean()}


def score(name, clf, Xtr, ytr, Xva, yva, labels, rows, preds, with_train=True):
    """Fit on train, score on val, append one result row."""
    with Timer() as tf:
        clf.fit(Xtr, ytr)
    with Timer() as tp:
        p_va = clf.predict(Xva)
    train_f1 = evaluate(ytr, clf.predict(Xtr))["macro_f1"] if with_train else np.nan
    row = {"model": name, **evaluate(yva, p_va), **macro_pr(yva, p_va, labels),
           "train_macro_f1": train_f1, "fit_s": round(tf.s, 2), "predict_s": round(tp.s, 2)}
    rows.append(row)
    preds[name] = p_va
    print(f"  {name:40s} macroF1={row['macro_f1']:.4f} acc={row['accuracy']:.4f} "
          f"fit={tf.s:.1f}s pred={tp.s:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["nb", "lr", "knn", "curve"])
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    train, val, _test = load_splits()  # test is loaded with the others but never used
    ytr, yva = train["product"], val["product"]
    labels = sorted(ytr.unique())
    vec, Xtr, Xva = fit_transform(train["norm_text"], val["norm_text"], **cfg["baseline"]["tfidf"])
    print(f"[m4] train={Xtr.shape} val={Xva.shape} classes={len(labels)}", flush=True)

    rows, preds = [], val[["complaint_id", "product"]].copy()
    best = {}

    print("[nb] M3 baseline and fit_prior=False")
    score("NB alpha=0.03 (M3 baseline)", MultinomialNB(alpha=0.03), Xtr, ytr, Xva, yva, labels, rows, preds)
    score("NB alpha=0.03, fit_prior=False", MultinomialNB(alpha=0.03, fit_prior=False),
          Xtr, ytr, Xva, yva, labels, rows, preds)

    if "lr" in args.parts:
        print("[lr] L2 logistic regression: C sweep x class_weight")
        sweep = []
        for cw in (None, "balanced"):
            for C in (0.1, 1, 10, 100):
                name = f"LR C={C} cw={cw or 'none'}"
                clf = LogisticRegression(C=C, class_weight=cw, max_iter=1000, random_state=seed)
                score(name, clf, Xtr, ytr, Xva, yva, labels, rows, preds)
                sweep.append({**rows[-1], "C": C, "class_weight": cw or "none"})
        sw = pd.DataFrame(sweep)
        sw.to_csv(TABLES / "lr_sweep.csv", index=False)
        top = sw.sort_values("macro_f1", ascending=False).iloc[0]
        best["lr"] = top["model"]
        best["lr_C"], best["lr_cw"] = float(top["C"]), top["class_weight"]
        print(f"[lr] best on val: {top['model']} ({top['macro_f1']:.4f})")

    if "knn" in args.parts:
        print("[knn] cosine vs euclidean, k sweep (brute force on sparse TF-IDF)")
        for metric in ("cosine", "euclidean"):
            for k in (1, 5, 15, 31):
                clf = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm="brute",
                                           weights="distance", n_jobs=-1)
                score(f"kNN {metric} k={k}", clf, Xtr, ytr, Xva, yva, labels, rows, preds,
                      with_train=False)
        knn = pd.DataFrame([r for r in rows if r["model"].startswith("kNN")])
        best["knn"] = knn.sort_values("macro_f1", ascending=False).iloc[0]["model"]
        # On L2-normalised TF-IDF, ||a-b||^2 = 2 - 2cos(a,b): both metrics rank neighbours identically.
        agree = {k: float((preds[f"kNN cosine k={k}"] == preds[f"kNN euclidean k={k}"]).mean())
                 for k in (1, 5, 15, 31)}
        json.dump(agree, open(TABLES / "knn_cos_vs_euc_agreement.json", "w"), indent=2)
        print("[knn] cosine vs euclidean prediction agreement:", agree)

    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "m4_validation_results.csv", index=False)
    preds.to_csv(TABLES / "m4_val_predictions.csv", index=False)

    # per-class results of the best model of each family, plus confusion matrices
    fam = {"NB": "NB alpha=0.03 (M3 baseline)", "LR": best.get("lr"), "kNN": best.get("knn")}
    pc = {}
    for lab, col in fam.items():
        if col and col in preds:
            rep = per_class_report(yva, preds[col], labels)
            pc["support"] = rep["support"]
            pc[f"{lab} recall"] = rep["recall"]
            pc[f"{lab} f1"] = rep["f1"]
            if lab != "NB":
                plot_confusion(confusion(yva, preds[col], labels),
                               FIGS / f"confusion_{lab.lower()}_val.png",
                               title=f"{col} (validation, row-normalised)")
    pd.DataFrame(pc).to_csv(TABLES / "per_class_best_models.csv")

    if "curve" in args.parts and "lr" in best:
        print("[curve] learning curve: NB vs best LR")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cw = None if best["lr_cw"] == "none" else best["lr_cw"]
        rng = np.random.RandomState(seed)
        curve = []
        for frac in (0.1, 0.25, 0.5, 1.0):
            if frac == 1.0:
                idx = np.arange(len(ytr))
            else:  # stratified subsample: the same fraction of every class
                idx = np.sort(np.concatenate([
                    rng.choice(g.index.to_numpy(), max(2, int(len(g) * frac)), replace=False)
                    for _, g in ytr.groupby(ytr)]))
            for nm, clf in (("NB alpha=0.03", MultinomialNB(alpha=0.03)),
                            (best["lr"], LogisticRegression(C=best["lr_C"], class_weight=cw,
                                                            max_iter=1000, random_state=seed))):
                clf.fit(Xtr[idx], ytr.iloc[idx])
                curve.append({"fraction": frac, "n_train": len(idx), "model": nm,
                              "train_macro_f1": evaluate(ytr.iloc[idx], clf.predict(Xtr[idx]))["macro_f1"],
                              "val_macro_f1": evaluate(yva, clf.predict(Xva))["macro_f1"]})
        cv = pd.DataFrame(curve)
        cv.to_csv(TABLES / "learning_curve.csv", index=False)
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for (nm, g), col in zip(cv.groupby("model", sort=False), ("#1f77b4", "#d62728")):
            ax.plot(g["n_train"], g["train_macro_f1"], "--", color=col, label=f"{nm} train")
            ax.plot(g["n_train"], g["val_macro_f1"], "-o", color=col, label=f"{nm} val")
        ax.set_xscale("log")
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Macro F1")
        ax.set_title("Learning curve (TF-IDF 1-2-grams, 10k features)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "learning_curve.png", dpi=150)

    json.dump({"seed": seed, "tfidf": cfg["baseline"]["tfidf"], "best": best,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, open(OUT / "m4_run.json", "w"), indent=2)
    print("[m4] done ->", OUT)


if __name__ == "__main__":
    main()
