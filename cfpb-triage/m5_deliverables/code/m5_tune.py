"""M5 tuning + fair comparison (Alex).

Equal-effort grid search on the VALIDATION split only:
  every model x every one of the 12 TF-IDF settings x its own hyper-parameter grid
  (NB: alpha x fit_prior = 14, LR: C x class_weight = 10, k-NN: k x weights = 12).
Selection rule (fixed before running): highest validation Macro F1; ties (4 dp) go to the
faster fit. The test split is never loaded here.

    python m5_deliverables/code/m5_tune.py [--n-jobs 4] [--resume] [--quick]

Writes (m5_deliverables/results/tables/):
    tuning_grid.csv            one row per (TF-IDF setting, model, hyper-parameters)
    search_space.json          the exact grids used (report this in the Results section)
    best_configs.json          FROZEN selection per model  <- everyone else reads this
    val_predictions_final.csv  val predictions of naive baselines + the 3 tuned models
    val_results_final.csv      val metrics of those models
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cfpb_triage.data.load import load_config
from cfpb_triage.evaluation.metrics import Timer
from cfpb_triage.models.baseline import majority_class, stratified_random
from cfpb_triage.models.knn import KNNCosine
from cfpb_triage.utils.seed import set_seed
from m5_common import (BEST_CONFIGS, LABELS, NAIVE, PRIMARY, QUICK_SPACE, SEARCH_SPACE, TABLES,
                       TUNED, ensure_dirs, features, fit_model, full_metrics, load_data,
                       param_grid, tfidf_grid, tfidf_id)
from m5_runlog import logged_run

GRID_CSV = TABLES / "tuning_grid.csv"


def _flat(tp: dict, model: str, params: dict, metrics: dict, extra: dict) -> dict:
    return {"tfidf_id": tfidf_id(tp), "stop_words": tp["stop_words"] or "none",
            "ngram": f"{tp['ngram_range'][0]}-{tp['ngram_range'][1]}",
            "max_features": tp["max_features"], "model": model,
            **{f"p_{k}": ("none" if v is None else v) for k, v in params.items()},
            **metrics, **extra}


def _fit_score(model, params, Xtr, ytr, Xva, yva, labels, seed):
    with Timer() as t:
        clf = fit_model(model, params, Xtr, ytr, seed)
    pred = clf.predict(Xva)
    train_f1 = full_metrics(ytr, clf.predict(Xtr), labels)["macro_f1"]
    return params, full_metrics(yva, pred, labels), {"train_macro_f1": train_f1,
                                                     "fit_s": round(t.s, 2)}


def run_setting(tp, train, val, space, labels, n_jobs, seed) -> list[dict]:
    """All models x all hyper-parameters for ONE TF-IDF setting (features fit on train)."""
    _, Xtr, Xva = features(train, val, tp)
    ytr, yva = train["product"], val["product"]
    rows = []
    for model in ("nb", "lr"):
        res = Parallel(n_jobs=n_jobs if model == "lr" else 1)(
            delayed(_fit_score)(model, p, Xtr, ytr, Xva, yva, labels, seed)
            for p in param_grid(space[model]))
        rows += [_flat(tp, model, p, m, e) for p, m, e in res]
    # k-NN: one neighbour search, every (k, weights) read off it
    with Timer() as t:
        knn = KNNCosine().fit(Xtr, ytr)
        preds = knn.predict_grid(Xva, space["knn"]["k"], space["knn"]["weights"])
    for (k, w), pred in preds.items():
        rows.append(_flat(tp, "knn", {"k": k, "weights": w}, full_metrics(yva, pred, labels),
                          {"train_macro_f1": np.nan, "fit_s": round(t.s, 2)}))
    return rows


def pick_best(grid: pd.DataFrame) -> dict:
    best = {}
    for model in TUNED:
        g = grid[grid["model"] == model].copy()
        g["_r"] = g[PRIMARY].round(4)
        top = g.sort_values(["_r", "fit_s"], ascending=[False, True]).iloc[0]
        tp = {"stop_words": None if top["stop_words"] == "none" else top["stop_words"],
              "ngram_range": [int(x) for x in top["ngram"].split("-")],
              "max_features": int(top["max_features"]), "sublinear_tf": True}
        params = {}
        for c in g.columns:
            if c.startswith("p_") and pd.notna(top[c]):
                v = top[c]
                params[c[2:]] = None if v == "none" else (
                    v.item() if hasattr(v, "item") else v)
        for k in ("k", "max_features"):   # csv round-trips ints as floats when NaNs are present
            if k in params:
                params[k] = int(params[k])
        if "fit_prior" in params:
            params["fit_prior"] = str(params["fit_prior"]) == "True"
        best[model] = {"tfidf": tp, "params": params,
                       "val_macro_f1": float(top[PRIMARY]), "n_configs_searched": int(len(g))}
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=2, help="parallel LR fits (RAM: ~1 GB each)")
    ap.add_argument("--resume", action="store_true", help="skip TF-IDF settings already in the csv")
    ap.add_argument("--quick", action="store_true", help="2 settings, tiny grids (smoke test)")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    ensure_dirs()
    space = QUICK_SPACE if args.quick else SEARCH_SPACE
    train, val = load_data()
    labels = sorted(train["product"].unique())

    with logged_run("m5_tune", vars(args), seed) as run:
        done, rows = set(), []
        if args.resume and GRID_CSV.exists():
            old = pd.read_csv(GRID_CSV)
            rows = old.to_dict("records")
            done = set(old["tfidf_id"])
        for i, tp in enumerate(tfidf_grid(cfg, args.quick), 1):
            if tfidf_id(tp) in done:
                continue
            t0 = time.perf_counter()
            rows += run_setting(tp, train, val, space, labels, args.n_jobs, seed)
            pd.DataFrame(rows).to_csv(GRID_CSV, index=False)          # checkpoint
            cur = pd.DataFrame(rows)
            print(f"[tune] {i}: {tfidf_id(tp)} done in {time.perf_counter() - t0:.0f}s | best so "
                  f"far " + ", ".join(f"{m}={cur[cur.model == m][PRIMARY].max():.4f}"
                                      for m in TUNED), flush=True)
        grid = pd.DataFrame(rows)
        json.dump({"tfidf": {k: v for k, v in cfg["tfidf"].items()}, **space},
                  open(TABLES / "search_space.json", "w"), indent=2, default=str)

        best = pick_best(grid)
        json.dump(best, open(BEST_CONFIGS, "w"), indent=2)
        print("[tune] FROZEN best configs ->", BEST_CONFIGS)

        # refit the winners once and save val predictions (naive baselines included)
        ytr, yva = train["product"], val["product"]
        preds = val[["complaint_id", "product", "group_size"]].copy()
        preds["majority"] = majority_class(ytr).predict([[0]] * len(val))
        preds["random"] = stratified_random(ytr, seed).predict([[0]] * len(val))
        for m in TUNED:
            b = best[m]
            _, Xtr, Xva = features(train, val, {**b["tfidf"], "ngram_range": tuple(
                b["tfidf"]["ngram_range"])})
            preds[m] = fit_model(m, b["params"], Xtr, ytr, seed).predict(Xva)
        preds.to_csv(TABLES / "val_predictions_final.csv", index=False)
        res = pd.DataFrame([{"model": m, "label": LABELS[m], **full_metrics(yva, preds[m], labels)}
                            for m in NAIVE + TUNED])
        res.to_csv(TABLES / "val_results_final.csv", index=False)
        print(res.round(4).to_string(index=False))
        run["outputs"] = [str(GRID_CSV), str(BEST_CONFIGS)]


if __name__ == "__main__":
    main()
