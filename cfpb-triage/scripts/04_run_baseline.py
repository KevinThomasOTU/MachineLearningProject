"""Entry point: 04_run_baseline (Alex). M3 naive benchmarks + course-level baseline.

Everything is fit on TRAIN and scored on VAL; the test split stays sealed until M5.
  - Majority class, Stratified Random (naive benchmarks, ignore the text)
  - Multinomial NB on one fixed TF-IDF setting (cfg["baseline"]["tfidf"]):
    alpha=1.0 (sklearn default) and the alpha with the best val Macro F1
Writes:
  results/tables/baseline_table.csv        one row per model, mean/std over seeds
  results/tables/baseline_runs.csv         one row per model x seed
  results/tables/nb_alpha_sweep.csv        train/val Macro F1 per alpha
  results/tables/per_class_nb.csv          per-class P/R/F1 of the tuned NB
  results/predictions/val_predictions.csv  ids + true label + each model's prediction
  results/figures/confusion_nb_val.png
  results/logs/baseline_run.json           config, versions, timings, git commit
"""
import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn

from cfpb_triage.data.load import load_config, resolve
from cfpb_triage.data.split import load_splits
from cfpb_triage.evaluation.metrics import Timer, confusion, evaluate, per_class_report
from cfpb_triage.evaluation.plots import plot_confusion
from cfpb_triage.evaluation.variability import fmt_mean_std, summarize
from cfpb_triage.features.tfidf import fit_transform
from cfpb_triage.models.baseline import majority_class, stratified_random
from cfpb_triage.models.naive_bayes import fit_nb, sweep_alpha
from cfpb_triage.utils.seed import set_seed


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=resolve("."), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    cfg = load_config()
    bcfg = cfg["baseline"]
    seeds = bcfg["seeds"]
    set_seed(cfg["seed"])
    train, val, _ = load_splits()  # test is loaded with the others but never used
    y_tr, y_va = train["product"], val["product"]
    labels = sorted(y_tr.unique())
    print(f"[04] train={len(train):,} val={len(val):,} classes={len(labels)}")

    runs = []
    preds = val[["complaint_id", "product", "group_size"]].copy()

    def record(model: str, seed, y_pred, fit_s: float, pred_s: float) -> None:
        runs.append({"model": model, "seed": seed, **evaluate(y_va, y_pred),
                     "fit_s": fit_s, "predict_ms_per_sample": 1000 * pred_s / len(val)})

    # naive benchmarks: X is ignored, so a column of zeros is enough
    X_dummy = np.zeros((len(val), 1))
    for seed in seeds:
        with Timer() as tf:
            maj = majority_class(y_tr)
        with Timer() as tp:
            p = maj.predict(X_dummy)
        record("Majority class", seed, p, tf.s, tp.s)
    preds["pred_majority"] = p

    for seed in seeds:
        set_seed(seed)
        with Timer() as tf:
            rnd = stratified_random(y_tr, seed)
        with Timer() as tp:
            p = rnd.predict(X_dummy)
        record("Stratified Random", seed, p, tf.s, tp.s)
        preds[f"pred_stratified_s{seed}"] = p

    # course-level baseline: TF-IDF (fit on train only) + Multinomial NB
    with Timer() as t_vec:
        vec, X_tr, X_va = fit_transform(train["norm_text"], val["norm_text"], **bcfg["tfidf"])
    print(f"[04] TF-IDF {bcfg['tfidf']} -> vocab={len(vec.vocabulary_):,} ({t_vec.s:.1f}s)")

    best_alpha, sweep = sweep_alpha(X_tr, y_tr, X_va, y_va, bcfg["nb_alphas"])
    print(f"[04] NB alpha sweep (val Macro F1):\n{sweep.round(4).to_string(index=False)}")
    print(f"[04] best alpha = {best_alpha}")

    for name, alpha, col in [("Multinomial NB (alpha=1, default)", 1.0, "pred_nb_default"),
                             (f"Multinomial NB (alpha={best_alpha:g}, tuned on val)",
                              best_alpha, "pred_nb")]:
        for seed in seeds:  # NB is deterministic; repeating it shows std = 0
            set_seed(seed)
            with Timer() as tf:
                nb = fit_nb(X_tr, y_tr, alpha)
            with Timer() as tp:
                p = nb.predict(X_va)
            record(name, seed, p, tf.s, tp.s)
        preds[col] = p

    # ---- outputs ----
    tables, figs = resolve("results/tables"), resolve("results/figures")
    pred_dir, logs = resolve("results/predictions"), resolve("results/logs")
    for d in (tables, figs, pred_dir, logs):
        d.mkdir(parents=True, exist_ok=True)

    runs = pd.DataFrame(runs)
    runs.to_csv(tables / "baseline_runs.csv", index=False)
    table = summarize(runs)
    timing = runs.groupby("model", sort=False)[["fit_s", "predict_ms_per_sample"]].mean()
    table = table.join(timing, on="model")
    table.insert(1, "macro_f1 (mean ± std)",
                 [fmt_mean_std(m, s) for m, s in zip(table["macro_f1_mean"], table["macro_f1_std"])])
    table.round(4).to_csv(tables / "baseline_table.csv", index=False)
    sweep.round(4).to_csv(tables / "nb_alpha_sweep.csv", index=False)
    per_class_report(y_va, preds["pred_nb"], labels).round(4).to_csv(tables / "per_class_nb.csv")
    preds.to_csv(pred_dir / "val_predictions.csv", index=False)
    plot_confusion(confusion(y_va, preds["pred_nb"], labels), figs / "confusion_nb_val.png",
                   title=f"Multinomial NB (alpha={best_alpha:g}), validation, row-normalised")

    log = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__},
        "seed": cfg["seed"], "baseline_cfg": bcfg,
        "n_train": len(train), "n_val": len(val), "n_classes": len(labels),
        "vocab_size": len(vec.vocabulary_), "best_alpha": best_alpha,
        "tfidf_fit_transform_s": round(t_vec.s, 3),
        "eval_split": "val (test sealed until M5)",
    }
    (logs / "baseline_run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(table[["model", "macro_f1 (mean ± std)", "accuracy_mean", "weighted_f1_mean",
                     "fit_s", "predict_ms_per_sample"]].round(4).to_string(index=False))
    print("[04] wrote results/tables/baseline_table.csv and friends")


if __name__ == "__main__":
    main()
