"""M5 one-shot TEST evaluation (Sayon). The ONLY script that scores the test split.

Protect the test set (spec sections 3 and 6): this script
  * refuses to run without --i-confirm-frozen (the whole team agreed best_configs.json is final),
  * refuses to run a second time (results/logs/TEST_USED.lock is created on the first run),
  * records the SHA-256 of best_configs.json and the git commit in the lock file and the log,
  * fits every model on TRAIN only with the frozen configs (same protocol as validation).
Do NOT delete the lock file to "try again": a second look at test would make it a validation set.

    python m5_deliverables/code/m5_final_test.py --i-confirm-frozen

Writes tables/test_results.csv (Macro F1 + 95% bootstrap CI, other metrics, val-test gap),
test_per_class.csv, test_predictions_final.csv, figures/m5_confusion_test.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from cfpb_triage.analysis.significance import bootstrap_ci
from cfpb_triage.data.load import load_config
from cfpb_triage.evaluation.metrics import confusion, per_class_report
from cfpb_triage.evaluation.plots import plot_confusion
from cfpb_triage.models.baseline import majority_class, stratified_random
from cfpb_triage.utils.seed import set_seed
from m5_common import (BEST_CONFIGS, FIGS, LABELS, LOGS, NAIVE, TABLES, TEST_PREDS, TUNED,
                       ensure_dirs, features, fit_model, full_metrics, load_data_with_test)
from m5_runlog import _git, logged_run

LOCK = LOGS / "TEST_USED.lock"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-confirm-frozen", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    if not args.i_confirm_frozen:
        sys.exit("Refusing to touch the test split: pass --i-confirm-frozen after the team has "
                 "agreed that best_configs.json is final.")
    if LOCK.exists():
        sys.exit(f"Test split already evaluated ({LOCK.read_text()}). Not running again.")
    if not BEST_CONFIGS.exists():
        sys.exit("best_configs.json missing: run m5_tune.py first.")
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    best = json.loads(BEST_CONFIGS.read_text())
    for m in best.values():
        m["tfidf"]["ngram_range"] = tuple(m["tfidf"]["ngram_range"])
    sha = hashlib.sha256(BEST_CONFIGS.read_bytes()).hexdigest()
    LOCK.write_text(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "best_configs_sha256": sha, "git_commit": _git("rev-parse", "HEAD")}))

    train, val, test = load_data_with_test()
    ytr, yte = train["product"], test["product"]
    labels = sorted(ytr.unique())
    with logged_run("m5_final_test", vars(args), seed, note=f"best_configs sha256={sha[:12]}") as run:
        preds = test[["complaint_id", "product", "group_size"]].copy()
        preds["majority"] = majority_class(ytr).predict([[0]] * len(test))
        preds["random"] = stratified_random(ytr, seed).predict([[0]] * len(test))
        for m in TUNED:
            b = best[m]
            _, Xtr, Xte = features(train, test, b["tfidf"])
            preds[m] = fit_model(m, b["params"], Xtr, ytr, seed).predict(Xte)
        preds.to_csv(TEST_PREDS, index=False)

        val_res = pd.read_csv(TABLES / "val_results_final.csv").set_index("model")
        rows = []
        for m in NAIVE + TUNED:
            ci = bootstrap_ci(yte, preds[m], n_boot=2000, seed=seed)
            rows.append({"model": m, "label": LABELS[m], **full_metrics(yte, preds[m], labels),
                         "macro_f1_ci_low": ci["ci_low"], "macro_f1_ci_high": ci["ci_high"],
                         "val_macro_f1": val_res.loc[m, "macro_f1"]})
        res = pd.DataFrame(rows)
        res["val_minus_test"] = res["val_macro_f1"] - res["macro_f1"]
        res.to_csv(TABLES / "test_results.csv", index=False)

        pc = {}
        for m in TUNED:
            rep = per_class_report(yte, preds[m], labels)
            pc["support"] = rep["support"]
            pc[f"{m}_precision"], pc[f"{m}_recall"], pc[f"{m}_f1"] = (
                rep["precision"], rep["recall"], rep["f1"])
        pd.DataFrame(pc).to_csv(TABLES / "test_per_class.csv")
        top = res[res["model"].isin(TUNED)].sort_values("macro_f1").iloc[-1]["model"]
        plot_confusion(confusion(yte, preds[top], labels), FIGS / "m5_confusion_test.png",
                       title=f"{LABELS[top]} (test, row-normalised)")
        print(res.round(4).to_string(index=False))
        run["outputs"] = [str(TABLES / "test_results.csv")]


if __name__ == "__main__":
    main()
