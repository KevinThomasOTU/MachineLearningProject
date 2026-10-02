"""Entry point: 06_compute_estimate (Alex). Timing probe for the M2 compute plan.

Times one run of each step on a subsample of the TRAIN split (queries come from
the VAL split; nothing is fit on val/test), using the most expensive ablation
setting (1-2 grams, 10,000 features, no stop-word removal). Then extrapolates to
  12 TF-IDF configs x 3 models (NB, LR, k-NN cosine+Euclidean) x n_seeds
for (a) the current cleaned data and (b) the full data before sample_size.
Scaling assumptions: TF-IDF, NB, LR ~ linear in n_train; brute-force k-NN
prediction ~ n_train x n_val. Writes results/tables/compute_estimate.csv.

Usage: python scripts/06_compute_estimate.py [--n-train 10000] [--n-query 2000]
                                             [--seeds 3] [--budget-hours 4]
"""
import argparse
import json
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from cfpb_triage.data.load import load_config, resolve
from cfpb_triage.data.split import load_splits
from cfpb_triage.utils.seed import set_seed

N_CONFIGS = 2 * 2 * 3  # stop words x n-gram ranges x max_features


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def full_size_from_log(default: int) -> int:
    """Rows after min_class_count, before sampling (from cleaning_log.json)."""
    path = resolve("results/logs/cleaning_log.json")
    if not path.exists():
        return default
    steps = json.loads(path.read_text(encoding="utf-8"))["steps"]
    return next((s["rows"] for s in steps if s["step"].startswith("min_class_count")), default)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-query", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget-hours", type=float, default=4.0)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    train, val, test = load_splits()
    tr = train.sample(min(args.n_train, len(train)), random_state=cfg["seed"])
    q = val.sample(min(args.n_query, len(val)), random_state=cfg["seed"])
    n_tr, n_q = len(tr), len(q)

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    Xtr, t_tfidf = timed(lambda: vec.fit_transform(tr["norm_text"]))  # fit on train only
    Xq, t_transform = timed(lambda: vec.transform(q["norm_text"]))
    _, t_nb = timed(lambda: MultinomialNB().fit(Xtr, tr["product"]))
    _, t_lr = timed(lambda: LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, tr["product"]))
    t_knn = 0.0
    for metric in ("cosine", "euclidean"):
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric, algorithm="brute")
        knn.fit(Xtr, tr["product"])
        _, t = timed(lambda: knn.predict(Xq))
        t_knn += t

    n_cur = len(train) + len(val) + len(test)  # test only counted, never used
    n_full = full_size_from_log(n_cur)
    fr_tr, fr_val = cfg["split"]["train"], cfg["split"]["val"]

    def per_run(n_total: int) -> dict:
        s_tr, s_val = fr_tr * n_total / n_tr, fr_val * n_total / n_q
        return {"tfidf_fit+transform": t_tfidf * s_tr + t_transform * s_val,
                "multinomial_nb_fit": t_nb * s_tr,
                "logreg_fit": t_lr * s_tr,
                "knn_predict_cos+euc": t_knn * s_tr * s_val}

    measured = {"tfidf_fit+transform": t_tfidf + t_transform, "multinomial_nb_fit": t_nb,
                "logreg_fit": t_lr, "knn_predict_cos+euc": t_knn}
    cur, full = per_run(n_cur), per_run(n_full)
    runs = N_CONFIGS * args.seeds
    tab = pd.DataFrame({"measured_s": measured, f"per_run_s_N={n_cur}": cur,
                        f"per_run_s_N={n_full}": full})
    tab.loc["TOTAL per run"] = tab.sum()
    tab.loc[f"GRID x{runs} runs (hours)"] = tab.loc["TOTAL per run"] * runs / 3600
    tab.loc[f"GRID x{runs} runs (hours)", "measured_s"] = float("nan")
    resolve("results/tables").mkdir(parents=True, exist_ok=True)
    tab.round(3).to_csv(resolve("results/tables/compute_estimate.csv"))

    print(f"[06] subsample: n_train={n_tr:,}, n_query={n_q:,}; grid = {N_CONFIGS} configs "
          f"x {args.seeds} seeds = {runs} runs (worst-case config timed)")
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 120):
        print(tab)
    h_cur = tab.loc[f"GRID x{runs} runs (hours)", f"per_run_s_N={n_cur}"]
    h_full = tab.loc[f"GRID x{runs} runs (hours)", f"per_run_s_N={n_full}"]
    if h_full <= args.budget_hours:
        rec = (f"full data ({n_full:,} rows) fits the {args.budget_hours:g} h budget "
               f"(~{h_full:.1f} h): sample_size can be set to null.")
    elif h_cur <= args.budget_hours:
        rec = (f"keep sample_size: current {n_cur:,} rows ~{h_cur:.1f} h fits the budget; "
               f"full {n_full:,} rows ~{h_full:.1f} h.")
    else:
        rec = (f"current {n_cur:,} rows ~{h_cur:.1f} h exceeds the {args.budget_hours:g} h budget: "
               "lower sample_size or subsample the k-NN training set.")
    print("[06] recommendation:", rec)


if __name__ == "__main__":
    main()
