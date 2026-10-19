"""Entry point: 07_discrepancy (Subangan). M3 discrepancy analysis.

Compares what we EXPECTED (theory, the M2 memo, the M2 compute estimate) with
what 04_run_baseline.py actually produced, and checks where the gaps come from.
Run 04_run_baseline.py first. Uses train + val only; test stays sealed.
Writes:
  results/tables/discrepancy.csv     check, expected, observed, gap, note
  results/tables/significance.csv    bootstrap CIs, paired bootstrap, McNemar
  results/tables/nb_by_nearcopy.csv  NB on val rows with / without a train near-copy
"""
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from cfpb_triage.analysis.distances import max_cosine_to_train
from cfpb_triage.analysis.significance import bootstrap_ci, mcnemar, paired_bootstrap
from cfpb_triage.analysis.smoothing import nb_from_scratch, nb_predict
from cfpb_triage.data.load import load_config, resolve
from cfpb_triage.data.split import load_splits
from cfpb_triage.evaluation.metrics import evaluate
from cfpb_triage.features.tfidf import fit_transform
from cfpb_triage.models.naive_bayes import fit_nb
from cfpb_triage.utils.seed import set_seed

NEAR_COPY = 0.8  # same threshold as the M2 leakage probe


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    tables = resolve("results/tables")
    runs = pd.read_csv(tables / "baseline_runs.csv")
    sweep = pd.read_csv(tables / "nb_alpha_sweep.csv")
    estimate = pd.read_csv(tables / "compute_estimate.csv", index_col=0)
    log = json.loads(resolve("results/logs/baseline_run.json").read_text(encoding="utf-8"))
    preds = pd.read_csv(resolve("results/predictions/val_predictions.csv"),
                        dtype={"complaint_id": str})
    train, val, _ = load_splits()
    val = val.set_index("complaint_id").loc[preds["complaint_id"]].reset_index()
    y = preds["product"]
    alpha = log["best_alpha"]
    mean = runs.groupby("model")[["macro_f1", "accuracy", "fit_s"]].mean()
    nb_name = next(m for m in mean.index if "tuned" in m)
    rows = []

    def add(check, expected, observed, note):
        gap = observed - expected if pd.notna(expected) else np.nan
        rows.append({"check": check, "expected": expected, "observed": observed,
                     "gap": gap, "note": note})

    # 1) naive benchmarks: closed-form expectations from the class priors
    q = train["product"].value_counts(normalize=True)          # what the model samples
    p = y.value_counts(normalize=True).reindex(q.index).fillna(0)  # true val mix
    k = len(q)
    add("Stratified Random Macro F1", float((2 * p * q / (p + q)).mean()),
        mean.loc["Stratified Random", "macro_f1"],
        "E[F1_c] ~ 2 p_c q_c / (p_c + q_c) = 1/K when train and val mixes match "
        "(M2 memo quoted ~0.091)")
    add("Stratified Random accuracy", float((p * q).sum()),
        mean.loc["Stratified Random", "accuracy"], "E[acc] = sum_c p_c q_c; M2 memo quoted ~0.31")
    top = q.idxmax()
    add("Majority class Macro F1", float(2 * p[top] / (1 + p[top]) / k),
        mean.loc["Majority class", "macro_f1"], f"only '{top[:25]}...' gets non-zero F1")
    add("Majority class accuracy", float(p[top]), mean.loc["Majority class", "accuracy"],
        "= share of the majority class in val")

    # 2) NB reproduced from the formulas vs sklearn (same TF-IDF, same alpha)
    _, X_tr, X_va = fit_transform(train["norm_text"], val["norm_text"],
                                  **cfg["baseline"]["tfidf"])
    hand = nb_predict(nb_from_scratch(X_tr, train["product"], alpha), X_va)
    sk = fit_nb(X_tr, train["product"], alpha).predict(X_va)
    add("NB from scratch vs sklearn: prediction agreement", 1.0, float((hand == sk).mean()),
        "same formulas -> should be identical")
    add("NB predictions match 04_run_baseline output", 1.0,
        float((sk == preds["pred_nb"].to_numpy()).mean()), "rerun is deterministic")

    # 3) compute: M2 estimate vs actual
    est = estimate.iloc[:, -1]  # per-run seconds at full data size
    add("TF-IDF fit+transform time (s)", float(est["tfidf_fit+transform"]),
        log["tfidf_fit_transform_s"], "M2 extrapolated from a 10k-row subsample")
    add("NB fit time (s)", float(est["multinomial_nb_fit"]), mean.loc[nb_name, "fit_s"],
        "M2 extrapolated from a 10k-row subsample")

    # 4) over-fitting and tuning optimism
    best = sweep.loc[sweep["alpha"] == alpha].iloc[0]
    add("NB train - val Macro F1 (tuned alpha)", np.nan,
        float(best["train_macro_f1"] - best["val_macro_f1"]), "generalisation gap")
    add("NB tuned - default alpha Macro F1 (val)", np.nan,
        float(best["val_macro_f1"] - sweep.loc[sweep["alpha"] == 1.0, "val_macro_f1"].iloc[0]),
        "gain from tuning alpha on val (optimistic until test in M5)")

    # 5) near-copies of training complaints (M2 probe: 22% of a 2k sample)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000)  # M2 settings
    S_tr = vec.fit_transform(train["norm_text"])  # fit on train only
    near = max_cosine_to_train(vec.transform(val["norm_text"]), S_tr) >= NEAR_COPY
    add("Share of val with a train near-copy (cos >= 0.8)", 0.22, float(near.mean()),
        "M2 estimate came from a 2,000-row val+test sample")
    sub = []
    for name, mask in [("all val", np.ones(len(y), bool)), ("has near-copy in train", near),
                       ("no near-copy in train", ~near)]:
        sub.append({"subset": name, "n": int(mask.sum()),
                    **evaluate(y[mask], preds.loc[mask, "pred_nb"])})
    sub = pd.DataFrame(sub)
    add("NB Macro F1: near-copy rows - other rows", np.nan,
        float(sub.iloc[1]["macro_f1"] - sub.iloc[2]["macro_f1"]),
        "how much easy template-like rows inflate the score")

    # 6) distinct complaints vs the live mix (weight each row by its duplicate group size)
    w = evaluate(y, preds["pred_nb"], sample_weight=preds["group_size"])
    add("NB Macro F1 weighted by group_size", mean.loc[nb_name, "macro_f1"], w["macro_f1"],
        "expected = unweighted score; gap = effect of collapsing templates")

    # ---- significance ----
    sig = []
    seed_cols = sorted(c for c in preds if c.startswith("pred_stratified_s"))
    for label, col in [("Majority class", "pred_majority"),
                       ("Stratified Random", seed_cols[0]),
                       ("NB default alpha", "pred_nb_default"), ("NB tuned alpha", "pred_nb")]:
        sig.append({"test": "bootstrap 95% CI", "model": label,
                    **bootstrap_ci(y, preds[col], seed=cfg["seed"])})
    for a, b in [("pred_nb", seed_cols[0]), ("pred_nb", "pred_nb_default")]:
        sig.append({"test": "paired bootstrap Macro F1 (A - B)", "model": f"{a} vs {b}",
                    **paired_bootstrap(y, preds[a], preds[b], seed=cfg["seed"])})
        sig.append({"test": "McNemar (accuracy)", "model": f"{a} vs {b}",
                    **mcnemar(y, preds[a], preds[b])})

    out = pd.DataFrame(rows)
    out.round(4).to_csv(tables / "discrepancy.csv", index=False)
    pd.DataFrame(sig).round(4).to_csv(tables / "significance.csv", index=False)
    sub.round(4).to_csv(tables / "nb_by_nearcopy.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60,
                           "display.max_columns", None):
        print(out.round(4).to_string(index=False))
        print(sub.round(4).to_string(index=False))
        print(pd.DataFrame(sig).round(4).to_string(index=False))
    print("[07] wrote results/tables/discrepancy.csv, significance.csv, nb_by_nearcopy.csv")


if __name__ == "__main__":
    main()
