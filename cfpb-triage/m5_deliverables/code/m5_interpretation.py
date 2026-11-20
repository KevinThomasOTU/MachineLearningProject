"""M5 model interpretation + theory checks (Subangan). Validation/train only.

  I1  Top terms per class: LR coefficients and NB log-likelihood ratios. Checks that the models
      learned product vocabulary (e.g. "escrow" -> Mortgage) and not artefacts (company names,
      boiler-plate). Anything suspicious goes into the limitations section.
  I2  Theory check for k-NN: on unit-length TF-IDF rows ||a-b||^2 = 2 - 2cos(a,b). We verify the
      identity numerically on random pairs and count all-zero rows (the M4 Euclidean failure).
  I3  Counts of all-zero val/train rows and NB's fitted prior vs the real class frequencies
      (why fit_prior=False changes the precision/recall balance).

    python m5_deliverables/code/m5_interpretation.py [--top 10]
Writes tables/top_terms_lr.csv, top_terms_nb.csv, theory_checks.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from cfpb_triage.data.load import load_config
from cfpb_triage.utils.seed import set_seed
from m5_common import TABLES, ensure_dirs, features, fit_model, load_data, read_best
from m5_runlog import logged_run


def top_terms(scores: np.ndarray, classes, vocab, n: int) -> pd.DataFrame:
    rows = []
    for k, c in enumerate(classes):
        for rank, j in enumerate(np.argsort(-scores[k])[:n], 1):
            rows.append({"product": c, "rank": rank, "term": vocab[j], "score": scores[k, j]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    ensure_dirs()
    train, val = load_data()
    best = read_best(cfg)
    ytr = train["product"]
    with logged_run("m5_interpretation", vars(args)) as run:
        # I1 -------------------------------------------------------------------------
        b = best["lr"]
        vec, Xtr, Xva = features(train, val, b["tfidf"])
        lr = fit_model("lr", b["params"], Xtr, ytr)
        top_terms(lr.coef_, lr.classes_, vec.get_feature_names_out(), args.top).to_csv(
            TABLES / "top_terms_lr.csv", index=False)

        b = best["nb"]
        vec_nb, Xtr_nb, _ = features(train, val, b["tfidf"])
        nb = fit_model("nb", b["params"], Xtr_nb, ytr)
        lp = nb.feature_log_prob_                           # log P(word | class)
        ratio = lp - (lp.sum(axis=0, keepdims=True) - lp) / (lp.shape[0] - 1)
        top_terms(ratio, nb.classes_, vec_nb.get_feature_names_out(), args.top).to_csv(
            TABLES / "top_terms_nb.csv", index=False)

        # I2 -------------------------------------------------------------------------
        rng = np.random.RandomState(cfg["seed"])
        i, j = rng.randint(0, Xva.shape[0], 2000), rng.randint(0, Xtr.shape[0], 2000)
        A, B = Xva[i], Xtr[j]
        cos = np.asarray(A.multiply(B).sum(axis=1)).ravel()
        diff = A - B
        euc2 = np.asarray(diff.multiply(diff).sum(axis=1)).ravel()
        norms = np.sqrt(np.asarray(Xtr.multiply(Xtr).sum(axis=1)).ravel())
        norms_v = np.sqrt(np.asarray(Xva.multiply(Xva).sum(axis=1)).ravel())
        keep = (np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel()) > 0) & \
               (np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel()) > 0)
        checks = {
            "identity_max_abs_error_nonempty_pairs": float(np.abs(euc2 - (2 - 2 * cos))[keep].max()),
            "n_pairs_checked": int(keep.sum()),
            "train_rows_all_zero": int((norms == 0).sum()),
            "train_rows_all_zero_share": float((norms == 0).mean()),
            "val_rows_all_zero": int((norms_v == 0).sum()),
            "val_rows_all_zero_share": float((norms_v == 0).mean()),
            "empty_row_distance_to_any_unit_vector": 1.0,
            "nb_class_prior_vs_frequency_max_abs_diff": float(np.abs(
                np.exp(nb.class_log_prior_) - ytr.value_counts(normalize=True)
                .reindex(nb.classes_).to_numpy()).max()) if b["params"].get("fit_prior", True)
            else None,
        }
        json.dump(checks, open(TABLES / "theory_checks.json", "w"), indent=2)
        print(json.dumps(checks, indent=2))
        run["outputs"] = [str(TABLES / "top_terms_lr.csv"), str(TABLES / "theory_checks.json")]


if __name__ == "__main__":
    main()
