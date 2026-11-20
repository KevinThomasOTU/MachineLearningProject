"""M5 significance testing between models (Subangan). No refitting: uses saved predictions.

For every pair among {Stratified Random, NB, LR, k-NN}:
  * paired bootstrap of  Macro F1(A) - Macro F1(B)  (same resampled rows for both models,
    2000 resamples, 95% percentile CI)   -> the primary comparison, because the primary metric
    is Macro F1 and McNemar only sees accuracy-type (right/wrong) differences
  * McNemar chi-square (continuity corrected) on per-row correctness -> supporting evidence
  * Holm step-down correction of both p-values over the family of all pairs (6 tests), so
    "significant" is not an artefact of testing many pairs.

    python m5_deliverables/code/m5_significance.py [--split val|test] [--n-boot 2000]

--split test is only allowed after Sayon's m5_final_test.py has written test predictions.
Output: results/tables/m5_significance_<split>.csv
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from cfpb_triage.analysis.significance import mcnemar, paired_bootstrap
from m5_common import LABELS, TABLES, TEST_PREDS, VAL_PREDS, ensure_dirs
from m5_runlog import logged_run

MODELS = ["random", "nb", "lr", "knn"]


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone step-down)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    ensure_dirs()
    path = VAL_PREDS if args.split == "val" else TEST_PREDS
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run the previous step first.")
    p = pd.read_csv(path, dtype={"complaint_id": str})
    y = p["product"].to_numpy()
    rows = []
    with logged_run("m5_significance", vars(args)) as run:
        for a, b in itertools.combinations(MODELS[::-1], 2):    # e.g. lr vs nb, knn vs nb ...
            pb = paired_bootstrap(y, p[a].to_numpy(), p[b].to_numpy(), n_boot=args.n_boot)
            mc = mcnemar(y, p[a].to_numpy(), p[b].to_numpy())
            rows.append({"A": LABELS[a], "B": LABELS[b], "macro_f1_diff": pb["diff"],
                         "ci_low": pb["ci_low"], "ci_high": pb["ci_high"],
                         "p_bootstrap": pb["p_value"], "mcnemar_p": mc["p_value"],
                         "A_right_B_wrong": mc["a_right_b_wrong"],
                         "A_wrong_B_right": mc["a_wrong_b_right"]})
        out = pd.DataFrame(rows)
        out["p_bootstrap_holm"] = holm(out["p_bootstrap"])
        out["mcnemar_p_holm"] = holm(out["mcnemar_p"])
        out["CI_excludes_0"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
        out.to_csv(TABLES / f"m5_significance_{args.split}.csv", index=False)
        print(out.round(4).to_string(index=False))
        run["outputs"] = [str(TABLES / f"m5_significance_{args.split}.csv")]


if __name__ == "__main__":
    main()
