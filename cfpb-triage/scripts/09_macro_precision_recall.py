"""Entry point: 09_macro_precision_recall. M3 memo baseline table columns.

Reads results/predictions/val_predictions.csv (written by 04_run_baseline.py) and writes
results/tables/baseline_macro_pr.csv: accuracy, macro precision, macro recall and macro F1
per model on the validation split (mean over the seeds for Stratified Random; the other
models are deterministic). Test split is not used.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from cfpb_triage.data.load import resolve

MODELS = {  # row label -> prediction columns in val_predictions.csv (prefix match)
    "Majority class": "pred_majority",
    "Stratified Random": "pred_stratified_s",
    "Multinomial NB (alpha=1, default)": "pred_nb_default",
    "Multinomial NB (tuned alpha)": "pred_nb",
}


def main() -> None:
    preds = pd.read_csv(resolve("results/predictions/val_predictions.csv"),
                        dtype={"complaint_id": str})
    y = preds["product"]
    rows = []
    for label, key in MODELS.items():
        cols = [c for c in preds if (c == key if key != "pred_stratified_s" else c.startswith(key))]
        scores = []
        for c in cols:
            p, r, f, _ = precision_recall_fscore_support(y, preds[c], average="macro",
                                                         zero_division=0)
            scores.append({"accuracy": accuracy_score(y, preds[c]), "macro_precision": p,
                           "macro_recall": r, "macro_f1": f})
        rows.append({"model": label, "n_runs": len(cols), **pd.DataFrame(scores).mean()})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(resolve("results/tables/baseline_macro_pr.csv"), index=False)
    print(out.round(3).to_string(index=False))
    print("[09] wrote results/tables/baseline_macro_pr.csv")


if __name__ == "__main__":
    main()
