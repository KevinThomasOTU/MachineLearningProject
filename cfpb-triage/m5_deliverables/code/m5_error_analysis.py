"""M5 error analysis (Kevin). Validation split only (test stays sealed until Sayon's script).

For the model with the best validation Macro F1 (or --model nb|lr|knn):
  E1  Per-class precision / recall / F1 for all three tuned models (where does Macro F1 go?)
  E2  The most frequent confusions (true -> predicted) with counts and share of the true class
  E3  Error rate by narrative length bucket and by template size (group_size)
  E4  Examples: seeded sample of misclassified narratives for the top confusion pairs, with an
      empty `reason` column for the team to fill by hand (label ambiguity? boilerplate? too
      short? genuinely wrong label?). This hand-coded column is what the Results section cites.
  E5  Model disagreement: rows all three models get wrong vs rows only one gets right

    python m5_deliverables/code/m5_error_analysis.py [--model lr] [--examples 8]
Writes tables/error_*.csv, figures/m5_confusion_val.png, figures/m5_error_by_length.png
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cfpb_triage.data.load import load_config
from cfpb_triage.evaluation.metrics import confusion, per_class_report
from cfpb_triage.evaluation.plots import plot_confusion
from cfpb_triage.utils.seed import set_seed
from m5_common import FIGS, LABELS, TABLES, TUNED, ensure_dirs, load_data, read_val_predictions
from m5_runlog import logged_run

LEN_BINS = [0, 25, 50, 100, 200, 400, np.inf]
LEN_LABELS = ["<25", "25-49", "50-99", "100-199", "200-399", "400+"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=TUNED, default=None)
    ap.add_argument("--examples", type=int, default=8, help="examples per confusion pair")
    ap.add_argument("--top-pairs", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    ensure_dirs()
    _, val = load_data()
    preds = read_val_predictions()
    df = val.merge(preds[["complaint_id"] + TUNED], on="complaint_id", validate="1:1",
                   suffixes=("", "_pred"))
    y = df["product"]
    labels = sorted(y.unique())
    f1s = {m: f1_score(y, df[m], average="macro", zero_division=0) for m in TUNED}
    model = args.model or max(f1s, key=f1s.get)
    df["pred"], df["wrong"] = df[model], df[model] != y
    print(f"[error] analysing {LABELS[model]} (val macro F1 {f1s[model]:.4f}); "
          f"{df['wrong'].sum()} / {len(df)} wrong")

    with logged_run("m5_error_analysis", vars(args)) as run:
        # E1 --------------------------------------------------------------------------
        pc = {}
        for m in TUNED:
            rep = per_class_report(y, df[m], labels)
            pc["support"] = rep["support"]
            pc[f"{m}_precision"], pc[f"{m}_recall"], pc[f"{m}_f1"] = (
                rep["precision"], rep["recall"], rep["f1"])
        pd.DataFrame(pc).to_csv(TABLES / "error_per_class.csv")

        # E2 --------------------------------------------------------------------------
        cm = confusion(y, df["pred"], labels)
        plot_confusion(cm, FIGS / "m5_confusion_val.png",
                       title=f"{LABELS[model]} (validation, row-normalised)")
        pairs = df[df["wrong"]].groupby(["product", "pred"]).size().rename("n").reset_index()
        pairs["share_of_true_class"] = pairs["n"] / pairs["product"].map(y.value_counts())
        pairs = pairs.sort_values("n", ascending=False)
        pairs.to_csv(TABLES / "error_top_confusions.csv", index=False)

        # E3 --------------------------------------------------------------------------
        df["n_tokens"] = df["norm_text"].fillna("").str.split().str.len()
        df["length_bucket"] = pd.cut(df["n_tokens"], LEN_BINS, labels=LEN_LABELS, right=False)
        df["template_size"] = pd.cut(df["group_size"], [0, 1, 5, np.inf],
                                     labels=["1 (unique)", "2-5", "6+"])
        sub = []
        for col in ("length_bucket", "template_size"):
            for lvl, part in df.groupby(col, observed=True):
                sub.append({"slice": col, "level": str(lvl), "n": len(part),
                            "error_rate": part["wrong"].mean(),
                            "macro_f1": f1_score(part["product"], part["pred"], average="macro",
                                                 zero_division=0)})
        s_all = pd.DataFrame(sub)
        s_all.to_csv(TABLES / "error_by_slice.csv", index=False)
        s = s_all[s_all["slice"] == "length_bucket"]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.bar(s["level"], s["error_rate"], color="#d62728")
        for i, (n, e) in enumerate(zip(s["n"], s["error_rate"])):
            ax.text(i, e, f"n={n}", ha="center", va="bottom", fontsize=7)
        ax.set(xlabel="Narrative length (tokens)", ylabel="Validation error rate",
               title=f"{LABELS[model]}: errors by narrative length")
        fig.tight_layout()
        fig.savefig(FIGS / "m5_error_by_length.png", dpi=150)
        plt.close(fig)

        # E4 --------------------------------------------------------------------------
        rng = np.random.RandomState(cfg["seed"])
        ex = []
        for r in pairs.head(args.top_pairs).itertuples():
            part = df[df["wrong"] & (df["product"] == r.product) & (df["pred"] == r.pred)]
            take = part.iloc[rng.choice(len(part), min(args.examples, len(part)), replace=False)]
            for row in take.itertuples():
                ex.append({"complaint_id": row.complaint_id, "true": row.product, "pred": row.pred,
                           "n_tokens": int(row.n_tokens), "group_size": row.group_size,
                           "text_first_600_chars": str(row.norm_text)[:600], "reason": ""})
        pd.DataFrame(ex).to_csv(TABLES / "error_examples_TO_LABEL.csv", index=False)

        # E5 --------------------------------------------------------------------------
        n_right = np.column_stack([df[m] == y for m in TUNED]).sum(axis=1)
        dis = pd.DataFrame({"rows": [int((n_right == k).sum()) for k in range(4)]},
                           index=["all 3 wrong", "1 right", "2 right", "all 3 right"])
        dis["share"] = dis["rows"] / len(df)
        dis.to_csv(TABLES / "error_model_agreement.csv")
        print(pairs.head(args.top_pairs).to_string(index=False))
        print(dis)
        run["outputs"] = [str(TABLES / "error_top_confusions.csv")]


if __name__ == "__main__":
    main()
