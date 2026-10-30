"""M4: paired bootstrap of Macro F1 differences on the validation split.

Uses the predictions saved by m4_initial_models.py (no refitting, test untouched):
    python m4_deliverables/code/m4_significance.py
Writes m4_deliverables/results/tables/m4_significance.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"
BASE = "NB alpha=0.03 (M3 baseline)"
CHALLENGERS = ["LR C=10 cw=none", "LR C=1 cw=none", "LR C=10 cw=balanced", "kNN cosine k=15"]
N_BOOT, SEED = 1000, 42


def main() -> None:
    p = pd.read_csv(TABLES / "m4_val_predictions.csv")
    y = p["product"].to_numpy()
    rng = np.random.RandomState(SEED)
    idx = [rng.randint(0, len(y), len(y)) for _ in range(N_BOOT)]
    f = lambda pred, i: f1_score(y[i], pred[i], average="macro", zero_division=0)
    base = p[BASE].to_numpy()
    base_f1 = np.array([f(base, i) for i in idx])
    rows = [{"model": BASE, "macro_f1": f1_score(y, base, average="macro", zero_division=0),
             "ci_low": np.percentile(base_f1, 2.5), "ci_high": np.percentile(base_f1, 97.5)}]
    for name in CHALLENGERS:
        pred = p[name].to_numpy()
        f1s = np.array([f(pred, i) for i in idx])
        d = f1s - base_f1
        rows.append({"model": name, "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                     "ci_low": np.percentile(f1s, 2.5), "ci_high": np.percentile(f1s, 97.5),
                     "diff_vs_nb": d.mean(), "diff_ci_low": np.percentile(d, 2.5),
                     "diff_ci_high": np.percentile(d, 97.5)})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(TABLES / "m4_significance.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
