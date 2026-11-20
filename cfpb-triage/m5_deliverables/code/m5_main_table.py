"""M5 main results table + main figure (Sayon). Regenerates both from saved results only.

Table columns per model: validation Macro F1 with 95% bootstrap CI, repeated-CV mean +- std,
test Macro F1 with 95% CI (if m5_final_test.py has run), accuracy, macro precision/recall,
fit time, and the selected configuration. Figure: (a) Macro F1 per model, three evidence
sources side by side; (b) per-class F1 of the three tuned models on validation.

    python m5_deliverables/code/m5_main_table.py
Writes tables/m5_main_table.csv, tables/m5_main_table.md, figures/m5_main_figure.png
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cfpb_triage.analysis.significance import bootstrap_ci
from cfpb_triage.evaluation.metrics import per_class_report
from m5_common import (BEST_CONFIGS, FIGS, LABELS, NAIVE, TABLES, TUNED, ensure_dirs, md_table,
                       read_val_predictions)
from m5_runlog import logged_run

COLORS = {"val": "#1f77b4", "cv": "#ff7f0e", "test": "#2ca02c"}


def cfg_str(m: str, best: dict) -> str:
    if m not in best:
        return "-"
    t, p = best[m]["tfidf"], best[m]["params"]
    return (f"{t['ngram_range'][0]}-{t['ngram_range'][1]}g, {t['max_features']} feat, "
            f"stop={t['stop_words'] or 'no'}; " + ", ".join(f"{k}={v}" for k, v in p.items()))


def main() -> None:
    ensure_dirs()
    preds = read_val_predictions()
    y = preds["product"]
    labels = sorted(y.unique())
    best = json.loads(BEST_CONFIGS.read_text()) if BEST_CONFIGS.exists() else {}
    val = pd.read_csv(TABLES / "val_results_final.csv").set_index("model")
    cv = pd.read_csv(TABLES / "cv_summary.csv").set_index("model") \
        if (TABLES / "cv_summary.csv").exists() else None
    test = pd.read_csv(TABLES / "test_results.csv").set_index("model") \
        if (TABLES / "test_results.csv").exists() else None
    with logged_run("m5_main_table") as run:
        rows = []
        for m in NAIVE + TUNED:
            ci = bootstrap_ci(y, preds[m], n_boot=2000)
            r = {"Model": LABELS[m], "Val Macro F1": val.loc[m, "macro_f1"],
                 "Val 95% CI": f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                 "CV Macro F1 (mean ± std)": f"{cv.loc[m, 'mean']:.3f} ± {cv.loc[m, 'std_over_folds']:.3f}"
                 if cv is not None and m in cv.index else "-",
                 "Test Macro F1": f"{test.loc[m, 'macro_f1']:.3f} [{test.loc[m, 'macro_f1_ci_low']:.3f}, "
                                  f"{test.loc[m, 'macro_f1_ci_high']:.3f}]"
                 if test is not None else "(not yet run)",
                 "Val accuracy": val.loc[m, "accuracy"],
                 "Val macro P": val.loc[m, "macro_precision"],
                 "Val macro R": val.loc[m, "macro_recall"],
                 "Selected configuration": cfg_str(m, best)}
            rows.append(r)
            r["_ci"] = (ci["ci_low"], ci["ci_high"])
        tab = pd.DataFrame(rows)
        out = tab.drop(columns="_ci")
        out.to_csv(TABLES / "m5_main_table.csv", index=False)
        (TABLES / "m5_main_table.md").write_text(md_table(out), encoding="utf-8")
        print(md_table(out))

        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
        names = [LABELS[m] for m in NAIVE + TUNED]
        x = np.arange(len(names))
        for off, key, lab in ((-0.27, "val", "Validation (95% bootstrap CI)"),
                              (0.0, "cv", "Repeated grouped CV (mean ± std)"),
                              (0.27, "test", "Test, one-shot (95% CI)")):
            for i, m in enumerate(NAIVE + TUNED):
                if key == "val":
                    v, lo, hi = val.loc[m, "macro_f1"], *tab.loc[i, "_ci"]
                    err = [[v - lo], [hi - v]]
                elif key == "cv" and cv is not None and m in cv.index:
                    v, err = cv.loc[m, "mean"], cv.loc[m, "std_over_folds"]
                elif key == "test" and test is not None:
                    v = test.loc[m, "macro_f1"]
                    err = [[v - test.loc[m, "macro_f1_ci_low"]], [test.loc[m, "macro_f1_ci_high"] - v]]
                else:
                    continue
                a.bar(i + off, v, 0.25, yerr=err, color=COLORS[key], capsize=2,
                      label=lab if i == 0 else None)
        a.set_xticks(x, names, rotation=25, ha="right", fontsize=8)
        a.set_ylabel("Macro F1 (primary metric)")
        a.set_title("(a) Model comparison, three evidence sources", fontsize=10)
        a.legend(fontsize=7)
        a.grid(axis="y", alpha=.3)

        pc = {m: per_class_report(y, preds[m], labels)["f1"] for m in TUNED}
        order = per_class_report(y, preds["lr"], labels).index
        w = 0.27
        for j, m in enumerate(TUNED):
            b.barh(np.arange(len(order)) + (j - 1) * w, pc[m].reindex(order), w, label=LABELS[m])
        b.set_yticks(range(len(order)), [o if len(o) < 34 else o[:31] + "..." for o in order],
                     fontsize=7)
        b.invert_yaxis()
        b.set_xlabel("Validation F1 (classes ordered by size, largest at top)")
        b.set_title("(b) Per-class F1", fontsize=10)
        b.legend(fontsize=7)
        b.grid(axis="x", alpha=.3)
        fig.tight_layout()
        fig.savefig(FIGS / "m5_main_figure.png", dpi=150)
        plt.close(fig)
        run["outputs"] = [str(TABLES / "m5_main_table.csv"), str(FIGS / "m5_main_figure.png")]


if __name__ == "__main__":
    main()
