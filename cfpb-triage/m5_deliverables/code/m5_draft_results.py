"""Draft Results section generator (Kevin). Fills numbers from the saved tables so that no
figure in the draft is typed by hand; the INTERPRETATION sentences stay `TODO(name)` markers
that the team must write and defend (spec section 7: question, approach, result, what the
result does and does not support).

    python m5_deliverables/code/m5_draft_results.py
Writes m5_deliverables/M5_RESULTS_DRAFT.md (re-run any time; keep hand edits in a copy).
"""
from __future__ import annotations

import json

import pandas as pd

from m5_common import LABELS, M5, TABLES, TUNED, md_table
from m5_runlog import logged_run

OUT = M5 / "M5_RESULTS_DRAFT.md"


def read(name: str):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


def table(df, cols=None, digits=3):
    return "_(table not generated yet)_" if df is None else md_table(
        df[cols] if cols else df, "{:." + str(digits) + "f}")


def para(question, approach, result, supports, owner):
    return (f"**Question.** {question}  \n**Approach.** {approach}  \n**Result.** {result}  \n"
            f"**What this supports / does not support.** TODO({owner}): {supports}\n")


def main() -> None:
    grid, cv, sig = read("tuning_grid.csv"), read("cv_summary.csv"), read("m5_significance_val.csv")
    val, test = read("val_results_final.csv"), read("test_results.csv")
    abl, nc = read("ablation_tfidf_effects.csv"), read("robust_nearcopy.csv")
    wt, tr, lc = read("robust_weighted.csv"), read("robust_truncation.csv"), read("learning_curve_all.csv")
    conf, sl = read("error_top_confusions.csv"), read("error_by_slice.csv")
    best = json.loads((TABLES / "best_configs.json").read_text()) \
        if (TABLES / "best_configs.json").exists() else {}
    main_tab = (TABLES / "m5_main_table.md").read_text(encoding="utf-8") \
        if (TABLES / "m5_main_table.md").exists() else "_(run m5_main_table.py)_"

    n_cfg = {m: int((grid["model"] == m).sum()) for m in TUNED} if grid is not None else {}
    L = ["# Results (DRAFT for Milestone 5)", "",
         "_Auto-generated numbers; every `TODO(...)` must be replaced by a sentence the named "
         "member can defend. Figures are in `results/figures/`._", "",
         "## 1. Experimental protocol and fair comparison", "",
         "All models use the same train/validation/test split (seed 42, group-aware), the same 12 "
         "TF-IDF settings fit on train only, the same primary metric (Macro F1, fixed in M2) and "
         "selection rule (best validation Macro F1; ties to the faster fit). Test was scored once, "
         "after the configurations were frozen.", "",
         f"Configurations searched per model: {n_cfg}." if n_cfg else "", "",
         "Selected configurations (validation):", ""]
    for m in TUNED:
        if m in best:
            L.append(f"- {LABELS[m]}: TF-IDF {best[m]['tfidf']}, hyper-parameters "
                     f"{best[m]['params']}, val Macro F1 {best[m]['val_macro_f1']:.4f}")
    L += ["", "## 2. Main result", "", main_tab, "",
          "![Main figure](results/figures/m5_main_figure.png)", ""]
    if val is not None:
        v = val.set_index("model")["macro_f1"]
        L.append(para("Does any model beat the naive benchmarks and the M3 NB baseline?",
                      "Equal-effort grid search on validation; headline numbers with 95% bootstrap CIs.",
                      "Validation Macro F1: " + ", ".join(f"{LABELS[m]} {v[m]:.3f}" for m in v.index) + ".",
                      "state which differences exceed the CI width, and that validation numbers are "
                      "optimistic because configurations were chosen on them.", "Alex"))
    L += ["## 3. Uncertainty", ""]
    if cv is not None:
        L += [table(cv, ["label", "mean", "std_over_folds", "min", "max", "n_runs"], 4), ""]
    L.append(para("How stable is Macro F1 under a different draw of the data?",
                  "Repeated 5-fold StratifiedGroupKFold on train+val (test excluded), frozen configs.",
                  "See table above; fold-to-fold std compared with the gaps between models.",
                  "which model orderings survive the variability.", "Sayon"))
    if sig is not None:
        L += ["Pairwise tests on validation (paired bootstrap of Macro F1 difference, McNemar, "
              "Holm-adjusted):", "",
              table(sig, ["A", "B", "macro_f1_diff", "ci_low", "ci_high", "p_bootstrap_holm",
                          "mcnemar_p_holm"], 4), ""]
    L.append(para("Are the model differences larger than resampling noise?",
                  "Paired bootstrap (2000 resamples) + McNemar, Holm-corrected over 6 pairs.",
                  "See table.", "a CI that excludes 0 supports a difference on THIS split only.",
                  "Subangan"))
    L += ["## 4. Ablation and sensitivity", ""]
    if abl is not None:
        L += [table(abl, ["model", "factor", "level", "reference", "mean_diff", "min_diff",
                          "max_diff"], 4), "",
              "![TF-IDF ablation](results/figures/m5_ablation_tfidf.png)", "",
              "![Hyper-parameter sensitivity](results/figures/m5_hyperparam_sensitivity.png)", ""]
    L.append(para("Which design choice (stop words, n-grams, vocabulary size, regularisation, k) matters?",
                  "Paired ablation over the tuning grid; hyper-parameter curves at the selected setting.",
                  "See table and figures.", "effects smaller than the CV std are not evidence.",
                  "Subangan"))
    L += ["## 5. Robustness", ""]
    for name, df in (("Near-copy subsets", nc), ("Template weighting", wt), ("Truncated input", tr),
                     ("Training-size curve", lc)):
        L += [f"**{name}**", "", table(df), ""]
    L.append(para("Is the score driven by near-duplicate complaints, template weighting, or long text?",
                  "Subset / re-weighted / truncated validation scoring of the frozen models.",
                  "See tables; figures `m5_truncation.png`, `m5_learning_curve.png`.",
                  "what these probes do not cover (e.g. other time periods).", "Alex"))
    L += ["## 6. Error analysis", ""]
    if conf is not None:
        L += [table(conf.head(8), None, 3), ""]
    if sl is not None:
        L += [table(sl), ""]
    L += ["![Confusion matrix](results/figures/m5_confusion_val.png)", "",
          para("Where and why do the models fail?",
               "Top confusions, error by narrative length / template size, hand-coded sample "
               "(`error_examples_TO_LABEL.csv`).", "See tables.",
               "fraction of errors that are label ambiguity vs model weakness, from the hand-coded "
               "sample.", "Kevin")]
    if test is not None:
        L += ["## 7. Test result (one-shot)", "",
              table(test, ["label", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
                           "val_macro_f1", "val_minus_test"], 4), ""]
    L += ["## 8. Draft conclusions", "",
          "TODO(Kevin): 3-4 bullets, each citing a table/figure above. Say what is supported, what "
          "is uncertain, negative results, and next steps.", "",
          "## 9. Draft limitations", "",
          "TODO(all): single time window (Apr-Jul 2024); labels are consumer-/company-assigned "
          "products (noisy boundary between credit-reporting and debt-collection); deduplication "
          "removes templates so scores describe unique complaints; CV configs chosen on val "
          "(optimistic); only TF-IDF features; test is a single split; no fairness/subgroup "
          "analysis by demographic (not available in data)."]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    with logged_run("m5_draft_results") as run:
        main()
        run["outputs"] = [str(OUT)]
