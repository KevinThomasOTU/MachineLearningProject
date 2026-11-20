"""M5 ablation / sensitivity from the tuning grid (Subangan). No refitting needed.

Reads tuning_grid.csv (written by m5_tune.py) and answers, for each model:
  A1  TF-IDF ablation: what does each factor (stop words, n-grams, vocabulary size) change?
      For every combination of the OTHER two factors we take the best hyper-parameters per level
      (so tuning effort is equal for both levels) and report level - reference, averaged over
      the combinations (paired design) with min/max.
  A2  Hyper-parameter sensitivity: Macro F1 against alpha (NB), C (LR), k (k-NN), at the model's
      selected TF-IDF setting, with the other hyper-parameter fixed at its selected value.
  A3  Over-fitting gap: train - val Macro F1 along the same hyper-parameter axes (NB, LR).

    python m5_deliverables/code/m5_ablation_effects.py
Writes tables/ablation_tfidf_effects.csv, ablation_hyperparam_curves.csv, and
figures/m5_ablation_tfidf.png, figures/m5_hyperparam_sensitivity.png
"""
from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from m5_common import FIGS, LABELS, PRIMARY, TABLES, TUNED, ensure_dirs, read_best
from cfpb_triage.data.load import load_config
from m5_runlog import logged_run

FACTORS = {"stop_words": "none", "ngram": "1-1", "max_features": 1000}   # reference levels
HP_AXIS = {"nb": "p_alpha", "lr": "p_C", "knn": "p_k"}
COLORS = {"nb": "#1f77b4", "lr": "#d62728", "knn": "#2ca02c"}


def tfidf_effects(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in TUNED:
        g = grid[grid["model"] == model]
        best_hp = g.groupby(["stop_words", "ngram", "max_features"])[PRIMARY].max().reset_index()
        for factor, ref in FACTORS.items():
            others = [f for f in FACTORS if f != factor]
            piv = best_hp.pivot_table(index=others, columns=factor, values=PRIMARY)
            ref_col = [c for c in piv.columns if str(c) == str(ref)][0]
            for level in piv.columns:
                if level == ref_col:
                    continue
                d = (piv[level] - piv[ref_col]).dropna()
                rows.append({"model": model, "factor": factor, "level": level,
                             "reference": ref, "mean_diff": d.mean(), "min_diff": d.min(),
                             "max_diff": d.max(), "n_pairs": len(d),
                             "best_at_level": piv[level].max()})
    return pd.DataFrame(rows)


def _matches(col: pd.Series, want) -> pd.Series:
    """Compare a csv column with a selected value (csv turns 10 into 10.0, None into 'none')."""
    if want is None:
        return col.astype(str) == "none"
    if isinstance(want, bool):
        return col.astype(str) == str(want)
    try:
        return pd.to_numeric(col, errors="coerce") == float(want)
    except (TypeError, ValueError):
        return col.astype(str) == str(want)


def hyperparam_curves(grid: pd.DataFrame, best: dict) -> pd.DataFrame:
    parts = []
    for model in TUNED:
        b = best[model]
        tid = f"sw={b['tfidf']['stop_words'] or 'none'}|ng={b['tfidf']['ngram_range'][0]}-" \
              f"{b['tfidf']['ngram_range'][1]}|mf={b['tfidf']['max_features']}"
        g = grid[(grid["model"] == model) & (grid["tfidf_id"] == tid)].copy()
        axis = HP_AXIS[model]
        for col in [c for c in g.columns
                    if c.startswith("p_") and c != axis and g[c].notna().any()]:
            g = g[_matches(g[col], b["params"][col[2:]])]
        g = g[[axis, PRIMARY, "train_macro_f1", "fit_s"]].rename(columns={axis: "value"})
        g.insert(0, "model", model)
        g.insert(1, "hyperparameter", axis[2:])
        parts.append(g.sort_values("value"))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    grid = pd.read_csv(TABLES / "tuning_grid.csv")
    best = read_best(cfg)
    with logged_run("m5_ablation_effects") as run:
        eff = tfidf_effects(grid)
        eff.to_csv(TABLES / "ablation_tfidf_effects.csv", index=False)
        hp = hyperparam_curves(grid, best)
        hp.to_csv(TABLES / "ablation_hyperparam_curves.csv", index=False)
        print(eff.round(4).to_string(index=False))

        # Figure 1: TF-IDF factor effects (mean paired difference, bar = min..max)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
        for ax, factor in zip(axes, FACTORS):
            for j, m in enumerate(TUNED):
                s = eff[(eff["factor"] == factor) & (eff["model"] == m)]
                for i, r in enumerate(s.itertuples()):
                    x = i + (j - 1) * 0.25
                    ax.errorbar(x, r.mean_diff, yerr=[[r.mean_diff - r.min_diff],
                                [r.max_diff - r.mean_diff]], fmt="o", color=COLORS[m], capsize=3,
                                label=LABELS[m] if i == 0 else None)
                ax.set_xticks(range(len(s)), [str(v) for v in s["level"]])
            ax.axhline(0, color="k", lw=.8)
            ax.set_title(f"{factor} (vs {FACTORS[factor]})", fontsize=9)
            ax.grid(alpha=.3)
        axes[0].set_ylabel("Δ val Macro F1 (mean, min-max)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "m5_ablation_tfidf.png", dpi=150)
        plt.close(fig)

        # Figure 2: hyper-parameter sensitivity with train curve (over-fitting gap)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, m in zip(axes, TUNED):
            g = hp[hp["model"] == m]
            ax.plot(range(len(g)), g[PRIMARY], "-o", color=COLORS[m], label="validation")
            if g["train_macro_f1"].notna().any():
                ax.plot(range(len(g)), g["train_macro_f1"], "--", color=COLORS[m], alpha=.6,
                        label="train")
            ax.set_xticks(range(len(g)), [f"{v:g}" for v in g["value"]])
            ax.set_xlabel(g["hyperparameter"].iloc[0])
            ax.set_title(LABELS[m], fontsize=9)
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Macro F1")
        fig.tight_layout()
        fig.savefig(FIGS / "m5_hyperparam_sensitivity.png", dpi=150)
        plt.close(fig)
        run["outputs"] = [str(TABLES / "ablation_tfidf_effects.csv")]


if __name__ == "__main__":
    main()
