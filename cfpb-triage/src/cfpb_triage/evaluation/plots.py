"""Confusion matrix and class-distribution plots (Sayon)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in scripts, CI and make
import matplotlib.pyplot as plt
import pandas as pd


def _short(label: str, n: int = 38) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def plot_class_distribution(labels, path: str | Path) -> pd.Series:
    """Horizontal bar chart of class counts on a log x-axis; returns the counts."""
    counts = pd.Series(labels).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(counts) + 1.5))
    ax.barh([_short(c) for c in counts.index], counts.to_numpy(), color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlim(right=counts.max() * 4)  # room for the count labels
    ax.set_xlabel("Number of complaints (log scale)")
    ax.set_title("Product class distribution (cleaned data)")
    for y, v in enumerate(counts.to_numpy()):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return counts


def plot_length_hist(n_tokens: pd.Series, path: str | Path) -> None:
    """Histogram of narrative length (tokens), log-scaled x-axis."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(n_tokens.clip(lower=1), bins=pd.Series(
        [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]).tolist(), color="#55A868")
    ax.set_xscale("log")
    ax.set_xlabel("Narrative length (tokens after masking, log scale)")
    ax.set_ylabel("Complaints")
    ax.set_title("Narrative length distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_length_box(df: pd.DataFrame, path: str | Path) -> None:
    """Box plot of narrative length per class (columns: product, n_tokens)."""
    order = df.groupby("product")["n_tokens"].median().sort_values().index
    data = [df.loc[df["product"] == c, "n_tokens"].clip(lower=1) for c in order]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.5))
    ax.boxplot(data, vert=False, showfliers=False)
    ax.set_yticks(range(1, len(order) + 1), [_short(c) for c in order])
    ax.set_xscale("log")
    ax.set_xlabel("Narrative length (tokens, log scale; outliers hidden)")
    ax.set_title("Narrative length by product")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
def plot_confusion(cm: pd.DataFrame, path: str | Path,
                   title: str = "Confusion matrix (row-normalised)") -> None:
    """Heatmap of a confusion matrix normalised per true class.

    Each row sums to 1, so the diagonal is per-class recall. Cells < 0.01 are
    left blank to keep it readable.
    """
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    n = len(cm)
    fig, ax = plt.subplots(figsize=(0.55 * n + 4.5, 0.45 * n + 3))
    im = ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), [_short(c, 22) for c in cm.columns], rotation=60,
                  ha="right", fontsize=8)
    ax.set_yticks(range(n), [_short(c, 32) for c in cm.index], fontsize=8)
    for i in range(n):
        for j in range(n):
            v = norm.iat[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    ax.set_xlabel("Predicted product")
    ax.set_ylabel("True product")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
