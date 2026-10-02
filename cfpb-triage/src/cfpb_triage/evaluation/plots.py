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
