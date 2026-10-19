"""Multinomial NB + alpha sweep (Alex)."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB


def fit_nb(X, y, alpha: float = 1.0) -> MultinomialNB:
    """Multinomial NB with Laplace/Lidstone smoothing alpha; class priors from train."""
    return MultinomialNB(alpha=alpha).fit(X, y)


def sweep_alpha(X_train, y_train, X_val, y_val, alphas) -> tuple[float, pd.DataFrame]:
    """Fit one NB per alpha on train and score Macro F1 on val.

    Returns (best_alpha, table). Ties go to the larger alpha (more smoothing).
    Train Macro F1 is kept so the over-fitting gap is visible.
    """
    rows = []
    for a in alphas:
        clf = fit_nb(X_train, y_train, a)
        rows.append({"alpha": a,
                     "train_macro_f1": f1_score(y_train, clf.predict(X_train),
                                                average="macro", zero_division=0),
                     "val_macro_f1": f1_score(y_val, clf.predict(X_val),
                                              average="macro", zero_division=0)})
    table = pd.DataFrame(rows)
    best = table.sort_values(["val_macro_f1", "alpha"], ascending=False).iloc[0]
    return float(best["alpha"]), table
