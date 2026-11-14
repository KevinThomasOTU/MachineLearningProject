"""L2 multinomial Logistic Regression (Alex).

Objective (per class k, softmax over K classes, n training rows):
    min_W  C * sum_i -log softmax(W x_i)[y_i]  +  0.5 * ||W||_F^2
so a LARGER C means WEAKER regularisation. class_weight='balanced' multiplies row i's loss by
n / (K * n_{y_i}); it trades precision for recall on rare classes (relevant for Macro F1).
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def fit_logreg(X, y, C: float = 1.0, class_weight=None, seed: int = 42,
               max_iter: int = 1000) -> LogisticRegression:
    """L2-penalised multinomial LR (sklearn default penalty, lbfgs solver)."""
    return LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter,
                              random_state=seed).fit(X, y)
