"""Laplace smoothing / Bayes demo 
Multinomial NB written from the formulas, used to check that sklearn's
MultinomialNB computes what the theory section says it does:

    P(c)       = N_c / N
    P(w | c)   = (count(w, c) + alpha) / (sum_w' count(w', c) + alpha * V)
    predict    = argmax_c  log P(c) + sum_w x_w log P(w | c)

With TF-IDF input, count(w, c) is the summed TF-IDF weight of word w in class c.
"""
from __future__ import annotations

import numpy as np


def nb_from_scratch(X, y, alpha: float = 1.0) -> dict:
    """Fit Multinomial NB by hand. X: (n, V) non-negative (sparse ok), y: labels."""
    y = np.asarray(y)
    classes = np.unique(y)  # sorted, same order as sklearn's classes_
    Y = (y[:, None] == classes[None, :]).astype(np.float64)        # (n, K) one-hot
    counts = np.asarray((X.T @ Y).T)                                # (K, V)
    V = counts.shape[1]
    log_denom = np.log(counts.sum(axis=1) + alpha * V)              # (K,)
    return {"classes": classes, "alpha": alpha,
            "log_prior": np.log(Y.sum(axis=0) / len(y)),
            "log_lik": np.log(counts + alpha) - log_denom[:, None],
            "log_denom": log_denom}


def nb_predict(model: dict, X) -> np.ndarray:
    """argmax_c of log P(c) + X @ log P(w|c)."""
    joint = np.asarray(X @ model["log_lik"].T) + model["log_prior"]
    return model["classes"][joint.argmax(axis=1)]


def unseen_word_logprob(model: dict) -> np.ndarray:
    """log P(w|c) for a word with count 0 in class c: log(alpha) - log(denominator).

    Without smoothing (alpha -> 0) this goes to -inf, so a single word never
    seen in class c would rule c out entirely; alpha keeps it finite.
    """
    return np.log(model["alpha"]) - model["log_denom"]
