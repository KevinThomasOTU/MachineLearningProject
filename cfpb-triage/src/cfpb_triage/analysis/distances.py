"""Cosine vs Euclidean on sparse TF-IDF"""
from __future__ import annotations

import numpy as np


def max_cosine_to_train(X_query, X_train, query_chunk: int = 2000,
                        train_chunk: int = 20000) -> np.ndarray:
    """For each query row, the highest cosine similarity to any TRAIN row.

    Assumes rows are L2-normalised (TfidfVectorizer default), so cosine = dot
    product. Both sides are chunked to bound memory (same idea as the M2
    cross-split leakage probe, but over every query row instead of a sample).
    """
    best = np.zeros(X_query.shape[0])
    for q0 in range(0, X_query.shape[0], query_chunk):
        Xq = X_query[q0:q0 + query_chunk]
        for t0 in range(0, X_train.shape[0], train_chunk):
            sims = (Xq @ X_train[t0:t0 + train_chunk].T).max(axis=1).toarray().ravel()
            best[q0:q0 + len(sims)] = np.maximum(best[q0:q0 + len(sims)], sims)
    return best


def euclidean_from_cosine(cos: np.ndarray) -> np.ndarray:
    """For unit-length vectors ||a - b|| = sqrt(2 - 2 cos(a, b)).

    So on L2-normalised TF-IDF, Euclidean and cosine k-NN rank neighbours the
    same way (theory note for M4/M5).
    """
    return np.sqrt(np.clip(2 - 2 * np.asarray(cos), 0, None))