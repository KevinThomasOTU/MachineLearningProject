"""k-NN with cosine similarity on sparse TF-IDF (Alex + Subangan).

Why a hand-written class instead of sklearn's KNeighborsClassifier?
  1. The neighbour search is the expensive part. Here it is done ONCE per TF-IDF setting
     for the largest k, and every (k, weighting) combination in the tuning grid is read off
     the same neighbour list. That makes a fair, equal-effort grid affordable.
  2. All-zero TF-IDF rows (no token of the narrative is in the vocabulary) are dropped from
     the reference set and queries with no tokens fall back to the majority class. This is the
     M4 finding: such rows sit at Euclidean distance 1 from every unit vector and wrecked
     Euclidean k-NN. For L2-normalised vectors ||a-b||^2 = 2 - 2cos(a,b), so cosine and
     Euclidean give the same neighbour ranking once empty rows are handled.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def _row_norms(X) -> np.ndarray:
    return np.sqrt(X.multiply(X).sum(axis=1)).A1


class KNNCosine:
    """Cosine k-NN. weights='uniform' (majority vote) or 'distance' (weight 1 / (1 - cos))."""

    def __init__(self, k: int = 15, weights: str = "distance", chunk: int = 300):
        self.k, self.weights, self.chunk = k, weights, chunk

    # ---- fit -----------------------------------------------------------------------
    def fit(self, X, y) -> "KNNCosine":
        X = sparse.csr_matrix(X)
        keep = _row_norms(X) > 0
        self.n_empty_ = int((~keep).sum())
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        self.prior_ = int(np.bincount(codes).argmax())       # majority class code
        self.X_, self.y_ = X[keep], codes[keep]
        self.XT_ = self.X_.T.tocsr()
        return self

    # ---- neighbour search (done once) ---------------------------------------------
    def neighbors(self, Xq, kmax: int) -> tuple[np.ndarray, np.ndarray]:
        """Indices and cosine similarities of the kmax nearest reference rows, best first."""
        Xq = sparse.csr_matrix(Xq)
        kmax = min(kmax, self.X_.shape[0])
        n = Xq.shape[0]
        idx = np.empty((n, kmax), dtype=np.int64)
        sim = np.empty((n, kmax), dtype=np.float64)
        for s in range(0, n, self.chunk):
            S = (Xq[s:s + self.chunk] @ self.XT_).toarray()
            top = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
            ts = np.take_along_axis(S, top, axis=1)
            order = np.argsort(-ts, axis=1, kind="stable")
            idx[s:s + self.chunk] = np.take_along_axis(top, order, axis=1)
            sim[s:s + self.chunk] = np.take_along_axis(ts, order, axis=1)
        return idx, sim

    def _vote(self, idx, sim, k: int, weights: str) -> np.ndarray:
        n = idx.shape[0]
        lab = self.y_[idx[:, :k]]
        w = (1.0 / np.clip(1.0 - sim[:, :k], 1e-9, None)) if weights == "distance" \
            else np.ones((n, k))
        counts = np.zeros((n, len(self.classes_)))
        rows = np.arange(n)
        for j in range(k):
            counts[rows, lab[:, j]] += w[:, j]
        return counts.argmax(axis=1)

    # ---- prediction ----------------------------------------------------------------
    def predict_grid(self, Xq, ks, weights_list=("uniform", "distance")) -> dict:
        """{(k, weights): predicted labels} for every combination, one neighbour search."""
        Xq = sparse.csr_matrix(Xq)
        idx, sim = self.neighbors(Xq, max(ks))
        empty = _row_norms(Xq) == 0
        out = {}
        for k in ks:
            for w in weights_list:
                code = self._vote(idx, sim, min(k, idx.shape[1]), w)
                code[empty] = self.prior_
                out[(k, w)] = self.classes_[code]
        return out

    def predict(self, Xq) -> np.ndarray:
        return self.predict_grid(Xq, [self.k], [self.weights])[(self.k, self.weights)]
