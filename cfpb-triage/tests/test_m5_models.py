"""M5 model tests (Alex). Synthetic data, so these always run (no CFPB data needed)."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from cfpb_triage.models.knn import KNNCosine
from cfpb_triage.models.logreg import fit_logreg


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = normalize(sparse.random(300, 50, density=0.2, random_state=rng, format="csr"))
    y = np.array(list("abc"))[rng.randint(0, 3, 300)]
    return X[:240], y[:240], X[240:], y[240:]


def test_knn_matches_sklearn_cosine_uniform(data):
    Xtr, ytr, Xq, _ = data
    ours = KNNCosine(k=7, weights="uniform").fit(Xtr, ytr).predict(Xq)
    ref = KNeighborsClassifier(7, metric="cosine", algorithm="brute").fit(Xtr, ytr).predict(Xq)
    assert (ours == ref).mean() > 0.97   # only exact vote ties can differ


def test_knn_grid_equals_single_calls(data):
    Xtr, ytr, Xq, _ = data
    grid = KNNCosine().fit(Xtr, ytr).predict_grid(Xq, [3, 9], ["uniform", "distance"])
    for (k, w), pred in grid.items():
        assert (pred == KNNCosine(k=k, weights=w).fit(Xtr, ytr).predict(Xq)).all()


def test_knn_empty_rows_dropped_and_fallback(data):
    Xtr, ytr, Xq, _ = data
    already_empty = int((np.asarray(Xtr.sum(axis=1)).ravel() == 0).sum())
    Xtr = sparse.vstack([Xtr, sparse.csr_matrix((10, Xtr.shape[1]))]).tocsr()
    ytr = np.concatenate([ytr, ["a"] * 10])
    knn = KNNCosine(k=5).fit(Xtr, ytr)
    assert knn.n_empty_ == already_empty + 10
    assert knn.X_.shape[0] == Xtr.shape[0] - knn.n_empty_
    majority = pd.Series(ytr).value_counts().idxmax()
    empty_q = sparse.csr_matrix((2, Xtr.shape[1]))
    assert set(knn.predict(empty_q)) == {majority}   # no tokens -> majority class


def test_logreg_regularisation_direction(data):
    Xtr, ytr, _, _ = data
    weak = fit_logreg(Xtr, ytr, C=100).coef_
    strong = fit_logreg(Xtr, ytr, C=0.01).coef_
    assert np.abs(weak).sum() > np.abs(strong).sum()   # larger C = weaker penalty
