"""M3 baseline tests (Sayon).
Small synthetic text data, so these always run (no CFPB data needed). They check
the metric helpers, that the benchmarks and NB are reproducible, that TF-IDF is
fit on train only, and that the hand-written NB matches sklearn.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from cfpb_triage.analysis.significance import _codes, _macro_f1, bootstrap_ci, paired_bootstrap
from cfpb_triage.analysis.smoothing import nb_from_scratch, nb_predict, unseen_word_logprob
from cfpb_triage.evaluation.metrics import evaluate, per_class_report
from cfpb_triage.evaluation.variability import summarize
from cfpb_triage.features.tfidf import fit_transform
from cfpb_triage.models.baseline import majority_class, stratified_random
from cfpb_triage.models.naive_bayes import fit_nb, sweep_alpha

WORDS = {"Credit card": "card fee interest statement charge limit",
         "Mortgage": "mortgage escrow home loan servicer foreclosure",
         "Debt collection": "collector debt called owed agency letter"}


def make_text(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Each class has its own words plus shared filler; imbalanced 60/30/10."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(list(WORDS), size=n, p=[0.6, 0.3, 0.1])
    filler = "the account bank i my was".split()
    text = [" ".join(rng.choice(WORDS[c].split() + filler, size=15)) for c in labels]
    return pd.DataFrame({"norm_text": text, "product": labels})


@pytest.fixture(scope="module")
def data():
    df = make_text()
    return df.iloc[:200].reset_index(drop=True), df.iloc[200:].reset_index(drop=True)


def test_evaluate_matches_sklearn(data):
    _, va = data
    y = va["product"]
    pred = y.sample(frac=1, random_state=1).to_numpy()  # shuffled = mostly wrong
    out = evaluate(y, pred)
    assert out["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert evaluate(y, y)["macro_f1"] == 1.0


def test_per_class_report_has_every_label(data):
    _, va = data
    rep = per_class_report(va["product"], ["Mortgage"] * len(va), sorted(WORDS))
    assert set(rep.index) == set(WORDS)
    assert rep["support"].sum() == len(va)
    assert rep.loc["Credit card", "f1"] == 0.0  # never predicted -> 0, no error


def test_stratified_random_is_reproducible(data):
    tr, va = data
    X = np.zeros((len(va), 1))
    a = stratified_random(tr["product"], seed=42).predict(X)
    b = stratified_random(tr["product"], seed=42).predict(X)
    c = stratified_random(tr["product"], seed=43).predict(X)
    assert (a == b).all()
    assert not (a == c).all()


def test_majority_predicts_train_mode(data):
    tr, va = data
    pred = majority_class(tr["product"]).predict(np.zeros((len(va), 1)))
    assert set(pred) == {tr["product"].mode()[0]}


def test_tfidf_fit_on_train_only():
    tr = pd.Series(["alpha beta", "beta gamma"])
    va = pd.Series(["delta epsilon beta"])  # delta/epsilon only exist in val
    vec, X_tr, X_va = fit_transform(tr, va, ngram_range=[1, 1])
    assert "delta" not in vec.vocabulary_ and "epsilon" not in vec.vocabulary_
    assert X_va.shape == (1, X_tr.shape[1])


def test_nb_beats_benchmarks_and_is_deterministic(data):
    tr, va = data
    _, X_tr, X_va = fit_transform(tr["norm_text"], va["norm_text"])
    best, table = sweep_alpha(X_tr, tr["product"], X_va, va["product"], [0.1, 1.0])
    assert best in (0.1, 1.0) and len(table) == 2
    p1 = fit_nb(X_tr, tr["product"], best).predict(X_va)
    p2 = fit_nb(X_tr, tr["product"], best).predict(X_va)
    assert (p1 == p2).all()
    rnd = stratified_random(tr["product"], 42).predict(np.zeros((len(va), 1)))
    assert evaluate(va["product"], p1)["macro_f1"] > evaluate(va["product"], rnd)["macro_f1"]


@pytest.mark.parametrize("alpha", [0.01, 1.0])
def test_nb_from_scratch_matches_sklearn(data, alpha):
    tr, va = data
    _, X_tr, X_va = fit_transform(tr["norm_text"], va["norm_text"])
    sk = fit_nb(X_tr, tr["product"], alpha)
    hand = nb_from_scratch(X_tr, tr["product"], alpha)
    np.testing.assert_allclose(hand["log_lik"], sk.feature_log_prob_, rtol=1e-10)
    np.testing.assert_allclose(hand["log_prior"], sk.class_log_prior_, rtol=1e-10)
    assert (nb_predict(hand, X_va) == sk.predict(X_va)).all()
    assert np.isfinite(unseen_word_logprob(hand)).all()


def test_fast_macro_f1_matches_sklearn(data):
    _, va = data
    y = va["product"].to_numpy()
    pred = np.roll(y, 3)
    k, t, p = _codes(y, pred)
    assert _macro_f1(t, p, k) == pytest.approx(f1_score(y, pred, average="macro"))


def test_bootstrap_ci_contains_point_estimate(data):
    _, va = data
    y = va["product"]
    pred = np.where(np.arange(len(y)) % 4 == 0, "Mortgage", y)
    ci = bootstrap_ci(y, pred, n_boot=200)
    assert ci["ci_low"] <= ci["macro_f1"] <= ci["ci_high"]
    same = paired_bootstrap(y, pred, pred, n_boot=200)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0


def test_summarize_mean_std():
    runs = pd.DataFrame({"model": ["a", "a", "b"], "macro_f1": [0.4, 0.6, 0.5],
                         "weighted_f1": [0, 0, 0], "accuracy": [0, 0, 0]})
    out = summarize(runs).set_index("model")
    assert out.loc["a", "macro_f1_mean"] == pytest.approx(0.5)
    assert out.loc["a", "macro_f1_std"] == pytest.approx(np.std([0.4, 0.6], ddof=1))
    assert out.loc["b", "macro_f1_std"] == 0.0 and out.loc["b", "n_runs"] == 1
