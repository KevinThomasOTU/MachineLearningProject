"""TF-IDF wrapper; fit on train only (Alex)."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(stop_words=None, ngram_range=(1, 1), max_features=None,
                     sublinear_tf: bool = True) -> TfidfVectorizer:
    """One TF-IDF setting from the ablation grid.

    norm_text is already lower-cased with masked tokens removed, so there is no
    extra preprocessing here. ngram_range may arrive from the yaml as a list.
    """
    return TfidfVectorizer(stop_words=stop_words, ngram_range=tuple(ngram_range),
                           max_features=max_features, sublinear_tf=sublinear_tf)


def fit_transform(train_text, *other_texts, **params):
    """Fit the vocabulary and IDF on TRAIN text only, then transform the others.

    Returns (vectorizer, X_train, *X_others), e.g.
        vec, Xtr, Xva = fit_transform(train["norm_text"], val["norm_text"], **cfg)
    """
    vec = build_vectorizer(**params)
    X_train = vec.fit_transform(train_text)
    return (vec, X_train, *(vec.transform(t) for t in other_texts))
