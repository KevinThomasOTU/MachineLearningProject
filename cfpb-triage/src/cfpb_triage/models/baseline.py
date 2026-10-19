"""Naive benchmarks: Majority class and Stratified Random (Sayon)."""
from sklearn.dummy import DummyClassifier


def stratified_random(train_labels, seed: int) -> DummyClassifier:
    """Fit a DummyClassifier that samples labels from the TRAIN class distribution.

    It ignores the input text, so X passed to fit/predict can be any array of
    the right length (e.g. the narratives themselves).
    """
    clf = DummyClassifier(strategy="stratified", random_state=seed)
    clf.fit([[0]] * len(train_labels), list(train_labels))
    return clf


def majority_class(train_labels) -> DummyClassifier:
    """Always predict the most frequent TRAIN class (also ignores the text)."""
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[0]] * len(train_labels), list(train_labels))
    return clf