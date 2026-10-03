"""Statistical EDA: duplicate rates, split-proportion test, top terms (Subangan).

Every TF-IDF model here is fit on the TRAIN split only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer

from cfpb_triage.data.load import resolve


def duplicate_rates(df: pd.DataFrame,
                    log_path: str = "results/logs/cleaning_log.json") -> dict:
    """Duplicate rates before vs after grouping.

    cleaned.csv keeps one row per group, so the 'before' numbers come from
    group_size (rows each group stood for) and the cleaning log.
    """
    log = json.loads(resolve(log_path).read_text(encoding="utf-8"))
    n_before = int(df["group_size"].sum())
    exact = int(log["dedupe"]["exact_duplicate_rows"])
    redundant = n_before - len(df)
    multi = df["group_size"] > 1
    return {
        "n_rows_before_grouping": n_before,
        "n_rows_after_grouping": int(len(df)),
        "exact_dup_rows_norm_text": exact,
        "exact_dup_rate_norm_text": exact / n_before,
        "redundant_rows_after_grouping": redundant,
        "redundant_rate_after_grouping": redundant / n_before,
        "near_dup_only_rows": redundant - exact,
        "rows_in_multi_row_groups": int(df.loc[multi, "group_size"].sum()),
        "share_rows_in_multi_row_groups": float(df.loc[multi, "group_size"].sum() / n_before),
        "largest_group": int(df["group_size"].max()),
        "groups_with_label_conflict": int(log.get("collapse", {}).get("groups_with_label_conflict", 0)),
    }


def cross_split_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                        threshold: float = 0.8, n_probe: int = 2000,
                        seed: int = 42) -> dict:
    """Check duplicates across splits after grouping.

    groups/norm_text shared across splits should be 0. residual_near_dup_rate is
    the share of a random sample of val+test rows whose most similar TRAIN row
    (TF-IDF cosine, vectorizer fit on train) is >= threshold, i.e. near-duplicates
    the grouping rule did not catch (e.g. templates from different consumers).
    """
    ids = [set(p["group_id"]) for p in (train, val, test)]
    txt = [set(p["norm_text"]) for p in (train, val, test)]
    shared_groups = len(ids[0] & ids[1]) + len(ids[0] & ids[2]) + len(ids[1] & ids[2])
    shared_text = len(txt[0] & txt[1]) + len(txt[0] & txt[2]) + len(txt[1] & txt[2])

    evald = pd.concat([val, test])
    probe = evald.sample(min(n_probe, len(evald)), random_state=seed)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000)
    Xtr = vec.fit_transform(train["norm_text"])
    Xp = vec.transform(probe["norm_text"])
    best = np.zeros(Xp.shape[0])
    for start in range(0, Xtr.shape[0], 20000):  # chunk to bound memory
        chunk = (Xp @ Xtr[start:start + 20000].T).max(axis=1).toarray().ravel()
        best = np.maximum(best, chunk)
    return {
        "groups_shared_across_splits": shared_groups,
        "norm_text_shared_across_splits": shared_text,
        "residual_probe_size": int(len(probe)),
        "residual_threshold": threshold,
        "residual_near_dup_rate": float((best >= threshold).mean()),
        "residual_max_sim_median": float(np.median(best)),
    }


def split_proportion_test(train: pd.DataFrame, val: pd.DataFrame,
                          test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Chi-square test of independence between split and class.

    H0: class proportions are the same in train/val/test (hence equal to the
    full data). A large p-value means no evidence of a mismatch.
    """
    counts = pd.DataFrame({name: part["product"].value_counts()
                           for name, part in [("train", train), ("val", val), ("test", test)]}
                          ).fillna(0).astype(int)
    chi2, p, dof, expected = chi2_contingency(counts.to_numpy())
    props = counts / counts.sum()
    full = counts.sum(axis=1) / counts.to_numpy().sum()
    dev = props.sub(full, axis=0).abs()
    table = counts.add_suffix("_n").join(props.add_suffix("_prop"))
    table["full_prop"] = full
    table["max_abs_dev_pp"] = dev.max(axis=1) * 100
    stats = {"chi2": float(chi2), "dof": int(dof), "p_value": float(p),
             "max_abs_dev_pp": float(dev.to_numpy().max() * 100),
             "min_expected_count": float(expected.min())}
    return table.sort_values("full_prop", ascending=False), stats


def top_terms_per_class(train: pd.DataFrame, n_terms: int = 15) -> pd.DataFrame:
    """Top terms per class by mean TF-IDF weight (vectorizer fit on train only)."""
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=5,
                          max_features=20000, token_pattern=r"(?u)\b[a-z][a-z]+\b")
    X = vec.fit_transform(train["norm_text"])
    vocab = np.array(vec.get_feature_names_out())
    rows = []
    for label in sorted(train["product"].unique()):
        mask = (train["product"] == label).to_numpy()
        mean_w = np.asarray(X[mask].mean(axis=0)).ravel()
        for rank, j in enumerate(np.argsort(-mean_w)[:n_terms], start=1):
            rows.append({"product": label, "rank": rank, "term": vocab[j],
                         "mean_tfidf": float(mean_w[j])})
    return pd.DataFrame(rows)


def interpretations(imb: dict, dup: dict, leak: dict, prop: dict,
                    terms: pd.DataFrame) -> list[str]:
    """One plain-language sentence per result, with numbers filled in from the run."""
    top = (terms[terms["rank"] <= 5].groupby("product")["term"]
           .apply(", ".join).to_dict())
    lines = [
        f"Imbalance: {imb['majority_class']} is {imb['majority_share']:.1%} of rows and "
        f"{imb['minority_class']} only {imb['minority_share']:.2%} (ratio "
        f"{imb['imbalance_ratio']:.1f}:1; normalised entropy {imb['normalized_entropy']:.2f}). "
        f"A stratified-random guesser gets accuracy ~{imb['expected_accuracy_stratified']:.2f} "
        f"but Macro F1 only ~{imb['expected_macro_f1_stratified']:.3f} (=1/K), which is "
        f"why Macro F1 is the primary metric.",
        f"Duplicates: {dup['exact_dup_rate_norm_text']:.1%} of the {dup['n_rows_before_grouping']:,} "
        f"cleaned narratives are exact copies of another narrative (after masking); grouping "
        f"(exact + near-duplicate) removes {dup['redundant_rate_after_grouping']:.1%} of rows "
        f"({dup['near_dup_only_rows']:,} caught only by the near-duplicate rule), leaving "
        f"{dup['n_rows_after_grouping']:,} distinct complaints. The largest group is one template "
        f"letter repeated {dup['largest_group']:,} times; {dup['groups_with_label_conflict']} groups "
        f"had the same text under different products.",
        f"Leakage: {leak['groups_shared_across_splits']} groups and "
        f"{leak['norm_text_shared_across_splits']} identical normalised texts are shared "
        f"across splits. Residual near-duplicates: {leak['residual_near_dup_rate']:.1%} of "
        f"{leak['residual_probe_size']:,} sampled val/test rows have a train row with cosine "
        f">= {leak['residual_threshold']} (median best match {leak['residual_max_sim_median']:.2f}).",
        f"Split balance: chi-square({prop['dof']}) = {prop['chi2']:.2f}, p = {prop['p_value']:.3f}; "
        f"largest class-proportion deviation from the full data is "
        f"{prop['max_abs_dev_pp']:.2f} percentage points, so the splits preserve the class mix.",
        "Top terms (train only, mean TF-IDF): " +
        "; ".join(f"{k}: {v}" for k, v in top.items()),
    ]
    return lines
