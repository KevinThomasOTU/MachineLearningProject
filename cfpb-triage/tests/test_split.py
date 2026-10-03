"""M2 leakage tests (Sayon).
Synthetic tests always run: a small raw frame that mimics the CFPB schema
(masked tokens, blank narratives, reworded 3-bureau duplicates) goes through
clean() and make_splits(). Real-data tests run only if data/processed/splits exists.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from cfpb_triage.data.clean import clean
from cfpb_triage.data.load import RAW_COLUMNS, resolve
from cfpb_triage.data.split import make_splits

TOL = 0.02  # max absolute difference in class proportion vs full data
BUREAUS = ["EQUIFAX, INC.", "Experian Information Solutions Inc.",
           "TRANSUNION INTERMEDIATE HOLDINGS, INC."]
PRODUCTS = {"Credit reporting or other personal consumer reports": 0.7,
            "Debt collection": 0.2, "Credit card": 0.1}
VOCAB = ("account report dispute late payment bank card fee charge loan collector "
         "identity theft fraud bureau balance interest called refused letter").split()


def make_raw(n_consumers: int = 900, seed: int = 0) -> pd.DataFrame:
    """Synthetic raw CFPB frame; ~1/3 of credit-reporting consumers file 3x."""
    rng = np.random.default_rng(seed)
    rows, cid = [], 1000
    for _ in range(n_consumers):
        prod = rng.choice(list(PRODUCTS), p=list(PRODUCTS.values()))
        body = " ".join(rng.choice(VOCAB, size=rng.integers(20, 60)))
        text = f"On XX/XX/XXXX I contacted XXXX about account XXXX1234. {body}"
        triple = prod.startswith("Credit reporting") and rng.random() < 0.33
        base = {"Date received": f"2024-05-{rng.integers(1, 29):02d}", "Product": prod,
                "Sub-product": "", "Issue": "Incorrect information on your report",
                "State": rng.choice(["CA", "TX", "NY"]), "ZIP code": f"{rng.integers(100, 999)}XX"}
        for k, comp in enumerate(BUREAUS if triple else [f"BANK {rng.integers(20)}"]):
            narr = text if k == 0 else text + " please correct this"
            rows.append({**base, "Company": comp, "Complaint ID": str(cid),
                         "Consumer complaint narrative": narr})
            cid += 1
    df = pd.DataFrame(rows)
    df.loc[df.index % 17 == 0, "Consumer complaint narrative"] = np.nan  # missing
    df.loc[df.index % 29 == 0, "Consumer complaint narrative"] = "   "    # blank
    return df[RAW_COLUMNS]


def make_cfg(seed: int = 42) -> dict:
    return {"seed": seed,
            "data": {"label_col": "Product", "text_col": "Consumer complaint narrative",
                     "min_class_count": 20, "sample_size": None},
            "split": {"train": 0.70, "val": 0.15, "test": 0.15}}


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory):
    log = tmp_path_factory.mktemp("logs") / "cleaning_log.json"
    return clean(make_raw(), make_cfg(), log_path=log)


@pytest.fixture(scope="module")
def splits(cleaned):
    return make_splits(cleaned, make_cfg())


def test_no_null_narratives_after_cleaning(cleaned):
    assert cleaned["narrative"].notna().all()
    assert cleaned["norm_text"].str.strip().ne("").all()
    assert not cleaned["norm_text"].str.contains(r"\bxxxx|xx/xx", regex=True).any()


def test_bureau_triplicates_share_group(cleaned):
    by_body = cleaned.assign(key=cleaned["norm_text"].str.replace(" please correct this", ""))
    assert (by_body.groupby("key")["group_id"].nunique() == 1).all()
    assert (cleaned.groupby("group_id").size() == 3).any()


def test_no_shared_ids(splits):
    ids = [set(s["complaint_id"]) for s in splits]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_splits_cover_all_rows(cleaned, splits):
    assert sum(len(s) for s in splits) == len(cleaned)


def test_groups_never_cross_splits(splits):
    gids = [set(s["group_id"]) for s in splits]
    assert not (gids[0] & gids[1] or gids[0] & gids[2] or gids[1] & gids[2])


def test_class_proportions_within_tolerance(cleaned, splits):
    full = cleaned["product"].value_counts(normalize=True)
    for part in splits:
        prop = part["product"].value_counts(normalize=True).reindex(full.index, fill_value=0)
        assert (prop - full).abs().max() <= TOL


def test_same_seed_identical_splits(cleaned, splits):
    again = make_splits(cleaned, make_cfg())
    for a, b in zip(splits, again):
        assert a["complaint_id"].tolist() == b["complaint_id"].tolist()


def test_different_seed_changes_splits(cleaned, splits):
    other = make_splits(cleaned, make_cfg(seed=7))
    assert set(other[2]["complaint_id"]) != set(splits[2]["complaint_id"])


def test_tfidf_fit_on_train_only(splits):
    """A term that appears only in the test split must not be in the vocabulary."""
    train, _, test = (s.copy() for s in splits)
    test.loc[test.index[0], "norm_text"] += " zzqtestonlyterm"
    vec = TfidfVectorizer().fit(train["norm_text"])
    assert "zzqtestonlyterm" not in vec.vocabulary_
    assert vec.transform(test["norm_text"]).shape[1] == len(vec.vocabulary_)


def test_one_row_per_group(tmp_path):
    """With one_row_per_group on, each group_id appears once and group_size adds up."""
    raw = make_raw()
    cfg = make_cfg()
    full = clean(raw, cfg, log_path=tmp_path / "a.json")
    cfg["data"]["one_row_per_group"] = True
    collapsed = clean(raw, cfg, log_path=tmp_path / "b.json")
    assert collapsed["group_id"].is_unique
    assert collapsed["group_size"].sum() == len(full)
    assert len(collapsed) == full["group_id"].nunique()


# ---------------------------------------------------------------- real data (optional)
SPLIT_DIR = resolve("data/processed/splits")
real = pytest.mark.skipif(not (SPLIT_DIR / "test.csv").exists(),
                          reason="real splits not built (run make splits)")


@real
def test_real_splits_disjoint():
    parts = [pd.read_csv(SPLIT_DIR / f"{n}.csv", dtype={"complaint_id": str})
             for n in ("train", "val", "test")]
    for col in ("complaint_id", "group_id"):
        sets = [set(p[col]) for p in parts]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), col


@real
def test_real_class_proportions():
    parts = [pd.read_csv(SPLIT_DIR / f"{n}.csv") for n in ("train", "val", "test")]
    full = pd.concat(parts)["product"].value_counts(normalize=True)
    for part in parts:
        prop = part["product"].value_counts(normalize=True).reindex(full.index, fill_value=0)
        assert (prop - full).abs().max() <= TOL
