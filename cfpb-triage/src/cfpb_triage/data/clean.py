"""Clean the raw complaints: drop empty narratives, fix labels, find duplicates (Kevin).

What happens, in order (row counts are printed and saved to results/logs/cleaning_log.json):
  1. drop rows with no narrative or no label
  2. tidy the Product labels and print the class map
  3. build norm_text: lowercase, remove the XXXX / XX/XX/XXXX masks, squash whitespace
  4. drop rows that are empty after that, and repeated Complaint IDs
  5. give every row a group_id (exact copies + near-duplicates, see assign_groups)
  6. if one_row_per_group is on, keep one row per group (majority label, and
     group_size says how many rows it replaced)
  7. drop classes smaller than min_class_count
  8. optionally sample down to sample_size
If one_row_per_group is off, the duplicates stay and the splitter keeps each
group together instead.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from cfpb_triage.data.load import resolve

OUTPUT_COLUMNS = [
    "complaint_id", "date_received", "product", "narrative", "norm_text",
    "group_id", "group_size", "state", "zip", "company",
]

# old CFPB product names mapped to the current ones (only matters if they show up)
LEGACY_PRODUCT_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports":
        "Credit reporting or other personal consumer reports",
    "Credit reporting": "Credit reporting or other personal consumer reports",
    "Payday loan, title loan, or personal loan":
        "Payday loan, title loan, personal loan, or advance loan",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Bank account or service": "Checking or savings account",
}

# masked dates like xx/xx/xxxx or xx/xx/2024 (needs at least one x, so real dates stay)
_DATE_MASK = re.compile(r"\b(?=[x\d/]*x)[x\d]{1,2}/[x\d]{1,2}/[x\d]{2,4}\b")
# masked words/numbers like xx, xxxx, xxxx1234 (leaves words like "exxon" alone)
_X_MASK = re.compile(r"\b(?:x{2,}\d*|\d+x{2,})\b")
_WS = re.compile(r"\s+")

DEFAULT_DEDUPE = {"block_on": ["date_received", "state", "zip", "issue"],
                  "prefix_chars": 200, "cosine_threshold": 0.80, "max_block_size": 2000}


# ----------------------------------------------------------------------------- text
def normalize_text(text: str) -> str:
    """Lowercase, strip the CFPB masks, squash whitespace."""
    t = str(text).lower()
    t = _DATE_MASK.sub(" ", t)
    t = _X_MASK.sub(" ", t)
    return _WS.sub(" ", t).strip()


def normalize_labels(products: pd.Series) -> tuple[pd.Series, dict]:
    """Tidy whitespace and merge old product names.

    Returns the cleaned labels and a {raw label: cleaned label} map for printing.
    """
    stripped = products.astype(str).str.replace(_WS, " ", regex=True).str.strip()
    normed = stripped.replace(LEGACY_PRODUCT_MAP)
    class_map = {raw: normed_val for raw, normed_val in
                 sorted(set(zip(products.astype(str), normed)))}
    return normed, class_map


# ----------------------------------------------------------------------------- grouping
class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        p = self.parent
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:  # point everything on the path at the root
            p[i], i = root, p[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def union_many(self, idx) -> None:
        idx = list(idx)
        for j in idx[1:]:
            self.union(idx[0], j)


def assign_groups(df: pd.DataFrame,
                  block_on: tuple[str, ...] = ("date_received", "state", "zip", "issue"),
                  prefix_chars: int = 200, cosine_threshold: float = 0.80,
                  max_block_size: int = 2000) -> tuple[pd.Series, dict]:
    """Give exact and near-duplicate narratives the same group_id.

    Comparing every pair of rows would be far too slow, so:
      A. rows with identical norm_text go in the same group.
      B. for near-duplicates, rows are first split into blocks by `block_on`
         (date, state, zip, issue). Inside a block two rows are joined if
           - the first `prefix_chars` characters of norm_text match, or
           - their TF-IDF cosine similarity is at least cosine_threshold.
         Blocks bigger than max_block_size only use the prefix rule.
    Rows linked this way (directly or through other rows) end up in one group,
    and group_id is the smallest complaint_id in it, so it doesn't depend on row order.

    Needs complaint_id, norm_text and every column in block_on.
    """
    n = len(df)
    uf = _UnionFind(n)
    pos = np.arange(n)
    stats = {"exact_links": 0, "prefix_links": 0, "cosine_links": 0,
             "blocks_checked": 0, "blocks_prefix_only": 0}

    # A. exact copies
    for idx in pd.Series(pos).groupby(df["norm_text"].to_numpy()).groups.values():
        if len(idx) > 1:
            uf.union_many(idx)
            stats["exact_links"] += len(idx) - 1

    # B. near-duplicates, only compared inside a block
    block_key = df[block_on[0]].fillna("").astype(str)
    for col in block_on[1:]:
        block_key = block_key + "|" + df[col].fillna("").astype(str)
    # B1. same block and same start of the text
    prefix_key = block_key + "||" + df["norm_text"].str[:prefix_chars]
    for idx in pd.Series(pos).groupby(prefix_key.to_numpy()).groups.values():
        if len(idx) > 1:
            roots = {uf.find(i) for i in idx}
            uf.union_many(idx)
            stats["prefix_links"] += len(roots) - 1

    # B2. cosine similarity within each block (X has the blocks one after another)
    blocks = [np.asarray(b) for b in
              pd.Series(pos).groupby(block_key.to_numpy()).groups.values() if len(b) > 1]
    stats["blocks_checked"] = len(blocks)
    small = [b for b in blocks if len(b) <= max_block_size]
    stats["blocks_prefix_only"] = len(blocks) - len(small)
    if small:
        order = np.concatenate(small)
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                              max_features=2**18, dtype=np.float32)
        X = vec.fit_transform(df["norm_text"].to_numpy()[order]).tocsr()  # rows are unit length
        start = 0
        for b in small:
            Xb = X[start:start + len(b)]
            start += len(b)
            sim = (Xb @ Xb.T).tocoo()
            mask = (sim.row < sim.col) & (sim.data >= cosine_threshold)
            for i, j in zip(b[sim.row[mask]], b[sim.col[mask]]):
                if uf.find(i) != uf.find(j):
                    uf.union(i, j)
                    stats["cosine_links"] += 1

    return _finalize_groups(df, uf), stats


def _finalize_groups(df: pd.DataFrame, uf: _UnionFind) -> pd.Series:
    roots = np.array([uf.find(i) for i in range(len(df))])
    ids = pd.to_numeric(df["complaint_id"], errors="coerce").to_numpy()
    min_id = pd.Series(ids).groupby(roots).transform("min").to_numpy()
    return pd.Series(min_id.astype(np.int64), index=df.index, name="group_id")


def collapse_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep a single row for each group_id.

    The group gets its most common product as the label (ties go to the first
    one alphabetically so reruns match). The row we keep is the lowest
    Complaint ID with that label, and group_size remembers how big the group was.
    """
    counts = (df.groupby(["group_id", "product"]).size().rename("n").reset_index()
                .sort_values(["group_id", "n", "product"], ascending=[True, False, True]))
    majority = counts.drop_duplicates("group_id").set_index("group_id")["product"]
    labels_per_group = counts.groupby("group_id").size()
    conflict = labels_per_group[labels_per_group > 1].index

    rep = df[df["product"].to_numpy() == df["group_id"].map(majority).to_numpy()]
    rep = (rep.assign(_id=pd.to_numeric(rep["complaint_id"], errors="coerce"))
              .sort_values("_id").drop_duplicates("group_id").drop(columns="_id"))
    stats = {"groups_with_label_conflict": int(len(conflict)),
             "rows_in_label_conflict_groups": int(df["group_id"].isin(conflict).sum()),
             "rows_removed_by_collapse": int(len(df) - len(rep))}
    return rep, stats


# ----------------------------------------------------------------------------- sampling
def sample_by_group(df: pd.DataFrame, sample_size: int, min_class_count: int,
                    seed: int) -> pd.DataFrame:
    """Shrink the data to roughly `sample_size` rows, keeping whole groups.

    Every class keeps about the same share of its rows, but never drops below
    min_class_count, so sampling can't wipe out a small class. A group counts
    towards its most common label.
    """
    if sample_size is None or len(df) <= sample_size:
        return df
    rng = np.random.default_rng(seed)
    frac = sample_size / len(df)
    g = df.groupby("group_id").agg(size=("product", "size"),
                                   label=("product", lambda s: s.mode().iloc[0]))
    keep = []
    for label, gl in g.groupby("label"):
        n_c = int(gl["size"].sum())
        target = max(round(n_c * frac), min(n_c, min_class_count))
        order = gl.iloc[rng.permutation(len(gl))]
        csum = order["size"].cumsum()
        keep.extend(order.index[(csum - order["size"]) < target])
    return df[df["group_id"].isin(set(keep))]


# ----------------------------------------------------------------------------- main
def clean(df: pd.DataFrame, cfg: dict,
          log_path: str | Path = "results/logs/cleaning_log.json") -> pd.DataFrame:
    """Clean the raw frame from load_raw() and return OUTPUT_COLUMNS.

    Also saves the row count after each step to `log_path` as json.
    """
    dcfg = cfg["data"]
    text_col, label_col = dcfg["text_col"], dcfg["label_col"]
    dedupe_cfg = {**DEFAULT_DEDUPE, **(cfg.get("dedupe") or {})}
    log = {"steps": [], "config": {"data": dcfg, "dedupe": dedupe_cfg, "seed": cfg["seed"]}}

    def step(name: str, frame: pd.DataFrame, **extra) -> None:
        prev = log["steps"][-1]["rows"] if log["steps"] else len(frame)
        entry = {"step": name, "rows": int(len(frame)), "dropped": int(prev - len(frame)), **extra}
        log["steps"].append(entry)
        print(f"[clean] {name:<32} rows={entry['rows']:>9,}  dropped={entry['dropped']:>9,}")

    step("raw", df)

    text = df[text_col]
    df = df[text.notna() & text.astype(str).str.strip().ne("")]
    step("drop_null_blank_narrative", df)

    df = df[df[label_col].notna() & df[label_col].astype(str).str.strip().ne("")]
    step("drop_null_label", df)

    out = pd.DataFrame({
        "complaint_id": df["Complaint ID"].astype(str).str.strip(),
        "date_received": pd.to_datetime(df["Date received"], errors="coerce")
                           .dt.strftime("%Y-%m-%d"),
        "narrative": df[text_col].astype(str),
        "state": df["State"],
        "zip": df["ZIP code"],
        "company": df["Company"],
        "issue": df["Issue"],
    })
    out["product"], class_map = normalize_labels(df[label_col])
    log["class_map"] = class_map
    print("[clean] product class map (raw -> normalised):")
    for raw, new in class_map.items():
        print(f"    {raw!r} -> {new!r}")

    out["norm_text"] = out["narrative"].map(normalize_text)
    out = out[out["norm_text"].ne("")]
    step("drop_empty_after_masking", out)

    out = out.drop_duplicates("complaint_id", keep="first")
    step("drop_repeated_complaint_id", out)

    out = out.reset_index(drop=True)
    out["group_id"], group_stats = assign_groups(out, **dedupe_cfg)
    out["group_size"] = out.groupby("group_id")["group_id"].transform("size")
    sizes = out.groupby("group_id").size()
    dedupe_summary = {
        **group_stats,
        "n_groups": int(len(sizes)),
        "rows_in_multi_row_groups": int(sizes[sizes > 1].sum()),
        "n_multi_row_groups": int((sizes > 1).sum()),
        "largest_group": int(sizes.max()),
        "exact_duplicate_rows": int(out.duplicated("norm_text").sum()),
    }
    log["dedupe"] = dedupe_summary
    step("assign_group_id", out, **{k: dedupe_summary[k] for k in
                                    ("n_groups", "rows_in_multi_row_groups")})

    if dcfg.get("one_row_per_group", False):
        out, collapse_stats = collapse_groups(out)
        log["collapse"] = collapse_stats
        step("one_row_per_group", out, **collapse_stats)

    counts = out["product"].value_counts()
    rare = counts[counts < dcfg["min_class_count"]]
    log["dropped_classes"] = {k: int(v) for k, v in rare.items()}
    out = out[~out["product"].isin(rare.index)]
    step(f"min_class_count>={dcfg['min_class_count']}", out, classes_dropped=len(rare))

    out = sample_by_group(out, dcfg.get("sample_size"), dcfg["min_class_count"], cfg["seed"])
    step(f"sample_size={dcfg.get('sample_size')}", out)

    n_tokens = out["norm_text"].str.split().str.len()
    log["short_narratives"] = {"lt_5_tokens": int((n_tokens < 5).sum()),
                               "lt_20_tokens": int((n_tokens < 20).sum()),
                               "median_tokens": float(n_tokens.median())}
    log["final_class_counts"] = {k: int(v) for k, v in out["product"].value_counts().items()}
    log["n_rows_final"] = int(len(out))
    log["n_groups_final"] = int(out["group_id"].nunique())
    log["n_classes_final"] = int(out["product"].nunique())

    log_path = resolve(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[clean] final: {len(out):,} rows, {out['product'].nunique()} classes, "
          f"{out['group_id'].nunique():,} groups; log -> {log_path}")

    out = out.sort_values("complaint_id", key=lambda s: pd.to_numeric(s, errors="coerce"))
    return out.reset_index(drop=True)[OUTPUT_COLUMNS]
