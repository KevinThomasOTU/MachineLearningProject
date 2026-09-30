"""Spot-check near-duplicate groups (Kevin, M2 step 4).

Re-runs cleaning in memory WITHOUT collapsing groups, then picks groups whose
rows are not all identical (i.e. joined by the prefix or cosine rule):
  - 10 random ones, and
  - 10 with the lowest text similarity (the riskiest merges).
Prints them and writes results/tables/spotcheck_groups.csv with an empty
`verdict` column to fill in (same_complaint / wrong_merge).
Does not touch data/interim/cleaned.csv or results/logs/cleaning_log.json.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from cfpb_triage.data.clean import clean
from cfpb_triage.data.load import load_config, load_raw, resolve

N_RANDOM, N_LOWEST, SHOW_CHARS = 10, 10, 250


def min_similarity(texts: list[str]) -> float:
    """Lowest pairwise TF-IDF cosine among a group's distinct texts."""
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    sim = (X @ X.T).toarray()
    return float(sim[np.triu_indices(len(texts), k=1)].min())


def main() -> None:
    cfg = load_config()
    cfg["data"]["one_row_per_group"] = False
    cfg["data"]["sample_size"] = None
    with tempfile.TemporaryDirectory() as tmp:
        df = clean(load_raw(cfg["data"]["raw_path"]), cfg, log_path=Path(tmp) / "log.json")

    distinct = df.groupby("group_id")["norm_text"].nunique()
    near = distinct[distinct > 1].index
    print(f"\n{len(near):,} groups contain more than one distinct text (joined by prefix/cosine)")

    rows = []
    for gid in near:
        g = df[df["group_id"] == gid]
        texts = g["norm_text"].drop_duplicates().head(10).tolist()
        rows.append({"group_id": gid, "n_rows": len(g), "n_distinct_texts": g["norm_text"].nunique(),
                     "min_similarity": round(min_similarity(texts), 3),
                     "companies": "; ".join(g["company"].dropna().unique()[:3]),
                     "dates": "; ".join(g["date_received"].dropna().unique()[:3]),
                     "state_zip": f"{g['state'].iloc[0]} {g['zip'].iloc[0]}",
                     "products": "; ".join(g["product"].unique()),
                     "text_a": texts[0][:SHOW_CHARS], "text_b": texts[1][:SHOW_CHARS],
                     "verdict": ""})
    summary = pd.DataFrame(rows)

    rng = np.random.default_rng(cfg["seed"])
    lowest = summary.nsmallest(N_LOWEST, "min_similarity")
    rest = summary.drop(lowest.index)
    random = rest.iloc[rng.choice(len(rest), size=min(N_RANDOM, len(rest)), replace=False)]
    pick = pd.concat([lowest.assign(why="lowest similarity"), random.assign(why="random")])

    out = resolve("results/tables/spotcheck_groups.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pick.to_csv(out, index=False)

    for i, r in enumerate(pick.itertuples(), 1):
        print(f"\n=== {i}/{len(pick)}  group {r.group_id}  ({r.why}, min sim {r.min_similarity}) ===")
        print(f"rows={r.n_rows}  distinct texts={r.n_distinct_texts}  {r.state_zip}  dates: {r.dates}")
        print(f"companies: {r.companies}\nproducts:  {r.products}")
        print(f"  A: {r.text_a}")
        print(f"  B: {r.text_b}")
    print(f"\nWrote {len(pick)} groups -> {out}  (fill the 'verdict' column)")


if __name__ == "__main__":
    main()
