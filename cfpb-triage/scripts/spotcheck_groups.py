"""Spot-check the near-duplicate groups (Kevin).

Runs the cleaning again in memory without collapsing the groups, then looks at
groups that aren't just exact copies (merged by the prefix or cosine rule) and
picks 20 to read: 10 random, and 10 where the texts are least alike (the most
likely wrong merges). Prints them and saves results/tables/spotcheck_groups.csv
with an empty `verdict` column to fill in.
It doesn't touch cleaned.csv or the cleaning log.
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
    """Smallest TF-IDF cosine between any two texts in a group."""
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
    print(f"\n{len(near):,} groups contain more than one distinct text (merged by prefix or cosine)")

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
    print(f"\nWrote {len(pick)} groups -> {out}  - fill in the 'verdict' column")


if __name__ == "__main__":
    main()
