"""Entry point: 03_eda. Import from cfpb_triage and call; keep logic in src/.
   EDA plots and sample rows (Sayon). Needs cleaned.csv and the splits.
  results/figures/class_distribution_log.png, narrative_length_hist.png,
                  narrative_length_by_class.png
  results/tables/narrative_length_by_class.csv, sample_io.csv
Sample rows come from the VALIDATION split (the test split stays untouched).
"""
from cfpb_triage.data.load import load_config, resolve
from cfpb_triage.data.split import load_splits, read_cleaned
from cfpb_triage.evaluation.plots import (plot_class_distribution, plot_length_box,
                                          plot_length_hist)
from cfpb_triage.models.baseline import stratified_random
from cfpb_triage.utils.seed import set_seed

N_SAMPLES = 5
MAX_CHARS = 300


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    figs, tabs = resolve("results/figures"), resolve("results/tables")
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    df = read_cleaned()
    df["n_tokens"] = df["norm_text"].str.split().str.len()
    plot_class_distribution(df["product"], figs / "class_distribution_log.png")
    plot_length_hist(df["n_tokens"], figs / "narrative_length_hist.png")
    plot_length_box(df, figs / "narrative_length_by_class.png")
    lengths = df.groupby("product")["n_tokens"].describe(percentiles=[0.25, 0.5, 0.75])
    lengths.round(1).to_csv(tabs / "narrative_length_by_class.csv")
    print("[03] figures ->", figs)

    train, val, _ = load_splits()
    baseline = stratified_random(train["product"], cfg["seed"])
    # one row per class (so minority classes appear), shuffled, first N_SAMPLES kept
    picks = (val.groupby("product").sample(1, random_state=cfg["seed"])
                .sample(frac=1, random_state=cfg["seed"]).head(N_SAMPLES))
    sample = picks[["complaint_id", "narrative", "product"]].copy()
    text = sample["narrative"].str.replace(r"\s+", " ", regex=True)
    sample["narrative"] = text.where(text.str.len() <= MAX_CHARS,
                                     text.str.slice(0, MAX_CHARS) + "…")
    sample = sample.rename(columns={"product": "true_product"})
    sample["stratified_random_pred"] = baseline.predict([[0]] * len(sample))
    sample.to_csv(tabs / "sample_io.csv", index=False)
    print(f"[03] {len(sample)} sample rows -> {tabs / 'sample_io.csv'} "
          "(narratives are CFPB-redacted with XXXX; check for residual PII before sharing)")


if __name__ == "__main__":
    main()
