"""Entry point: 03_eda_stats. Import from cfpb_triage and call; keep logic in src/.

Statistical EDA (Subangan). Needs cleaned.csv and the splits. Writes to results/tables/:
  imbalance_summary.json, class_distribution.csv, duplicate_rates.json,
  cross_split_leakage.json, split_proportions.csv, split_chi2.json,
  top_terms_per_class.csv, eda_interpretations.md
"""
import json

from cfpb_triage.analysis.eda_stats import (cross_split_leakage, duplicate_rates,
                                            interpretations, split_proportion_test,
                                            top_terms_per_class)
from cfpb_triage.analysis.imbalance import imbalance_report
from cfpb_triage.data.load import load_config, resolve
from cfpb_triage.data.split import load_splits, read_cleaned
from cfpb_triage.utils.seed import set_seed


def dump(obj, name: str) -> None:
    path = resolve("results/tables") / name
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"[03s] wrote {path.name}")


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    resolve("results/tables").mkdir(parents=True, exist_ok=True)
    df = read_cleaned()
    train, val, test = load_splits()

    imb, class_tab = imbalance_report(df["product"])
    class_tab.to_csv(resolve("results/tables/class_distribution.csv"), index=False)
    dump(imb, "imbalance_summary.json")

    dup = duplicate_rates(df)
    dump(dup, "duplicate_rates.json")

    leak = cross_split_leakage(train, val, test, seed=cfg["seed"])
    dump(leak, "cross_split_leakage.json")

    prop_tab, prop = split_proportion_test(train, val, test)
    prop_tab.to_csv(resolve("results/tables/split_proportions.csv"))
    dump(prop, "split_chi2.json")

    terms = top_terms_per_class(train)
    terms.to_csv(resolve("results/tables/top_terms_per_class.csv"), index=False)

    lines = interpretations(imb, dup, leak, prop, terms)
    md = "# EDA interpretations (auto-generated from run)\n\n" + "\n".join(f"- {l}" for l in lines)
    resolve("results/tables/eda_interpretations.md").write_text(md + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
