"""Entry point: 02_make_splits. Import from cfpb_triage and call; keep logic in src/.

data/interim/cleaned.csv -> data/processed/splits/{train,val,test}.csv
"""
from cfpb_triage.data.load import load_config
from cfpb_triage.data.split import make_splits, read_cleaned, save_splits
from cfpb_triage.utils.seed import set_seed


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    df = read_cleaned()
    print(f"[02] {len(df):,} cleaned rows, {df['group_id'].nunique():,} groups")
    train, val, test = make_splits(df, cfg)
    save_splits(train, val, test)


if __name__ == "__main__":
    main()
