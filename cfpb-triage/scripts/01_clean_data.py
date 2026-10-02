"""Entry point: 01_clean_data. Import from cfpb_triage and call; keep logic in src/.

Reads the raw csv, cleans it, and writes data/interim/cleaned.csv
(plus the row-count log in results/logs/cleaning_log.json).
"""
import time

from cfpb_triage.data.clean import clean
from cfpb_triage.data.load import load_config, load_raw, resolve

OUT_PATH = "data/interim/cleaned.csv"


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    raw = load_raw(cfg["data"]["raw_path"])
    print(f"[01] loaded {len(raw):,} raw rows in {time.time() - t0:.1f}s")

    df = clean(raw, cfg)

    out = resolve(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[01] wrote {len(df):,} rows to {OUT_PATH} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
