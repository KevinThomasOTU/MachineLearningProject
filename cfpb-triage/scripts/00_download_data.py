"""Entry point: 00_download_data. Import from cfpb_triage and call; keep logic in src/.

We don't scrape anything. This just tells you where to get the archive file
and checks that it's in the right place with the columns we need.
"""
import sys

import pandas as pd

from cfpb_triage.data.load import RAW_COLUMNS, load_config, resolve

INSTRUCTIONS = """
The CFPB narratives archive (April - July 2024) needs to be at:
    {path}

To get it:
  1. Go to https://www.consumerfinance.gov/data-research/consumer-complaints/
     and find the narratives archive.
  2. Download the April - July 2024 file (unzip it if needed).
  3. Save it as {path}
  4. Write the download date and file link in docs/DATA_CARD.md.

CFPB stopped publishing narratives on Aug 14, 2026, so this archive is the
dataset we're stuck with. Don't commit it (data/raw is git-ignored).
"""


def main() -> int:
    cfg = load_config()
    path = resolve(cfg["data"]["raw_path"])
    if not path.exists():
        print(INSTRUCTIONS.format(path=path))
        print("[00] raw file not found, stopping.")
        return 1

    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in RAW_COLUMNS if c not in header]
    size_mb = path.stat().st_size / 1e6
    print(f"[00] found {path} ({size_mb:,.1f} MB, {len(header)} columns)")
    if missing:
        print(f"[00] missing columns: {missing}")
        return 1
    print("[00] all the columns we need are there:", ", ".join(RAW_COLUMNS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
