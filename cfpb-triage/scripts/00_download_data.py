"""Entry point: 00_download_data. Import from cfpb_triage and call; keep logic in src/.

Does NOT download or scrape anything. Prints how to obtain the frozen CFPB
narratives archive file and checks that it is in place with the expected columns.
"""
import sys

import pandas as pd

from cfpb_triage.data.load import RAW_COLUMNS, load_config, resolve

INSTRUCTIONS = """
CFPB narratives archive (April 2024 - July 2024) is required at:
    {path}

How to obtain it (manual, no scraping):
  1. Open https://www.consumerfinance.gov/data-research/consumer-complaints/
     and go to the Consumer Complaint Database narratives archive.
  2. Download the April 2024 - July 2024 CSV file (unzip if needed).
  3. Save it as  {path}
  4. Record the download date and the exact file name/link in docs/DATA_CARD.md.
Note: CFPB stopped publishing new narratives on Aug 14, 2026, so this archive
snapshot is the project's frozen dataset. Do not commit it (data/raw is git-ignored).
"""


def main() -> int:
    cfg = load_config()
    path = resolve(cfg["data"]["raw_path"])
    if not path.exists():
        print(INSTRUCTIONS.format(path=path))
        print("[00] MISSING raw file -> stopping.")
        return 1

    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in RAW_COLUMNS if c not in header]
    size_mb = path.stat().st_size / 1e6
    print(f"[00] found {path} ({size_mb:,.1f} MB, {len(header)} columns)")
    if missing:
        print(f"[00] ERROR: missing expected columns: {missing}")
        return 1
    print("[00] all required columns present:", ", ".join(RAW_COLUMNS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
