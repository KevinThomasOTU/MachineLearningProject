"""Load raw CFPB CSV (Kevin). Read only needed columns; return DataFrame."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

# Repo root: src/cfpb_triage/data/load.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]

RAW_COLUMNS = [
    "Complaint ID",
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "ZIP code",
]


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_raw(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """ Reads only `usecols` (default: RAW_COLUMNS) as strings so ZIP codes and
    IDs keep leading zeros. Raises FileNotFoundError / ValueError with a
    readable message if the file or a required column is missing.
    """
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python scripts/00_download_data.py` "
            "for download instructions."
        )
    usecols = list(usecols or RAW_COLUMNS)
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in usecols if c not in header]
    if missing:
        raise ValueError(f"Missing expected columns in {path.name}: {missing}")
    return pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
