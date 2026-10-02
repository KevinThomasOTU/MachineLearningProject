"""Load the raw CFPB CSV (Kevin). Only reads the columns we need."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

# repo root, two levels above src/cfpb_triage
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
    """Read the yaml config (relative paths are taken from the repo root)."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Turn a path from the config into one relative to the repo root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_raw(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load the raw complaints csv.

    Everything is read as a string so ZIP codes keep their leading zeros.
    """
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Couldn't find the raw data at {path}. "
            "Run scripts/00_download_data.py for instructions."
        )
    usecols = list(usecols or RAW_COLUMNS)
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in usecols if c not in header]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
