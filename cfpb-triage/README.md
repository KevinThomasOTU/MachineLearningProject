# CFPB Complaint Triage (Team 32, CSCI 3052)
Compare Multinomial NB, Logistic Regression and k-NN on imbalanced complaint text:
predict the `Product` of a CFPB complaint from its consumer narrative. Primary metric: Macro F1.

## Setup
    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    pip install -e .

## Data
Download the CFPB narratives archive (April–July 2024) manually and save it as
`data/raw/complaints.csv`. Run `python scripts/00_download_data.py` for instructions;
it checks the file and its columns. Raw and processed data are git-ignored.
CFPB stopped publishing narratives on Aug 14, 2026, so this archive is a frozen snapshot.

## Run Milestone 2
    make m2          # = make data splits eda test
Without make (e.g. Windows), run in order:
    python scripts/00_download_data.py   # check raw file
    python scripts/01_clean_data.py      # -> data/interim/cleaned.csv, results/logs/cleaning_log.json
    python scripts/02_make_splits.py     # -> data/processed/splits/{train,val,test}.csv
    python scripts/03_eda.py             # -> results/tables/, results/figures/
    pytest -q                            # leakage tests

All settings (seed, min_class_count, sample_size, dedupe, split ratios) are in `configs/config.yaml`.

## Docs
- `docs/DATA_CARD.md`: source, license, fields, cleaning and row counts, biases, PII
- `docs/SPLIT_STRATEGY.md`: group-aware stratified split
- `docs/M2_APPROVAL_MEMO.md`: Milestone 2 memo
- `docs/PROPOSAL_UPDATES.md`: changes since the proposal
- `AI_USE.md`: AI tool usage log
