# CFPB Complaint Triage

**Team 32 (KASS Tech), CSCI 3052 — Machine Learning**

Predict which financial **product** a consumer complaint is about (credit reporting, debt collection, mortgage, …)
from the text of the complaint, using the CFPB Consumer Complaint Database. We compare three classical models
on TF-IDF features against a chance-level baseline, on heavily imbalanced data, and study what drives their
differences.

| | |
|---|---|
| **Task** | Multiclass text classification |
| **Input** | `Consumer complaint narrative` |
| **Label** | `Product` (11 classes after cleaning) |
| **Primary metric** | Macro F1 (every product counts equally) |
| **Models** | Stratified Random baseline, Multinomial Naive Bayes, L2 Logistic Regression, k-NN (cosine and Euclidean) |
| **Features** | TF-IDF, fit on the training split only |

## Team

| Member | Role |
|---|---|
| Kevin Thomas | Project Coordinator, Data Lead & Documentation Lead |
| Alex Facey | Machine-Learning Modelling Lead |
| Subangan Sundaralingam | Mathematical Theory, Statistical Analysis & Model Interpretation Lead |
| Sayon Kirubaharan | Evaluation, Reproducibility & Communication Lead |

## Dataset

CFPB Consumer Complaint Database, **narratives archive, April–July 2024**
(`CCDB_Export_6_April_2024_through_July_2024`). CFPB stopped publishing narratives on Aug 14, 2026, so this
archive is a **frozen snapshot**: there is no newer narrative data to add.

What the raw file looks like, and what we do about it:

| Issue in the raw data | How we handle it |
|---|---|
| Most rows (~67%) have no narrative (opt-in) | Drop blank/null narratives |
| Masked tokens (`XXXX`, `XX/XX/XXXX`) | Removed from the model input (`norm_text`) |
| Template letters and 3-bureau filings: about half of all narratives are exact copies, more are lightly reworded | Group exact and near-duplicates (`group_id`), then keep **one row per group** with a `group_size` count |
| The same text filed under different products | Majority label per group; conflicts are counted and documented |
| Heavy class imbalance | Macro F1, stratified splits, per-class reporting |
| Some very short narratives | Reported in the data card |

Numbers from the real run: 839,903 raw rows → 276,275 usable narratives → **97,655 distinct complaints** in 11 classes.
Details, row counts per step and limitations are in [docs/DATA_CARD.md](docs/DATA_CARD.md).

The dataset is **not stored in git** (it is ~500 MB). Everyone gets the same file; check it with
`Get-FileHash data\raw\complaints.csv -Algorithm SHA256` against the hash in the data card.

## Method overview

1. **Clean** — drop empty narratives, normalise labels, strip masks, build `norm_text`, assign `group_id`
   (exact match, plus near-duplicates: same date/state/ZIP/issue and a matching prefix or TF-IDF cosine ≥ 0.80),
   collapse each group to one row.
2. **Split** — 70/15/15 train/validation/test, seed 42, **stratified by product and group-aware**
   (`StratifiedGroupKFold`), so no complaint and none of its near-copies appear in more than one split.
3. **Featurise** — TF-IDF fit on **train only**; validation and test are only transformed.
4. **Model** — Stratified Random baseline, Multinomial NB (alpha sweep), L2 Logistic Regression,
   k-NN with cosine and Euclidean distance.
5. **Ablate** — stop words {none, English} × n-grams {(1,1), (1,2)} × max features {1,000, 5,000, 10,000},
   repeated over several seeds.
6. **Evaluate** — tune on validation, touch the test set once; report Macro F1 (mean ± std over seeds),
   per-class precision/recall, confusion matrices, and paired significance tests between models.

Design rules: fixed seeds, no fitting on validation/test data, leakage tests, logic in `src/`, thin scripts.
See [docs/SPLIT_STRATEGY.md](docs/SPLIT_STRATEGY.md).

## Setup

Requires Python 3.10+.

    python -m venv .venv
    .venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

Put the dataset at `data/raw/complaints.csv`, then check it:

    python scripts/00_download_data.py

## Running the pipeline

With `make`:

    make m2          # data + splits + eda + tests + compute estimate

Without `make` (e.g. Windows), run in order:

    python scripts/00_download_data.py   # check the raw file and its columns
    python scripts/01_clean_data.py      # -> data/interim/cleaned.csv, results/logs/cleaning_log.json
    python scripts/02_make_splits.py     # -> data/processed/splits/{train,val,test}.csv
    python scripts/03_eda_stats.py       # statistical EDA -> results/tables/
    python scripts/03_eda.py             # plots and sample rows -> results/figures/, results/tables/
    pytest -q                            # leakage and reproducibility tests
    python scripts/06_compute_estimate.py  # timing probe for the ablation grid

Modelling milestones add:

    python scripts/04_run_baseline.py    # baseline and the three models
    python scripts/05_run_ablations.py   # TF-IDF ablation grid over seeds

Cleaning takes about 95 s, the split about 1 min. All settings (seed, `min_class_count`, `sample_size`,
`one_row_per_group`, dedupe thresholds, split ratios, TF-IDF grid) are in [configs/config.yaml](configs/config.yaml).

## Repository layout

    configs/config.yaml        all settings
    data/                      raw, interim, processed (git-ignored; only .gitkeep tracked)
    docs/                      data card, split strategy, milestone memos, proposal updates
    notebooks/                 EDA notebook
    reports/                   proposal, progress and final reports
    results/                   figures, tables, logs (cleaning_log.json is tracked)
    scripts/                   thin entry points (00_ … 06_) that call src/
    src/cfpb_triage/
      data/                    load, clean (grouping), split
      features/                TF-IDF wrapper (train-only fit)
      models/                  baseline, naive_bayes, logreg, knn
      analysis/                imbalance, EDA statistics, distances, smoothing, significance
      evaluation/              metrics, plots, variability over seeds
      experiments/             config-driven ablation runner
      utils/                   seeding
    tests/                     split/leakage tests

## Status

| Milestone | Content | Status |
|---|---|---|
| Proposal | Problem, data, methods | Done (`reports/proposal`) |
| M2 | Data card, cleaning, group-aware splits, EDA, leakage tests, compute plan, approval memo | In progress |
| Modelling | Baseline, NB, LR, k-NN, TF-IDF ablations, significance tests | Planned (stubs in `src/`) |
| Progress / Final reports | Results, interpretation, contributions | Planned |

## Documentation

- [docs/DATA_CARD.md](docs/DATA_CARD.md) — source, license, fields, cleaning steps with row counts, biases, PII
- [docs/SPLIT_STRATEGY.md](docs/SPLIT_STRATEGY.md) — stratified group split, leakage rules, why Macro F1
- [docs/M2_APPROVAL_MEMO.md](docs/M2_APPROVAL_MEMO.md) — Milestone 2 memo
- [docs/PROPOSAL_UPDATES.md](docs/PROPOSAL_UPDATES.md) — changes since the proposal
- [AI_USE.md](AI_USE.md) — log of AI tool use
- [CONTRIBUTIONS.md](CONTRIBUTIONS.md) — who did what

## Contributing

One branch per person per task (e.g. `kevin/m2-data-cleaning`), small commits, and a pull request into `master`
that is **reviewed by someone other than its author**. Do not commit data files. Log AI tool use in `AI_USE.md`.

## Limitations

- Results describe performance on **distinct** complaint texts, not the live mix where template letters repeat.
- Only complaints with an opt-in narrative are included, which may differ from all complainants.
- Four-month window; no data after July 2024, so no drift analysis is possible.
- Some label noise: identical text filed under different products.
- Edited template variants that are not grouped can still make validation/test slightly easier; this is measured
in the EDA and reported.