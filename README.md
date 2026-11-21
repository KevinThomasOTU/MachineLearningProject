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
| **Models** | Stratified Random baseline, Multinomial Naive Bayes, L2 Logistic Regression, k-NN (cosine) |
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
   k-NN with cosine similarity (Euclidean k-NN was dropped in M5: on unit-length TF-IDF rows it ranks neighbours
   exactly like cosine once all-zero rows are handled, see `m4_deliverables/code/m4_knn_empty_rows.py`).
5. **Tune and ablate (M5)** — every model gets the same equal-effort grid search on validation: 12 TF-IDF settings
   (stop words {none, English} × n-grams {(1,1), (1,2)} × max features {1,000, 5,000, 10,000}) × a per-model
   hyper-parameter grid (NB: alpha × fit_prior, LR: C × class_weight, k-NN: k × weights). Selection rule fixed in
   advance: best validation Macro F1.
6. **Evaluate** — the test split is scored **once**, after the selected configurations are frozen (lock file
   `m5_deliverables/results/logs/TEST_USED.lock`). Uncertainty: 95% bootstrap CIs, repeated grouped
   cross-validation on train+val, and Holm-corrected paired tests; plus ablation, robustness and error analysis.

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

    python scripts/04_run_baseline.py    # M3: naive benchmarks + Multinomial NB baseline (validation split)
    python scripts/07_discrepancy.py     # M3: expected-vs-observed analysis and significance tests
    python scripts/08_check_reproducible.py  # M3: re-runs 04 and checks the metrics are identical
    python scripts/09_macro_precision_recall.py  # M3: macro precision/recall columns for the memo table
    python scripts/05_run_ablations.py   # placeholder; the M5 ablation lives in m5_deliverables/ (below)

Cleaning takes about 95 s, the split about 1 min. All settings (seed, `min_class_count`, `sample_size`,
`one_row_per_group`, dedupe thresholds, split ratios, TF-IDF grid) are in [configs/config.yaml](configs/config.yaml).

## Reproducing the M3 baseline

Release `v0.3-m3`. Tested on Python 3.14.5, Windows 11, CPU only. Package versions are pinned in
`requirements.txt`. With the splits from `02_make_splits.py` in place (see above):

    pip install -r requirements.txt && pip install -e .
    make m3          # pytest, 04_run_baseline, 07_discrepancy, 08_check_reproducible

Without `make`, run `pytest -q`, then `scripts/04_run_baseline.py`, `scripts/07_discrepancy.py` and
`scripts/08_check_reproducible.py` in that order (about 2.5 minutes in total). Expected: 23 tests pass, tuned
Multinomial NB scores Macro F1 0.623 on validation, and the last script prints `PASS`.

| Output | File |
|---|---|
| Baseline table | `results/tables/baseline_table.csv` (per-seed rows in `baseline_runs.csv`; macro precision/recall in `baseline_macro_pr.csv`) |
| NB alpha sweep, per-class scores | `results/tables/nb_alpha_sweep.csv`, `per_class_nb.csv` |
| Discrepancy analysis | `results/tables/discrepancy.csv`, `significance.csv`, `nb_by_nearcopy.csv` |
| Confusion matrix | `results/figures/confusion_nb_val.png` |
| Run and environment logs | `results/logs/baseline_run.json`, `repro_check.json` |

All M3 scores are on the **validation** split. The test split is not used until M5. The memo is
[docs/M3_TECHNICAL_MEMO.md](docs/M3_TECHNICAL_MEMO.md).

## Reproducing the M4 and M5 results

M4 (initial models, validation only) is in `m4_deliverables/` (run `python m4_deliverables/code/m4_initial_models.py`).
M5 (final experiments) is in `m5_deliverables/`. Same setup and data as above, then from `cfpb-triage/`:

    make m5-tune       # equal-effort tuning -> best_configs.json, val predictions (about 45 min)
    make m5-analysis   # repeated CV, robustness, ablation, interpretation, significance, error analysis
    make m5-test       # ONE-SHOT test evaluation + test significance + main table (run once only)
    make m5-log        # render the experiment log and regenerate the draft Results section

Without `make`, each step is `python m5_deliverables/code/<script>.py` (see the Makefile for the order).
`m5_final_test.py` refuses to run without `--i-confirm-frozen` and refuses a second run. Every script appends to
`m5_deliverables/results/logs/experiment_log.jsonl`.

Headline M5 results (Macro F1, the primary metric; full tables in `m5_deliverables/results/tables/`):

| Model | Validation | 5-fold grouped CV (mean ± std) | Test (one-shot, 95% CI) |
|---|---|---|---|
| Majority class | 0.062 | 0.062 ± 0.000 | 0.062 |
| Stratified Random | 0.094 | 0.090 ± 0.002 | 0.092 |
| Multinomial NB | 0.648 | 0.646 ± 0.006 | 0.657 [0.642, 0.671] |
| k-NN (cosine) | 0.665 | 0.653 ± 0.006 | 0.667 [0.651, 0.682] |
| **L2 Logistic Regression** | **0.713** | **0.708 ± 0.016** | **0.703 [0.687, 0.717]** |

LR beats NB and k-NN on validation and test (Holm-adjusted p = 0.006); k-NN vs NB is not separable on test.
Vocabulary size and regularisation matter most; the best vocabulary (10,000 features) is at the edge of the grid.
Most errors are the credit-reporting / debt-collection boundary and a few small classes. The submission document
is [m5_deliverables/M5_Submission_Team32.pdf](m5_deliverables/M5_Submission_Team32.pdf) (draft Results section,
results package index, experiment log); its source is
[m5_deliverables/M5_RESULTS_DRAFT.md](m5_deliverables/M5_RESULTS_DRAFT.md).

| M5 output | Location |
|---|---|
| Tuning grid, frozen configs, search space | `m5_deliverables/results/tables/tuning_grid.csv`, `best_configs.json`, `search_space.json` |
| Main table and figure | `results/tables/m5_main_table.csv`, `results/figures/m5_main_figure.png` |
| Uncertainty | `cv_summary.csv`, `m5_significance_val.csv`, `m5_significance_test.csv` |
| Ablation and sensitivity | `ablation_tfidf_effects.csv`, `ablation_hyperparam_curves.csv`, `robust_*.csv`, `learning_curve_all.csv` |
| Error analysis, interpretation | `error_*.csv`, `top_terms_*.csv`, `theory_checks.json` |
| Test result | `test_results.csv`, `test_per_class.csv` |
| Experiment log | `results/logs/experiment_log.md` and `.jsonl` |

## Repository layout

    configs/config.yaml        all settings
    data/                      raw, interim, processed (git-ignored; only .gitkeep tracked)
    docs/                      data card, split strategy, milestone memos, proposal updates
    notebooks/                 EDA notebook
    reports/                   proposal, progress and final reports
    results/                   figures, tables, logs (cleaning_log.json is tracked)
    m4_deliverables/           M4 progress report, slides, initial-model code and results
    m5_deliverables/           M5 code (code/), results (results/), draft Results, submission PDF
    scripts/                   thin entry points (00_ … 06_) that call src/
    src/cfpb_triage/
      data/                    load, clean (grouping), split
      features/                TF-IDF wrapper (train-only fit)
      models/                  baseline, naive_bayes, logreg, knn (cosine, empty-row safe)
      analysis/                imbalance, EDA statistics, distances, smoothing, significance
      evaluation/              metrics, plots, variability over seeds
      experiments/             config-driven ablation runner
      utils/                   seeding
    tests/                     split/leakage, baseline and M5 model tests (27 tests)

## Status

| Milestone | Content | Status |
|---|---|---|
| Proposal | Problem, data, methods | Done (`reports/proposal`) |
| M2 | Data card, cleaning, group-aware splits, EDA, leakage tests, compute plan, approval memo | Done |
| M3 | Pipeline, naive benchmarks, Multinomial NB baseline, baseline table, discrepancy analysis, start/stop/continue | Done (`v0.3-m3`) |
| M4 | L2 Logistic Regression and k-NN next to the NB baseline, significance tests, progress report and presentation | Done (`m4_deliverables/`) |
| M5 | Equal-effort tuning, repeated CV and bootstrap uncertainty, one-shot test evaluation, ablation, robustness, error analysis, draft Results and limitations | Done (`m5_deliverables/`) |
| M6 | Final report, presentation, repository release, contribution and AI-use statements | Planned |
| M7 | Individual reflection and oral defence | Planned |

## Documentation

- [docs/DATA_CARD.md](docs/DATA_CARD.md) — source, license, fields, cleaning steps with row counts, biases, PII
- [docs/SPLIT_STRATEGY.md](docs/SPLIT_STRATEGY.md) — stratified group split, leakage rules, why Macro F1
- [docs/M2_APPROVAL_MEMO.md](docs/M2_APPROVAL_MEMO.md) — Milestone 2 memo
- [docs/M3_TECHNICAL_MEMO.md](docs/M3_TECHNICAL_MEMO.md) — Milestone 3 technical memo (baseline, discrepancy analysis, start/stop/continue)
- [docs/M3_DISCREPANCY_SECTION.md](docs/M3_DISCREPANCY_SECTION.md) — full discrepancy write-up behind memo section 3
- [m5_deliverables/M5_Submission_Team32.pdf](m5_deliverables/M5_Submission_Team32.pdf) — Milestone 5 submission (draft Results, results package index, experiment log)
- [m5_deliverables/M5_RESULTS_DRAFT.md](m5_deliverables/M5_RESULTS_DRAFT.md) — draft Results section (source of the PDF; numbers filled from the result tables)
- [docs/UPDATED_PROPOSAL.md](docs/UPDATED_PROPOSAL.md) — changes since the proposal
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
- The model signal partly comes from company names (e.g. experian, synchrony, mohela, coinbase among the top terms), a
  shortcut that may not transfer to other periods.
- The credit-reporting / debt-collection boundary is ambiguous, so part of the error is label noise rather than model weakness.
- Cross-validation scores are slightly optimistic: the configurations were chosen on validation rows that lie in the CV pool.
- Only TF-IDF features and three classical models were compared; the best vocabulary size is at the edge of the grid.
- Edited template variants that are not grouped can still make validation/test slightly easier; this is measured
in the EDA and reported.