# Baseline Reproduction - Technical Memo

Team 32 (KASS Tech): Kevin Thomas, Alex Facey, Subangan Sundaralingam, Sayon Kirubaharan

**Date:** October 19, 2026

**Subject:** Baseline Reproduction - CFPB Consumer Complaint Triage (predict `Product` from the complaint narrative)

## 1. Repository Submission

- **Repository URL:** https://github.com/KevinThomasOTU/MachineLearningProject
- **Release Tag:** v0.3-m3 (https://github.com/KevinThomasOTU/MachineLearningProject/releases/tag/v0.3-m3)

## 2. Reproducible Environment & Repository Setup

The codebase for this assignment is tagged as v0.3-m3 in the main repository. Reproducibility is provided by a pinned `requirements.txt` (Python 3.14.5, scikit-learn 1.9.1) in the project folder. Users can recreate the environment with `pip install -r requirements.txt` and `pip install -e .`, then run `make m3` (or the four `python` commands in the README) to execute the tests, baseline, discrepancy analysis and reproducibility check in about 2.5 minutes on a laptop CPU. All stochastic steps (the split, model seeds, bootstrap) use seed 42 (`configs/config.yaml`). `scripts/08_check_reproducible.py` re-runs the baseline and confirms every metric is identical (maximum difference 0.0), and logs package versions and split-file hashes. 23 unit tests pass: 12 split-leakage tests from M2 and 11 new baseline tests.

## 3. Working Data Pipeline

The pipeline is modularized in `src/cfpb_triage/` and driven by the numbered scripts in `scripts/`. It executes the following steps on the raw dataset (CFPB complaints with narratives, April-July 2024):

1. **Ingestion & Cleaning:** Drops rows without a narrative (839,903 raw rows to 276,275), removes masked tokens such as `XXXX`, normalises labels, and groups exact and near-duplicate narratives (template letters, 3-bureau filings), keeping one row per group: 97,655 distinct complaints in 11 classes.
2. **Splitting:** Performs a stratified, group-aware 70/15/15 train/validation/test split (`StratifiedGroupKFold`) so no duplicate group appears in two splits. The class mix is preserved, from 51.3% credit reporting down to 0.46% for the smallest class. All scores in this memo are on the validation split (n = 14,649); the test split is sealed until M5.
3. **Preprocessing:** TF-IDF features (1-2-grams, 10,000 features, sublinear term frequency), fit on the training split only and only applied to validation.

## 4. Baseline Methodology

To establish a foundation for future model development, two baseline approaches were evaluated:

- **Naive Benchmark:** A Stratified Random classifier (`DummyClassifier(strategy="stratified")`) that draws labels from the training class mix and ignores the text, averaged over seeds 42-44. A Majority Class classifier (always "Credit reporting or other personal consumer reports", 51% of rows) is shown for reference.
- **Course-Level Baseline:** Multinomial Naive Bayes with Laplace smoothing on the TF-IDF features, the standard probabilistic text classifier. The smoothing parameter alpha was chosen from {0.001, 0.01, 0.03, 0.1, 0.3, 1} by validation Macro F1 (alpha = 0.03). Because the data is heavily imbalanced, Macro F1 (every product counts equally) is the primary metric.

## 5. Baseline Table

The models were evaluated on the validation split. Precision, recall and F1 are macro-averaged over the 11 classes. Macro F1 is mean ± std over seeds (Naive Bayes is deterministic, so its std is 0).

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---|---|
| Naive Benchmark (Stratified Random) | 0.306 | 0.092 | 0.092 | 0.092 ± 0.002 |
| Naive Benchmark (Majority Class) | 0.513 | 0.047 | 0.091 | 0.062 ± 0.000 |
| Multinomial NB, alpha = 1 (default) | 0.769 | 0.728 | 0.509 | 0.516 ± 0.000 |
| **Course-Level Baseline (Multinomial NB, alpha = 0.03)** | **0.787** | **0.776** | **0.602** | **0.623 ± 0.000** |

The tuned baseline beats Stratified Random by +0.530 Macro F1 (paired bootstrap 95% CI 0.515-0.544; its own 95% CI is 0.609-0.637). Alpha was tuned on the same validation split, so 0.623 is slightly optimistic until the test split is used in M5. The per-class confusion matrix is in `results/figures/confusion_nb_val.png`.

## 6. Discrepancy Analysis

We compared the results with what theory and our M2 plan predicted (`scripts/07_discrepancy.py`, `results/tables/discrepancy.csv`).

- **Match:** The benchmarks land on their closed-form values: Stratified Random scores Macro F1 0.092 against 1/11 = 0.091 expected and accuracy 0.306 against 0.305, and the Majority Class matches exactly (0.062, 0.513). A Naive Bayes written directly from the smoothing formulas gives identical predictions to scikit-learn on all validation rows. The share of validation complaints with a near-copy in training is 22.9%, against the 22% estimated in M2.
- **Discrepancy:** Rare classes are weak: recall is 0.03 for *Debt or credit management* (69 validation rows), 0.28 for *Payday loan*, 0.40 for *Money transfer* and 0.45 for *Prepaid card*, mostly predicted as *Checking or savings* or *Credit card*, because the class prior and small training vocabularies favour the large classes. The model also over-fits moderately (train Macro F1 0.705 vs validation 0.623), and tuning alpha added +0.107 over the default.
- **Leakage and weighting:** Complaints with a training near-copy are easier (accuracy 0.93 vs 0.75) but lift Macro F1 by only 0.03, so leakage does not explain the score. Weighting each complaint by its duplicate-group size lowers Macro F1 to 0.587, so the live mix (where templates repeat) is somewhat harder than distinct complaints.
- **Compute:** M2 over-estimated Naive Bayes fit time by about 8x (1.30 s expected vs 0.17 s measured), because timings on a small subsample include fixed overhead.

## 7. Start / Stop / Continue Plan

Reflecting on the workflow for this baseline phase, the following adjustments will be made for the next milestone:

- **Start:** Running L2 Logistic Regression and k-NN (cosine and Euclidean) through the same pipeline and table, then the 12-setting TF-IDF ablation over 3 seeds with bootstrap CIs. We will also test `fit_prior=False` in Naive Bayes to target the rare-class recall gap.
- **Stop:** Treating the validation score used for tuning as a final result (final numbers will come from one pass over the sealed test split), and extrapolating compute from 10,000-row samples (we will budget from measured full-size times instead).
- **Continue:** Group-aware splits with train-only fitting, the pinned environment with the automatic reproducibility check before each milestone, modular code in `src/` with thin scripts, and one reviewed pull request per person with AI use logged in `AI_USE.md`.
