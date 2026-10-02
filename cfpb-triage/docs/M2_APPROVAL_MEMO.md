# Milestone 2 Approval Memo: Team 32 (KASS Tech), CSCI 3052
**Project:** Product triage of CFPB consumer complaint narratives
**Prepared by:** Kevin Thomas (Coordinator / Data Lead), with Alex, Subangan, Sayon
**Date:** <FILL: YYYY-MM-DD>

## 1. Data source and license
CFPB Consumer Complaint Database, narratives archive file covering April–July 2024 (`data/raw/complaints.csv`, downloaded <FILL: date>). Public federal data; terms of use to be verified by <NAME>, <DATE>. Because CFPB stopped publishing narratives on Aug 14, 2026, this archive is a fixed, frozen snapshot. Details: `docs/DATA_CARD.md`.

## 2. Data card summary
- Raw rows: <FILL FROM RUN: n_rows_raw>; rows with a narrative: <FILL FROM RUN>; after cleaning: <FILL FROM RUN>; after keeping one row per duplicate group, the final modelling set is <FILL FROM RUN: n_rows_final> distinct complaints in <FILL FROM RUN: n_classes_final> classes.
- Label: `Product` (normalised). Input: `Consumer complaint narrative` (masked tokens removed).
- Dropped classes (< 200 rows): <FILL FROM RUN: dropped_classes>.

## 3. Sample inputs and outputs
| complaint_id | narrative (first 150 chars, normalised) | product (label) |
|---|---|---|
| <FILL> | <FILL> | <FILL> |
| <FILL> | <FILL> | <FILL> |
| <FILL> | <FILL> | <FILL> |

Model output (from M3 on): one predicted `Product` per narrative, plus per-class scores for NB/LR.

## 4. Initial EDA findings
- Class distribution: majority class <FILL FROM SUBANGAN: name> = <FILL>% of rows; minority class <FILL> = <FILL>%; imbalance ratio <FILL FROM SUBANGAN>; normalised entropy <FILL FROM SUBANGAN>. (Figure: `results/figures/<FILL>`)
- Narrative length: median <FILL FROM SUBANGAN/RUN> tokens; <FILL FROM RUN: lt_20_tokens> narratives under 20 tokens.
- Duplication: <FILL FROM RUN: dedupe.exact_duplicate_rows> narratives (<FILL>%) are exact copies of another narrative, mostly credit-repair template letters in credit reporting and debt collection; the largest template appears <FILL FROM RUN: dedupe.largest_group> times. <FILL FROM RUN: collapse.groups_with_label_conflict> identical texts were filed under more than one product (label noise). We keep one row per duplicate group.
- Narrative coverage: only <FILL FROM RUN>% of raw complaints include a narrative.

## 5. Target and task
Multiclass text classification: `Consumer complaint narrative` → `Product`. Models (M3+): stratified-random baseline, Multinomial Naive Bayes, L2 Logistic Regression, k-NN (cosine and Euclidean) on TF-IDF features, with an ablation over stop words, n-gram range and vocabulary size.

## 6. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Narratives discontinued Aug 14, 2026 | Frozen archive snapshot; documented; no refresh needed for the course |
| Near-duplicate leakage (3-bureau filings, template letters) | Exact + near-duplicate grouping; one row per group; group-aware splits; leakage tests |
| Collapsing templates changes the evaluated distribution | Report results as performance on distinct complaints; keep `group_size` for a weighted sensitivity check |
| Label noise (same text under different products) | Majority label per group; conflict count reported |
| Severe class imbalance | Macro F1 primary metric; stratified splits; per-class reporting |
| Rare classes too small to evaluate | min_class_count = 200 |
| k-NN compute cost | ~98k distinct complaints after collapsing; sparse TF-IDF; `sample_size` available if needed |
| License terms unverified | Verification assigned to <NAME> by <DATE> |

## 7. Train / validation / test strategy
After cleaning, each duplicate group is one row, so we have 97,655 distinct complaints. We split them 70/15/15 (seed 42) using scikit-learn's StratifiedGroupKFold with 20 folds. Each fold is stratified on Product and only holds whole duplicate groups. 14 folds go to train, 3 to validation and 3 to test. Since every `group_id` is in exactly one split, the same complaint (or a near-copy of it) can't show up in both training and evaluation. Split sizes: train 68,357, validation 14,649, test 14,649. No class's share differs from the full data by more than 0.01 percentage points (chi-square p = <FILL FROM SUBANGAN: split_chi2.json p_value>). TF-IDF and all model parameters are fit on train only, hyperparameters are tuned on validation, and test is used once for the final Macro F1. All 12 leakage tests in `tests/test_split.py` pass.

## 8. Primary metric
Macro F1 on the test set. It weights every product equally, so the dominant credit-reporting class cannot mask poor minority-class performance. Secondary: per-class F1, accuracy, weighted F1, confusion matrix.

## 9. Feasibility and compute plan
We timed one run of each step on a 10,000-row sample of the training set, using the most expensive setting (1–2-grams, 10,000 features), on a laptop with an Intel Core Ultra 5 125U (12 cores) and 15 GB RAM. Measured times: TF-IDF 2.9 s, Multinomial NB 0.2 s, logistic regression 2.4 s, and k-NN (cosine + Euclidean) predicting 2,000 validation rows 1.7 s. Scaled up to all 97,655 complaints, one run takes about 123 s, and the full ablation grid (12 TF-IDF settings × 3 models × 3 seeds = 36 runs) takes about 1.2 h. k-NN is the slowest part (about 70% of the time) because brute-force search grows with n_train × n_val. This is well within our budget, so we use all 97,655 distinct complaints with no sampling (`sample_size: null`; see `results/tables/compute_estimate.csv`). Everything runs on CPU with a fixed seed and can be reproduced with `make m2`.
