# Milestone 2 Approval Memo: Team 32 (KASS Tech), CSCI 3052
**Project:** Product triage of CFPB consumer complaint narratives
**Prepared by:** Kevin Thomas (Coordinator / Data Lead), with Alex, Subangan, Sayon
**Date:** <FILL: YYYY-MM-DD>

## 1. Data source and license
CFPB Consumer Complaint Database, narratives archive file covering April–July 2024 (`data/raw/complaints.csv`, downloaded <FILL: date>). Public federal data; terms of use to be verified by <NAME>, <DATE>. Because CFPB stopped publishing narratives on Aug 14, 2026, this archive is a fixed, frozen snapshot. Details: `docs/DATA_CARD.md`.

## 2. Data card summary
- Raw rows: <FILL FROM RUN: n_rows_raw>; rows with a narrative: <FILL FROM RUN>; after cleaning and filtering: <FILL FROM RUN>; final modelling set: <FILL FROM RUN: n_rows_final> rows, <FILL FROM RUN: n_classes_final> classes, <FILL FROM RUN: n_groups_final> duplicate groups.
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
- Duplication: <FILL FROM RUN: rows_in_multi_row_groups> rows (<FILL>%) belong to multi-row groups, mostly same-day filings against the three credit bureaus.
- Narrative coverage: only <FILL FROM RUN>% of raw complaints include a narrative.

## 5. Target and task
Multiclass text classification: `Consumer complaint narrative` → `Product`. Models (M3+): stratified-random baseline, Multinomial Naive Bayes, L2 Logistic Regression, k-NN (cosine and Euclidean) on TF-IDF features, with an ablation over stop words, n-gram range and vocabulary size.

## 6. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Narratives discontinued Aug 14, 2026 | Frozen archive snapshot; documented; no refresh needed for the course |
| Near-duplicate leakage (3-bureau filings, reworded templates) | Exact + near-duplicate grouping; group-aware splits; leakage tests |
| Severe class imbalance | Macro F1 primary metric; stratified splits; per-class reporting |
| Rare classes too small to evaluate | min_class_count = 200 |
| k-NN compute cost | 100k group-aware stratified sample; sparse TF-IDF |
| License terms unverified | Verification assigned to <NAME> by <DATE> |

## 7. Train / validation / test strategy
<!-- Replace with Alex's paragraph (Section 2) once his numbers are in. -->
70/15/15, seed 42, stratified on Product, group-aware so no `group_id` appears in more than one split. Sizes: train <FILL FROM ALEX>, val <FILL FROM ALEX>, test <FILL FROM ALEX>. Largest per-class proportion difference vs full data: <FILL FROM ALEX/SAYON> percentage points. Test set is held out until final evaluation. Leakage tests: <FILL FROM SAYON: N passed / N total>.

## 8. Primary metric
Macro F1 on the test set. It weights every product equally, so the dominant credit-reporting class cannot mask poor minority-class performance. Secondary: per-class F1, accuracy, weighted F1, confusion matrix.

## 9. Feasibility and compute plan
<!-- Replace with Alex's paragraph (Section 2) once 06_compute_estimate.py has run on real data. -->
- Cleaning pipeline tested end-to-end on a synthetic ~330k-narrative file: ~70 s on a laptop. Real run time: <FILL FROM RUN: 01 total seconds> s.
- Modelling set: 100,000 rows × ≤ 10,000 TF-IDF features (sparse); NB/LR train in seconds to minutes; k-NN brute-force sparse search on ~70k training rows is feasible in batches.
- All steps are reproducible from `make m2` (download check → clean → split → EDA → tests), seed 42, with configuration in `configs/config.yaml`.
