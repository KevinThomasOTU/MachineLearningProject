# Split Strategy

## 1. Stratified group split
- 70 / 15 / 15 train / validation / test, seed 42 (`configs/config.yaml`).
- Implementation (`src/cfpb_triage/data/split.py`): `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)` on label = `product`, groups = `group_id`. Folds 0–2 → test, 3–5 → validation, 6–19 → train.
- Every `group_id` lies in exactly one split; class proportions per split match the full data (chi-square test and a ±2 pp tolerance test). Sizes: train 68,357, validation 14,649, test 14,649. The largest class-proportion difference from the full data is 0.01 percentage points (chi-square(20) = 0.07, p = 1.000).
- Classes need ≥ 20 groups to appear in every fold; `min_class_count = 200` removes classes too small to evaluate (none were removed: the smallest class, Debt or credit management, has 451 rows).

## 2. Dedupe rule (group_id) and one row per group
Implemented in `src/cfpb_triage/data/clean.py`:
1. identical normalised text (lowercase, `XXXX`/`XX/XX/XXXX` masks removed, whitespace collapsed) → same group;
2. near-duplicates: same Date received + State + ZIP + Issue **and** (same first 200 normalised characters **or** TF-IDF cosine ≥ 0.80) → same group;
3. connected components of these links form the groups;
4. each group is then collapsed to **one row** (majority label; `group_size` keeps the original count), so every row in the splits is a distinct complaint text.

About half of all narratives are exact copies (mostly credit-repair template letters), so without step 4 a single template could appear thousands of times in the test set. A spot-check of 20 merged groups found 0 wrong merges (`results/tables/spotcheck_groups.csv`).

**Residual similarity:** 22.0% of 2,000 sampled val/test complaints have a training complaint with TF-IDF cosine ≥ 0.8 (median best match 0.40). No group or identical text is shared across splits. These are edited variants of template letters from different consumers that fall outside the grouping blocks. They are not exact leakage, but they may make the task slightly easier than for entirely novel complaints; we report this as a limitation (`results/tables/cross_split_leakage.json`).

## 3. No fitting on validation or test
- TF-IDF vocabulary/IDF weights, NB/LR parameters and k-NN reference sets are fit on **train only**; val/test are only transformed.
- All EDA statistics that fit a model (top terms, residual-similarity vectorizer) are fit on train.

## 4. Validation for tuning, test touched once
- Hyper-parameters (TF-IDF configuration, NB alpha, LR C, k and distance for k-NN) are chosen by validation Macro F1.
- The test split is evaluated **once**, after all choices are frozen. Sample rows shown in EDA come from validation, not test.

## 5. Leakage tests (`tests/test_split.py`)
No shared Complaint IDs across splits; no `group_id` in two splits; class proportions within ±2 pp of the full data; same seed ⇒ identical splits (different seed ⇒ different); no null/blank narratives after cleaning; reworded 3-bureau copies share a group; one row per group when collapsing is on; TF-IDF vocabulary built from train only.

## 6. Optional temporal sanity check
The archive covers April–July 2024. As a robustness check (not the primary result), train on April–June and evaluate on July (same pipeline). A large Macro F1 drop relative to the random split would signal temporal drift in wording or product mix.

## 7. Why Macro F1
Credit reporting is 51.3% of complaints. Accuracy rewards predicting the majority class: a stratified-random guesser already reaches accuracy ≈ Σ p_k² = 0.31. Macro F1 averages per-class F1 with equal weight, so each product counts the same. Its chance level for a stratified-random predictor is exactly 1/K (derivation in `src/cfpb_triage/analysis/imbalance.py`) = 0.091 for our K = 11 classes, a clear floor for every model.
