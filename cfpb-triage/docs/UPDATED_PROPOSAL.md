# Updated Proposal: Automated Consumer Complaint Triage

**Team 32 (KASS Tech), CSCI 3052** · Kevin Thomas, Alex Facey, Subangan Sundaralingam, Sayon Kirubaharan  
Updates the M1 proposal (Sept. 18, 2026) · Milestone 2, Oct. 2, 2026

## 1. Summary

The research question is unchanged: how do Multinomial Naive Bayes, L2 Logistic Regression and k-NN (cosine and Euclidean) compare on TF-IDF features when classifying imbalanced complaint text, and how much do stop words, n-grams and vocabulary size matter? After inspecting the data, the task is now fixed: **predict the `Product` (11 classes) of a CFPB complaint from its `Consumer complaint narrative`**, scored mainly by **Macro F1**. The modelling set is **97,655 distinct complaints**, and the methods and timeline below are otherwise as in M1.

## 2. What changed since M1

| Topic | M1 | Now | Why |
|---|---|---|---|
| Data | CFPB Complaint Database | Narratives archive file, Apr–Jul 2024 (839,903 rows; 276,332 with a narrative) | CFPB stopped publishing narratives on Aug 14, 2026, so the archive is a frozen snapshot. Results are reproducible, but no newer data exists. |
| License | "public domain / permissive" | CFPB states the data is "freely available for anyone to use, analyze, and build on"; no formal license text found; we attribute the CFPB | Checked on 2026-10-02 (data card). |
| Data point / label | each complaint; product or issue | one row per group of exact or near-duplicate narratives (276,275 → 97,655); label is `Product` (11 classes, smallest 451 rows) | 50.6% of narratives are exact copies (template letters, or one complaint sent to all three bureaus). Left in, one template would count thousands of times and leak across splits. |
| Split | strict stratified | stratified **and group-aware**, 70/15/15, seed 42 (train 68,357 / val 14,649 / test 14,649) | No duplicate group can appear in two splits; 12 automated leakage tests pass. |
| Compute | not estimated | about 123 s per run, about 1.2 h for the full grid on a laptop CPU, no sampling needed | Measured on a 10,000-row subsample and scaled up. |

## 3. Problem and data (from the M2 EDA)

- **Input / label:** narrative text (masked tokens such as `XXXX` removed) → `Product`. Only 32.9% of raw complaints have a narrative (opt-in); the median narrative has 119 tokens and 5.4% have fewer than 20.
- **Imbalance:** credit reporting is 51.3% of rows and the smallest class 0.46% (111:1). A Stratified Random predictor gets accuracy ≈ 0.31 but Macro F1 only ≈ 1/K = **0.091**, which is why Macro F1 is the primary metric.

## 4. Methods (unchanged from M1)

- **Benchmark:** Stratified Random prediction from the training class distribution.
- **Models:** Multinomial Naive Bayes (Laplace α sweep), L2 Logistic Regression, k-NN with cosine and Euclidean distance.
- **Ablation:** stop words {none, English} × n-grams {(1,1), (1,2)} × max features {1,000, 5,000, 10,000} = 12 TF-IDF settings, run for each model over several seeds.
- **Protocol:** TF-IDF and all parameters are fit on train only, hyper-parameters are tuned on validation, and the test split is used once. Reported: Macro F1 (mean ± std over seeds), per-class precision and recall, confusion matrices, time per sample.
- **Theory focus:** Bayes' theorem with Laplace smoothing; cosine vs Euclidean distance in sparse TF-IDF space.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Narratives discontinued; no new data | Frozen snapshot, documented; no refresh needed |
| Near-copies across splits (22.0% of sampled val/test complaints have a training complaint with cosine ≥ 0.8, mostly edited templates) | Group-aware split; 0 shared groups or texts; reported as a limitation |
| Results describe distinct complaints, not the live mix where templates repeat | Stated in the report; `group_size` kept for a weighted check |
| Label noise (345 groups with the same text under different products) | Majority label per group; count reported |
| Severe imbalance and short texts | Macro F1, stratified splits, per-class reporting, error analysis |

## 6. Timeline and roles

| Milestone | Date | Deliverable | Status |
|---|---|---|---|
| M1 Proposal & contract | Sept 18 | Proposal, contract | Done |
| M2 Data approval & EDA | Oct 2 | Data card, EDA, split code, leakage tests, compute plan, updated proposal | Delivered with this memo |
| M3 Baseline reproduction | Oct 19 | TF-IDF pipeline, Stratified Random benchmark, Naive Bayes baseline | Next |
| M4 Progress presentation | Oct 30 | Theory write-up, 5-minute video, initial validation results | Planned |
| M5 Final experiments | Nov 20 | Logistic Regression and k-NN, ablation logs, error analysis | Planned |
| M6 / M7 Final report, defence | Dec 4 / Dec 7 | 6–8 page report, repository release, AI use statement; individual reflection and oral defence | Planned |

Roles: Kevin Thomas, data and documentation; Alex Facey, modelling; Subangan Sundaralingam, theory and statistics; Sayon Kirubaharan, evaluation and reproducibility.

## References

[1] Consumer Financial Protection Bureau, "Consumer Complaint Database," CFPB Open Data, 2026. https://www.consumerfinance.gov/data-research/consumer-complaints/  
[2] Consumer Financial Protection Bureau, "How we share complaint data," 2026. https://www.consumerfinance.gov/complaint/data-use/  
[3] C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008.  
[4] S. Raschka, "Naive Bayes and Text Classification I: Algorithms and Implementation," arXiv:1410.5329, 2014.  
[5] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, vol. 12, pp. 2825–2830, 2011.
