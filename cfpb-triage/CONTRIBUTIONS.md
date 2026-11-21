# Contributions

Draft built from the Git history (all branches, M1 to M5) and the roles in the README. **Every member must
read this, correct anything inaccurate, and confirm it before the Milestone 6 submission** (spec section 15).
Evidence: commits and pull requests in the repository, `AI_USE.md`, and the experiment log
`m5_deliverables/results/logs/experiment_log.jsonl`.

## By member

| Member | Role | Main contributions (evidence) |
|---|---|---|
| Kevin Thomas | Project Coordinator, Data Lead, Documentation Lead | Proposal and updated proposal; raw loader and config (`load.py`); cleaning and near-duplicate grouping (`clean.py`, `01_clean_data.py`); EDA notebook and data card; M2 approval memo; M3 technical memo and discrepancy write-up; M4 initial models, significance tests, progress report and presentation; M5 error analysis (`m5_error_analysis.py`), draft Results generator (`m5_draft_results.py`), Makefile M5 targets, integration and submission package |
| Alex Facey | Machine-Learning Modelling Lead | Stratified group split (`split.py`, `02_make_splits.py`); compute estimate; TF-IDF wrapper; Naive Bayes and alpha sweep; M3 baseline run (`04_run_baseline.py`); M5 cosine k-NN and LR modules, model tests, `m5_common.py`, equal-effort tuning (`m5_tune.py`) and robustness analysis (`m5_robustness.py`) |
| Subangan Sundaralingam | Theory, Statistical Analysis, Interpretation Lead | Imbalance and EDA statistics; `SPLIT_STRATEGY.md`; significance tests, smoothing analysis and distance helpers (`analysis/`); M3 discrepancy analysis (`07_discrepancy.py`); M5 pairwise significance with Holm correction, TF-IDF ablation and hyper-parameter sensitivity, interpretation and theory checks (`m5_significance.py`, `m5_ablation_effects.py`, `m5_interpretation.py`) |
| Sayon Kirubaharan | Evaluation, Reproducibility, Communication Lead | Naive baselines and seeding (`baseline.py`, `seed.py`); metrics, plots and EDA figures; notebook and M3 tests; reproducibility check; M5 experiment log (`m5_runlog.py`), repeated grouped CV (`m5_repeated_cv.py`), one-shot test evaluation (`m5_final_test.py`) and main table/figure (`m5_main_table.py`) |

## By activity category

| Category | Lead | Support |
|---|---|---|
| Problem formulation | Kevin | all |
| Data preparation | Kevin | Alex (splits) |
| Software / pipeline | Alex, Kevin | Sayon |
| Modelling | Alex | Subangan (theory) |
| Experiments | Alex, Sayon | Subangan |
| Analysis (statistics, error analysis) | Subangan, Kevin | Sayon |
| Visualization | Sayon | Subangan, Kevin |
| Project management, integration | Kevin | all |
| Writing | Kevin | all (each writes their own Results paragraphs) |
| Presentation | all | |

## M5 note on who ran what

For the Milestone 5 deadline, Kevin ran Sayon's and Subangan's remaining scripts on the real data
(`m5_repeated_cv.py`, `m5_final_test.py`, `m5_main_table.py`, `m5_runlog.py --render`; `m5_significance.py`,
`m5_ablation_effects.py`, `m5_interpretation.py`). The experiment log records these runs under Kevin's account.
Sayon and Subangan wrote those scripts, review the outputs, and are responsible for explaining them.
