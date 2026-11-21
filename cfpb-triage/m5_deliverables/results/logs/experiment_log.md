# M5 experiment log

Generated from `experiment_log.jsonl` (append-only; failed and negative runs kept).

| # | UTC time | script | status | wall (s) | commit | args | note |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-21T00:56:57+00:00 | m5_tune | ok | 298.1 | b9f06af* | n_jobs=2, resume=False, quick=True |  |
| 2 | 2026-11-21T01:40:18+00:00 | m5_tune | ok | 2580.9 | b9f06af* | n_jobs=2, resume=False, quick=False |  |
| 3 | 2026-11-21T01:50:16+00:00 | m5_robustness | ok | 567.4 | 74ee035* | parts=['curve', 'nearcopy', 'weighted', 'truncation'] |  |
| 4 | 2026-11-21T03:38:57+00:00 | m5_significance | ok | 2.4 | 706b387* | split=val, n_boot=2000 |  |
| 5 | 2026-11-21T03:39:00+00:00 | m5_ablation_effects | ok | 0.6 | 706b387* |  |  |
| 6 | 2026-11-21T03:39:29+00:00 | m5_error_analysis | ok | 1.1 | 706b387* | model=None, examples=8, top_pairs=5 |  |
| 7 | 2026-11-21T03:39:47+00:00 | m5_interpretation | ok | 43.2 | 706b387* | top=10 |  |
| 8 | 2026-11-21T03:42:52+00:00 | manual_note | ok | 0.0 | 706b387* |  | Deviation: m5_repeated_cv.py --repeats 2 was stopped after ~1.5 folds (about 8 min per fold, too slow for the deadline); re-run with --repeats 1 --folds 5. Uncertainty from CV therefore uses 5 folds (1 repeat). |
| 9 | 2026-11-21T03:45:06+00:00 | m5_final_test | ok | 119.0 | 706b387* | i_confirm_frozen=True | best_configs sha256=eb2c2c71520b |
| 10 | 2026-11-21T03:45:15+00:00 | m5_significance | ok | 3.1 | 706b387* | split=test, n_boot=2000 |  |
| 11 | 2026-11-21T03:51:43+00:00 | m5_repeated_cv | ok | 526.0 | 706b387* | repeats=1, folds=5, models=['nb', 'lr', 'knn'] |  |
| 12 | 2026-11-21T03:51:51+00:00 | m5_main_table | ok | 1.8 | 706b387* |  |  |
| 13 | 2026-11-21T03:51:53+00:00 | m5_draft_results | ok | 0.0 | 706b387* |  |  |

`*` = uncommitted changes when the run started.