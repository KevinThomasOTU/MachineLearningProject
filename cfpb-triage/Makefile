.PHONY: data splits eda test compute m2 baseline discrepancy repro m3 all
data:
	python scripts/00_download_data.py && python scripts/01_clean_data.py
splits:
	python scripts/02_make_splits.py
eda:
	python scripts/03_eda_stats.py && python scripts/03_eda.py
test:
	pytest -q
compute:
	python scripts/06_compute_estimate.py
m2: data splits eda test compute

baseline:
	python scripts/04_run_baseline.py
discrepancy:
	python scripts/07_discrepancy.py
repro:
	python scripts/08_check_reproducible.py
m3: test baseline discrepancy repro

all: data splits m3

# ---- Milestone 5 -------------------------------------------------------------------
.PHONY: m5-tune m5-analysis m5-test m5-log

m5-tune:
	python m5_deliverables/code/m5_tune.py --n-jobs 2

m5-analysis:
	python m5_deliverables/code/m5_repeated_cv.py
	python m5_deliverables/code/m5_robustness.py
	python m5_deliverables/code/m5_ablation_effects.py
	python m5_deliverables/code/m5_interpretation.py
	python m5_deliverables/code/m5_significance.py --split val
	python m5_deliverables/code/m5_error_analysis.py

# run ONCE, after the team has frozen best_configs.json (creates results/logs/TEST_USED.lock)
m5-test:
	python m5_deliverables/code/m5_final_test.py --i-confirm-frozen
	python m5_deliverables/code/m5_significance.py --split test
	python m5_deliverables/code/m5_main_table.py

m5-log:
	python m5_deliverables/code/m5_runlog.py --render
	python m5_deliverables/code/m5_draft_results.py
