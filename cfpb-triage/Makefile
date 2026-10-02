.PHONY: data splits eda test compute m2
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

