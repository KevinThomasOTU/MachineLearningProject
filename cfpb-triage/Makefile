.PHONY: data splits test m2
data:
	python scripts/00_download_data.py && python scripts/01_clean_data.py
splits:
	python scripts/02_make_splits.py
test:
	pytest -q
m2: data splits test
