#!/usr/bin/env bash
# Scaffold for KASS Tech (Team 32) CFPB complaint triage project.
# Usage: bash setup_project.sh [project_dir]   (default: cfpb-triage)
set -euo pipefail
ROOT="${1:-cfpb-triage}"
PKG="$ROOT/src/cfpb_triage"

mkdir -p "$ROOT"/{configs,docs,notebooks,scripts,tests,.github}
mkdir -p "$ROOT"/data/{raw,interim,processed/splits}
mkdir -p "$ROOT"/results/{figures,tables,logs}
mkdir -p "$ROOT"/reports/{proposal,progress,final}
mkdir -p "$PKG"/{data,features,models,evaluation,analysis,experiments,utils}

# package markers
for d in "" data features models evaluation analysis experiments utils; do
  touch "$PKG/${d:+$d/}__init__.py"
done
touch "$ROOT"/data/raw/.gitkeep "$ROOT"/data/interim/.gitkeep "$ROOT"/data/processed/.gitkeep
touch "$ROOT"/results/figures/.gitkeep "$ROOT"/results/tables/.gitkeep "$ROOT"/results/logs/.gitkeep

# stub helper
stub() { # path, docstring
  printf '"""%s"""\n' "$2" > "$1"
}

# ---- M2 modules ----
stub "$PKG/data/load.py"        "Load raw CFPB CSV (Kevin). Read only needed columns; return DataFrame."
stub "$PKG/data/clean.py"       "Drop null narratives, normalise labels, dedupe; log row counts (Kevin)."
stub "$PKG/data/split.py"       "Stratified, dedupe-aware train/val/test split with fixed seed (Alex)."
stub "$PKG/analysis/imbalance.py" "Class imbalance stats: ratio, entropy, per-class counts (Subangan)."
stub "$PKG/utils/seed.py"       "Global seeding helper (Sayon)."
# ---- M3+ modules ----
stub "$PKG/features/tfidf.py"   "TF-IDF wrapper; fit on train only (Alex)."
stub "$PKG/models/baseline.py"  "Stratified Random benchmark (Sayon)."
stub "$PKG/models/naive_bayes.py" "Multinomial NB + alpha sweep (Alex)."
stub "$PKG/models/logreg.py"    "L2 Logistic Regression (Alex)."
stub "$PKG/models/knn.py"       "k-NN with cosine/Euclidean (Alex + Subangan)."
stub "$PKG/evaluation/metrics.py" "Macro F1, per-class P/R, confusion matrix, timing (Sayon)."
stub "$PKG/evaluation/plots.py" "Confusion matrix and class-distribution plots (Sayon)."
stub "$PKG/evaluation/variability.py" "Macro F1 mean/std over seeds or folds (Sayon)."
stub "$PKG/analysis/smoothing.py" "Laplace smoothing / Bayes demo (Subangan)."
stub "$PKG/analysis/distances.py" "Cosine vs Euclidean on sparse TF-IDF (Subangan)."
stub "$PKG/analysis/significance.py" "Paired bootstrap / McNemar between models (Subangan)."
stub "$PKG/experiments/ablation.py" "Config-driven ablation runner (Sayon)."

# scripts (thin entry points)
for s in 00_download_data 01_clean_data 02_make_splits 03_eda 04_run_baseline 05_run_ablations; do
  printf '"""Entry point: %s. Import from cfpb_triage and call; keep logic in src/."""\n' "$s" > "$ROOT/scripts/$s.py"
done

# tests
cat > "$ROOT/tests/test_split.py" << 'PY'
"""M2 leakage tests (Sayon). TODO:
- no shared IDs across train/val/test
- class proportions within tolerance of full data
- same seed -> identical splits
- duplicate narratives never cross splits
"""
PY
touch "$ROOT/tests/__init__.py"

# config
cat > "$ROOT/configs/config.yaml" << 'YML'
seed: 42
data:
  raw_path: data/raw/complaints.csv
  label_col: Product
  text_col: Consumer complaint narrative
  min_class_count: 100     # TODO decide in M2
  sample_size: null        # TODO decide in M2
split:
  train: 0.70
  val: 0.15
  test: 0.15
tfidf:
  stop_words: [null, english]
  ngram_range: [[1, 1], [1, 2]]
  max_features: [1000, 5000, 10000]
YML

# docs
cat > "$ROOT/docs/DATA_CARD.md" << 'MD'
# Data Card: CFPB Consumer Complaint Database
- **Source / URL / download date:** TODO
- **License / terms (verified by, date):** TODO
- **Fields used:** TODO
- **Label definition and class list:** TODO
- **Cleaning steps and row counts:** TODO
- **Splits (ratios, seed, dedupe policy):** TODO
- **Known biases, limitations, PII notes:** TODO
MD
printf '# Split Strategy\nTODO (Subangan)\n' > "$ROOT/docs/SPLIT_STRATEGY.md"
printf '# AI Use Log\n\n| Date | Member | Tool | Prompt/purpose | What we used | How verified/tested |\n|---|---|---|---|---|---|\n' > "$ROOT/AI_USE.md"
printf '# Contributions\nTODO (fill from Git history in M6)\n' > "$ROOT/CONTRIBUTIONS.md"

cat > "$ROOT/README.md" << 'MD'
# CFPB Complaint Triage (Team 32, CSCI 3052)
Compare Multinomial NB, Logistic Regression and k-NN on imbalanced complaint text.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Run (fill in as milestones land)
    make data     # download + clean
    make splits   # leakage-free splits
    make test
See docs/PROJECT_PLAN.md.
MD

cat > "$ROOT/requirements.txt" << 'TXT'
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
pyyaml
jupyter
pytest
tqdm
TXT

cat > "$ROOT/pyproject.toml" << 'TOML'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "cfpb_triage"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["src"]
TOML

cat > "$ROOT/Makefile" << 'MK'
.PHONY: data splits test m2
data:
	python scripts/00_download_data.py && python scripts/01_clean_data.py
splits:
	python scripts/02_make_splits.py
test:
	pytest -q
m2: data splits test
MK

cat > "$ROOT/.gitignore" << 'GI'
.venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
data/raw/*
data/interim/*
data/processed/*
!data/**/.gitkeep
.DS_Store
GI

cat > "$ROOT/.github/pull_request_template.md" << 'MD'
## What / why
## Checklist
- [ ] Tests pass
- [ ] No data files committed
- [ ] AI use logged in AI_USE.md (or none)
- [ ] Peer reviewer assigned (not the author)
MD

cp "$(dirname "$0")/PROJECT_PLAN.md" "$ROOT/docs/PROJECT_PLAN.md" 2>/dev/null || true

echo "Created $ROOT"; command -v tree >/dev/null && tree -a -I '.git' "$ROOT" || find "$ROOT" -type f | sort
