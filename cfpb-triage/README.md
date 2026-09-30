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
