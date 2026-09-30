# Proposal Updates (since M1)

1. **New dataset snapshot.** We now use the CFPB Consumer Complaint Database *narratives archive* file for April–July 2024 instead of a live download. It is a fixed file (`data/raw/complaints.csv`, downloaded <FILL: date>), so results are reproducible.

2. **Narrative-publication-ended risk.** CFPB stopped publishing consumer complaint narratives on Aug 14, 2026. The archive is therefore the final available snapshot. We cannot add newer data or test for drift after July 2024; this is documented as a limitation in the data card.

3. **Near-duplicate grouping.** Exact-match deduplication is not enough: consumers often file the same reworded complaint against Equifax, Experian and TransUnion. We now assign a `group_id` from exact normalised-text matches plus near-duplicates (same date, state, ZIP and issue, and a matching 200-character prefix or TF-IDF cosine ≥ 0.80). Duplicates are grouped rather than deleted.

4. **Updated split.** Splits are 70/15/15, stratified by Product and **group-aware**: all rows of a duplicate group go to the same split, which prevents near-duplicate leakage into validation and test. We also drop classes with fewer than 200 rows and use a 100,000-row group-aware stratified sample for compute (see `configs/config.yaml`).
