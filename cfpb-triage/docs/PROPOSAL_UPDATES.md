# Proposal Updates (since M1)

1. **New dataset snapshot.** We now use the CFPB Consumer Complaint Database *narratives archive* file for April–July 2024 instead of a live download. It is a fixed file (`data/raw/complaints.csv`, downloaded 2026-09-30), so results are reproducible.

2. **Narrative-publication-ended risk.** CFPB stopped publishing consumer complaint narratives on Aug 14, 2026. The archive is therefore the final available snapshot. We cannot add newer data or test for drift after July 2024; this is documented as a limitation in the data card.

3. **Near-duplicate grouping and collapsing.** About half of all narratives are exact copies of another narrative (mostly credit-repair template letters), and consumers often file the same reworded complaint against Equifax, Experian and TransUnion. We assign a `group_id` from exact normalised-text matches plus near-duplicates (same date, state, ZIP and issue, and a matching 200-character prefix or TF-IDF cosine ≥ 0.80), then keep **one row per group** (majority label; `group_size` records the original count). The modelling set is therefore made of distinct complaint texts.

4. **Updated split.** Splits are 70/15/15, stratified by Product and group-aware, so no duplicate group can leak into validation or test. We drop classes with fewer than 200 rows. After collapsing duplicates the dataset is small enough that no sampling is needed (`sample_size: null`; see `configs/config.yaml`).
