# Data Card: CFPB Consumer Complaint Database (Narratives Archive, Apr–Jul 2024)

## Source
- **Publisher:** Consumer Financial Protection Bureau (CFPB), Consumer Complaint Database
- **URL:** https://www.consumerfinance.gov/data-research/consumer-complaints/ (narratives archive section). Exact file link: https://files.consumerfinance.gov/f/documents/CCDB_Export_6_April_2024_through_July_2024.zip
- **File / date range:** April 2024 – July 2024 narratives archive, saved as `data/raw/complaints.csv` (not committed)
- **Download date:** 2026-09-30, downloaded by kevin
- **File size / raw rows:** <FILL FROM RUN: 00 output MB> MB / <FILL FROM RUN: n_rows_raw> rows
- **SHA-256 of complaints.csv:** <FILL: hash> (every teammate should get the same value; PowerShell: `Get-FileHash data\raw\complaints.csv -Algorithm SHA256`)
- **Frozen snapshot:** CFPB stopped publishing consumer narratives on Aug 14, 2026. No newer narrative data will appear, so this archive file is the project's fixed dataset and cannot be refreshed.

## License / terms
CFPB complaint data is published by a U.S. federal agency for public use. Exact terms of use and attribution requirements: **to be verified by <>, <DATE>** (link: <FILL>).

## Fields used
| Raw column | Cleaned column | Use |
|---|---|---|
| Complaint ID | complaint_id | row key |
| Consumer complaint narrative | narrative, norm_text | **model input** (norm_text) |
| Product | product | **label** |
| Date received, State, ZIP code, Issue | date_received, state, zip (Issue only for grouping) | near-duplicate blocking |
| Company | company | analysis only (not a model feature) |
| Sub-product | — | loaded, not used in M2 |

## Label definition and class list
Label = `Product` after whitespace normalisation and merging of legacy CFPB product names (full raw→clean map in `results/logs/cleaning_log.json` → `class_map`). Classes with fewer than `min_class_count` = 200 rows are removed.

| Class | Rows (final) | Share |
|---|---|---|
| <FILL FROM RUN: final_class_counts, one row per class> | | |

Dropped classes (< 200 rows): <FILL FROM RUN: dropped_classes>

## Cleaning steps and row counts (from `results/logs/cleaning_log.json`)
| Step | Rows after | Dropped |
|---|---|---|
| Raw file | <FILL FROM RUN: steps[raw].rows> | — |
| Drop null/blank narrative | <FILL FROM RUN> | <FILL FROM RUN> |
| Drop null label | <FILL FROM RUN> | <FILL FROM RUN> |
| Drop narratives empty after mask removal | <FILL FROM RUN> | <FILL FROM RUN> |
| Drop repeated Complaint IDs | <FILL FROM RUN> | <FILL FROM RUN> |
| Assign group_id (no rows removed) | <FILL FROM RUN> | 0 |
| min_class_count ≥ 200 | <FILL FROM RUN> | <FILL FROM RUN> |
| Group-aware stratified sample (100,000) | <FILL FROM RUN: n_rows_final> | <FILL FROM RUN> |

**Text normalisation (`norm_text`):** lowercase; remove masked dates (`XX/XX/XXXX`, `XX/XX/2024`) and masked tokens (`XX`, `XXXX`, `XXXX1234`); collapse whitespace. Real words containing "xx" (e.g. "Exxon") and unmasked dates are kept. Dollar amounts (`{$500.00}`) are kept.

**Near-duplicate grouping (`group_id`):** duplicate rows are *grouped, not deleted*.
1. Exact: identical `norm_text` → same group.
2. Near-duplicate: rows are blocked on (Date received, State, ZIP, Issue). Within a block, two rows are linked if the first 200 normalised characters match **or** their TF-IDF (word 1–2-gram, sublinear tf) cosine similarity is ≥ 0.80. Blocks with more than 2,000 rows use the prefix rule only.
3. Connected components (union-find) form groups; `group_id` = smallest Complaint ID in the group.

This targets the main duplication pattern: one consumer filing the same, slightly reworded complaint against Equifax, Experian and TransUnion on the same day. Blocking makes it fast (roughly linear; no all-pairs comparison). Result: <FILL FROM RUN: dedupe.n_multi_row_groups> multi-row groups covering <FILL FROM RUN: dedupe.rows_in_multi_row_groups> rows; <FILL FROM RUN: dedupe.exact_duplicate_rows> exact-duplicate rows; <FILL FROM RUN: dedupe.cosine_links> links found only by cosine similarity; largest group <FILL FROM RUN: dedupe.largest_group> rows. Threshold sanity check: <FILL: N spot-checked groups, M false merges, by NAME>.

**Sampling:** group-aware stratified downsample to 100,000 rows (seed 42). Each class keeps its natural share, except that no class drops below 200 rows. Small classes are therefore slightly over-represented relative to the full file; full-file counts are in the log.

## Splits
70 / 15 / 15 train/val/test, seed 42, stratified on `product`, group-aware (all rows of a `group_id` in exactly one split). Files: `data/processed/splits/{train,val,test}.csv`. Sizes: <FILL FROM ALEX: train/val/test rows>. See `docs/SPLIT_STRATEGY.md`.

## Known biases and limitations
- **Opt-in narratives:** only consumers who consented to publication have narratives (<FILL FROM RUN: % of raw rows with narrative>%). They may differ systematically from all complainants.
- **Heavy imbalance:** credit reporting dominates (<FILL FROM SUBANGAN: majority share>%; imbalance ratio <FILL FROM SUBANGAN>). This is why Macro F1 is the primary metric.
- **Four-month window (Apr–Jul 2024):** seasonal and topical effects; product taxonomy fixed to 2024 names.
- **Templated letters:** credit-repair style templates recur across *different* consumers (different date/ZIP). These are only grouped when the normalised text is identical, so some template leakage across splits remains possible.
- **Short narratives:** <FILL FROM RUN: short_narratives.lt_20_tokens> narratives have < 20 tokens after masking (median <FILL FROM RUN: median_tokens> tokens).
- **Masking side effects:** removing XXXX tokens removes information (company names, amounts, dates) the CFPB redacted.
- **Frozen data:** narratives ended Aug 14, 2026. No new data for later re-evaluation or drift checks.

## PII
CFPB scrubs narratives before publication and replaces personal details with `XXXX`. ZIP codes may be partially masked. We do not attempt re-identification, we strip mask tokens from model input, and we do not commit raw or processed data (`.gitignore`). Any quoted example narrative in reports is truncated and checked for residual personal details.
