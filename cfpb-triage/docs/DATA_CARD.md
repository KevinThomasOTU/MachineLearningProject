# Data Card: CFPB Consumer Complaint Database (Narratives Archive, Apr–Jul 2024)

## Source
- **Publisher:** Consumer Financial Protection Bureau (CFPB), Consumer Complaint Database
- **URL:** https://www.consumerfinance.gov/data-research/consumer-complaints/ (narratives archive section). Exact file link: https://files.consumerfinance.gov/f/documents/CCDB_Export_6_April_2024_through_July_2024.zip
- **File / date range:** April 2024 – July 2024 narratives archive, saved as `data/raw/complaints.csv` (not committed)
- **Download date:** 2026-09-30, downloaded by kevin
- **File size / raw rows:** 517.1 MB / 839,903 rows (16 columns)
- **SHA-256 of complaints.csv:** `B5414EC55705DA56307A5058EE928322099C1303D32D77A0061B132C418F88C1` (every teammate should get the same value; PowerShell: `Get-FileHash data\raw\complaints.csv -Algorithm SHA256`)
- **Frozen snapshot:** CFPB stopped publishing consumer narratives on Aug 14, 2026. No newer narrative data will appear, so this archive file is the project's fixed dataset and cannot be refreshed.

## License / terms
CFPB states that "All complaint data we publish is freely available for anyone to use, analyze, and build on" (link: https://www.consumerfinance.gov/data-research/consumer-complaints/, checked 2026-10-02 by Kevin Thomas). No formal license text or attribution requirement was found on that page or on the "How we share complaint data" page; the data is published by a U.S. federal agency. We attribute it to the CFPB Consumer Complaint Database and make no re-identification attempts.


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
| Credit reporting or other personal consumer reports | 50,092 | 51.3% |
| Debt collection | 12,803 | 13.1% |
| Credit card | 10,188 | 10.4% |
| Checking or savings account | 9,599 | 9.8% |
| Mortgage | 3,877 | 4.0% |
| Money transfer, virtual currency, or money service | 3,085 | 3.2% |
| Student loan | 2,456 | 2.5% |
| Vehicle loan or lease | 2,328 | 2.4% |
| Payday loan, title loan, personal loan, or advance loan | 1,594 | 1.6% |
| Prepaid card | 1,182 | 1.2% |
| Debt or credit management | 451 | 0.5% |
| **Total (11 classes)** | **97,655** | 100% |

Dropped classes (< 200 rows): none. All 11 products in the file already use current CFPB names, so the legacy-name merge changed nothing.

## Cleaning steps and row counts (from `results/logs/cleaning_log.json`)
| Step | Rows after | Dropped |
|---|---|---|
| Raw file | 839,903 | — |
| Drop null/blank narrative | 276,332 | 563,571 |
| Drop null label | 276,332 | 0 |
| Drop narratives empty after mask removal | 276,275 | 57 |
| Drop repeated Complaint IDs | 276,275 | 0 |
| Assign group_id (no rows removed) | 276,275 | 0 |
| Keep one row per group | 97,655 | 178,620 |
| min_class_count ≥ 200 | **97,655** | 0 |

No sampling is applied (`sample_size: null`): the distinct complaints already fit the compute budget.

**Text normalisation (`norm_text`):** lowercase; remove masked dates (`XX/XX/XXXX`, `XX/XX/2024`) and masked tokens (`XX`, `XXXX`, `XXXX1234`); collapse whitespace. Real words containing "xx" (e.g. "Exxon") and unmasked dates are kept. Dollar amounts (`{$500.00}`) are kept.

**Near-duplicate grouping (`group_id`):**
1. Exact: identical `norm_text` → same group.
2. Near-duplicate: rows are blocked on (Date received, State, ZIP, Issue). Within a block, two rows are linked if the first 200 normalised characters match **or** their TF-IDF (word 1–2-gram, sublinear tf) cosine similarity is ≥ 0.80. Blocks with more than 2,000 rows use the prefix rule only.
3. Connected components (union-find) form groups; `group_id` = smallest Complaint ID in the group.

Step 1 catches credit-repair template letters submitted word-for-word by many different consumers; step 2 catches one consumer filing the same, slightly reworded complaint against Equifax, Experian and TransUnion on the same day, and edited variants of a template that share a block with another copy. Blocking makes it fast (roughly linear; no all-pairs comparison; the full run takes about 95 s). Result on the 276,275 cleaned narratives: 30,981 multi-row groups covering 209,601 rows (75.9%); 139,710 rows (50.6%) are exact copies of another narrative; 18,054 extra links came from the prefix rule and 20,856 from cosine similarity; the largest group has 7,788 rows (one template letter). In total 97,655 distinct groups remain.

Threshold sanity check: 20 groups containing more than one distinct text were spot-checked on 2026-09-30 (10 with the lowest within-group similarity, 10 random): 0 wrong merges; 13 were one consumer filing to several companies and 7 were shared template letters (merged by design). Threshold kept at 0.80. Details: `results/tables/spotcheck_groups.csv` (`scripts/spotcheck_groups.py`).

**One row per group (`one_row_per_group: true`):** about half of all narratives are exact copies of another narrative (duplication is concentrated in credit reporting and debt collection). Keeping every copy would let a single template count thousands of times in the test set and saturate k-NN neighbourhoods, so each group is reduced to one representative row:
- label = the group's majority product (ties → alphabetically first); the kept row is the lowest Complaint ID with that label;
- `group_size` records how many rows the group had (sum over the dataset = rows before collapsing), so duplicate frequency is still available for analysis or weighting;
- 345 groups (26,932 rows before collapsing) contained the same text filed under more than one product; they take the majority label, which is a source of label noise.

Effect: 178,620 duplicate rows removed; credit reporting falls from 77.5% of narratives to 51.3% of distinct complaints.

## Splits
70 / 15 / 15 train/val/test, seed 42, stratified on `product`, group-aware (each `group_id` in exactly one split; after collapsing, one row per group). Files: `data/processed/splits/{train,val,test}.csv`. Sizes: train 68,357 / validation 14,649 / test 14,649 rows. See `docs/SPLIT_STRATEGY.md`.

## Known biases and limitations
- **Opt-in narratives:** only consumers who consented to publication have narratives (276,332 of 839,903 raw rows, 32.9%). They may differ systematically from all complainants.
- **Heavy imbalance:** credit reporting dominates (51.3% of rows); the largest-to-smallest class ratio is 111:1 (50,092 vs 451, Debt or credit management). This is why Macro F1 is the primary metric.
- **Four-month window (Apr–Jul 2024):** seasonal and topical effects; product taxonomy fixed to 2024 names.
- **Templated letters:** identical templates are collapsed to one row, so results measure performance on *distinct* complaint texts, not on the day-to-day mix a live triage system would see (where templates repeat). Edited template variants are only grouped when they share a (date, state, ZIP, issue) block with another copy, so some template similarity across splits remains possible.
- **Label noise:** the same text is sometimes filed under different products (see label-conflict count above).
- **Short narratives:** 5,308 of the 97,655 final narratives (5.4%) have < 20 tokens after masking, and 190 have < 5 (median 119 tokens).
- **Masking side effects:** removing XXXX tokens removes information (company names, amounts, dates) the CFPB redacted.
- **Frozen data:** narratives ended Aug 14, 2026. No new data for later re-evaluation or drift checks.

## PII
CFPB scrubs narratives before publication and replaces personal details with `XXXX`. ZIP codes may be partially masked. We do not attempt re-identification, we strip mask tokens from model input, and we do not commit raw or processed data (`.gitignore`). Any quoted example narrative in reports is truncated and checked for residual personal details.
