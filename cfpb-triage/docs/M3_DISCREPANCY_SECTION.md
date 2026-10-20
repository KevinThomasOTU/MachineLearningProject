# M3 Discrepancy Analysis (memo section 3)

Source: `scripts/07_discrepancy.py` → `results/tables/discrepancy.csv`, `significance.csv`, `nb_by_nearcopy.csv`. Validation split (n = 14,649); the test split is unused.

| Check | Expected | Observed |
|---|---|---|
| Stratified Random Macro F1 / accuracy | 0.091 / 0.305 (closed form, M2 memo) | 0.092 / 0.306 |
| Majority class Macro F1 / accuracy | 0.062 / 0.513 | 0.062 / 0.513 |
| NB from the formulas vs scikit-learn (prediction agreement) | 100% | 100% |
| TF-IDF / NB fit time (M2 estimate) | 19.8 s / 1.30 s | 16.7 s / 0.17 s |
| Val rows with a train near-copy (cosine ≥ 0.8) | 22% (M2 sample) | 22.9% (all of val) |
| NB Macro F1 weighted by duplicate-group size | 0.623 | 0.587 |

Both naive benchmarks land on their closed-form values (Stratified Random 0.0917 vs 1/K = 0.0909; majority class exact at 0.0616 Macro F1 and 0.513 accuracy), so the evaluation code is behaving as the theory predicts. Our Multinomial NB written directly from the Laplace-smoothing formulas gives the same predictions as scikit-learn on all 14,649 validation rows, so the baseline is the model described in our theory section. Tuned NB (alpha = 0.03) reaches Macro F1 0.623 (95% bootstrap CI 0.609–0.637), which is 0.530 above Stratified Random (paired bootstrap CI 0.515–0.544); it over-fits moderately (train 0.705 vs validation 0.623), and choosing alpha on validation adds 0.107 over the default alpha = 1, so this figure is slightly optimistic until the test split is used in M5. The weak spot is rare classes: *Debt or credit management* has recall 0.03 (69 validation rows), *Payday loan* 0.28, *Money transfer* 0.40 and *Prepaid card* 0.45, mostly predicted as *Checking or savings* or *Credit card*, because the class prior and the small training vocabularies of these classes favour the large classes. Near-copies of training complaints are much easier (accuracy 0.93 vs 0.75) but lift Macro F1 by only 0.03, so residual leakage does not explain the score; weighting each complaint by its duplicate-group size lowers Macro F1 by 0.036, so performance on the live mix, where templates repeat, is somewhat worse than on distinct complaints. Finally, the M2 compute plan over-estimated NB fit time by about 8× (1.30 s vs 0.17 s) because timings on a 10,000-row subsample include fixed overhead that does not scale linearly; M5 budgets should use the measured full-size times.
