| Model | Val Macro F1 | Val 95% CI | CV Macro F1 (mean ± std) | Test Macro F1 | Val accuracy | Val macro P | Val macro R | Selected configuration |
|---|---|---|---|---|---|---|---|---|
| Majority class | 0.062 | [0.061, 0.062] | 0.062 ± 0.000 | 0.062 [0.061, 0.062] | 0.513 | 0.047 | 0.091 | - |
| Stratified Random | 0.094 | [0.089, 0.098] | 0.090 ± 0.002 | 0.092 [0.087, 0.096] | 0.309 | 0.094 | 0.094 | - |
| Multinomial NB | 0.648 | [0.634, 0.662] | 0.646 ± 0.006 | 0.657 [0.642, 0.671] | 0.805 | 0.755 | 0.629 | 1-2g, 10000 feat, stop=english; alpha=0.03, fit_prior=True |
| L2 Logistic Regression | 0.713 | [0.698, 0.728] | 0.708 ± 0.016 | 0.703 [0.687, 0.717] | 0.846 | 0.806 | 0.677 | 1-2g, 10000 feat, stop=english; C=3.0, class_weight=None |
| k-NN (cosine) | 0.665 | [0.649, 0.680] | 0.653 ± 0.006 | 0.667 [0.651, 0.682] | 0.818 | 0.788 | 0.613 | 1-1g, 10000 feat, stop=no; k=15, weights=distance |