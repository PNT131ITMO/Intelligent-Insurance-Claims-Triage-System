# CatBoost model candidate selection — 2026-10-19

**Status:** candidate selection for the next evaluation stage; no final model has been trained or deployed.

**Primary candidate:** `expert_without_cat11_v22` (247 features). It has the lowest pooled out-of-fold (OOF) Log Loss among the four completed, comparable CatBoost runs. Log Loss is the primary selection metric; lower is better.

| Feature set | Features | OOF Log Loss | OOF ROC-AUC | Five fold models, GiB |
|---|---:|---:|---:|---:|
| `expert_base_only` (`expert_base_only_v2` run) | 25 | 0.442167 | 0.787119 | 0.621 |
| **`expert_without_cat11_v22`** | **247** | **0.438446** | **0.791097** | **1.692** |
| `expert_without_cat11_and_rounded_numeric` | 235 | 0.438478 | 0.790995 | 1.483 |
| `expert_without_cat11_and_rounded_numeric_and_num_pair_sums` | 169 | 0.438688 | 0.790897 | 1.819 |

**Alternative when artifact size matters:** `expert_without_cat11_and_rounded_numeric` (235 features). Its OOF Log Loss is higher by only 0.000033, while its five saved fold models occupy about 0.209 GiB less than the primary candidate. The 169-feature variant has fewer input columns, but its OOF Log Loss is higher by 0.000242 and its saved fold models are larger than those of the primary candidate; fewer features did not make this run cheaper to store.

All four runs use the same processed-data checksums, five stratified folds (seed 42), and CatBoost settings. Their OOF `ID`, target, and fold columns match row for row. A paired per-row Log Loss comparison gives these differences relative to the primary candidate (positive means worse):

| Alternative | Mean OOF Log Loss difference | 95% paired row-bootstrap interval |
|---|---:|---:|
| 235 features | +0.000033 | [-0.000332, +0.000392] |
| 169 features | +0.000242 | [-0.000141, +0.000673] |

Intervals use 1,000 bootstrap resamples of matched OOF rows (seed 42). They describe uncertainty conditional on these fitted models and folds; they do not account for model-selection bias or variation from retraining. Both intervals include zero, so the small differences should not be presented as proven performance improvements. Model sizes above sum the five `.cbm` fold files, not a future final model.

The `expert_full` feature set has not been run, so these experiments do not establish that removing the 11-way interactions improves performance over the complete expert feature set. The next stage is OOF error and probability analysis before final training and inference design.

Sources: [ablation tracker](ablation_candidates.csv), [primary run](../experiments/expert_without_cat11_v22/overall_metrics.csv), [235-feature run](../experiments/expert_without_cat11_and_rounded_numeric/overall_metrics.csv), [169-feature run](../experiments/expert_without_cat11_and_rounded_numeric_and_num_pair_sums/overall_metrics.csv), and [base run](../experiments/expert_base_only_v2/overall_metrics.csv).
