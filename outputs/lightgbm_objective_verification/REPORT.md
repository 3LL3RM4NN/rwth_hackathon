# Independent verification: LightGBM L1 (MAE) vs L2 (mean) objective

**VERDICT: VERIFIED.** Both models were re-trained from scratch; their predictions are **bit-for-bit identical** to the stored predictions of `outputs/lightgbm_objective_screening/` (max / mean absolute difference 0, 0 differing rows out of 1,795,968 for each model), and every metric recomputed with independent code matches the existing value to floating-point precision (≤ 2e-13). The large L1/L2 difference is real, not an implementation or evaluation artefact.

## What was independent
- Matrices built **twice** (once per model) with the production feature code and compared byte-for-byte: fit X/y, validation X/y, and all 241 test matrices identical. 1,266,767 training rows (25% sample, seed 0), 549,504 validation rows, 32 features (listed in `verification_summary.json`), 245 training households, 241 scored, 84 days, train targets 2022-03-16..2023-02-15, validation 2023-02-16..2023-03-15, nothing after 2023-03-15. Same hyperparameters (500-tree cap, lr 0.05, 31 leaves, subsample/colsample 0.8, seed 0); best iteration L1 = 497, L2 = 500 (cap), early stopping on each objective's own validation loss.
- **Features recomputed with plain pandas straight from the raw CSVs** (no cumulative-sum windows): `y_last, y_lag1h, lag_1d/2d/7d, mean_same_slot_4wk, roll_mean_1h/6h/24h/7d, roll_std_24h` and five temperature features, 72 random (household, day, slot) rows: max abs difference 8.9e-7 (float32 rounding).
- **Leakage**, per model, on 24 random household-days: (a) every target and weather value stamped after D-1 11:45 set to 9999, (b) only the target day D set to 9999: all forecast-day features unchanged, `lag_1d` NaN for all slots after 11:45, and the trained model's predictions unchanged. No target-day actual enters any feature.
- Actuals taken **directly from the raw CSVs** (complete 96-interval days), all metrics recomputed from first principles (no `src.evaluation`, no shared metric code). Pipeline and raw-CSV actuals agree exactly.

## Direct comparison (like-for-like set: 1,783,680 intervals / 18,580 household-days)
| Metric | Existing L1 | Verified L1 | Diff | Existing L2 | Verified L2 | Diff |
|---|--:|--:|--:|--:|--:|--:|
| Overall interval MAE | 0.151480 | 0.151480 | 1e-16 | 0.167639 | 0.167639 | 8e-17 |
| Mean household MAE | 0.152665 | 0.152665 | 6e-17 | 0.168576 | 0.168576 | 3e-17 |
| Interval RMSE | 0.298842 | 0.298842 | 6e-17 | 0.286036 | 0.286036 | 0 |
| Household daily-energy MAE | 7.12811 | 7.12811 | 9e-16 | 5.61854 | 5.61854 | 9e-16 |
| Portfolio-day MAE | 1206.92 | 1206.92 | 2e-13 | 510.18 | 510.18 | 0 |
| Portfolio-day RMSE | 1462.40 | 1462.40 | 0 | 825.41 | 825.41 | 0 |
| Portfolio-day MAPE % | 25.1497 | 25.1497 | 4e-15 | 8.4180 | 8.4180 | 0 |
| Portfolio bias % | −21.3737 | −21.3737 | 0 | +3.1015 | +3.1015 | 4e-16 |
| Underforecast MWh | 100.102 | 100.102 | 0 | 14.257 | 14.257 | 0 |
| Overforecast MWh | 1.279 | 1.279 | 0 | 28.598 | 28.598 | 0 |

(The existing screening scores only intervals that the seasonal baseline and the stored AutoGluon forecast also cover; that excludes 128 household-days (12,288 intervals) that the full 1,795,968-interval set contains. On the full set the headline numbers shift slightly: L1 portfolio-day MAE 1,217.6 / bias −21.5% / MAPE 25.18%; L2 514.1 / +3.0% / 8.40% — same conclusion; see `verified_metrics_full_evaluation_set.csv`. An earlier run of this script exposed this as the only source of a ≈1% metric difference — predictions were already identical.)

## Why L1 and L2 differ (distribution check and diagnostics)
Target distribution is strongly right-skewed (test: mean 0.259, median 0.106; 9.4% exact zeros).
| | actual | L1 | L2 |
|---|--:|--:|--:|
| Test mean / median | 0.259 / 0.106 | 0.204 / 0.121 | 0.267 / 0.199 |
| Validation mean / median | 0.392 / 0.243 | 0.335 / 0.256 | 0.409 / 0.347 |
| Training-sample mean / median | 0.288 / 0.128 | 0.224 / 0.133 | 0.288 / 0.208 |
| Total predicted energy (test, MWh) | 465.7 | 365.8 | 479.9 |
L1 reproduces the **median** (0.121 vs 0.106; 0.133 vs 0.128 in training) but misses the mean by 21%; L2 reproduces the **mean** (0.288 vs 0.288 in training; 0.267 vs 0.259 in test). Representative households (`distribution_representative_households_validation.csv`) show the same pattern for low, mid and high consumers (e.g. household 1218377: actual mean 0.259 / median 0.072, L1 mean 0.135 / median 0.102, L2 mean 0.262 / median 0.238).

Portfolio totals (84 days): mean actual 5,544 kWh/day; L1 predicts 4,355 (ratio 0.785), L2 5,713 (ratio 1.030).
1. **Shift in the forecast level — yes, this is the main effect.** L1's predicted/actual ratio is 0.785, L2's 1.030.
2. **High-consumption households — partly.** L1 under-forecasts in every household-level quintile (−2.9 … −8.5 kWh/day; the top two quintiles carry 55% of the total under-forecast). L2 removes the bias in the top quintile (−0.06) and slightly over-forecasts low consumers (+1.0 … +1.1 kWh/day). By interval level, L1's error is concentrated in high actuals (−0.30 kWh for 0.5–1 kWh intervals, −0.79 above 1 kWh).
3. **Reduced systematic household under-forecasting — yes.** Share of under-forecast household-days 77% (L1) → 38% (L2); mean signed household daily error −5.34 → +0.76 kWh.
4. **Cancellation — yes.** Cancellation of household daily errors in the portfolio total (1 − |Σe|/Σ|e|): 24% (L1) vs 59% (L2). L1's errors share a sign, so they add up; L2's cancel.
Household accuracy: L1 is better on interval MAE (0.1515 vs 0.1676) and household MAE (0.1527 vs 0.1686) — it is the right model for the typical interval — while L2 is better on interval RMSE (0.2860 vs 0.2988), household daily-energy MAE (5.62 vs 7.13 kWh) and every portfolio metric, as reported earlier.

## Conclusion
No implementation or evaluation discrepancy was found. The −21% portfolio bias of the MAE-objective model is the expected consequence of a median-targeting objective on a right-skewed, zero-inflated target (sum of conditional medians < sum of conditional means), and the mean objective removes it.

Files: `metrics/verification_summary.json`, `comparison_vs_existing.csv`, `verified_metrics.csv`, `verified_metrics_full_evaluation_set.csv`, `prediction_sanity.csv`, `sample_household_days_predictions.csv` (30 household-days × 96 slots with actual / L1 / L2 / difference), `distribution_*.csv`, `diagnostics_*.csv`, `predictions/verified_predictions.parquet`; log `outputs/lightgbm_objective_verification_run.log`.
