# Simple global LightGBM (direct, D-1 11:45) vs pooled AutoGluon — 12-week screening

Same sample as the AutoGluon screening: 12 weeks / 84 days / 241 scored households / 1,783,680 intervals. Reference A = stored pooled AutoGluon predictions (not retrained). `ag_pooled_weather_calendar` = the AutoGluon weather+calendar screening, shown for context. Error = forecast − actual.

## Design (all in `src/lgbm_features.py`, `scripts/run_lightgbm.py`)
- **Direct strategy, one global model**: rows = household × target day × slot; every feature is anchored at the origin D-1 11:45 or is a target lag older than it. No predicted value is ever reused, nothing is recursive. The slot encodes the horizon (step = slot + 49).
- Features (32): calendar (slot, hour, quarter, dow + sin/cos, weekend, month, national holiday); target lags `lag_1d` (only slots ≤ 11:45, else NaN), `lag_2d`, `lag_7d`, `mean_same_slot_4wk`; origin-anchored `y_last`, `y_lag1h`, rolling mean 1h/6h/24h/7d, std 24h; 9 historical temperature features (latest obs at D-1 11:00, lags, means, changes); `household` categorical. Variant `lgbm_pv` adds the PV ownership flag. No Group, AffectsTimePoint, HeatPump/Other, humidity/wind/solar.
- LightGBM 4.7.0, objective l1 (MAE), 500 trees, lr 0.05, 31 leaves, subsample 0.8, colsample 0.8, seed 0, early stopping (50) on the last 28 dev days; trained on 25% random rows (seed 0) of the 365-day window (target days 2022-03-16..2023-02-15; validation 2023-02-16..2023-03-15; nothing after 2023-03-15).
- Cutoff test: all targets (D-1 afternoon, all of D) and all weather stamped after D-1 11:45 set to 9999 → no feature changes (30 household-days).

## Results (common timestamps)
| Metric | Seasonal mean 4w | Pooled AutoGluon | LightGBM | LightGBM + PV flag |
|---|--:|--:|--:|--:|
| Mean household MAE | 0.1829 | 0.1565 | **0.1527** (−2.5%) | **0.1523** (−2.7%) |
| Overall MAE | 0.1822 | 0.1557 | 0.1515 | 0.1511 |
| RMSE | 0.3187 | 0.2993 | 0.2988 | 0.2981 |
| sMAPE % / MAPE % | 80.0 / 100.3 | 77.6 / 71.1 | 75.7 / 66.0 | 75.3 / 66.0 |
| Household daily-energy MAE (kWh) | 7.15 | 7.11 | 7.13 | 7.10 |
| Portfolio-day MAE (kWh) | 930 | 1,233 | 1,207 | 1,199 |
| Portfolio-day bias | +5.5% | −18.2% | −21.4% | −21.2% |
| Under / over-forecast (MWh) | 35.0 / 60.3 | 97.8 / 13.7 | 103.7 / 4.9 | 102.9 / 5.0 |
| Portfolio days under-forecast | 38% | 88% | 94% | 94% |
| Households better than AutoGluon | – | – | 189 / 241 | 192 / 241 |
| Fit / predict time | – | ~226 s / ~300 s | **12 s / 6 s** | 12 s / 7 s |

## Reading
- A simple LightGBM beats the pooled AutoGluon on household accuracy (MAE −2.5%, MAPE −5 pts, sMAPE −2 pts; 78% of households better) and is ~20× faster to fit and ~50× faster to predict. Household daily-energy MAE is unchanged.
- It does **not** fix the procurement problem: the L1 objective gives a stronger under-forecast than AutoGluon (portfolio bias −21%, under-forecast on 94% of days); the 4-week seasonal mean is still the best portfolio-level / 1:1-cost forecast.
- PV flag: ≈0.3% gain — negligible, and confounded (known PV status ⇔ almost exactly the treatment group).
- Most important features (gain): lag_2d (39%), mean_same_slot_4wk (23%), household (11%), lag_7d, rolling means; temperature is minor (~2%).
- Caveats: not tuned; `household` id adds household-level information the AutoGluon pooled model lacked; best_iteration hit the 500-tree cap (more trees might help slightly); 25% row sample; 12 sampled weeks; one seed.
