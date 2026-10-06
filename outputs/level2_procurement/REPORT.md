# Level 2 — Day-ahead procurement analysis (pooled AutoGluon, D-1 11:45 cutoff)

Analysis of **existing predictions only** (no model trained). Sample: the 12 sampled weeks / 84 forecast days / 241 households of `outputs/level1_fast_screening/` (1,783,680 scored intervals, 18,580 complete household-days, ~221 households present per day, mean portfolio actual 5,504 kWh/day). Primary model: pooled AutoGluon. Comparison: `seasonal_mean_4weeks` on identical timestamps (recomputed deterministically from raw data).
**Sign convention: error = forecast − actual; >0 over-forecast (over-procurement), <0 under-forecast (under-procurement).** Prices: none in the data, so costs are parametric (under-cost : over-cost = 1:1 … 3:1), not real prices.

## Summary table
| Metric | Pooled AutoGluon | Seasonal mean 4w |
|---|--:|--:|
| Household-interval MAE [kWh/15 min] | **0.1557** | 0.1822 |
| Household-interval RMSE | **0.2993** | 0.3187 |
| MAPE % (actual ≥ 0.05 kWh; 31.1% of intervals excluded, 9.4% exactly zero) | **71.1** | 100.3 |
| sMAPE % | **77.6** | 80.0 |
| Median absolute error [kWh/15 min] | **0.062** | 0.090 |
| Household daily-energy MAE [kWh/day] | **7.11** | 7.15 |
| Household daily bias [kWh/day] | −4.52 | +1.36 |
| Portfolio-interval MAE [kWh/15 min, all households] | 13.83 | **11.82** |
| Portfolio-day MAE [kWh/day] | 1,233 | **930** |
| Portfolio-day bias [kWh/day] (mean actual 5,504) | **−1,000 (−18%)** | +301 (+5.5%) |
| Portfolio over-forecast / under-forecast, interval level [MWh, 84 days] | 13.7 / **97.8** | 60.3 / 35.0 |
| Portfolio days under- / over-forecast | **88% / 12%** | 38% / 62% |
| Cumulative signed error [MWh] | **−84.0** | +25.3 |

## Answers
1. **Accuracy.** At household level the pooled model is clearly more accurate than the baseline (MAE −14.5%, RMSE −6%, MAPE 71% vs 100%); but MAPE/sMAPE stay high (≈70–78%) because individual heat-pump households are spiky and 31% of intervals are below 0.05 kWh. Daily-energy MAE per household is only equal to the baseline (7.11 vs 7.15 kWh).
2. **Systematic bias: yes — strong under-forecast.** Pooled model: portfolio bias −1,000 kWh/day (−18%); under-forecast on 88% of days (74/84) and on 80–91% of portfolio intervals in every hour of the day. All three seasons except spring are biased low (household bias/interval: winter −0.078, autumn −0.065, summer −0.035, spring −0.009); weekends worse (−0.065) than weekdays (−0.040); worst hours 00:00–02:00 and 18:00–23:00. Only the first two weeks after the training cut (late March, April) are near unbiased; from mid-May on every sampled week is under-forecast (see `time_of_day_metrics.csv`, plots 3–4). The baseline is nearly unbiased (+5.5%). Likely contributors (not isolated here): the median-seeking MAE/L1 objective on zero-inflated, spiky loads, and a model that is never refit after 2023-03-15.
3. **Cancellation across households.** Little. Mean household-day absolute error is 7.11 kWh; the portfolio-day absolute error per household is 5.57 kWh, i.e. only ≈22% cancels, because the error is mostly a common one-sided bias, not independent noise. The baseline cancels ≈41% (portfolio-day MAE 930 kWh ≈ 4.2 kWh/household). So the pooled model wins at household level but **loses at portfolio level** (portfolio-day MAE 1,233 vs 930 kWh; portfolio-interval MAE 13.8 vs 11.8 kWh).
4. **Energy mis-procured (84 sampled days, ≈462 MWh actual).** Pooled: 97.8 MWh under-procured vs 13.7 MWh over-procured at interval level (net −84.0 MWh, −18%). Baseline: 35.0 MWh under / 60.3 MWh over (net +25.3 MWh, +5.5%). Daily-net level: pooled under 93.8 / over 9.8 MWh; baseline 26.4 / 51.7 MWh.
5. **Cost sensitivity** (portfolio-interval, cost per kWh consumed, over-forecast cost = 1):

| under:over | pooled | seasonal mean 4w | pooled × hindsight uplift* |
|--:|--:|--:|--:|
| 1 : 1 | 0.241 | **0.206** | 0.192 (×1.21) |
| 1.5 : 1 | 0.347 | 0.244 | 0.236 (×1.28) |
| 2 : 1 | 0.453 | 0.282 | 0.269 (×1.34) |
| 3 : 1 | 0.664 | 0.358 | 0.318 (×1.41) |

   At every ratio the biased-low pooled forecast is **more expensive than the naive baseline** at portfolio level (+17% at 1:1, +86% at 3:1), and the gap grows with the under-cost. (At household-day net level the two tie at 1:1 (0.286 vs 0.287) and the pooled model is worse once under-cost is higher: 0.403 vs 0.345 at 1.5:1, 0.753 vs 0.520 at 3:1.) *Hindsight uplift = a single scalar fitted on these same 84 days to show the size of the bias; it is in-sample, an illustration and not a valid method.
6. **Procurement-relevant weakness: yes.** Accuracy metrics favour the pooled model, but the procurement outcome is dominated by a one-sided portfolio bias that does not average out. The optimal forecast under asymmetric costs is a quantile (critical ratio = under/(under+over) = 0.50, 0.60, 0.67, 0.75 for the four ratios); the current point forecast sits well below even the median at portfolio level (under-forecast in 80–91% of intervals). This points to quantile/uncertainty forecasting and bias control, i.e. Level 3.

## Descriptive PV vs non-PV (pooled model; NOT evidence for separate models)
| Group | Household-interval MAE | Household bias [kWh/15 min] | Daily-energy MAE [kWh] | Daily bias [kWh] | Days under / over |
|---|--:|--:|--:|--:|--:|
| PV (131) | 0.148 | −0.048 | 7.01 | −4.59 | 70% / 30% |
| non-PV (110) | 0.165 | −0.046 | 7.24 | −4.44 | 76% / 24% |
PV and non-PV households are under-forecast by almost the same amount; non-PV households are somewhat harder. All non-PV households are treatment households, so this contrast is confounded with the intervention.

## Limitations
- 12 sampled weeks only (not the full year); seasonal means are 3 weeks each.
- Portfolio = the households with data on each day (≈221 of 241 per day); actual and forecast are summed over the same households.
- MAPE excludes intervals with actual < 0.05 kWh (31%); do not rank on MAPE alone.
- Costs are parametric; no price data; no imbalance-settlement rules or hourly/half-hourly products modelled.
- Target is net received energy (no PV production/export data).
- Single seed; model fixed after 2023-03-15 (no refit), so drift and the L1 objective cannot be separated here.

## Recommended next experiment
**Bias-corrected / quantile forecasting of the same pooled model:** refit the pooled AutoGluon with quantile levels (e.g. 0.5/0.6/0.75, using its probabilistic outputs) or a mean-objective (RMSE/L2) variant, and evaluate the same 12 weeks on portfolio-day bias and the cost ratios above. This tests directly whether the portfolio under-forecast is removable and gives the quantile forecasts needed for Level 3, without new features.

Files: `metrics.csv`, `portfolio_daily_metrics.csv`, `household_daily_metrics.csv`, `time_of_day_metrics.csv`, `cost_sensitivity.csv`, `summary_table.csv`, `run_metadata.json`, `plots/1…6`, log `outputs/level2_procurement_run.log`.
