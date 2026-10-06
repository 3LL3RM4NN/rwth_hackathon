# Experiment decision log (branch `johannes_l2`)

Selected final candidate: **global LightGBM with L2 (squared-error) objective**, chosen on the controlled 12-week / 84-day screening experiment (experiments 6–10). It has **not** been run over the complete 2023-03-16 – 2024-02-27 test year, and no full-year LightGBM result exists. See [`METHODS.md`](METHODS.md).

Conventions: error = forecast − actual; MAE in kWh per 15-minute interval unless stated; "hh" = household. **Cutoff** is the information cutoff of the forecast: experiments 1–3 used the earlier **D−1 23:45** cutoff on the full test year (404 households); experiments 4–10 use the operational **D−1 11:45** cutoff (241 households), 4 on the full test year (Level 1b) plus a 12-week control, and 5–10 on the **12-week screening sample** (84 days). **Full-year evaluations: experiments 1–3 and the Level 1b part of 4. 12-week screening: the 12-week control in 4 and experiments 5–10.** Numbers with different cutoffs or samples are *not* directly comparable. Test period 2023-03-16 – 2024-02-27; development data end 2023-03-15.

## Decision table

| # | Experiment | Question | Main result | Decision |
|--:|---|---|---|---|
| 1 | Naive / seasonal baselines (`naive_1day`, `naive_7day`, `seasonal_mean_4weeks`); full year, 404 hh, 23:45 cutoff | What does a consumption-only benchmark achieve? | Overall MAE 0.1919 / 0.2019 / **0.1836**; mean hh MAE 0.1933 / 0.2038 / 0.1854; RMSE 0.3692 / 0.3778 / 0.3133; household daily-energy MAE **5.43** / 7.72 / 7.73 kWh; bias ≈ 0 | Kept as reference. `naive_1day` is invalid at an 11:45 cutoff; `naive_7day` and the 4-week mean remain valid (lags ≥ 7 days) |
| 2 | Pooled AutoGluon TimeSeries (SeasonalNaive + Recursive/Direct tabular, ensemble); full year, 404 hh, 23:45 cutoff | Does an ML ensemble beat the baselines? | Overall MAE **0.1578**, mean hh MAE 0.1593 (−14 % vs 4-week mean), RMSE 0.2977; bias −0.066 (under-forecast); household daily-energy MAE 7.58 (worse than `naive_1day`); fit 231 s, rolling prediction 1,439 s | Accepted as ML reference. Later **superseded** by LightGBM: in the controlled 12-week screening, the LightGBM L1 variant had better household-level MAE than AutoGluon (0.1527 vs 0.1565) and the selected L2 variant had substantially better portfolio-level performance (portfolio-day MAE 510 vs 1,233 kWh, bias +3.1 % vs −18.2 %) and household daily-energy MAE (5.62 vs 7.11 kWh), although its household interval MAE is higher (0.1686); LightGBM also has much lower computational cost (see 6, 7) |
| 3 | AutoGluon + static household features (survey, PV flag, region); full year, 404 hh, 23:45 cutoff | Do static household characteristics add power? | Mean hh MAE 0.1593 → 0.1592 (**−0.08 %**); 230 households improved / 174 worsened | **Rejected** (negligible). `Group` and `AffectsTimePoint` audited and excluded (intervention label / reveals visit timing); HeatPump/Other exist for only 17 households (4 % of rows) and were not used |
| 4 | PV vs non-PV separate AutoGluon models, 11:45 cutoff; Level 1b full year (241 hh) and 12-week control | Does splitting by PV ownership help? | Level 1b full year: mean hh MAE 0.1596 (split) vs 0.1603 (pooled, *23:45 cutoff*, not like-for-like). 12-week control vs a pooled 11:45 model: mean hh MAE 0.1548 vs 0.1564 (**−0.97 %**), daily-energy MAE +2.5 % worse, split better in only 4 of 12 weeks; non-PV −2.9 %, PV +0.8 % | **Rejected**: small, uneven gain; every non-PV household is a treatment household, so the effect is confounded with the intervention ([report](outputs/level1_fast_screening/REPORT.md)) |
| 5 | AutoGluon + historical temperature + calendar (weekend, month, national holiday); 12 weeks, 11:45 cutoff | Does weather/calendar information reduce the portfolio under-forecast? | Mean hh MAE 0.1565 → 0.1567 (+0.13 %); RMSE −0.4 %; household daily-energy MAE −3.3 %; portfolio-day MAE 1,233 → 1,161 kWh; **portfolio bias −18.2 % → −18.7 %**; 138 households better / 103 worse | **Rejected for AutoGluon**: bias not reduced. Weather (anchored at D−1 11:00) and calendar features are nevertheless part of the LightGBM feature set; no separate LightGBM ablation was run |
| 6 | LightGBM, L1 / MAE objective; 12 weeks, 11:45 cutoff | Can a simple explicit-feature global LightGBM match AutoGluon and be faster/more transparent? | Mean hh MAE **0.1527** vs 0.1565 (−2.5 %); overall MAE 0.1515 vs 0.1557; fit 12 s, predict 6 s (AutoGluon ≈ 226 s / 297 s); but **portfolio bias −21.4 %**, portfolio-day MAE 1,207 kWh, under-forecast on 94 % of days. Adding the PV flag: mean hh MAE 0.1523 (−0.26 %) | Accepted as the better *household* model and as the feature pipeline for all further work; its **objective was superseded** by experiment 7 (PV flag rejected) ([report](outputs/lightgbm_screening/REPORT.md)) |
| 7 | LightGBM, **L2 / squared-error** objective; 12 weeks | Does a mean-type objective remove the systematic portfolio under-forecast? | Portfolio-day MAE **510** kWh (−58 % vs L1), MAPE **8.4 %**, bias **+3.1 %**; cost 1:1 / 3:1 = 42.9 / 71.4 vs 101.4 / 301.6 MWh-eq (L1); better than L1 in 12/12 weeks. Household trade-off: mean hh MAE 0.1686 (+10 %), but RMSE 0.2860 (best) and household daily-energy MAE 5.62 (−21 %) | **Accepted — selected final candidate** (screening evidence only; no full-year evaluation yet) ([report](outputs/lightgbm_objective_screening/REPORT.md)) |
| 8 | LightGBM, quantile α = 0.60; 12 weeks | Does a mildly high quantile fix the bias with less household cost? | Bias **−2.7 %**; portfolio-day MAE 818 kWh, MAPE 16.5 %; cost 1:1 / 3:1 = 68.7 / 149.8 MWh-eq; mean hh MAE 0.1593 | Not selected; **secondary candidate** (smallest absolute bias, better household MAE than L2, but 60 % higher portfolio-day MAE and cost) |
| 9 | LightGBM, quantile α = 0.75; 12 weeks | Does a higher quantile suit asymmetric costs? | Portfolio bias **+34.6 %**; portfolio-day MAE 1,958 kWh, MAPE 29.9 %; cost 164 / 169 MWh-eq; mean hh MAE 0.1970; portfolio-day coverage 94 % (not 75 %) | **Rejected**: household quantiles do not add up to a portfolio quantile |
| 10 | Independent verification of L1 vs L2 (rebuilt matrices, retrained from scratch, independent metrics from raw CSVs) | Is the large L1/L2 gap real or an implementation/evaluation bug? | Predictions **bit-identical** to the stored ones (0 of 1,795,968 differ per model); metrics match to ≤ 2e-13; leakage perturbation tests passed; L1 predicted/actual energy ratio 0.785 vs L2 1.030; household-error cancellation in the portfolio total 24 % (L1) vs 59 % (L2) | **Verified** ([report](outputs/lightgbm_objective_verification/REPORT.md)) |

Procurement-analysis context (Level 2) for the pooled AutoGluon model: [`outputs/level2_procurement/REPORT.md`](outputs/level2_procurement/REPORT.md) — portfolio bias −18 %, 88 % of days under-forecast; this motivated experiments 7–9.

## Final conclusion

**Selected final candidate: global LightGBM with L2 objective** (selected on the 12-week screening; not yet evaluated over the full test year). It is trained on a household-level squared-error loss and its household forecasts are summed afterwards; it is not trained on a portfolio loss. The L1 / MAE model gives the best household-level interval MAE (0.1527) but behaves like a conditional-median model on a right-skewed, zero-inflated target and under-forecasts the aggregate portfolio by 21 %, which is costly for day-ahead procurement. The L2 model is 10 % worse on household MAE but reduces portfolio-day MAE by 58 %, brings the portfolio bias to +3 %, and is cheapest at every tested under:over cost ratio.

### L1 vs L2 vs quantiles (12-week screening, 241 households, 84 days, 1,783,680 intervals; independently verified)

| Metric | L1 / MAE | **L2 / mean** | Quantile 0.60 | Quantile 0.75 |
|---|--:|--:|--:|--:|
| *Portfolio / procurement* | | | | |
| Portfolio-day MAE (kWh) | 1,207 | **510** | 818 | 1,958 |
| Portfolio-day RMSE (kWh) | 1,462 | **825** | 1,030 | 2,656 |
| Portfolio-day MAPE (%) | 25.1 | **8.4** | 16.5 | 29.9 |
| Portfolio bias (%) | −21.4 | **+3.1** | −2.7 | +34.6 |
| Underforecast / overforecast (MWh) | 100.1 / 1.3 | 14.3 / 28.6 | 40.6 / 28.1 | 2.3 / 162.2 |
| % days underforecast | 94 | 36 | 69 | 6 |
| Cost 1:1 / 1.5:1 / 2:1 / 3:1 (MWh-eq) | 101 / 151 / 201 / 302 | **43 / 50 / 57 / 71** | 69 / 89 / 109 / 150 | 164 / 166 / 167 / 169 |
| *Household* | | | | |
| Mean / median household MAE | **0.1527 / 0.1423** | 0.1686 / 0.1558 | 0.1593 / 0.1485 | 0.1970 / 0.1845 |
| Overall interval MAE | **0.1515** | 0.1676 | 0.1585 | 0.1964 |
| Interval RMSE | 0.2988 | **0.2860** | 0.2929 | 0.3205 |
| MAPE / sMAPE (%) | **66.0 / 75.7** | 90.9 / 83.3 | 82.6 / 77.6 | 128.0 / 86.3 |
| Household daily-energy MAE (kWh) | 7.13 | **5.62** | 6.24 | 10.79 |

Reference on the same sample: `seasonal_mean_4weeks` portfolio-day MAE 930 kWh, MAPE 15.1 %, bias +5.5 %, cost 1:1 78 MWh-eq; pooled AutoGluon portfolio-day MAE 1,233 kWh, MAPE 24.1 %, bias −18.2 %.

### Status and scope
- All LightGBM results (6–9) and the verification (10) are on the 12-week screening sample with one seed, no hyperparameter tuning and no refitting during the test period; the L2 model reached the 500-tree cap. The L2 model has not been run or evaluated over the full test year; that evaluation is still outstanding, and nothing here should be read as a full-year result.
- Excluded throughout: `Group`, `AffectsTimePoint`, HeatPump/Other consumption, any actual weather of the forecast day. The target is net received energy; PV production is not observed.
- Details, definitions and limitations: [`METHODS.md`](METHODS.md).
