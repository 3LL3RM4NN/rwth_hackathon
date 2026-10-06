# LightGBM objective screening for portfolio procurement (D-1 11:45 cutoff)

Features, cutoff, split, 25% row sample, 365-day window, 500 trees / lr 0.05 / 31 leaves / seed 0, household set (241) and the 84 sampled days are **identical** to `outputs/lightgbm_screening/` (the l1 model here reproduces it exactly, max |diff| = 0). Only the objective differs; early stopping uses each model's own validation loss. L2 and both quantile models hit the 500-tree cap (not converged; not tuned). Error = forecast − actual; portfolio-day = sum of all scored households per forecast day (≈221 households/day on common complete household-days). Cost = under-cost × max(actual−pred,0) + over-cost × max(pred−actual,0) with over-cost = 1; costs are in kWh-equivalents, for evaluation only (nothing tuned on the test weeks).

## Portfolio / procurement (primary)
| | MAE l1 (current) | L2 (mean) | Quantile 0.60 | Quantile 0.75 | seasonal mean 4w |
|---|--:|--:|--:|--:|--:|
| Portfolio-day MAE [kWh] | 1,207 | **510** | 818 | 1,958 | 930 |
| Portfolio-day MAPE [%] (households aggregated first) | 25.1 | **8.4** | 16.5 | 29.9 | 15.1 |
| Portfolio-day RMSE [kWh] | 1,462 | **825** | 1,030 | 2,656 | 1,366 |
| Portfolio bias [kWh/day] | −1,176 | +171 | **−148** | +1,904 | +301 |
| Portfolio bias % | −21.4 | +3.1 | **−2.7** | +34.6 | +5.5 |
| Underforecast [MWh] | 100.1 | 14.3 | 40.6 | **2.3** | 26.4 |
| Overforecast [MWh] | **1.3** | 28.6 | 28.1 | 162.2 | 51.7 |
| Net over−under [MWh] | −98.8 | +14.3 | **−12.4** | +159.9 | +25.3 |
| % days underforecast | 94.0 | 35.7 | 69.0 | 6.0 | 38.1 |
| Cost 1:1 total [MWh-eq] / per day [kWh-eq] | 101.4 / 1,207 | **42.9 / 510** | 68.7 / 818 | 164.5 / 1,958 | 78.1 / 930 |
| Cost 1.5:1 | 151.4 / 1,803 | **50.0 / 595** | 89.0 / 1,059 | 165.6 / 1,971 | 91.3 / 1,087 |
| Cost 2:1 | 201.5 / 2,399 | **57.1 / 680** | 109.3 / 1,301 | 166.7 / 1,985 | 104.5 / 1,244 |
| Cost 3:1 | 301.6 / 3,590 | **71.4 / 850** | 149.8 / 1,784 | 169.0 / 2,012 | 130.9 / 1,558 |
| Coverage actual ≤ pred: portfolio-day | 6% | 64% | 31% | 94% | 62% |
| Coverage actual ≤ pred: household×interval | 55.6% | 69.6% | 64.2% | 77.0% | 62.7% |

(Stored pooled AutoGluon for reference: portfolio-day MAPE 24.1%, portfolio-day MAE 1,233, bias −18.2%, cost 1:1 / 3:1 = 103.6 / 291.1 MWh-eq.)

## Household accuracy (same 241 households, 1,783,680 intervals)
| | l1 (current) | L2 | q0.60 | q0.75 | seasonal mean 4w | pooled AutoGluon |
|---|--:|--:|--:|--:|--:|--:|
| Mean HH MAE | **0.1527** | 0.1686 | 0.1593 | 0.1970 | 0.1829 | 0.1565 |
| Median HH MAE | **0.1423** | 0.1558 | 0.1485 | 0.1845 | 0.1673 | 0.1441 |
| Overall MAE | **0.1515** | 0.1676 | 0.1585 | 0.1964 | 0.1822 | 0.1557 |
| RMSE | 0.2988 | **0.2860** | 0.2929 | 0.3205 | 0.3187 | 0.2993 |
| MAPE % / sMAPE % | **66.0 / 75.7** | 90.9 / 83.3 | 82.6 / 77.6 | 128.0 / 86.3 | 100.3 / 80.0 | 71.1 / 77.6 |
| Daily-energy MAE [kWh] | 7.13 | **5.62** | 6.24 | 10.79 | 7.15 | 7.11 |
| Interval portfolio MAE [kWh/15 min] | 13.46 | **8.01** | 10.05 | 20.99 | 11.82 | 13.83 |
| Households better / worse than l1 | – | 5 / 236 | 23 / 218 | 2 / 239 | – | 52 / 189 |

(MAPE/sMAPE/MAE favour the median-type l1 because most household intervals are near zero; RMSE and daily-energy MAE favour the mean.)

## Ranking
1. **Procurement cost (all ratios 1:1–3:1):** L2 < q0.60 < l1 (q0.75 is worst at 1:1 and 1.5:1; at 2:1–3:1 it passes l1 only because l1's under-forecast is penalised more). L2 is cheapest at every ratio, with the lead growing from 1.6× (vs q0.60 at 1:1) to 2.1× at 3:1.
2. **Portfolio-day MAE:** L2 (510) < q0.60 (818) < l1 (1,207) < q0.75 (1,958). Portfolio-day MAPE gives the same order: L2 (8.4%) < q0.60 (16.5%) < l1 (25.1%) < q0.75 (29.9%); only L2 beats the 4-week seasonal mean (15.1%).
3. **Portfolio bias (|%|):** q0.60 (−2.7%) < L2 (+3.1%) < l1 (−21.4%) < q0.75 (+34.6%).
4. **Household MAE:** l1 < q0.60 < L2 < q0.75.

## Trade-off and robustness
- The −21% portfolio bias of the MAE model **is removable by the objective**: L2 gives +3.1%, q0.60 gives −2.7%. The l1 bias was caused by the median-seeking objective on zero-inflated, spiky household loads; the sum of per-household medians understates the portfolio total.
- Price of the fix: household MAE +10% (L2) / +4% (q0.60) and worse MAPE/sMAPE, but RMSE (−4%) and household daily-energy MAE (−21%) are better with L2. For a portfolio buyer the household-interval MAE is the wrong yardstick; portfolio-day error falls by 58%.
- q0.75 over-shoots (+35%, covers 94% of days): the quantile that is nominally right for a 3:1 cost ratio is far too high at portfolio level because household-level quantiles do not add up to a portfolio quantile (portfolio-day coverage is 94%, not 75%; household-interval coverage 77%).
- L2 is stable over the sample: better than l1 in 12/12 weeks and 70/84 days (mean daily |error| −697 kWh, s.e. 94); portfolio bias within −5.7% … +11% in every week, by season +0.4 … +6.4%; better than the seasonal mean on 56/84 days (−419 kWh, s.e. 93). The l1 model is −13% … −31% in every season.
- Caveats: 12 sampled weeks, one seed, L2/quantile models not converged at 500 trees, quantile-coverage figures are descriptive only; no hyperparameter was tuned on the test weeks; costs are parametric.

## Recommendation
Carry the **L2 (mean regression) objective** to the full-year test evaluation: lowest procurement cost at every ratio, lowest portfolio-day MAE/RMSE, small positive bias (+3%), best household RMSE and daily-energy MAE, stable across weeks. Keep **q0.60 as secondary** (smallest absolute bias, better household MAE than L2). Do not use q0.75 as a point forecast. For asymmetric costs above ~1.5:1, the next step is a small bias/quantile calibration on the *development* period (e.g. a portfolio-level scale factor), not a higher household quantile.
