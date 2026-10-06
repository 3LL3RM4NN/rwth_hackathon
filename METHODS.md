# Methods — day-ahead household load forecasting (branch `johannes_l2`)

Selected final candidate: **one global LightGBM regressor with a squared-error (L2) objective**, trained on household × day × quarter-hour rows, direct (non-recursive) multi-horizon forecasts, evaluated at household and portfolio level. It was selected on the controlled **12-week / 84-day screening experiment**; it has **not** yet been run or evaluated over the complete 2023-03-16 – 2024-02-27 test year, so no full-year performance is claimed.
Code: [`src/lgbm_features.py`](src/lgbm_features.py) (features), [`src/covariates.py`](src/covariates.py) (weather/calendar), [`scripts/run_lightgbm_objectives.py`](scripts/run_lightgbm_objectives.py) (training/evaluation), [`scripts/verify_lightgbm_objectives.py`](scripts/verify_lightgbm_objectives.py) (independent verification). Decision history: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## 1. Objective

Forecast the next day's electricity consumption of each heat-pump household (target `kWh_received_Total`, 15-minute resolution) so that the aggregate volume can be bought on the day-ahead market. The target is the **net received energy**; the data contain no PV production, export or returned-energy series.

## 2. Forecast protocol

| Item | Definition |
|---|---|
| Forecast day | D (UTC calendar day) |
| Information cutoff (forecast origin) | **D−1 11:45** |
| Forecast horizon | D 00:00 – D 23:45, 96 × 15-minute intervals (steps 49–144 after the origin) |
| Allowed information | observations with timestamp ≤ D−1 11:45 only |
| Not used | any value from D−1 12:00 onward, any target-day (D) observation, any weather observation after the cutoff, any weather from day D |

The forecast is one day ahead and operational: a day-D forecast could really be produced at D−1 11:45. There is **no weather forecast** in the data; weather enters only as *historical observations* (Section 5). All timestamps are UTC; a normal day has exactly 96 intervals.

Missing data are never interpolated. A day with fewer than 96 observed intervals is unusable as history (set to NaN as a whole day) and is not scored; the only exception is day D−1, whose morning slots (00:00–11:45) are taken from the raw series, because a completeness mask would depend on afternoon values after the cutoff.

## 3. Training protocol

| Item | Value (verified in code / `outputs/lightgbm_objective_screening/`) |
|---|---|
| Development period | ends **2023-03-15**; nothing later is used for training, validation or feature construction of training rows |
| Held-out test period | **2023-03-16 – 2024-02-27** (chronological, last ≈20% of the calendar; no shuffling) |
| Training window | **365 days** of target days, 2022-03-16 – 2023-03-15 |
| Fit / validation split inside the window | fit on target days 2022-03-16 – 2023-02-15 (337 days); **internal validation** on the last 28 days, 2023-02-16 – 2023-03-15, used only for early stopping |
| Households | 245 with a known PV flag are used for training (the 165 with unknown PV status are excluded); 241 of them have usable test days and are scored |
| Row eligibility | target day D with day D−1 present and ≥ 7 usable days in D−42 … D−2; rows with a missing target are dropped |
| Row sampling | a random **25 %** of the fit rows (NumPy generator, seed 0) → 1,266,767 training rows; all 549,504 validation rows are used |
| Test simulation | the model is fitted **once and frozen**; it is not refitted during the test period (only the inputs move forward day by day). In the screening experiment it is applied to the 84 sampled test days only |
| Library | LightGBM **4.7.0**, `LGBMRegressor`, CPU, `n_jobs=8` |
| Hyperparameters (not tuned) | `n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.8` (`subsample_freq=1`), `colsample_bytree=0.8`, `random_state=0`, early stopping 50 rounds on the validation loss of the model's own objective; `household` passed as `categorical_feature` |
| Final objective | `objective="regression"` (L2); it reached the 500-tree cap (best iteration 500), i.e. it is not converged |
| Post-processing | predictions are clipped at 0 kWh |

Quoted screening results use a deterministic sample of **12 complete Monday–Sunday test weeks (84 days)**, 3 per season, chosen from data availability only ([`outputs/level1_fast_screening/selected_weeks.json`](outputs/level1_fast_screening/selected_weeks.json)). All LightGBM results in this repository (including the L1/L2/quantile comparison) are screening results on these 84 days; **no LightGBM model has been run over the complete test year**. Only the baselines, the AutoGluon experiments 2–3 and the Level 1b PV-split run (part of experiment 4) were evaluated over a full test year (see [`EXPERIMENTS.md`](EXPERIMENTS.md)).

## 4. Mathematical formulation

Let *i* be a household, *d* a forecast day and *h ∈ {1,…,96}* the quarter-hour slot of day *d*. Let $y_{i,d,h}$ be the actual consumption, $x_{i,d,h}$ the feature vector (Section 5, built only from information available at $d{-}1$ 11:45) and $f_\theta$ the global LightGBM model. The forecast is

$$
\hat y_{i,d,h} = \max\bigl(0,\; f_\theta(x_{i,d,h})\bigr).
$$

The selected model minimises the household-level squared error over the (sampled) training rows $\mathcal S$:

$$
\theta^{*}
=
\arg\min_{\theta}
\sum_{(i,d,h)\in\mathcal S}
\left(
y_{i,d,h}-f_\theta(x_{i,d,h})
\right)^2 .
$$

Each of the 96 slots is predicted **directly** from origin-anchored features; no predicted value is ever fed back into a later prediction (the slot index encodes the horizon).

Predictions stay **household-level**. For procurement they are aggregated *after* prediction, over the scored households:

$$
\hat Y_{d,h} = \sum_i \hat y_{i,d,h},
\qquad
\hat E_d = \sum_{h=1}^{96} \hat Y_{d,h},
\qquad
E_d = \sum_i \sum_{h=1}^{96} y_{i,d,h}.
$$

The model is **not trained on a portfolio-level loss**. Portfolio performance is only an evaluation criterion; it informed the choice among objectives (Section 7), but the training loss itself is household-level.

## 5. Exact feature vector (32 features, model-matrix order)

From `FEATURES_BASE` in [`src/lgbm_features.py`](src/lgbm_features.py). "Cutoff availability" refers to the origin D−1 11:45. Features 1–12 are known in advance; 13–23 depend on past consumption; 24–32 on past temperature.

| # | Feature | Group | Meaning | Availability at the cutoff |
|--:|---|---|---|---|
| 1 | `household` | household identity | integer code of the household (categorical) | static |
| 2 | `slot` | calendar / horizon | quarter-hour index 0–95 of the target day (step after origin = slot + 49) | known |
| 3 | `hour` | calendar | slot // 4 | known |
| 4 | `quarter` | calendar | slot % 4 | known |
| 5 | `dow` | calendar | day of week of D (0 = Monday) | known |
| 6 | `is_weekend` | calendar | dow ≥ 5 | known |
| 7 | `month` | calendar | month of D | known |
| 8 | `is_holiday` | calendar | German national public holiday (`holidays` package, no state/school holidays) | known |
| 9 | `slot_sin` | calendar | sin(2π·slot/96) | known |
| 10 | `slot_cos` | calendar | cos(2π·slot/96) | known |
| 11 | `dow_sin` | calendar | sin(2π·dow/7) | known |
| 12 | `dow_cos` | calendar | cos(2π·dow/7) | known |
| 13 | `lag_1d` | consumption lag | same slot on D−1 (t−96) | **only for slots 0–47 (00:00–11:45); NaN for slots 48–95** because D−1 afternoon is after the cutoff |
| 14 | `lag_2d` | consumption lag | same slot on D−2 (t−192) | available |
| 15 | `lag_7d` | consumption lag | same slot on D−7 (t−672) | available |
| 16 | `mean_same_slot_4wk` | seasonal statistic | mean of the same slot on D−7, D−14, D−21, D−28 (the seasonal-mean baseline as a feature) | available |
| 17 | `y_last` | recent level | consumption at the origin (D−1 11:45) | available; same for all 96 slots |
| 18 | `y_lag1h` | recent level | consumption 1 h before the origin | available; same for all 96 slots |
| 19 | `roll_mean_1h` | rolling | mean of the 4 slots ending at the origin | available; same for all slots |
| 20 | `roll_mean_6h` | rolling | mean of the 24 slots ending at the origin | available; same for all slots |
| 21 | `roll_mean_24h` | rolling | mean of the 96 slots ending at the origin | available; same for all slots |
| 22 | `roll_mean_7d` | rolling | mean of the 672 slots ending at the origin | available; same for all slots |
| 23 | `roll_std_24h` | rolling | std of the 96 slots ending at the origin | available; same for all slots |
| 24 | `temp_latest` | weather (historical) | station temperature at D−1 11:00 (latest hourly value complete at the cutoff) | available; constant over the day |
| 25 | `temp_lag1h` | weather | temperature 1 h before that value | as above |
| 26 | `temp_lag6h` | weather | 6 h before | as above |
| 27 | `temp_lag24h` | weather | 24 h before | as above |
| 28 | `temp_lag7d` | weather | 7 days before | as above |
| 29 | `temp_mean6h` | weather | mean of the 6 hours ending at D−1 11:00 | as above |
| 30 | `temp_mean24h` | weather | mean of the 24 hours ending at D−1 11:00 | as above |
| 31 | `temp_chg6h` | weather | `temp_latest` − value 6 h earlier | as above |
| 32 | `temp_chg24h` | weather | `temp_latest` − value 24 h earlier | as above |

Notes
- **Only `lag_1d` becomes structurally NaN for later slots** (slots 48–95, i.e. 12:00–23:45 of day D). All other features are defined for all 96 slots; any feature can also be NaN when the underlying data are missing (LightGBM handles NaN natively).
- Consumption windows (17–23) use raw values for the D−1 morning slots and the completeness-masked daily matrix for earlier days; day D−1 afternoon and day D are never read.
- Weather: each household is mapped to its station via `households.csv → Weather_ID` and `data/weather_data_hourly/<Weather_ID>.csv` (temperature only; other variables are missing for several stations). Missing hourly values are forward-filled (backward-looking). **Assumption:** the anchor hour D−1 11:00 presumes that hourly values are stamped at the *end* of the hour (so the 11:00 value is complete at the 11:45 cutoff). This convention is not documented in the data sheet; it was inferred from the sunshine/temperature pattern. If values were instead stamped at the start of the hour, the 11:00 value would include 15 minutes after the cutoff.
- Excluded on purpose: `Group` (control/treatment assignment), `AffectsTimePoint` (reveals the intervention timing), HeatPump/Other consumption, PV-ownership flag (tested separately, no useful gain, confounded with treatment status), actual weather of day D.

## 6. Leakage controls

1. **Post-cutoff perturbation test** (`verify_cutoff_features` in `scripts/run_lightgbm*.py`, repeated in `scripts/verify_lightgbm_objectives.py`): for random household-days, every target value (masked and raw series) and every weather observation **stamped after D−1 11:45 — D−1 afternoon and all of day D — was overwritten with 9999**, the features were regenerated, and all forecast-day features were asserted unchanged; `lag_1d` must be NaN for slots ≥ 48. The production runs checked 30 household-days. The verification script checked 24 household-days **per model (L1 and L2)** in two modes (everything after the cutoff, and only target day D), and additionally asserted that **the trained model's predictions were unchanged**. All checks passed.
2. **Independent feature recomputation:** 72 random (household, day, slot) rows were recomputed with plain pandas directly from the raw CSVs (lags, rolling statistics and temperature features); maximum difference 8.9e-7 (float32 rounding).
3. **Bitwise-identical matrices:** the feature matrices built separately for the L1 and L2 models were identical byte for byte.
4. **Independent reproduction:** [`outputs/lightgbm_objective_verification/REPORT.md`](outputs/lightgbm_objective_verification/REPORT.md) retrained L1 and L2 from scratch and reproduced the stored predictions **bit for bit** (0 of 1,795,968 predictions differ for each model). Metrics recomputed from raw-CSV actuals with independent code match the stored ones to ≤ 2e-13 on the like-for-like evaluation set.
5. The AutoGluon runs at the 11:45 cutoff (experiments 4–5 in `EXPERIMENTS.md`) used the same rule (context ending at the origin) with a perturbation test on inputs and, after training, on predictions; the earlier AutoGluon runs used a D−1 23:45 cutoff.

## 7. Model selection (objective function)

The feature set, cutoff, split, 25 % row sample and hyperparameters were held fixed; only the objective was changed ([`outputs/lightgbm_objective_screening/REPORT.md`](outputs/lightgbm_objective_screening/REPORT.md), controlled screening: 12 sampled weeks, 241 households, 84 days — not a full-year test). Each model is trained on a **household-level** loss; the portfolio metrics are computed afterwards by summing household forecasts per day. Nothing was tuned on the test weeks.

| | L1 (MAE) | **L2 (mean)** | Quantile 0.60 | Quantile 0.75 |
|---|--:|--:|--:|--:|
| Portfolio-day MAE (kWh) | 1,207 | **510** | 818 | 1,958 |
| Portfolio-day MAPE (%) | 25.1 | **8.4** | 16.5 | 29.9 |
| Portfolio bias (%) | −21.4 | **+3.1** | −2.7 | +34.6 |
| Underforecast / overforecast (MWh) | 100.1 / 1.3 | 14.3 / 28.6 | 40.6 / 28.1 | 2.3 / 162.2 |
| Cost 1:1 / 3:1 (MWh-eq) | 101 / 302 | **43 / 71** | 69 / 150 | 164 / 169 |
| Mean household MAE | **0.1527** | 0.1686 | 0.1593 | 0.1970 |
| Interval RMSE | 0.2988 | **0.2860** | 0.2929 | 0.3205 |
| Household daily-energy MAE (kWh) | 7.13 | **5.62** | 6.24 | 10.79 |

L1 gives the best household-interval MAE but behaves like a conditional-median model on a right-skewed, zero-inflated target (test mean 0.259 vs median 0.106 kWh): the sum of household medians understates the portfolio total, giving a −21 % bias and under-forecasting on 94 % of days. L2 estimates the conditional mean, so household errors cancel in the sum (cancellation 24 % → 59 %). L2 was therefore selected as the final candidate, on this screening evidence, because it gives the best portfolio-day MAE, RMSE, MAPE and procurement cost at every tested cost ratio while remaining a household-level model, at a price of +10 % household MAE. Quantile 0.60 is the closest alternative (smallest absolute bias); quantile 0.75 over-forecasts the portfolio by 35 % and is not usable as a point forecast. The model was **not** optimised for portfolio error — only the choice among objectives was informed by it.

## 8. Evaluation

Error is defined as **forecast − actual** (positive = over-forecast / over-procurement). Metrics are computed on identical timestamps, over complete household-days only.

- Household level: mean/median household interval MAE, overall interval MAE, RMSE, safe MAPE (actual ≥ 0.05 kWh), sMAPE, **household daily-energy MAE** (|Σ_h ŷ − Σ_h y| per household-day).
- Portfolio level (households summed first, per day): **portfolio-day MAE and RMSE, portfolio-day MAPE** (mean of |Ê_d − E_d| / E_d, not an average of household MAPEs), **portfolio bias** (kWh/day and % of actual), **under- and over-forecast energy** (MWh), % of days under-forecast, quantile coverage (actual ≤ prediction).
- **Asymmetric procurement cost** (evaluation only, no real prices): cost = r · max(E_d − Ê_d, 0) + max(Ê_d − E_d, 0), with under:over ratio r ∈ {1, 1.5, 2, 3}.

Portfolio metrics matter because day-ahead procurement buys the *aggregate* volume: errors that share a sign add up across households, whereas independent errors cancel. A model with the best household MAE can therefore still be the most expensive to procure with (L1), and household accuracy alone does not rank forecasts for this use case. Household-level metrics are kept to show the trade-off.

## 9. Limitations

- No real day-ahead/intraday prices or settlement rules; costs are parametric ratios.
- LightGBM results, including the selection of the L2 objective, come from the **12-week screening sample** (84 days, 241 households). The selected L2 model has not been evaluated over the full 2023-03-16 – 2024-02-27 test year, so full-year performance is not established. One seed; hyperparameters not tuned (the L2 model hit the 500-tree cap) and a 25 % training-row sample was used.
- The model is not refitted during the held-out period.
- The target is net received energy; PV production, export and returned energy are not observed, and PV ownership is only a static flag.
- Confounding: every household with a known "no PV" flag is a treatment (heat-pump optimisation) household, and the PV flag is missing for 165 households, mostly controls; a PV/non-PV contrast is therefore confounded with the intervention, and `Group`/`AffectsTimePoint` were excluded.
- Weather enters only as historical temperature anchored at the cutoff (no forecasts); the end-of-hour timestamp convention of the weather data is an inference.
- Holidays are German national holidays only (household state unknown).
- Households with fewer than 7 usable days in the preceding 42 days are not forecast; the 4-week seasonal mean and stored AutoGluon forecasts cover slightly fewer household-days than the LightGBM models (the reported common set has 1,783,680 intervals; the full LightGBM set 1,795,968).
