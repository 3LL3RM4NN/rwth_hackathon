# Day-ahead consumption forecast: model comparison

**Status:** in progress (6 Oct 2026). Rows marked ⏳ are still running.

## 1. What we forecast

- **What:** each household's **electricity drawn from the grid, in kWh per hour** (`kWh_received_Total`), for every hour of **tomorrow**. That is 24 hours, or 23/25 on clock-change days.
- **When:** at **10:00 the day before**, just before the Swiss day-ahead market closes at 11:00.
- **Portfolio:** we sum the forecasts of **255 households** fixed at the end of 2022. That total is what a supplier actually buys, so it is our **main score**.
- **Test:** one full year, **1 Mar 2023 – 27 Feb 2024**, that no model saw during training or tuning.
- **No PV production:** it is not in the data, so it cannot be forecast. For PV owners we forecast their net grid consumption, which already includes their solar.

## 2. Rules (identical for every model)

At 10:00 the day before, a model may only use what would really be known then.

| Allowed | Not allowed |
|---|---|
| Meter data up to the end of the **day before yesterday** (yesterday is not complete at 10:00) | Any consumption from yesterday or tomorrow |
| Calendar: hour, weekday, Swiss holidays | **Tomorrow's actual measured weather** |
| Measured weather up to the day before yesterday, plus the **typical** temperature for that date and hour | |
| Household info: PV yes/no, size, heat-pump type, visit status | |

**Data periods:**

| Period | Dates | Used for |
|---|---|---|
| Train | ≤ 29 Oct 2022 | Fitting models |
| Tune | Nov – Dec 2022 | Early stopping, choosing ensemble members |
| Calibrate | Jan – Feb 2023 | First window of errors for the uncertainty ranges |
| **Test** | **Mar 2023 – Feb 2024** | Final scoring only |

Every trained model gets **at most 15 minutes** of training and its default settings.

### Weather variants ("tracks")

- **Strict (`W0`):** follows the rules above. **These are real, usable forecasts.**
- **Consumption only (`target-only`):** past consumption only, also usable.
- **Actual weather (`W2_oracle`):** uses tomorrow's measured weather, i.e. a perfect weather forecast. **Not a usable forecast.** It only shows the best case if we had an excellent weather forecast.

## 3. Models and inputs

| Family | Models | Inputs | Trained on our data |
|---|---|---|---|
| Baselines | Same hour last week · 7-day average · blend | Past consumption | – |
| Trees | scikit-learn HistGradientBoosting (HGB) · LightGBM · CatBoost | Lagged consumption (2, 3, 7, 14 days), recent averages, calendar, weather, household info | Yes |
| Deep learning | N-HITS · TFT · DeepAR | Last 28 days of hourly consumption, temperature, calendar | Yes |
| Deep learning | PatchTST | Last 28 days of hourly consumption only | Yes |
| Pretrained | Chronos-2 · TiRex-2 · TimesFM-3 | Last 28 days of consumption, temperature, day-off flag | No (zero-shot) |
| Pretrained | Toto 2.0 | Last 28 days of consumption only | No (zero-shot) |
| Ensemble | Mean of the top 3, selected on the **tune** period | – | – |

**Which models we may use as our forecast:**
- ✅ Every **strict** and **consumption-only** model.
- ❌ **Actual-weather** versions: an upper bound only.
- ❌ **TimesFM-3:** its weights are licensed for non-commercial use, so it is for comparison only.
- ⚠️ **Pretrained models** learned from large public datasets before we used them. The tree and deep-learning models use only the provided data.

**Benchmark:**
- **Floor:** the simple baselines. The best is the blend, at **15.7%** portfolio error.
- **Reference:** our original model, HGB strict, at **14.6%**.
- **Ceiling:** the actual-weather versions, at about **8%**.

## 4. How to read the results table

| Column | Meaning |
|---|---|
| **Hourly nMAE** | Average absolute gap between forecast and actual, per hour, divided by average consumption (282 kWh per hour). **Main score; lower is better.** 14% means off by about 40 kWh per hour on average. |
| **vs HGB (pp, 95% CI)** | Difference in hourly nMAE from HGB strict, in percentage points, with a 95% interval. **Entirely below 0:** reliably better. **Entirely above 0:** reliably worse. **Spans 0:** no real difference. |
| **Bias (kWh/h)** | Average of forecast minus actual. **+** means it buys too much, **−** too little. Near 0 is good. |
| **Daily nMAE** | Same as hourly nMAE but on **daily totals**. Hourly misses partly cancel, so it is lower. Uses the 362 complete days only. |
| **Household nMAE** | Same error measure for **single homes**, pooled over all household-hours. Shows how unpredictable one home is. |
| **Regret (€/MWh)** | **Simulated extra cost** of buying the forecast instead of having perfect knowledge: a shortfall costs +50 €/MWh and a surplus loses 40 €/MWh (assumed prices). Per MWh consumed; 6.4 €/MWh is about 6,400 € per GWh. |

The portfolio total in each hour is summed over the households whose meters reported in that hour (96.9% on average). Forecast and actual always cover the same households.

## 5. Results so far (test year, portfolio of 255 households)

**Strict weather / consumption only (usable forecasts)**

| Model | Hourly nMAE | vs HGB (pp, 95% CI) | Bias (kWh/h) | Daily nMAE | Household nMAE | Regret €/MWh |
|---|---|---|---|---|---|---|
| **CatBoost** | **14.28%** | [−0.58, −0.04] | +0.6 | **11.55%** | 44.8% | **6.42** |
| LightGBM | 14.51% | [−0.30, +0.18] | −0.2 | 11.74% | 44.8% | 6.53 |
| HGB (scikit-learn) | 14.58% | – | +0.2 | 11.90% | 44.8% | 6.56 |
| Toto 2.0 (consumption only) | 14.88% | [−0.60, +1.45] | −8.4 | 12.14% | **43.6%** | 6.84 |
| Blend of baselines | 15.70% | [+0.27, +1.95] | −5.9 | 13.06% | 45.4% | 7.17 |
| Chronos-2 | 16.95% | [+0.17, +5.08] | +15.6 | 14.23% | 46.3% | 7.35 |
| N-HITS | 17.46% | [+1.94, +4.02] | +5.0 | 13.33% | 49.1% | 7.77 |
| TFT* | 20.10% | [+4.26, +7.26] | +10.9 | 15.68% | 57.0% | 8.85 |
| Same hour last week | 20.42% | [+4.11, +7.19] | +1.3 | 17.25% | 55.6% | 9.17 |
| PatchTST · DeepAR · TiRex-2 · TimesFM-3 · Ensemble | ⏳ | | | | | |

\*TFT is slow on CPU and got only about 55 training steps in the 15-minute budget. A lighter TFT setting is planned.

**Actual weather (upper bound, not usable)**

| Model | Hourly nMAE | Daily nMAE |
|---|---|---|
| HGB | 8.01% | 5.54% |
| LightGBM | 8.04% | 5.54% |
| CatBoost | 8.09% | 5.60% |
| N-HITS | 14.97% | 10.98% |
| Chronos-2 · TFT · DeepAR · Ensemble | ⏳ | |

Full table: `results/leaderboard.csv`.

### Findings so far

1. **Tree models lead.** CatBoost is the best usable model, and its small gain over HGB is statistically clear.
2. **Toto 2.0 is the surprise.** With consumption history only and no training, it ties the tree models on the portfolio and has the lowest household error.
3. **Deep learning and Chronos-2 do not beat the trees** under the same rules and budget.
4. **Weather is the biggest lever.** The tree models drop from about 14.5% to about 8% with actual weather. A good day-ahead weather forecast would recover part of that gap.

## 6. Uncertainty

We handle uncertainty in two places: the forecast itself (Level 3) and our results.

### 6a. A range for every forecast hour

For every hour of tomorrow we give a **range** in addition to the forecast.

1. Take the portfolio's **relative errors over the last 56 days**, using only days already known at bid time.
2. Do it **separately for each hour of the day**.
3. Apply the 10th and 90th percentile of those errors to tomorrow's forecast.

*Example:* the forecast for 18:00 is 300 kWh. Recent 18:00 errors ranged from −12% to +15%, so the range is about 264–345 kWh.

- **Why on the portfolio:** household ranges cannot simply be added up, so the portfolio range comes from the portfolio's own errors.
- **Same method for every model**, for fairness. Some pretrained models output their own ranges, but those are per household.

**How well the range works (HGB strict):**

| Check | Target | Achieved |
|---|---|---|
| Actual below the low end (10%) | 10% of hours | 12.8% |
| Actual below the high end (90%) | 90% of hours | 89.0% |
| Actual inside the 10–90% range | 80% of hours | 76%, slightly too narrow |

Because the range is a percentage of the forecast, it is wider in winter, when consumption is higher (`results/portfolio_weeks.png`).

### 6b. Using the range to decide how much to buy (Level 2–3)

If buying too little costs more than buying too much, buy **above** the forecast, and vice versa. The cost ratio sets the exact point in the range: **τ = c_under / (c_under + c_over)**. With 50 / 40 €/MWh, τ = 0.556.

| Cost situation | Effect of bidding at τ instead of the forecast |
|---|---|
| Lopsided (one error 4× more expensive) | **21–33% lower simulated cost** |
| Near-equal (our 50/40 assumption) | No gain, slightly worse |

The range pays off when one kind of mistake is much more expensive than the other. Details: `results/regret_sweep.csv`, `results/regret_sweep.png`, `results/quantile_coverage.csv`.

### 6c. Uncertainty of our results

The test year has only so many weeks, so a small difference between two models can be luck. We resample the test year in **7-day blocks** (2,000 times) to get a 95% interval for each model comparison. That is the "vs HGB" column.

**Caveats:** results are retrospective (a simulation on past data), and the prices are assumed, not real market prices.

## 7. Run

```bash
# core: data, baselines, trees, evaluation
uv venv --python 3.12 .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/python -m utils.data                 # raw CSVs -> data/processed/*.parquet
.venv/bin/python -m scripts.run_pipeline       # baselines + HGB, Level 2-3 analysis
.venv/bin/python -m scripts.run_trees          # LightGBM, CatBoost

# deep learning (separate env, CPU)
uv venv --python 3.12 .venv-nf && uv pip install -p .venv-nf neuralforecast==3.2.2 polars numpy holidays pyarrow pandas
.venv-nf/bin/python -m scripts.run_neural nhits W0          # models: nhits tft patchtst deepar; tracks: W0 W2_oracle target-only

# pretrained models (one env each; Apple GPU)
uv venv --python 3.12 .venv-fm && uv pip install -p .venv-fm chronos-forecasting==2.3.2 polars numpy holidays pyarrow pandas
.venv-fm/bin/python -m scripts.run_foundation chronos2 W0   # also: tirex2 (tirex-2), timesfm3 (timesfm[mlx]), toto2 (toto-models) target-only

.venv/bin/python -m scripts.ensemble           # top-3 ensembles, selected on the tune period
.venv/bin/python -m scripts.leaderboard        # -> results/leaderboard.csv
```

## 8. Files

- `utils/data.py`: loader. Selects columns by name, builds a gap-free hourly grid, shifts weather to hour-start timestamps.
- `utils/features.py`: features, with checks that nothing uses data after the cutoff.
- `utils/splits.py`: data periods, portfolio households, common forecast file format.
- `utils/evaluate.py`: metrics, portfolio, regret, uncertainty ranges, block bootstrap.
- `scripts/run_pipeline.py`, `run_trees.py`, `run_neural.py`, `run_foundation.py`, `ensemble.py`, `leaderboard.py`: model runners and scoring.
- `results/`: leaderboard, metrics tables and figures.
