# Day-ahead consumption forecast: model comparison

**Status:** complete (6 Oct 2026). **Best model: LightGBM with extra features ("v2"), 14.07% portfolio error.**

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

### Strict weather only

Every model follows the rules above, with either strict weather (past measured weather plus typical values) or consumption history only.

For reference only, we once checked what **perfect knowledge of tomorrow's weather** would give: about **8%** portfolio error (best: Chronos-2 at 7.97%). That is not a usable forecast, so it is not part of the comparison. It shows that a good weather forecast would be the biggest single improvement.

## 3. Models and inputs

| Family | Models | Inputs | Trained on our data |
|---|---|---|---|
| Baselines | Same hour last week · 7-day average · blend | Past consumption | – |
| Trees | scikit-learn HistGradientBoosting (HGB) · LightGBM · CatBoost | Lagged consumption (2, 3, 7, 14 days), recent averages, calendar, weather, household info | Yes |
| Deep learning | N-HITS · TFT | Last 28 days of hourly consumption, temperature, calendar | Yes |
| Deep learning | PatchTST | Last 28 days of hourly consumption only | Yes |
| Pretrained | Chronos-2 | Last 28 days of consumption, temperature, day-off flag | No (zero-shot) |
| Pretrained | Toto 2.0 | Last 28 days of consumption only | No (zero-shot) |
| Trees v2 | LightGBM · CatBoost with extra "recent consumption shape" features (section 7) | As trees, plus recent shape | Yes |
| Ensemble | Mean of the top 3, selected on the **tune** period | – | – |

**Which models we may use as our forecast:** all of the above. The pretrained models (Chronos-2, Toto 2.0) learned from large public datasets before we used them; all other models use only the provided data.

**Stopped and not reported:** DeepAR, TimesFM-3 and TiRex-2.
- **DeepAR:** the same family as N-HITS and TFT, which were clearly behind the trees.
- **TimesFM-3 and TiRex-2:** the same kind of model as Chronos-2, which was also behind. They also need hours of compute, and TimesFM-3's licence forbids commercial use.

**Benchmark:**
- **Floor:** the simple baselines. The best is the blend, at **15.7%** portfolio error.
- **Reference:** our original model, HGB strict, at **14.6%**.
- **Best achieved:** LightGBM v2, at **14.07%**.

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

## 5. Results (test year, portfolio of 255 households)

| Model | Hourly nMAE | vs HGB (pp, 95% CI) | Bias (kWh/h) | Daily nMAE | Household nMAE | Regret €/MWh |
|---|---|---|---|---|---|---|
| **LightGBM v2** † | **14.07%** | [−0.86, −0.06] | +0.4 | **11.44%** | 44.1% | **6.33** |
| CatBoost v2 † | 14.12% | [−0.93, +0.10] | +1.5 | 11.45% | 44.6% | 6.33 |
| CatBoost | 14.28% | [−0.58, −0.04] | +0.6 | 11.55% | 44.8% | 6.42 |
| Ensemble (top 3 on tune) | 14.40% | [−0.99, +0.95] | +5.8 | 11.65% | 44.2% | 6.38 |
| LightGBM | 14.51% | [−0.30, +0.18] | −0.2 | 11.74% | 44.8% | 6.53 |
| HGB (scikit-learn) | 14.58% | – | +0.2 | 11.90% | 44.8% | 6.56 |
| Toto 2.0 (consumption only) | 14.88% | [−0.60, +1.45] | −8.4 | 12.14% | **43.6%** | 6.84 |
| PatchTST (consumption only) | 15.21% | [−0.34, +1.77] | −16.3 | 12.22% | 46.9% | 7.13 |
| Blend of baselines | 15.70% | [+0.27, +1.95] | −5.9 | 13.06% | 45.4% | 7.17 |
| 7-day average | 16.15% | [+0.79, +2.34] | +1.4 | 13.55% | 46.1% | 7.24 |
| Chronos-2 | 16.95% | [+0.17, +5.08] | +15.6 | 14.23% | 46.3% | 7.35 |
| Same hour 2 days ago | 17.11% | [+1.35, +3.86] | +1.2 | 12.45% | 54.6% | 7.68 |
| N-HITS | 17.46% | [+1.94, +4.02] | +5.0 | 13.33% | 49.1% | 7.77 |
| TFT* | 20.10% | [+4.26, +7.26] | +10.9 | 15.68% | 57.0% | 8.85 |
| Same hour last week | 20.42% | [+4.11, +7.19] | +1.3 | 17.25% | 55.6% | 9.17 |

\* TFT is slow on CPU and got only about 55 training steps in the 15-minute budget.
† Follow-up experiment (section 7): the test year had already been seen when these features were chosen.

Full table: `results/leaderboard.csv`.

### Findings

1. **Tree models with good features win.** LightGBM v2 is the best model and reliably better than our original model.
2. **Recent consumption shape was the most useful extra input** (section 7).
3. **Toto 2.0 is the strongest model without training.** With consumption history only, it is close to the trees on the portfolio and has the lowest household error.
4. **Deep learning and Chronos-2 do not beat the trees** under the same rules and budget. Chronos-2 is excellent with perfect weather (7.97%) but weak with strict weather.
5. **The ensemble did not help.** It picked Chronos-2 as a member because Chronos-2 ranked 3rd on the two winter tuning months, but it was weak over the full year. Choosing on a short, one-season period can mislead.
6. **Weather is the biggest remaining gap.** Perfect weather would give about 8%; we are at 14%.

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

## 7. Improvement experiment (v2)

We added three groups of features that use only past data, and tested them on LightGBM:

| Feature group | What it adds | Tuning-period nMAE |
|---|---|---|
| Base (current features) | – | 14.01% |
| Weather persistence | Deviation from normal temperature 2 days before, temperature trend, hourly temperature and sunshine 2 days before, "expected temperature" = normal + recent deviation | 13.54% |
| **Recent consumption shape** | Last 6 hours available, morning and evening level 2 days before, same-weekday averages over 1–4 weeks, recent trend | **13.03%** ← selected |
| Peer groups | Average consumption of similar households 2 and 7 days back: same weather station, same station + PV status, all households | 13.79% |
| Weather + recent | | 13.03% |
| All three | | 13.28% |

**Protocol:**
1. The selection rule was fixed in advance: the lowest tuning-period error wins.
2. LightGBM and CatBoost were retrained on the winning set.
3. A blend with Toto 2.0 was tested on the tuning period; its best weight was 0, so no blend.
4. The test year was scored once.

**Result on the test year:** LightGBM 14.51% → **14.07%**, CatBoost 14.28% → **14.12%**.

**Caveat:** the test year had already been seen when we designed these features, so this is an exploratory follow-up, not an independent test.

## 8. Run

```bash
# core: data, baselines, trees, evaluation
uv venv --python 3.12 .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/python -m utils.data                 # raw CSVs -> data/processed/*.parquet
.venv/bin/python -m scripts.run_pipeline       # baselines + HGB, Level 2-3 analysis
.venv/bin/python -m scripts.run_trees          # LightGBM, CatBoost
.venv/bin/python -m scripts.run_improve        # v2 feature experiment (selection on tune block)

# deep learning (separate env, CPU)
uv venv --python 3.12 .venv-nf && uv pip install -p .venv-nf neuralforecast==3.2.2 polars numpy holidays pyarrow pandas
.venv-nf/bin/python -m scripts.run_neural nhits W0          # models: nhits tft (W0), patchtst (target-only)

# pretrained models (one env each; Apple GPU)
uv venv --python 3.12 .venv-fm && uv pip install -p .venv-fm chronos-forecasting==2.3.2 polars numpy holidays pyarrow pandas
.venv-fm/bin/python -m scripts.run_foundation chronos2 W0   # Toto 2.0: separate env with toto-models, track target-only

.venv/bin/python -m scripts.ensemble           # top-3 ensembles, selected on the tune period
.venv/bin/python -m scripts.leaderboard        # -> results/leaderboard.csv
```

## 9. Files

- `utils/data.py`: loader. Selects columns by name, builds a gap-free hourly grid, shifts weather to hour-start timestamps.
- `utils/features.py`: features, with checks that nothing uses data after the cutoff.
- `utils/features_extra.py`: v2 feature groups (weather persistence, recent shape, peer groups), with leakage checks.
- `utils/splits.py`: data periods, portfolio households, common forecast file format.
- `utils/evaluate.py`: metrics, portfolio, regret, uncertainty ranges, block bootstrap.
- `scripts/run_pipeline.py`, `run_trees.py`, `run_improve.py`, `run_neural.py`, `run_foundation.py`, `ensemble.py`, `leaderboard.py`: model runners and scoring.
- `results/`: leaderboard, metrics tables and figures.
