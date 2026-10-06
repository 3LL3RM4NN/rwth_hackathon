# Day-ahead consumption forecast: model comparison

Hourly forecast of grid electricity consumption for each household on the next day. The households are then summed into a 255-household portfolio, which is the quantity a supplier buys on the day-ahead market. The comparison covers simple baselines, tree models, deep learning and pretrained time-series models, all under the same rules.

**Status:** in progress (6 Oct 2026). Rows marked ⏳ in the table are still running.

## Forecast rules (identical for every model)

| | |
|---|---|
| Forecast made | 10:00 on the day before delivery (the Swiss day-ahead gate closes at 11:00) |
| Target | kWh per hour of grid consumption (`kWh_received_Total`), for each hour of the next local day (23/24/25 h) |
| Meter data allowed | Up to the end of 2 days before delivery. Yesterday is not complete at 10:00. |
| Weather allowed (**strict**) | Measured weather up to 2 days before, plus the typical temperature for that date and hour |
| Weather **actual** (best case only) | The measured weather of the delivery day. It stands in for a perfect weather forecast, so it shows an upper bound and is **not** a usable forecast. |
| Data periods | Train ≤ 29 Oct 2022 · tune Nov–Dec 2022 · calibrate Jan–Feb 2023 · **test 1 Mar 2023 – 27 Feb 2024** (never used before final scoring) |
| Portfolio | 255 households fixed at the end of 2022. Each hour is summed over the households whose meters reported (96.9% on average). |
| Training budget | At most 15 min per trained model, default settings |

PV production is not in the data and cannot be forecast. For PV owners the target is their net grid consumption, which already includes the effect of their panels.

## Models

| Family | Models | Inputs | Trained on our data |
|---|---|---|---|
| Baselines | Same hour last week · 7-day average · blend | Past consumption | – |
| Trees | scikit-learn HistGradientBoosting · LightGBM · CatBoost | Lagged consumption, calendar, weather, household info | Yes |
| Deep learning | N-HITS · TFT · DeepAR (PatchTST: consumption only) | Last 28 days of hourly consumption, temperature, calendar | Yes |
| Pretrained | Chronos-2 · TiRex-2 · TimesFM-3 (Toto 2.0: consumption only) | Last 28 days of hourly consumption, temperature, day-off flag | No (zero-shot) |
| Ensemble | Mean of the top 3, selected on the tuning period | – | – |

Pretrained models were trained by their developers on large public datasets. TimesFM-3 weights are licensed for non-commercial use only, so it is included for comparison and is not a deliverable model.

## Results so far (test year, portfolio of 255 households)

nMAE is the mean absolute error divided by mean consumption (282 kWh per hour), so lower is better. The CI column is the 95% interval of the difference from our original model (HGB, strict), from a 7-day block bootstrap. Regret is the simulated cost of buying the forecast, assuming 50 €/MWh to cover a shortfall and 40 €/MWh for surplus.

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
| Same hour last week | 20.42% | [+4.11, +7.19] | +1.3 | 17.25% | 55.6% | 9.17 |
| TFT · PatchTST · DeepAR · TiRex-2 · TimesFM-3 · Ensemble | ⏳ | | | | | |

**Actual weather (upper bound, not usable)**

| Model | Hourly nMAE | Daily nMAE |
|---|---|---|
| HGB | 8.01% | 5.54% |
| LightGBM | 8.04% | 5.54% |
| CatBoost | 8.09% | 5.60% |
| N-HITS | 14.97% | 10.98% |
| Chronos-2 · TFT · DeepAR · Ensemble | ⏳ | |

### Findings so far

1. **Tree models lead.** CatBoost is the best usable model, and its small gain over HGB is statistically clear.
2. **Toto 2.0 is the surprise.** With consumption history only and no training, it is tied with the tree models at portfolio level and has the lowest household-level error.
3. **Deep learning and Chronos-2 do not beat the trees** under the same rules and budget. Chronos-2 over-forecasts by about 16 kWh per hour.
4. **Weather is the biggest lever.** Every tree model drops from about 14.5% to about 8% with actual weather. A good day-ahead weather forecast would recover part of that gap.

Full table: `results/leaderboard.csv`.

### Uncertainty and buying (Levels 2–3)

- Uncertainty ranges come from the portfolio errors of the last 56 days.
- Bidding a cost-optimal quantile instead of the forecast cut simulated regret by 21–33% when costs were lopsided (1:4 and 4:1), and did not help when they were near-symmetric.
- The 10–90% range covered 76% of hours.
- Details: `results/regret_sweep.csv`, `results/quantile_coverage.csv`, `results/regret_sweep.png`, `results/portfolio_weeks.png`.

## Run

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

.venv/bin/python -m scripts.ensemble           # top-3 ensembles, selected on the tuning period
.venv/bin/python -m scripts.leaderboard        # -> results/leaderboard.csv
```

Track names: `W0` = strict weather, `W2_oracle` = actual weather (upper bound), `target-only` = consumption only.

## Files

- `utils/data.py`: loader. Selects columns by name, builds a gap-free hourly grid, shifts weather to hour-start timestamps.
- `utils/features.py`: features, with checks that nothing uses data after the cutoff.
- `utils/splits.py`: data periods, portfolio households, common forecast file format.
- `utils/evaluate.py`: metrics, portfolio, regret, quantile bids, block bootstrap.
- `scripts/run_pipeline.py`, `run_trees.py`, `run_neural.py`, `run_foundation.py`, `ensemble.py`, `leaderboard.py`: model runners and scoring.
