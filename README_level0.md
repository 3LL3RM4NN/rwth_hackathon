# Level 0 (+ Levels 2–3): day-ahead consumption forecast

Hourly forecast of grid consumption for every household on the next day, plus a portfolio total with an uncertainty range and a buying rule.

## Run

```bash
uv venv --python 3.12 .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/python -m utils.data              # raw CSVs -> data/processed/*.parquet (~10 s)
.venv/bin/python -m scripts.run_pipeline    # features, models, evaluation -> results/ (~1 min)
```

## Setup

| | |
|---|---|
| Forecast made | 10:00 on the day before (Swiss day-ahead gate closes at 11:00) |
| Data used | Meter data up to the end of 2 days before. Yesterday is not complete at 10:00. |
| Target | kWh per hour of grid consumption (`kWh_received_Total`), for each hour of the next local day |
| Model | One gradient-boosted tree model shared by all households (scikit-learn `HistGradientBoostingRegressor`) |
| Inputs | Past consumption (same hour 2, 3, 7 and 14 days back; 7- and 28-day averages), calendar (hour, weekday, Swiss holidays, season), weather, household info (PV flag, survey data, heat-pump visit status) |
| Train / tune / calibrate / test | ≤ 29 Oct 2022 / Nov–Dec 2022 / Jan–Feb 2023 / **1 Mar 2023 – 27 Feb 2024** |

Two weather versions:
- **Strict:** measured weather up to 2 days before, plus the typical temperature for that day and hour.
- **Actual weather:** the measured weather of the target day. This stands in for a perfect day-ahead weather forecast, which the data does not contain.

PV production cannot be forecast because it is not in the data. For PV owners the target is their net grid consumption, which already includes the effect of their panels.

## Results (test year, hourly, kWh per hour)

The portfolio is 255 households and averages 282 kWh per hour. It is summed over the households whose meters reported in each hour (96.9% of them on average).

| Model | Portfolio MAE | Portfolio nMAE | Daily nMAE | Household nMAE (pooled) |
|---|---|---|---|---|
| Same hour last week (baseline) | 57.7 | 20.4% | 17.2% | 56.4% |
| Average of last 7 days (baseline) | 45.6 | 16.2% | 13.5% | 46.7% |
| Blend of baselines | 44.3 | 15.7% | 13.1% | 46.0% |
| **Our model, strict weather** | **41.2** | **14.6%** | **11.9%** | **45.4%** |
| Our model, actual weather | 22.6 | 8.0% | 5.5% | 43.1% |

- nMAE is MAE divided by mean actual consumption.
- Daily nMAE uses only the 362 complete days.

Levels 2–3 (simulated buying cost):
- Assumed prices: under-buying costs 50 €/MWh, over-buying 40 €/MWh, with a sweep over the ratio.
- Uncertainty ranges come from the last 56 days of portfolio errors.
- Bidding the estimated quantile instead of the forecast cut retrospective regret by 21–33% when costs were lopsided (1:4 and 4:1). With near-equal costs it did not help.
- The 10–90% range covered 76% of hours.

Details are in `results/*.csv`; the figures are `results/portfolio_weeks.png` and `results/regret_sweep.png`.

## Files

- `utils/data.py`: loader. Selects columns by name, builds a gap-free hourly grid, shifts weather timestamps to the start of the hour.
- `utils/features.py`: features, with checks that no feature uses data after the cutoff.
- `utils/evaluate.py`: metrics, portfolio, buying regret, quantile bids, bootstrap.
- `scripts/run_pipeline.py`: end-to-end runner.
