You are working on an energy forecasting challenge for E.ON.

## Objective

Build a simple, robust **baseline day-ahead electricity consumption forecasting model** using the provided household smart-meter data.

The goal is to predict the next day's electricity consumption for each household at 15-minute resolution.

Each day contains up to **96 × 15-minute observations**, so the model should produce up to 96 forecasts for the following day.

The baseline should prioritize:

* correctness
* simplicity
* reproducibility
* clear evaluation
* avoiding data leakage

Do NOT build an advanced ML model yet. This is a baseline that we will later compare against models such as LightGBM or TFT.

## Dataset

The project contains:

`data/15min/<Household_ID>.csv`

The files are semicolon-separated and contain:

* `Household_ID`
* `Group`
* `AffectsTimePoint`
* `Timestamp`
* `kWh_received_Total`
* `kWh_received_HeatPump`
* `kWh_received_Other`

The target for this baseline is:

`kWh_received_Total`

The data is recorded at 15-minute resolution.

There are approximately 410 households with different amounts of historical data.

There are also household metadata and weather files, but **do not use weather or household metadata in the first baseline**. We want a simple consumption-only benchmark.

## Forecasting task

For each household:

Given all observations available up to the end of day D, predict the 96 quarter-hour consumption values for day D+1.

For example:

Historical data:
2023-01-01 → 2023-12-31

Forecast:
2024-01-01, 00:00 through 23:45

The forecast should be generated separately for each household.

## Baseline model

Implement a **seasonal naive baseline** first.

For every quarter-hour of the day:

```
forecast(D+1, interval) = actual(D, interval)
```

In other words, tomorrow's consumption profile is assumed to be identical to today's.

Also implement a second baseline:

```
forecast(D+1, interval) = actual(D-7, interval)
```

This represents a same-weekday seasonal naive forecast.

Call these:

* `naive_1day`
* `naive_7day`

If sufficient historical data exists, optionally implement:

```
forecast(D+1, interval) =
    mean(consumption at the same 15-minute interval
         over the previous 4 occurrences of that weekday)
```

Call this:

* `seasonal_mean_4weeks`

Keep this model simple. Do not add machine learning at this stage.

## Important: no data leakage

This is critical.

The challenge explicitly requires using only past data to predict the future.

Do NOT randomly split the time series.

Use a chronological evaluation.

For example:

* Training/history: earlier observations
* Validation: subsequent period
* Test: most recent period

Ideally reserve the final ~20% of available dates as the test period.

The forecast for every test day must only use observations that would actually have been available before that forecast day.

Do not use:

* future consumption
* future rolling statistics
* future averages
* future weather
* future metadata
* randomly shuffled observations

Make the temporal split explicit in the code.

## Data preprocessing

Create a clean preprocessing pipeline that:

1. Reads the household CSV files.
2. Parses `Timestamp` correctly.
3. Sorts observations chronologically.
4. Checks for duplicate timestamps.
5. Checks for missing timestamps.
6. Determines whether each household has complete 15-minute days.
7. Handles incomplete days sensibly.

Do not silently interpolate target consumption values.

If a day is incomplete, document how it is treated and make the choice configurable.

The code should also report:

* number of households
* date range
* number of observations
* number of complete days
* number of incomplete days
* missing intervals

## Evaluation

Evaluate the forecasts on the held-out test period.

Calculate at least:

* MAE
* RMSE
* MAPE, but handle zero/small actual values safely
* sMAPE

Calculate metrics:

1. aggregated across all households and timestamps
2. per household
3. optionally per forecast horizon / quarter-hour

The most important metrics should be clearly reported.

Also calculate the average daily energy error if practical.

## Visualizations

Create a small number of useful plots:

1. Actual vs predicted consumption for a representative test day.
2. Actual vs predicted consumption for several consecutive days.
3. Average actual daily load profile vs average forecast daily profile.
4. Distribution of per-household MAE.

Do not create excessive plots.

Save the plots to a sensible `outputs/` directory.

## Project structure

Inspect the existing repository before creating files.

Reuse existing utilities and conventions where appropriate.

Create a clean structure similar to:

```
src/
    data.py
    baseline.py
    evaluation.py
    visualization.py

scripts/
    run_baseline.py

outputs/
    metrics/
    plots/
```

Adjust this structure if the repository already has an established structure.

Do not unnecessarily rewrite existing code.

## Configuration

Make important settings configurable, such as:

* test fraction
* minimum history
* forecast horizon
* baseline type
* minimum completeness of a day

Avoid hard-coding paths throughout the code.

## Reproducibility

The complete baseline should be runnable with one command, for example:

```
python scripts/run_baseline.py
```

or an appropriate equivalent based on the existing project setup.

The script should produce:

* evaluation metrics
* saved predictions
* plots
* a concise summary of the experiment

Save predictions in a machine-readable format such as CSV or Parquet.

Each prediction should contain at least:

* `Household_ID`
* `Timestamp`
* `actual`
* `prediction`
* `model`

## Important modeling consideration

The dataset contains households with different amounts of historical data.

Do not discard the entire household simply because it has less than one year of history.

The baseline should use whatever historical data is available, while clearly reporting households/days where a particular baseline cannot make a prediction.

For example, the 7-day baseline requires at least 7 days of history.

## Validation of implementation

Before considering the implementation complete:

1. Test the preprocessing on several household files.
2. Verify that timestamps are correctly aligned.
3. Verify that every prediction timestamp is strictly after the data used to generate it.
4. Manually inspect a few forecasts.
5. Check that the 1-day naive forecast exactly equals the previous day's corresponding 15-minute values.
6. Check that the 7-day naive forecast exactly equals the corresponding values from 7 days earlier.
7. Run the full evaluation.
8. Report any data-quality issues discovered.

## Final output

After implementing everything, give me a concise report containing:

### Dataset

* households used
* date range
* number of observations
* number of complete/incomplete days

### Experimental setup

* train/validation/test dates
* forecast horizon
* handling of missing/incomplete days

### Results

A table like:

| Model                | MAE | RMSE | sMAPE |
| -------------------- | --: | ---: | ----: |
| Previous day         | ... |  ... |   ... |
| Previous week        | ... |  ... |   ... |
| 4-week seasonal mean | ... |  ... |   ... |

### Interpretation

Briefly explain:

* which baseline performs best
* whether the results vary substantially between households
* whether there are obvious periods where the baseline fails

### Next step

Do NOT implement the next model yet.

Instead, explain what features and modeling approach you would recommend for the next iteration using LightGBM, based on what you observed in the baseline results.

## Constraints

* Do not use future information.
* Do not randomly shuffle the time series.
* Do not use external datasets.
* Do not use weather yet.
* Do not use PV metadata yet.
* Do not implement deep learning.
* Keep the baseline understandable enough that it can be explained in a technical presentation.
* Prefer simple, maintainable code over premature abstraction.

Start by inspecting the repository and existing data structure. Then implement the baseline, run it, inspect the results, and report what you found.
