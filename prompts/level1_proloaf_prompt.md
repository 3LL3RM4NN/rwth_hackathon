# Prompt: Level 1 — Grouped Day-Ahead Forecasting with ProLoaF

Copy everything below the line into a fresh Claude Code session in this repo.

---

You are working in the `rwth_hackathon` repo for the "Bringing the Heat" E.ON forecasting
challenge. Read `README.md` first for full context. In short: E.ON needs day-ahead (next-day)
forecasts of household energy consumption to buy the right amount of electricity on the
day-ahead market. We have 15-minute smart-meter data for 410 heat-pump households
(`data/15min/<Household_ID>.csv`, columns `Household_ID;Group;AffectsTimePoint;Timestamp;
kWh_received_Total;kWh_received_HeatPump;kWh_received_Other`) and hourly weather data for 8
stations (`data/weather_data_hourly/<Weather_ID>.csv`). Household metadata lives in
`data/smart_meter_meta_data/`:
- `households.csv` — `Household_ID;Group;Weather_ID;Installation_HasPVSystem;...` (links each
  household to its nearest weather station and flags whether it has a PV system — missing/blank
  for households never surveyed on this).
- `meta_data.csv` — building type, living area, heat-pump type, EV ownership, etc. (393/410
  households).
- `smart_meter_data_15min_overview.csv` — per-household data coverage.

A Level 0 baseline (single global forecasting model, no grouping) may already exist elsewhere in
the repo — check for it, but don't assume it does.

## Goal: Level 1

The dataset mixes two very different consumption profiles: households with only a heat pump, and
households that also have a PV system (much more volatile, PV self-consumption suppresses midday
net draw). Build **separate forecasting models per group** instead of one global model, following
Level 1 option (b) from the README:

> Identify PV owners by their energy consumption patterns, then build separate forecasting models
> for households with and without PV systems.

Treat the PV flag as a validation target, not a shortcut: engineer features from the 15-minute
consumption pattern (e.g. midday-vs-baseline draw ratio on clear days, correlation with the
nearest station's solar-radiation signal from `weather_data_overview`/`weather_data_hourly`,
seasonal amplitude) and train/evaluate a simple classifier or rule-based detector against the
`Installation_HasPVSystem` labels where available. Report how well pattern-based detection
recovers the known flag — this matters more than squeezing out the last bit of accuracy. Use the
known flag (not your detector's output) to define the two groups for the forecasting step itself,
and note explicitly that this is a simplification; it's fine to also show forecasting results
using your detector's groups as a secondary comparison if time allows.

Then, for each group (PV / non-PV), build a day-ahead forecasting model for **aggregate group-level
consumption** (sum `kWh_received_Total` across the group's households per timestamp) — this
matches the actual business need (procuring an aggregate volume on the day-ahead market), and
sidesteps the sparse/incomplete-coverage problem of forecasting 400 individual series. Document
how you handle households with missing or partial coverage at a given timestamp.

## Modeling: use ProLoaF

Use **ProLoaF** (https://github.com/sogno-platform/proloaf), the Fraunhofer/Sogno probabilistic
load-forecasting library, as the forecasting engine for both group models. Its workflow:

1. Install it (e.g. `pip install "proloaf @ git+https://github.com/sogno-platform/proloaf.git"` or
   clone as a submodule) and add it as a dependency in `pyproject.toml` (this project uses
   Poetry).
2. ProLoaF is driven by per-"station" folders under `targets/<name>/` containing
   `preprocessing.json` and `config.json`. Create one such folder per group, e.g.
   `targets/pv_group/` and `targets/non_pv_group/`.
3. `preprocessing.json` points at your raw aggregated CSV and describes how to clean/resample it
   into the single combined CSV ProLoaF trains on (merge in weather features here — radiation,
   temperature — resampled/aligned to your chosen resolution).
4. `config.json` defines the encoder/decoder feature groups, history length, forecast horizon
   (24h ahead), target column(s), the LSTM encoder-decoder hyperparameters, and train/val split.
   Look at `targets/opsd/` in the ProLoaF repo as a reference example.
5. Run via `python src/preprocess.py --station <name>` then `python src/train.py --station <name>`,
   then evaluate with `src/evaluate.py`.
6. Pick a consistent time resolution for both the target and horizon (hourly is ProLoaF's typical
   granularity in its examples; resampling the 15-minute consumption data up to hourly is a
   reasonable, defensible choice — justify whatever you pick).

If ProLoaF proves awkward to wire up in the time available, it's fine to fall back to a simpler
model (e.g. gradient boosting or an LSTM you write directly with the existing
pandas/numpy/scikit-learn/torch-free stack) for one or both groups, but attempt ProLoaF first
and explain in your writeup why you kept or dropped it.

## Hard constraints

- **No leakage from the future.** Split chronologically: train on the earlier ~80% of each
  group's aggregated series, test on the most recent ~20%. Do not shuffle or split randomly.
- Only use information that would actually be available before the forecast day (e.g. don't use
  same-day weather actuals as if they were forecasts — note this as a known simplification if you
  use actuals as a stand-in for a day-ahead weather forecast, since we don't have one).

## Deliverables

- Code under a new `src/` module: data loading/aggregation, PV-pattern
  feature engineering + detector, ProLoaF target/config generation, training/evaluation glue.
- The `targets/pv_group/` and `targets/non_pv_group/` ProLoaF config folders.
- A short report (markdown or notebook) summarizing: the PV-detection approach and how it compares
  to the ground-truth flag, the two trained models' day-ahead forecast accuracy (e.g. MAE, RMSE,
  MAPE on the held-out test period), and a brief discussion of whether grouping actually helped
  versus a single ungrouped model (if a Level 0 baseline exists in the repo, compare against it).
- Keep the scope realistic for a one-day hackathon: a working, clearly-explained two-group
  pipeline with honest metrics beats an ambitious but unfinished one.
