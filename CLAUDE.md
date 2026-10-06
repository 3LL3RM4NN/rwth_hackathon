# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

One-day hackathon repo ("Bringing the Heat") for an E.ON day-ahead load-forecasting
challenge. The full problem statement, data dictionary, and the four complexity levels
(Level 0 baseline → Level 3 uncertainty) are in `README.md` — read it before doing any
modeling work, it is the spec. `discussion.md` tracks live decisions (candidate model
families: ProLoaF, Crystal). `prompts/level1_proloaf_prompt.md` is a self-contained
task brief for the Level 1 (PV-group detection + grouped forecasting) workstream using
ProLoaF — treat it as the current plan of record for that level unless told otherwise.

`src/` now contains the Level 1 pipeline (see below); there are still no tests and no
lint config.

## Environment

Poetry-managed Python project (`pyproject.toml`), requires Python >=3.12 — but the
`poetry` binary itself is **not installed** in this sandbox, so the working venv here
was created with `uv` instead. Both describe the same dependency set; if `poetry` is
available in your environment, prefer it and keep `pyproject.toml`/`poetry.lock` as the
source of truth (and run `poetry lock` after any manual `pyproject.toml` edit, since
`uv` was used to add `lightgbm` to `pyproject.toml` without being able to regenerate
`poetry.lock`).

```bash
# with poetry (preferred, once installed):
poetry install
poetry run python -m src.forecast

# with uv (used in this sandbox instead):
uv venv && source .venv/bin/activate
uv pip install pandas matplotlib polars numpy scikit-learn seaborn jupyterlab lightgbm
python3 -m src.<module>
```

There is no lint/format/test tooling configured — don't assume `ruff`/`pytest`/etc. exist
until they're added to `pyproject.toml`.

## Level 1 pipeline (`src/`)

Run in order (each stage writes its outputs to `reports/`, consumed by the next stage):

```bash
python3 -m src.pv_features    # per-household PV-pattern features -> reports/pv_features.csv
python3 -m src.pv_detection   # classifier vs surveyed PV flag -> reports/pv_detect*.{csv,json}
python3 -m src.aggregate      # hourly group aggregates (PV / non-PV / combined) -> reports/*_hourly.csv
python3 -m src.forecast       # LightGBM day-ahead model per group -> reports/forecast_metrics.json, grouping_comparison.json
```

- `src/data_loading.py` — all raw-CSV readers (households/meta/overview/weather/15-min);
  everything else imports from here rather than re-parsing CSVs.
- `src/pv_features.py` — turns a household's 15-min `kWh_received_Total` into PV-pattern
  summary features (midday/night suppression ratios, seasonal contrast, sunshine
  correlation). `household_hourly_total()` here is also reused by `aggregate.py`.
- `src/pv_detection.py` — trains/cross-validates the PV classifier against the surveyed
  `Installation_HasPVSystem` flag; this flag, not the detector's output, is what defines
  the forecasting groups in `aggregate.py` (the detector is a validation exercise, used
  only to additionally score the 165 unsurveyed households).
- `src/aggregate.py` — builds the hourly group-sum series + weighted multi-station
  weather for a group; handles the meter-rollout/partial-coverage problem by trimming to
  a "stable window" (≥70% of the group's eventual households reporting).
- `src/forecast.py` — LightGBM day-ahead (24h horizon) model per group; see its docstring
  for the no-leakage lag-feature design (only lags ≥24h are safe across all 24 horizons).

Full write-up of methodology, results, and known simplifications: `reports/level1_report.md`.
ProLoaF (the brief's primary choice) was not installed/used — installing its setup code
from an external git repo wasn't approved for this sandboxed session, so LightGBM is used
as the documented fallback instead; see the report's §1 for details before assuming
ProLoaF config folders (`targets/<name>/`) exist anywhere.

## Data layout and how it joins together

All data lives under `data/` and is CSV, **semicolon-separated**, timestamps are UTC
(`YYYY-MM-DD HH:MM:SS+00:00`). Note the actual on-disk column order in `data/15min/*.csv`
is `Household_ID;AffectsTimePoint;Group;Timestamp;...` — this differs from the order
listed in the README's table; don't rely on positional column assumptions, use header
names.

- **`data/15min/<Household_ID>.csv`** (one file per household, 410 files): the core
  target series. Columns: `Household_ID`, `Group` (`treatment`/`control`), `AffectsTimePoint`
  (`before visit`/`after visit`, relative to a heat-pump optimisation visit), `Timestamp`,
  `kWh_received_Total`, `kWh_received_HeatPump`, `kWh_received_Other`. 15-minute resolution,
  coverage varies per household (2019-05-20 to 2024-02-27 overall). No PV-production or
  grid-return columns exist in this dataset.
- **`data/smart_meter_meta_data/households.csv`**: one row per household (411 lines incl.
  header, i.e. ~410 households). This is the join hub: `Household_ID` → `Weather_ID` (links
  to the matching file in `data/weather_data_hourly/`) and `Installation_HasPVSystem`
  (ground-truth PV flag, blank/missing where never surveyed — don't coerce blank to False).
- **`data/smart_meter_meta_data/meta_data.csv`**: survey answers (building type, living
  area, heat-pump type, DHW production method, etc.), keyed by `Household_ID`. Only covers
  393/410 households. Column meanings are documented in
  `data/smart_meter_meta_data/meta_data_variables.csv` — check there instead of guessing
  at a column's meaning from its name.
- **`data/smart_meter_meta_data/smart_meter_data_15min_overview.csv`**: per-household
  coverage stats (earliest/latest timestamp, day counts before/after visit). Useful for
  filtering out households with too little data before aggregating a group.
- **`data/weather_data_hourly/<Weather_ID>.csv`**: 8 weather stations, hourly resolution
  (temperature, dew point, humidity, precipitation, pressure, sunshine duration, wind
  speed). Reached from a household via `households.csv.Weather_ID`.
- **`data/weather_data_overview/weather_variables.csv`** and
  **`weather_variables_availability.csv`**: describes which weather variables exist at
  which resolution (hourly vs daily) and which are actually populated per station — check
  availability before assuming a station has a given column populated.

Resolution mismatch is a first-class problem: smart-meter data is 15-min, weather is
hourly. You must explicitly choose and justify a resampling strategy (upsample weather,
aggregate consumption to hourly, or otherwise) — this is called out in the README as
something graders look for, not an incidental detail.

## Hard modeling constraints (apply to every level)

- **No future leakage.** Splits must be chronological — train on the earlier ~80% of a
  series, test on the most recent ~20%. Never shuffle/randomly split a time series here.
- Day-ahead forecasts must only use information available before the forecast day (e.g.
  same-day weather *actuals* are not a legitimate stand-in for a day-ahead weather
  *forecast* — if used as a simplification due to no forecast data being available, say so
  explicitly in any write-up).
- Grouped/aggregate forecasting (summing `kWh_received_Total` across a group's households
  per timestamp) is the expected approach for group-level models, per
  `prompts/level1_proloaf_prompt.md` — this sidesteps per-household sparse coverage. When
  aggregating, document how timestamps with partial household coverage are handled.
