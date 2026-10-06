# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

One-day hackathon repo ("Bringing the Heat") for an E.ON day-ahead load-forecasting
challenge. The full problem statement, data dictionary, and the four complexity levels
(Level 0 baseline → Level 3 uncertainty) are in `README.md` — read it before doing any
modeling work, it is the spec. `discussion.md` tracks live decisions (candidate model
families: ProLoaF, Crystal). `prompts/level1_proloaf_prompt.md` is a self-contained
task brief for the Level 1 (PV-group detection + grouped forecasting) workstream using
ProLoaF — it's the historical starting point, not the current scope (see below).

**Work here now spans Levels 1-3, not just Level 1.** `src/` contains one forecasting
pipeline built up across all three: PV-group detection + grouped models (Level 1),
grouped-vs-ungrouped comparison on day-ahead-procurement-relevant metrics, not just raw
accuracy (Level 2), and prediction-interval uncertainty with an honest calibration check
(Level 3). Level 0 (a single baseline model) was never a separate deliverable --
`all_households_group`, built for the Level 2 comparison, already serves as that
baseline. `reports/report.md` is the single write-up covering all of it; there are still
no tests and no lint config.

## Environment

`uv`-managed Python project (`pyproject.toml` + `uv.lock`), requires Python >=3.12. This
repo previously targeted Poetry (hence `[project]`'s PEP 621 layout), but Poetry is no
longer used: `pyproject.toml`'s `[build-system]` has been dropped and
`[tool.uv] package = false` is set, since there's no installable "utils" package to
build anyway — the real code is `src/`, imported as `src.<module>`, never installed.
`uv.lock` is the committed, authoritative dependency lock; regenerate it with `uv lock`
after any `pyproject.toml` edit.

```bash
uv sync                         # creates/updates .venv from uv.lock
source .venv/bin/activate
python3 -m src.<module>
```

(`uv pip install <pkg>` also works for ad-hoc additions, but then run `uv lock` + `uv sync`
afterwards so `uv.lock` and the venv stay in sync with `pyproject.toml`.)

There is no lint/format/test tooling configured — don't assume `ruff`/`pytest`/etc. exist
until they're added to `pyproject.toml`.

## Forecasting pipeline, Levels 1-3 (`src/`)

Run in order (each stage writes its outputs to `reports/`, consumed by the next stage):

```bash
python3 -m src.pv_features    # per-household PV-pattern features -> reports/pv_features.csv
python3 -m src.pv_detection   # classifier vs surveyed PV flag -> reports/pv_detect*.{csv,json}
python3 -m src.aggregate      # 15-min group aggregates (PV / non-PV / combined) -> reports/*_15min.csv
python3 -m src.forecast       # LightGBM day-ahead model per group, 15-min steps -> reports/forecast_metrics.json, grouping_comparison.json
python3 -m src.feature_ablation  # optional: leave-one-feature-out impact on portfolio-wide MAPE -> reports/feature_ablation.json
```

- `src/data_loading.py` — all raw-CSV readers (households/meta/overview/weather/15-min);
  everything else imports from here rather than re-parsing CSVs.
- `src/pv_features.py` — turns a household's 15-min `kWh_received_Total` into PV-pattern
  summary features (midday/night suppression ratios, seasonal contrast, sunshine
  correlation), computed at an intentionally coarse hourly grain via
  `household_hourly_total()` regardless of the aggregation pipeline's resolution.
  `household_resampled_total(household_id, freq)` is the generalised version
  `aggregate.py` reuses at `freq="15min"`.
- `src/pv_detection.py` — trains/cross-validates the PV classifier against the surveyed
  `Installation_HasPVSystem` flag (0.937 ROC AUC out-of-fold), then scores the 165
  unsurveyed households -> `reports/pv_detector_scores.csv` (`detector_pred_pv`).
  `aggregate.py`'s `household_pv_labels()` consumes this: surveyed flag where known, this
  detector's output as a fallback for the rest -- so run `pv_detection.py` before
  `aggregate.py` (the documented pipeline order already does this).
- `src/aggregate.py` — builds the group-sum **and per-household-average**
  (`kWh_mean_per_active_household`) consumption series at its **native 15-min resolution**
  (not downsampled to match weather) + weighted multi-station weather **upsampled** from
  hourly to 15-min via time-based linear interpolation; groups are 158 PV / 252 no-PV /
  410 all-households (every household on disk lands in one group or the other, none
  excluded); handles the meter-rollout/partial-coverage problem by trimming to a window
  with an absolute floor on active households (`MIN_HOUSEHOLDS = 30`), not a fraction of
  the group's eventual size — the per-household average is far more stable across a
  changing household count than the raw sum is, so an absolute floor is enough and
  recovers ~1-2 extra years of history versus an earlier, sum-based/70%-relative-threshold
  version of this pipeline.
- `src/forecast.py` — LightGBM day-ahead model per group at 15-min steps (horizon = 96
  steps = 24h), trained on `kWh_mean_per_active_household` (not the raw group total --
  see `aggregate.py` above for why); predictions are rescaled back to group/portfolio kWh
  totals at evaluation time by multiplying by the *actual* historical
  `active_household_count` for that timestamp. Matches the actual day-ahead market use
  case: every feature is anchored to a fixed **gate-closure cutoff of 11:45 AM (before
  noon) the day before delivery** (`CUTOFF_HOUR`/`CUTOFF_MINUTE`), not to midnight of the
  delivery day — bids have to be submitted before gate closure, so the last ~12h of the
  previous day isn't actually known at bid time either (an earlier version of this
  pipeline assumed it was, via a midnight-anchored origin). Target and weather lag/rolling
  features are looked up via `Series.reindex` at fixed offsets *before this cutoff*
  (always safe by construction, for all 96 targets of a day at once), not via a per-row
  constant-steps-before-*t* shift. See its docstring for the full derivation (including
  why the same-time-of-day lookback needs k>=2 days now, not k>=1) and the
  fully-vectorised feature construction (no per-row Python loop, no per-day groupby
  either). Same-day weather *actuals* are never used as a forecast stand-in (an earlier
  version of this pipeline did that as a brief-sanctioned simplification; that's a real
  fix now, not just a disclosed shortcut). Also trains two `objective="quantile"` models
  per group (`LOWER_QUANTILE`/`UPPER_QUANTILE` = 0.05/0.95) for a 90% prediction interval
  (Level 3-style uncertainty), reporting PICP (realised coverage) and mean interval
  width — found to be overconfident in practice (PICP 74-85% vs the 90% nominal target,
  worse for the noisier PV group), reported honestly rather than tuned away; see the
  report's "Uncertainty" subsection in §4.
- `src/feature_ablation.py` — leave-one-feature-out check against the "Grouped,
  portfolio-wide" MAPE from `forecast.py`'s own grouped-vs-ungrouped comparison: retrains
  pv_group/non_pv_group with each of `FEATURE_COLUMNS` dropped in turn (same fixed
  train/test rows throughout) and reports the MAPE delta. Notably, gain-based feature
  importance and this held-out leave-one-out impact disagree for a few features (e.g.
  `horizon` ranks top by gain but costs nothing to remove) — see the report's "Feature
  ablation" subsection in §4.

Full write-up of methodology, results, and known simplifications: `reports/report.md`.
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
something graders look for, not an incidental detail. The forecasting pipeline
(`src/aggregate.py`) upsamples weather to 15-min via time-based linear interpolation and
keeps consumption at its native 15-min resolution, rather than the reverse (an earlier
version of this pipeline aggregated consumption up to hourly instead) — see
`src/aggregate.py`'s module docstring for why that's a real simplification for the two
cumulative-style weather columns specifically.

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
