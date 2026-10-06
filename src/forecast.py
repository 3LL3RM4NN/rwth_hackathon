"""Day-ahead, group-level forecasting model (LightGBM), built on the
15-minute-resolution aggregates from ``src/aggregate.py``.

Matching the actual day-ahead market use case
----------------------------------------------
Day-ahead market bids have to be submitted by a fixed gate-closure time the
day *before* delivery -- modelled here as ``CUTOFF_HOUR:CUTOFF_MINUTE``
(11:45 AM, before noon) on day D-1 for delivery day D. Every feature for every one of day D's
96 15-minute targets is therefore computed as of that single, fixed cutoff
timestamp, not as of midnight or as of the target time itself: nothing from
after 11:45 on D-1 is ever used, including the rest of D-1 itself (11:45
onward) and all of day D before it's forecast. This is stricter than an
"origin = midnight" framing would be (that would implicitly assume all of day
D-1 is already known, which isn't true at real bidding time).

For a target ``t`` on day D, ``cutoff = D.normalize() - CUTOFF_GAP`` (CUTOFF_GAP
= 12h15m, i.e. the gap from 11:45 back to the next midnight). Two kinds of
feature live on either side of this:

- **Cutoff-anchored history** (target + weather): ``lag_24h``/``lag_48h``/
  ``lag_168h`` and ``rolling_mean_24h_asof_cutoff`` are looked up *at* fixed
  offsets *before the cutoff itself* (via ``Series.reindex`` on timestamps
  derived from each row's own cutoff) -- so they always mean exactly what
  their name says ("24h before the cutoff"), identically for all 96 targets
  of a given day, regardless of which of the 96 is being predicted.
- **Same-time-of-day history** (target only): ``rolling_mean_same_timeofday_7d``
  looks up ``t`` itself at fixed day-multiples in the past (``t - k*1day`` for
  k=2..8). This one is inherently anchored to ``t``'s own clock time, not to
  the cutoff, so it needs its own safety argument: it's safe only once
  ``k*24h`` is large enough to land before the cutoff for every target in the
  day, including the *last* one (23:45) -- which requires k>=2 (not k>=1, as
  it would under the old midnight-cutoff framing), since the cutoff is now
  12h15m *before* midnight rather than right at it. See the derivation in
  ``build_supervised_table``.

``lead_time_steps`` (how many 15-min steps separate the cutoff from this
specific target, 49..144) is computed and kept in the table for description/
analysis, but *not* used as a model input: since ``CUTOFF_HOUR``/``MINUTE``
are fixed constants, ``lead_time_steps`` is always exactly ``horizon +
MARGIN_STEPS`` for every row -- a pure additive-constant transform of
``horizon`` -- so it carries zero information a tree ensemble doesn't already
get from ``horizon`` alone (confirmed empirically: 0 feature importance when
included). It would only diverge from `horizon` if the cutoff varied (e.g. by
weekday), which it doesn't here.

Feature construction is fully vectorised (no per-row Python loop): cutoff
timestamps are derived once from each row's own calendar day, then looked up
via ``Series.reindex`` / ``Series.shift`` over the whole continuous 15-min
index.
"""

from __future__ import annotations

import json

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from src import aggregate

STEPS_PER_HOUR = 4  # 15-min resolution
STEPS_PER_DAY = 24 * STEPS_PER_HOUR  # 96

CUTOFF_HOUR = 11  # day-ahead bid gate closure: 11:45 AM (24h clock, before noon) on the day before delivery
CUTOFF_MINUTE = 45
CUTOFF_STEP_OF_DAY = CUTOFF_HOUR * STEPS_PER_HOUR + CUTOFF_MINUTE // 15  # 47
# Gap from the cutoff back to the *next* midnight (12h15m) -- used to derive
# each row's own cutoff timestamp from its calendar day via subtraction, in
# minutes/pd.Timedelta(unit="m") throughout to avoid a generic-unit
# DeprecationWarning that pd.Timedelta(hours=..., minutes=...) triggers on
# this numpy/pandas version.
CUTOFF_GAP = pd.Timedelta((24 * STEPS_PER_HOUR - CUTOFF_STEP_OF_DAY) * 15, unit="m")
# How many 15-min steps separate the cutoff from midnight of the delivery day
# -- i.e. every target's lead_time_steps = horizon + MARGIN_STEPS.
MARGIN_STEPS = STEPS_PER_DAY - CUTOFF_STEP_OF_DAY  # 49

LAG_HOURS_FROM_CUTOFF = [24, 48, 168]  # a day / 2 days / a week before the cutoff
ROLLING_WINDOW_STEPS = STEPS_PER_DAY  # 24h trailing window for the "as of cutoff" rolling mean

# Safety margin for the same-time-of-day lookback (k*1day before t): needs
# k*96 - h >= MARGIN_STEPS for every horizon h up to 95 (the last of the day),
# i.e. k >= (95 + MARGIN_STEPS) / 96 = 1.5 -> k >= 2.
SAME_TIMEOFDAY_MIN_DAYS_BACK = 2
SAME_TIMEOFDAY_LOOKBACK_DAYS = 7  # how many weekly terms to average (k = 2..8)

WEATHER_FEATURES = [
    "Temperature_avg_hourly",
    "DewPoint_hourly",
    "Humidity_avg_hourly",
    "Precipitation_total_hourly",
    "Sunshine_duration_hourly",
    "WindSpeed_hourly",
]
TARGET = "kWh_total_group_sum"
TRAIN_FRACTION = 0.8


def _hours_before(timestamps: pd.DatetimeIndex, hours: int) -> pd.DatetimeIndex:
    return timestamps - pd.Timedelta(hours * 60, unit="m")


def load_group_15min(name: str) -> pd.DataFrame:
    df = pd.read_csv(f"reports/{name}_15min.csv", index_col=0, parse_dates=True)
    return df


def build_supervised_table(df: pd.DataFrame) -> pd.DataFrame:
    s = df[TARGET]
    idx = s.index
    print(f"Building supervised table over {len(idx)} 15-min timestamps (vectorised)...")

    table = pd.DataFrame(index=idx)
    table["y"] = s

    # Every one of a day's 96 targets shares the same cutoff: 11:45 the day
    # before. cutoff_time is a per-row Series here only because it's cheapest
    # to compute that way (idx.normalize() is already vectorised); its actual
    # *value* only depends on which calendar day the row's target falls on.
    cutoff_time = idx.normalize() - CUTOFF_GAP

    # Cutoff-anchored history: looked up via reindex at a fixed offset before
    # each row's own cutoff, so e.g. lag_24h always means "24h before the
    # cutoff", identically for all 96 targets of that day (unlike shifting by
    # a constant number of steps relative to t itself, which would make a
    # fixed-name lag feature's actual recency drift across the day).
    rolling_24h_mean = s.rolling(ROLLING_WINDOW_STEPS).mean()  # trailing 24h mean ending at each timestamp
    for hours in LAG_HOURS_FROM_CUTOFF:
        table[f"lag_{hours}h"] = s.reindex(_hours_before(cutoff_time, hours)).to_numpy()
    table["rolling_mean_24h_asof_cutoff"] = rolling_24h_mean.reindex(cutoff_time).to_numpy()

    # Same-time-of-day history: mean of "this exact 15-min-of-day" across the
    # past several days (k=2..8, not 1..7 -- see module docstring for why k=1
    # (yesterday) is unsafe here: the cutoff sits 12h15m before midnight, so
    # for late-day targets "yesterday, same time" would still be in the
    # future relative to the cutoff).
    same_timeofday = pd.concat(
        [
            s.shift(STEPS_PER_DAY * k)
            for k in range(SAME_TIMEOFDAY_MIN_DAYS_BACK, SAME_TIMEOFDAY_MIN_DAYS_BACK + SAME_TIMEOFDAY_LOOKBACK_DAYS)
        ],
        axis=1,
    )
    table["rolling_mean_same_timeofday_7d"] = same_timeofday.mean(axis=1, skipna=True)

    # Baseline-only (not a model feature): "same 15-min-of-day, one week ago",
    # anchored to t itself rather than to the cutoff. Exactly 7 days is safely
    # >= the 2-day minimum above, so this is just as leakage-safe as the
    # smoothed version -- it exists separately because a *cutoff*-anchored
    # column (like lag_168h below) is constant across all 96 targets of a
    # day, which makes it useless as a "naive forecast" (it wouldn't track
    # the daily shape at all); this one does, by design.
    table["naive_same_timeofday_last_week"] = s.shift(STEPS_PER_DAY * 7)

    # Weather gets the same cutoff-anchored treatment as the target -- same-day
    # (or even same-morning) actuals are never used (see module docstring).
    for feat in WEATHER_FEATURES:
        base = df[feat]
        base_rolling_24h_mean = base.rolling(ROLLING_WINDOW_STEPS).mean()
        table[f"{feat}_lag_24h"] = base.reindex(_hours_before(cutoff_time, 24)).to_numpy()
        table[f"{feat}_rolling_mean_24h_asof_cutoff"] = base_rolling_24h_mean.reindex(cutoff_time).to_numpy()

    table["origin"] = idx.normalize()
    table["target_time"] = idx
    # horizon = 15-min-of-day index (0..95): which of the day's 96 targets this is.
    table["horizon"] = idx.hour * STEPS_PER_HOUR + idx.minute // 15
    # lead_time_steps = steps from the cutoff to this target (49..144): how far
    # ahead of the decision point this specific prediction actually is. No
    # longer equivalent to `horizon` now that the cutoff isn't at midnight.
    table["lead_time_steps"] = table["horizon"] + MARGIN_STEPS
    table["hour"] = idx.hour
    table["minute"] = idx.minute
    table["dow"] = idx.dayofweek
    table["month"] = idx.month
    table["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    return table.reset_index(drop=True)


def chronological_split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = table["origin"].drop_duplicates().sort_values()
    n_train = int(len(origins) * TRAIN_FRACTION)
    train_origins = set(origins.iloc[:n_train])
    train = table[table["origin"].isin(train_origins)]
    test = table[~table["origin"].isin(train_origins)]
    return train, test


WEATHER_FEATURE_COLUMNS = [f"{feat}_lag_24h" for feat in WEATHER_FEATURES] + [
    f"{feat}_rolling_mean_24h_asof_cutoff" for feat in WEATHER_FEATURES
]

FEATURE_COLUMNS = (
    [f"lag_{h}h" for h in LAG_HOURS_FROM_CUTOFF]
    + ["rolling_mean_24h_asof_cutoff", "rolling_mean_same_timeofday_7d"]
    + WEATHER_FEATURE_COLUMNS
    + ["hour", "minute", "dow", "month", "is_weekend", "horizon"]
)


def train_and_evaluate(name: str, n_households: int) -> dict:
    print(f"Loading reports/{name}_15min.csv...")
    df = load_group_15min(name)
    table = build_supervised_table(df)
    usable = table.dropna(subset=FEATURE_COLUMNS + ["y"])
    dropped = len(table) - len(usable)
    print(f"{len(usable)}/{len(table)} rows usable after dropping missing features/target ({dropped} dropped)")

    train, test = chronological_split(usable)

    print(f"Training LightGBM on {len(train)} rows (400 estimators)...")
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=0,
        verbosity=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["y"])
    print(f"Evaluating on {len(test)} held-out rows...")
    pred = model.predict(test[FEATURE_COLUMNS])

    mae = mean_absolute_error(test["y"], pred)
    rmse = mean_squared_error(test["y"], pred) ** 0.5
    mape = mean_absolute_percentage_error(test["y"], pred)

    by_horizon = (
        pd.DataFrame({"horizon": test["horizon"], "y": test["y"], "pred": pred})
        .assign(abs_err=lambda d: (d["y"] - d["pred"]).abs())
        .groupby("horizon")["abs_err"]
        .mean()
    )

    # Naive day-ahead baseline for context: "same 15-min-of-day, same day of
    # week, last week" (anchored to the target itself, not the cutoff -- see
    # build_supervised_table), the single most defensible no-model forecast
    # available at bid time. Unlike the model's own features, this single
    # (non-averaged) lookup has no fallback for the rare row whose "exactly
    # 7 days ago" point falls inside a data gap (e.g. the 2023-10-29 outage),
    # so those few rows are excluded from this comparison specifically.
    naive_valid = test.dropna(subset=["naive_same_timeofday_last_week"])
    naive_mae = mean_absolute_error(naive_valid["y"], naive_valid["naive_same_timeofday_last_week"])

    result = {
        "name": name,
        "n_households": n_households,
        "n_rows_total": int(len(table)),
        "n_rows_dropped_missing_features": int(dropped),
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)),
        "train_origin_range": [str(train["origin"].min()), str(train["origin"].max())],
        "test_origin_range": [str(test["origin"].min()), str(test["origin"].max())],
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "mae_per_household": float(mae / n_households),
        "naive_lag168_mae": float(naive_mae),
        "mae_by_horizon": by_horizon.round(3).to_dict(),
        "feature_importance": pd.Series(
            model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False).to_dict(),
    }

    test_out = test[["origin", "horizon", "target_time", "y"]].copy()
    test_out["pred"] = pred
    test_out.to_csv(f"reports/{name}_test_predictions.csv", index=False)

    return result


if __name__ == "__main__":
    results = {}
    for name, pv in [("pv_group", True), ("non_pv_group", False), ("all_known_group", None)]:
        print(f"\n=== Training {name} ===")
        n_households = len(aggregate.group_household_ids(pv))
        res = train_and_evaluate(name, n_households)
        results[name] = res
        print(f"\n=== {name} ===")
        print(f"households={res['n_households']}  train_rows={res['n_rows_train']}  test_rows={res['n_rows_test']}")
        print(f"test period: {res['test_origin_range']}")
        print(f"MAE={res['mae']:.2f} kWh/15min  RMSE={res['rmse']:.2f} kWh/15min  MAPE={res['mape']*100:.1f}%")
        print(f"MAE per household={res['mae_per_household']:.4f} kWh/15min")
        print(f"naive (same 15-min-of-day, last week) MAE={res['naive_lag168_mae']:.2f} kWh/15min")

    with open("reports/forecast_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Fair grouped-vs-ungrouped comparison: each model above was evaluated on
    # its own series' last 20% of days, which differ in length/start date
    # across groups (series start at different points due to meter rollout),
    # so their MAE numbers above aren't directly comparable to each other.
    # Re-score all three on the single latest common test window instead.
    print("\nRe-scoring all groups on the common held-out test window...")
    common_start = max(r["test_origin_range"][0] for r in results.values())
    comparison = {}
    for name, res in results.items():
        preds = pd.read_csv(
            f"reports/{name}_test_predictions.csv", parse_dates=["origin", "target_time"]
        )
        preds = preds[preds["origin"] >= common_start]
        mae = mean_absolute_error(preds["y"], preds["pred"])
        rmse = mean_squared_error(preds["y"], preds["pred"]) ** 0.5
        mape = mean_absolute_percentage_error(preds["y"], preds["pred"])
        comparison[name] = {
            "n_households": res["n_households"],
            "n_rows": int(len(preds)),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "mae_per_household": float(mae / res["n_households"]),
        }

    grouped_mae_sum = comparison["pv_group"]["mae"] + comparison["non_pv_group"]["mae"]
    grouped_hh_sum = comparison["pv_group"]["n_households"] + comparison["non_pv_group"]["n_households"]
    comparison["grouped_combined"] = {
        "n_households": grouped_hh_sum,
        "mae": grouped_mae_sum,
        "mae_per_household": grouped_mae_sum / grouped_hh_sum,
    }
    comparison["common_test_window_start"] = common_start

    print(f"\n=== Fair comparison on common test window (from {common_start}) ===")
    print(
        f"Ungrouped single model : MAE/hh={comparison['all_known_group']['mae_per_household']:.4f}  "
        f"MAPE={comparison['all_known_group']['mape']*100:.1f}%"
    )
    print(
        f"Grouped (PV + non-PV)  : MAE/hh={comparison['grouped_combined']['mae_per_household']:.4f}  "
        f"(PV MAPE={comparison['pv_group']['mape']*100:.1f}%, non-PV MAPE={comparison['non_pv_group']['mape']*100:.1f}%)"
    )

    with open("reports/grouping_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
