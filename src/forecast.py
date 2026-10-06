"""Day-ahead, group-level forecasting model (LightGBM), built on the
15-minute-resolution aggregates from ``src/aggregate.py``.

Forecast setup
--------------
A forecast is "issued" at ``origin`` = midnight UTC of day D+1, using only data
from day D and earlier, to predict all 96 15-minute values of day D+1
(``horizon`` h = 0..95 fifteen-minute steps since midnight, target timestamp
``t = origin + h * 15min``).

Only lag features with lag >= 24h (96 steps) are used, because for h > 0 a lag
< 96 steps would reach into day D+1 itself (not yet known at the time the
forecast is issued) -- e.g. a naive "previous step" feature is safe for h=0 but
leaks future information for h=20. Anchoring every lag to t (not to origin) at
>=96 steps keeps every lag strictly inside day D or earlier for every horizon
-- the same argument as before, just re-scaled from hours to 15-min steps.

Weather is held to the exact same no-leakage rule as the target series: there
is no day-ahead weather *forecast* in this dataset, and same-day weather
*actuals* at the target timestamp are not used as a stand-in for one (that
would leak information not actually available at forecast time). Instead,
each weather column gets the same lag_24h / rolling_mean_24_48 treatment as
the target -- i.e. "yesterday, same time" and "yesterday's daily average" --
which is the same kind of persistence assumption a naive day-ahead weather
forecast would make, not a shortcut around the leakage rule.
``aggregate.py``'s own linear-interpolation-to-15min simplification for that
weather data is still inherited here.

Feature construction below is fully vectorised (no per-row Python loop): since
``origin`` is always just ``t.normalize()`` (midnight of t's own calendar day)
and ``horizon`` is always just how far past that midnight t is, every lag/
rolling feature is a plain ``Series.shift``/``rolling`` over the full
continuous 15-min index, computed once for the whole series.
"""

from __future__ import annotations

import json

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from src import aggregate

STEPS_PER_HOUR = 4  # 15-min resolution
STEPS_PER_DAY = 24 * STEPS_PER_HOUR  # 96
LAG_HOURS = [24, 48, 168]  # yesterday / 2 days ago / same weekday last week
LAG_STEPS = {f"lag_{h}h": h * STEPS_PER_HOUR for h in LAG_HOURS}
ROLLING_WINDOW_STEPS = STEPS_PER_DAY  # 24h window, anchored to end at t-96 steps (see module docstring)
SAME_TIMEOFDAY_LOOKBACK_DAYS = 7

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


def load_group_15min(name: str) -> pd.DataFrame:
    df = pd.read_csv(f"reports/{name}_15min.csv", index_col=0, parse_dates=True)
    return df


def build_supervised_table(df: pd.DataFrame) -> pd.DataFrame:
    s = df[TARGET]
    idx = s.index
    print(f"Building supervised table over {len(idx)} 15-min timestamps (vectorised)...")

    table = pd.DataFrame(index=idx)
    table["y"] = s

    for name, lag in LAG_STEPS.items():
        table[name] = s.shift(lag)

    # Smoothed recent history: mean of the 24h window ending at t-96 steps
    # (t-191..t-96), i.e. entirely within day D or earlier for every horizon.
    table["rolling_mean_24_48"] = s.shift(ROLLING_WINDOW_STEPS).rolling(ROLLING_WINDOW_STEPS).mean()

    # Mean of "this exact 15-min-of-day" across the past week (7 lags spaced
    # one day apart) -- all are >=1 day old, so always safe regardless of h.
    same_timeofday = pd.concat(
        [s.shift(STEPS_PER_DAY * k) for k in range(1, SAME_TIMEOFDAY_LOOKBACK_DAYS + 1)], axis=1
    )
    table["rolling_mean_same_timeofday_7d"] = same_timeofday.mean(axis=1, skipna=True)

    # Weather gets the same no-leakage lag/rolling treatment as the target --
    # same-day actuals are never used (see module docstring).
    for feat in WEATHER_FEATURES:
        base = df[feat]
        table[f"{feat}_lag_24h"] = base.shift(LAG_STEPS["lag_24h"])
        table[f"{feat}_rolling_mean_24_48"] = (
            base.shift(ROLLING_WINDOW_STEPS).rolling(ROLLING_WINDOW_STEPS).mean()
        )

    table["origin"] = idx.normalize()
    table["target_time"] = idx
    # horizon = 15-min-of-day index (0..95); origin is always midnight of t's
    # own day, so this is equivalent to (t - origin) / 15min without the
    # Timedelta-division deprecation warning that triggers on this pandas version.
    table["horizon"] = idx.hour * STEPS_PER_HOUR + idx.minute // 15
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
    f"{feat}_rolling_mean_24_48" for feat in WEATHER_FEATURES
]

FEATURE_COLUMNS = (
    list(LAG_STEPS.keys())
    + ["rolling_mean_24_48", "rolling_mean_same_timeofday_7d"]
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
    # week, last week" (lag_168h), the single most defensible no-model
    # forecast available at origin time.
    naive_mae = mean_absolute_error(test["y"], test["lag_168h"])

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
