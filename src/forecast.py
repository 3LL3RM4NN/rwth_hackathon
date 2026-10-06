"""Day-ahead, group-level forecasting model (LightGBM), built on the hourly
aggregates from ``src/aggregate.py``.

ProLoaF (the LSTM encoder-decoder engine the task brief asks for primarily) was
not installed in this environment: pulling and executing its setup code from
`git+https://github.com/sogno-platform/proloaf.git` was declined for this
sandboxed session (installing arbitrary code from an agent-chosen external repo
needs explicit user approval, and the user chose the documented fallback
instead). LightGBM gradient boosting is used in its place, as the task brief
explicitly allows.

Forecast setup
--------------
A forecast is "issued" at ``origin`` = midnight UTC of day D+1, using only data
from day D and earlier, to predict all 24 hourly values of day D+1
(``horizon`` h = 0..23, target timestamp t = origin + h).

Only lag features with lag >= 24h are used, because for h > 0 a lag < 24h
would reach into day D+1 itself (not yet known at the time the forecast is
issued) -- e.g. a naive "previous hour" feature is safe for h=0 but leaks
future information for h=5. Anchoring every lag to t (not to origin) at >=24h
keeps every lag strictly inside day D or earlier for every horizon.

Weather is the one deliberate exception/simplification: there is no day-ahead
weather *forecast* in this dataset, so same-day weather *actuals* at the
target timestamp t are used as a stand-in, per the task brief. This is a
known source of optimism in the reported accuracy and is called out again in
the report.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from src import aggregate

LAGS_HOURS = [24, 48, 168]  # yesterday / 2 days ago / same weekday last week
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


def load_group_hourly(name: str) -> pd.DataFrame:
    df = pd.read_csv(f"reports/{name}_hourly.csv", index_col=0, parse_dates=True)
    return df


def build_supervised_table(df: pd.DataFrame) -> pd.DataFrame:
    s = df[TARGET]
    origins = pd.date_range(
        df.index.min().normalize() + pd.Timedelta(days=1),
        df.index.max().normalize(),
        freq="D",
        tz="UTC",
    )

    rows = []
    for origin in origins:
        for h in range(24):
            t = origin + pd.Timedelta(hours=h)
            if t not in s.index:
                continue
            row = {"origin": origin, "horizon": h, "target_time": t, "y": s.get(t, np.nan)}
            for lag in LAGS_HOURS:
                row[f"lag_{lag}"] = s.get(t - pd.Timedelta(hours=lag), np.nan)
            # Smoothed recent history, anchored so its most recent point is
            # t-24h (i.e. entirely within day D or earlier for every horizon).
            window = s.reindex(pd.date_range(t - pd.Timedelta(hours=47), t - pd.Timedelta(hours=24), freq="h"))
            row["rolling_mean_24_48"] = window.mean()
            same_hour_week = [s.get(t - pd.Timedelta(hours=24 * k), np.nan) for k in range(1, 8)]
            row["rolling_mean_same_hour_7d"] = np.nanmean(same_hour_week)
            for feat in WEATHER_FEATURES:
                row[feat] = df[feat].get(t, np.nan)
            row["hour"] = t.hour
            row["dow"] = t.dayofweek
            row["month"] = t.month
            row["is_weekend"] = int(t.dayofweek >= 5)
            rows.append(row)

    table = pd.DataFrame(rows)
    return table


def chronological_split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = table["origin"].drop_duplicates().sort_values()
    n_train = int(len(origins) * TRAIN_FRACTION)
    train_origins = set(origins.iloc[:n_train])
    train = table[table["origin"].isin(train_origins)]
    test = table[~table["origin"].isin(train_origins)]
    return train, test


FEATURE_COLUMNS = (
    [f"lag_{lag}" for lag in LAGS_HOURS]
    + ["rolling_mean_24_48", "rolling_mean_same_hour_7d"]
    + WEATHER_FEATURES
    + ["hour", "dow", "month", "is_weekend", "horizon"]
)


def train_and_evaluate(name: str, n_households: int) -> dict:
    df = load_group_hourly(name)
    table = build_supervised_table(df)
    usable = table.dropna(subset=FEATURE_COLUMNS + ["y"])
    dropped = len(table) - len(usable)

    train, test = chronological_split(usable)

    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=0,
        verbosity=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["y"])
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

    # Naive day-ahead baseline for context: "same hour, same day of week, last
    # week" (lag_168), the single most defensible no-model forecast available
    # at origin time.
    naive_mae = mean_absolute_error(test["y"], test["lag_168"])

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
        n_households = len(aggregate.group_household_ids(pv))
        res = train_and_evaluate(name, n_households)
        results[name] = res
        print(f"\n=== {name} ===")
        print(f"households={res['n_households']}  train_rows={res['n_rows_train']}  test_rows={res['n_rows_test']}")
        print(f"test period: {res['test_origin_range']}")
        print(f"MAE={res['mae']:.2f} kWh/h  RMSE={res['rmse']:.2f} kWh/h  MAPE={res['mape']*100:.1f}%")
        print(f"MAE per household={res['mae_per_household']:.4f} kWh/h")
        print(f"naive (same hour, last week) MAE={res['naive_lag168_mae']:.2f} kWh/h")

    with open("reports/forecast_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Fair grouped-vs-ungrouped comparison: each model above was evaluated on
    # its own series' last 20% of days, which differ in length/start date
    # across groups (series start at different points due to meter rollout),
    # so their MAE numbers above aren't directly comparable to each other.
    # Re-score all three on the single latest common test window instead.
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
