"""Day-ahead forecasting with CatBoost, as a direct model-family comparison
against ``src/forecast.py``'s LightGBM results.

Reuses ``forecast.py``'s data prep (``build_supervised_table``,
``chronological_split``, ``FEATURE_COLUMNS``, ``TARGET``) and metric
definitions (per-household vs. rescaled-to-total MAE/RMSE, the naive
same-time-last-week baseline, the 90% quantile prediction interval with
PICP/width) unchanged, so the two models are compared on identical rows,
features, train/test split, and metrics -- only the model family differs.
See ``forecast.py``'s module docstring for what each of those choices means
and why; this module doesn't repeat that reasoning, just the model swap.

CatBoost's hyperparameters are chosen to roughly match LightGBM's effective
capacity rather than independently tuned: ``iterations=400`` (vs.
``n_estimators=400``), ``depth=5`` (2**5=32 leaves, close to LightGBM's
``num_leaves=31``), ``learning_rate=0.05`` (same), ``min_data_in_leaf=20``
(vs. ``min_child_samples=20``). Neither model's hyperparameters were tuned
against the test set -- this is a fair-ish, not a best-effort, comparison of
the two libraries' out-of-the-box behaviour on this exact pipeline, not a
claim that either is better once properly tuned.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from src import aggregate
from src.forecast import (
    FEATURE_COLUMNS,
    LOWER_QUANTILE,
    NOMINAL_COVERAGE,
    UPPER_QUANTILE,
    build_supervised_table,
    chronological_split,
    load_group_15min,
)

CATBOOST_PARAMS = dict(
    iterations=400,
    learning_rate=0.05,
    depth=5,
    min_data_in_leaf=20,
    random_seed=0,
    verbose=False,
)


def train_quantile_predict(train: pd.DataFrame, test: pd.DataFrame, alpha: float) -> np.ndarray:
    model = CatBoostRegressor(loss_function=f"Quantile:alpha={alpha}", **CATBOOST_PARAMS)
    model.fit(train[FEATURE_COLUMNS], train["y"])
    return model.predict(test[FEATURE_COLUMNS])


def train_and_evaluate(name: str, n_households: int) -> dict:
    print(f"Loading reports/{name}_15min.csv...")
    df = load_group_15min(name)
    table = build_supervised_table(df)
    usable = table.dropna(subset=FEATURE_COLUMNS + ["y"])
    dropped = len(table) - len(usable)
    print(f"{len(usable)}/{len(table)} rows usable after dropping missing features/target ({dropped} dropped)")

    train, test = chronological_split(usable)

    print(f"Training CatBoost on {len(train)} rows (400 iterations)...")
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train[FEATURE_COLUMNS], train["y"])
    print(f"Evaluating on {len(test)} held-out rows...")
    pred = model.predict(test[FEATURE_COLUMNS])

    mae = mean_absolute_error(test["y"], pred)
    mse = mean_squared_error(test["y"], pred)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(test["y"], pred)

    count = test["active_household_count"]
    y_total = test["y"] * count
    pred_total = pred * count
    mae_total = mean_absolute_error(y_total, pred_total)
    mse_total = mean_squared_error(y_total, pred_total)
    rmse_total = mse_total ** 0.5

    print(
        f"Training quantile models ({LOWER_QUANTILE:.0%}/{UPPER_QUANTILE:.0%}, "
        f"{NOMINAL_COVERAGE:.0%} nominal interval)..."
    )
    pred_lower = train_quantile_predict(train, test, LOWER_QUANTILE)
    pred_upper = train_quantile_predict(train, test, UPPER_QUANTILE)
    n_crossed = int((pred_lower > pred_upper).sum())
    if n_crossed:
        print(f"  {n_crossed} rows had quantile crossing (lower > upper); clamped.")
        pred_lower, pred_upper = np.minimum(pred_lower, pred_upper), np.maximum(pred_lower, pred_upper)

    picp = float(((test["y"] >= pred_lower) & (test["y"] <= pred_upper)).mean())
    interval_width = pred_upper - pred_lower
    mean_interval_width = float(interval_width.mean())
    mean_interval_width_total = float((interval_width * count).mean())

    naive_valid = test.dropna(subset=["naive_same_timeofday_last_week"])
    naive_mae = mean_absolute_error(naive_valid["y"], naive_valid["naive_same_timeofday_last_week"])
    naive_mae_total = mean_absolute_error(
        naive_valid["y"] * naive_valid["active_household_count"],
        naive_valid["naive_same_timeofday_last_week"] * naive_valid["active_household_count"],
    )

    result = {
        "name": name,
        "n_households": n_households,
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)),
        "train_origin_range": [str(train["origin"].min()), str(train["origin"].max())],
        "test_origin_range": [str(test["origin"].min()), str(test["origin"].max())],
        "mae_per_household": float(mae),
        "mse_per_household": float(mse),
        "rmse_per_household": float(rmse),
        "mape": float(mape),
        "mae_total": float(mae_total),
        "mse_total": float(mse_total),
        "rmse_total": float(rmse_total),
        "naive_mae_per_household": float(naive_mae),
        "naive_mae_total": float(naive_mae_total),
        "nominal_coverage": NOMINAL_COVERAGE,
        "picp": picp,
        "mean_interval_width_per_household": mean_interval_width,
        "mean_interval_width_total": mean_interval_width_total,
        "n_quantile_crossings": n_crossed,
        "feature_importance": pd.Series(
            model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False).to_dict(),
    }

    test_out = test[["origin", "horizon", "target_time", "y", "active_household_count"]].copy()
    test_out["pred"] = pred
    test_out["pred_lower"] = pred_lower
    test_out["pred_upper"] = pred_upper
    test_out.to_csv(f"reports/{name}_catboost_test_predictions.csv", index=False)

    return result


if __name__ == "__main__":
    try:
        with open("reports/forecast_metrics.json") as f:
            lgbm_results = json.load(f)
    except FileNotFoundError:
        lgbm_results = {}
        print("(reports/forecast_metrics.json not found -- run `python3 -m src.forecast` first "
              "for a side-by-side comparison; continuing with CatBoost-only results.)")

    results = {}
    for name, pv in [("pv_group", True), ("non_pv_group", False), ("all_households_group", None)]:
        print(f"\n=== Training {name} (CatBoost) ===")
        n_households = len(aggregate.group_household_ids(pv))
        res = train_and_evaluate(name, n_households)
        results[name] = res
        print(f"\n=== {name} (CatBoost) ===")
        print(f"households={res['n_households']}  train_rows={res['n_rows_train']}  test_rows={res['n_rows_test']}")
        print(f"test period: {res['test_origin_range']}")
        print(
            f"MAE/household={res['mae_per_household']:.4f} kWh/15min  "
            f"MSE/household={res['mse_per_household']:.4f}  RMSE/household={res['rmse_per_household']:.4f}  "
            f"MAPE={res['mape']*100:.1f}%"
        )
        print(f"MAE (rescaled to group total)={res['mae_total']:.2f} kWh/15min  "
              f"MSE (rescaled)={res['mse_total']:.2f}")
        print(f"naive (same 15-min-of-day, last week) MAE/household={res['naive_mae_per_household']:.4f} kWh/15min")
        print(
            f"{res['nominal_coverage']*100:.0f}% prediction interval: PICP={res['picp']*100:.1f}%  "
            f"mean width/household={res['mean_interval_width_per_household']:.4f} kWh/15min  "
            f"({res['n_quantile_crossings']} quantile crossings)"
        )

        if name in lgbm_results:
            lgbm = lgbm_results[name]
            print(f"\n--- {name}: LightGBM vs CatBoost (same rows/features/split) ---")
            print(f"{'metric':30s} {'LightGBM':>12s} {'CatBoost':>12s} {'delta':>10s}")
            for label, key, fmt in [
                ("MAPE (%)", "mape", lambda v: v * 100),
                ("MAE/household", "mae_per_household", lambda v: v),
                ("MSE/household", "mse_per_household", lambda v: v),
                ("RMSE/household", "rmse_per_household", lambda v: v),
                ("PICP (%)", "picp", lambda v: v * 100),
                ("mean interval width/hh", "mean_interval_width_per_household", lambda v: v),
            ]:
                lgbm_v = fmt(lgbm[key])
                cb_v = fmt(res[key])
                print(f"{label:30s} {lgbm_v:12.4f} {cb_v:12.4f} {cb_v - lgbm_v:+10.4f}")

    with open("reports/forecast_metrics_catboost.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote reports/forecast_metrics_catboost.json")
