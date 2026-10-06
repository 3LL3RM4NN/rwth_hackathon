"""Day-ahead, group-level forecasting model (LightGBM), built on the
15-minute-resolution aggregates from ``src/aggregate.py``.

Target: per-household average, not group total
------------------------------------------------
The model is trained on ``kWh_mean_per_active_household`` (the group's summed
consumption divided by however many households were reporting at that
instant), not the raw group total. This isn't a per-household model -- it's
still one model per group -- but the *ratio* is far more stable across
``aggregate.py``'s household-rollout ramp-up than the raw sum is, which is
what lets the usable history start ~1.5-2 years earlier than a sum-based
version of this pipeline could (see ``aggregate.py``'s module docstring).
Predictions are rescaled back to group/portfolio kWh totals at evaluation
time by multiplying by the *actual* historical ``active_household_count`` for
that timestamp -- realistic for a desk that knows its own contract count in
advance, though it does assume that count (not just the consumption rate) is
knowable ahead of the forecast, same spirit as the weather-actuals
simplification elsewhere in this pipeline.

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

Uncertainty: quantile regression, not a probabilistic model
-------------------------------------------------------------
Alongside the point-forecast model, two extra ``LGBMRegressor`` models per
group are trained with ``objective="quantile"`` at ``LOWER_QUANTILE``/
``UPPER_QUANTILE`` (0.05/0.95, a 90% nominal prediction interval), same
features and train/test rows throughout. This is the standard way to get
interval estimates out of gradient boosting, but it's three independently
fit models, not one joint distribution -- the three outputs aren't
guaranteed consistent with each other (rare "quantile crossing", where the
lower prediction ends up above the upper one for a given row, is detected
and clamped, not silently ignored). Reported alongside the point metrics:
PICP (realised coverage -- the fraction of actual test values that fall
inside the predicted interval, which should land close to 90% if the
intervals are well calibrated) and mean interval width (sharpness -- a
trivially wide interval gets perfect coverage for free, so width has to be
read together with PICP, not alone).
"""

from __future__ import annotations

import json

import numpy as np
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
TARGET = "kWh_mean_per_active_household"
TRAIN_FRACTION = 0.8

LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95
NOMINAL_COVERAGE = UPPER_QUANTILE - LOWER_QUANTILE  # 0.90

LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    random_state=0,
    verbosity=-1,
)


def _hours_before(timestamps: pd.DatetimeIndex, hours: int) -> pd.DatetimeIndex:
    return timestamps - pd.Timedelta(hours * 60, unit="m")


def mean_squared_percentage_error(y_true, y_pred) -> float:
    """L2 analogue of MAPE: mean of squared *relative* errors, i.e. the
    percentage-scale counterpart to MSE the same way MAPE is the
    percentage-scale counterpart to MAE. Floors the denominator at
    ``np.finfo(float64).eps`` (not 0), matching sklearn's own
    ``mean_absolute_percentage_error`` so both percentage metrics treat
    near-zero actuals the same way. Scale-invariant to the
    per-household-vs-rescaled-to-total distinction used everywhere else in
    this module, for the same reason MAPE is (see train_and_evaluate):
    rescaling both y_true and y_pred by the same factor leaves the ratio
    unchanged, so there's no separate "_total" version, same as MAPE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), np.finfo(np.float64).eps)
    return float(np.mean(((y_pred - y_true) / denom) ** 2))


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
    # Not a model feature -- carried through purely so evaluation can rescale
    # the per-household-average prediction back to a group/portfolio kWh
    # total (see module docstring).
    table["active_household_count"] = df["active_household_count"]
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


def train_quantile_predict(train: pd.DataFrame, test: pd.DataFrame, alpha: float) -> np.ndarray:
    model = LGBMRegressor(objective="quantile", alpha=alpha, **LGBM_PARAMS)
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

    print(f"Training LightGBM on {len(train)} rows (400 estimators)...")
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[FEATURE_COLUMNS], train["y"])
    print(f"Evaluating on {len(test)} held-out rows...")
    pred = model.predict(test[FEATURE_COLUMNS])

    print(
        f"Training quantile models ({LOWER_QUANTILE:.0%}/{UPPER_QUANTILE:.0%}, "
        f"{NOMINAL_COVERAGE:.0%} nominal interval)..."
    )
    pred_lower = train_quantile_predict(train, test, LOWER_QUANTILE)
    pred_upper = train_quantile_predict(train, test, UPPER_QUANTILE)
    # Independently-fit quantile models aren't guaranteed consistent with each
    # other; clamp the rare row where it happens rather than silently ignoring it.
    n_crossed = int((pred_lower > pred_upper).sum())
    if n_crossed:
        print(f"  {n_crossed} rows had quantile crossing (lower > upper); clamped.")
        pred_lower, pred_upper = np.minimum(pred_lower, pred_upper), np.maximum(pred_lower, pred_upper)

    picp = float(((test["y"] >= pred_lower) & (test["y"] <= pred_upper)).mean())
    interval_width = pred_upper - pred_lower
    mean_interval_width = float(interval_width.mean())
    mean_interval_width_total = float((interval_width * test["active_household_count"]).mean())

    # mae/mse/rmse/mape are on the kWh-per-household-per-15min scale, since
    # that's what the model actually predicts now (see module docstring).
    # mse_total/rmse_total rescale both actual and predicted by the
    # *realized* historical active_household_count for that row to get back
    # to group-total kWh/15min -- MAPE is scale-invariant to this rescaling
    # (the count cancels in the ratio since it multiplies both actual and
    # predicted identically), so there's only one mape, not a separate
    # "total" version. mse is the plain L2/squared error (what the model's
    # default regression objective actually minimizes); rmse = sqrt(mse) is
    # just mse brought back to the target's own units for interpretability.
    mae = mean_absolute_error(test["y"], pred)
    mse = mean_squared_error(test["y"], pred)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(test["y"], pred)
    # mspe/rmspe: L2's percentage-scale counterpart, same way mape is MAE's
    # (see mean_squared_percentage_error's docstring) -- also scale-invariant,
    # so also just one value, no "_total" version.
    mspe = mean_squared_percentage_error(test["y"], pred)
    rmspe = mspe ** 0.5

    count = test["active_household_count"]
    y_total = test["y"] * count
    pred_total = pred * count
    mae_total = mean_absolute_error(y_total, pred_total)
    mse_total = mean_squared_error(y_total, pred_total)
    rmse_total = mse_total ** 0.5

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
    naive_mae_total = mean_absolute_error(
        naive_valid["y"] * naive_valid["active_household_count"],
        naive_valid["naive_same_timeofday_last_week"] * naive_valid["active_household_count"],
    )

    result = {
        "name": name,
        "n_households": n_households,
        "n_rows_total": int(len(table)),
        "n_rows_dropped_missing_features": int(dropped),
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)),
        "train_origin_range": [str(train["origin"].min()), str(train["origin"].max())],
        "test_origin_range": [str(test["origin"].min()), str(test["origin"].max())],
        "mae_per_household": float(mae),
        "mse_per_household": float(mse),
        "rmse_per_household": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
        "rmspe": float(rmspe),
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
        "mae_by_horizon": by_horizon.round(3).to_dict(),
        "feature_importance": pd.Series(
            model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False).to_dict(),
    }

    test_out = test[["origin", "horizon", "target_time", "y", "active_household_count"]].copy()
    test_out["pred"] = pred
    test_out["pred_lower"] = pred_lower
    test_out["pred_upper"] = pred_upper
    test_out.to_csv(f"reports/{name}_test_predictions.csv", index=False)

    return result


if __name__ == "__main__":
    results = {}
    for name, pv in [("pv_group", True), ("non_pv_group", False), ("all_households_group", None)]:
        print(f"\n=== Training {name} ===")
        n_households = len(aggregate.group_household_ids(pv))
        res = train_and_evaluate(name, n_households)
        results[name] = res
        print(f"\n=== {name} ===")
        print(f"households={res['n_households']}  train_rows={res['n_rows_train']}  test_rows={res['n_rows_test']}")
        print(f"test period: {res['test_origin_range']}")
        print(
            f"MAE/household={res['mae_per_household']:.4f} kWh/15min  "
            f"MSE/household={res['mse_per_household']:.4f}  RMSE/household={res['rmse_per_household']:.4f}  "
            f"MAPE={res['mape']*100:.1f}%  RMSPE={res['rmspe']*100:.1f}%"
        )
        print(f"MAE (rescaled to group total)={res['mae_total']:.2f} kWh/15min  "
              f"MSE (rescaled)={res['mse_total']:.2f}")
        print(f"naive (same 15-min-of-day, last week) MAE/household={res['naive_mae_per_household']:.4f} kWh/15min")
        print(
            f"{res['nominal_coverage']*100:.0f}% prediction interval: PICP={res['picp']*100:.1f}%  "
            f"mean width/household={res['mean_interval_width_per_household']:.4f} kWh/15min  "
            f"({res['n_quantile_crossings']} quantile crossings)"
        )

    with open("reports/forecast_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Fair grouped-vs-ungrouped comparison: each model above was evaluated on
    # its own series' last 20% of days, which differ in length/start date
    # across groups (series start at different points due to meter rollout),
    # so their numbers above aren't directly comparable to each other.
    # Re-score all three on the single latest common test window instead.
    print("\nRe-scoring all groups on the common held-out test window...")
    common_start = max(r["test_origin_range"][0] for r in results.values())
    comparison = {}
    preds_by_name = {}
    for name, res in results.items():
        preds = pd.read_csv(
            f"reports/{name}_test_predictions.csv", parse_dates=["origin", "target_time"]
        )
        preds = preds[preds["origin"] >= common_start]
        preds_by_name[name] = preds
        mae = mean_absolute_error(preds["y"], preds["pred"])
        mse = mean_squared_error(preds["y"], preds["pred"])
        rmse = mse ** 0.5
        mape = mean_absolute_percentage_error(preds["y"], preds["pred"])
        mspe = mean_squared_percentage_error(preds["y"], preds["pred"])
        rmspe = mspe ** 0.5
        y_total = preds["y"] * preds["active_household_count"]
        pred_total = preds["pred"] * preds["active_household_count"]
        mae_total = mean_absolute_error(y_total, pred_total)
        mse_total = mean_squared_error(y_total, pred_total)
        rmse_total = mse_total ** 0.5
        comparison[name] = {
            "n_households": res["n_households"],
            "n_rows": int(len(preds)),
            "mae_per_household": float(mae),
            "mse_per_household": float(mse),
            "rmse_per_household": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
            "rmspe": float(rmspe),
            "mae_total": float(mae_total),
            "mse_total": float(mse_total),
            "rmse_total": float(rmse_total),
        }

    # Naive sum of each group's own total-scale MAE -- kept for contrast, but
    # this *overstates* the real combined-bid error: it implicitly assumes the
    # PV and non-PV groups' forecast errors are perfectly correlated (always
    # wrong in the same direction by the same amount), when in reality
    # independent errors partially cancel once you actually add the two
    # groups' bids together. See "portfolio-wide" below for the real number.
    grouped_mae_sum = comparison["pv_group"]["mae_total"] + comparison["non_pv_group"]["mae_total"]
    grouped_hh_sum = comparison["pv_group"]["n_households"] + comparison["non_pv_group"]["n_households"]
    comparison["grouped_naive_mae_sum"] = {
        "n_households": grouped_hh_sum,
        "mae_total": grouped_mae_sum,
    }

    # Portfolio-wide error: what the day-ahead desk actually cares about if
    # PV and non-PV are bid as one combined position. Each group's prediction
    # is per-household-average, so first rescale both actual and predicted by
    # that *group's own* active_household_count back to a group kWh total,
    # then sum the two groups' totals per 15-min step, then score the combined
    # series. This is the real error the "grouped" approach would produce as a
    # single bid, directly comparable to all_households_group's error (also
    # rescaled to a portfolio total via its own count), since both are now a
    # single portfolio-wide kWh/15min forecast.
    pv_preds = preds_by_name["pv_group"][["target_time", "y", "pred", "active_household_count"]].copy()
    pv_preds["y_total"] = pv_preds["y"] * pv_preds["active_household_count"]
    pv_preds["pred_total"] = pv_preds["pred"] * pv_preds["active_household_count"]
    nonpv_preds = preds_by_name["non_pv_group"][["target_time", "y", "pred", "active_household_count"]].copy()
    nonpv_preds["y_total"] = nonpv_preds["y"] * nonpv_preds["active_household_count"]
    nonpv_preds["pred_total"] = nonpv_preds["pred"] * nonpv_preds["active_household_count"]

    portfolio = pv_preds[["target_time", "y_total", "pred_total"]].merge(
        nonpv_preds[["target_time", "y_total", "pred_total"]],
        on="target_time",
        suffixes=("_pv", "_nonpv"),
    )
    portfolio_y = portfolio["y_total_pv"] + portfolio["y_total_nonpv"]
    portfolio_pred = portfolio["pred_total_pv"] + portfolio["pred_total_nonpv"]
    portfolio_mae = mean_absolute_error(portfolio_y, portfolio_pred)
    portfolio_mse = mean_squared_error(portfolio_y, portfolio_pred)
    portfolio_rmse = portfolio_mse ** 0.5
    portfolio_mape = mean_absolute_percentage_error(portfolio_y, portfolio_pred)
    portfolio_mspe = mean_squared_percentage_error(portfolio_y, portfolio_pred)
    portfolio_rmspe = portfolio_mspe ** 0.5
    comparison["grouped_portfolio_wide"] = {
        "n_households": grouped_hh_sum,
        "n_rows": int(len(portfolio)),
        "mae_total": float(portfolio_mae),
        "mse_total": float(portfolio_mse),
        "rmse_total": float(portfolio_rmse),
        "mape": float(portfolio_mape),
        "mspe": float(portfolio_mspe),
        "rmspe": float(portfolio_rmspe),
    }
    comparison["common_test_window_start"] = common_start

    print(f"\n=== Fair comparison on common test window (from {common_start}) ===")
    print(
        f"Ungrouped single model      : portfolio MAE={comparison['all_households_group']['mae_total']:.2f} "
        f"MSE={comparison['all_households_group']['mse_total']:.2f} kWh/15min  "
        f"MAPE={comparison['all_households_group']['mape']*100:.1f}%  "
        f"RMSPE={comparison['all_households_group']['rmspe']*100:.1f}%  "
        f"(MAE/hh={comparison['all_households_group']['mae_per_household']:.4f})"
    )
    print(
        f"Grouped, portfolio-wide     : portfolio MAE={comparison['grouped_portfolio_wide']['mae_total']:.2f} "
        f"MSE={comparison['grouped_portfolio_wide']['mse_total']:.2f} kWh/15min  "
        f"MAPE={comparison['grouped_portfolio_wide']['mape']*100:.1f}%  "
        f"RMSPE={comparison['grouped_portfolio_wide']['rmspe']*100:.1f}%  "
        f"(PV MAPE={comparison['pv_group']['mape']*100:.1f}%, non-PV MAPE={comparison['non_pv_group']['mape']*100:.1f}%)"
    )
    print(
        f"Grouped, naive MAE sum      : portfolio MAE={comparison['grouped_naive_mae_sum']['mae_total']:.2f}  "
        f"(sums each group's own total-scale MAE -- overstates the real combined error, see code comment)"
    )

    with open("reports/grouping_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
