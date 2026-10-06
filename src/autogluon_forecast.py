"""AutoGluon-TimeSeries benchmark (consumption only), isolated from the baselines.

Missing data: the daily matrices from `src.data` keep NaN for absent rows / unusable days. They are turned into a
*regular* 15-min grid and NaN targets are passed to AutoGluon as NaN (no interpolation here). AutoGluon's models then
handle NaN themselves (tabular models drop NaN targets/lag features, SeasonalNaive falls back to earlier values);
this is backward-looking inside the input window only. Leading NaN rows of a household are trimmed.

Forecast protocol (day-ahead): predictor is fitted ONCE on history <= train_end. For every test day T the model is
given data up to and including T-1 23:45 (last `context_days` days) and asked for 96 steps. Weights are never refit
on test data; only the context moves forward.
"""
import time

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

MODEL_NAME = "autogluon_timeseries"
OFFSETS = pd.to_timedelta(np.arange(96) * 15, unit="min").to_numpy()


def _frame(blocks: dict[str, tuple[pd.DatetimeIndex, np.ndarray]], static: pd.DataFrame | None = None) -> TimeSeriesDataFrame:
    """blocks: household -> (days, values[days, 96]) -> regular 15-min TimeSeriesDataFrame.

    `static` (optional): static household features indexed by Household_ID, attached as AutoGluon static_features."""
    ids, ts, y = [], [], []
    for hh, (days, vals) in blocks.items():
        t = (days.to_numpy()[:, None] + OFFSETS[None, :]).ravel()
        ids.append(np.full(len(t), hh, dtype=object))
        ts.append(t)
        y.append(vals.ravel())
    idx = pd.MultiIndex.from_arrays([np.concatenate(ids), np.concatenate(ts)], names=["item_id", "timestamp"])
    sf = static.loc[list(blocks)].rename_axis("item_id") if static is not None else None
    return TimeSeriesDataFrame(pd.DataFrame({"target": np.concatenate(y)}, index=idx), static_features=sf)


def _slice(mat: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, min_valid_days: int):
    """Days [start, end] of one household, leading all-NaN days trimmed; None if too little data."""
    w = mat.loc[start:end]
    valid = w.notna().any(axis=1).to_numpy()
    if valid.sum() < min_valid_days:
        return None
    first = np.argmax(valid)
    w = w.iloc[first:]
    return w.index, w.to_numpy()


def build_train_frame(mats, train_end: pd.Timestamp, train_days: int, min_valid_days: int, static=None):
    start = train_end - pd.Timedelta(days=train_days - 1)
    blocks = {}
    for hh, mat in mats.items():
        b = _slice(mat, start, train_end, min_valid_days)
        if b is not None:
            blocks[hh] = b
    return _frame(blocks, static)


def fit(train_data, model_dir, cfg: dict) -> TimeSeriesPredictor:
    predictor = TimeSeriesPredictor(
        prediction_length=cfg.get("prediction_length", 96), freq="15min", target="target", eval_metric="MAE",
        quantile_levels=[0.5], path=str(model_dir), verbosity=2,
        known_covariates_names=cfg.get("known_covariates") or None)
    predictor.fit(
        train_data, presets=cfg["preset"], hyperparameters=cfg["hyperparameters"],
        time_limit=cfg["time_limit"], num_val_windows=cfg["num_val_windows"],
        enable_ensemble=cfg["enable_ensemble"], random_seed=cfg["seed"])
    return predictor


def rolling_predict(predictor, mats, test_range, cfg: dict, log_every: int = 20, static=None) -> pd.DataFrame:
    """Day-ahead forecasts for every test day T from data <= T-1 only. Returns long frame (no actuals)."""
    out, t0 = [], time.time()
    origins = pd.date_range(test_range[0] - pd.Timedelta(days=1), test_range[1] - pd.Timedelta(days=1))
    if cfg.get("max_origins"):
        origins = origins[: cfg["max_origins"]]
    for k, O in enumerate(origins):
        T = O + pd.Timedelta(days=1)
        blocks = {}
        for hh, mat in mats.items():
            if T not in mat.index or not np.isfinite(mat.loc[T].to_numpy()).any():
                continue                                     # nothing to score for this household-day
            b = _slice(mat, O - pd.Timedelta(days=cfg["context_days"] - 1), O, cfg["min_context_valid_days"])
            if b is not None and b[0][-1] == O:              # context must end on origin day O (T-1)
                blocks[hh] = b
        if not blocks:
            continue
        data = _frame(blocks, static)
        assert data.index.get_level_values("timestamp").max() < T      # no data from the forecast day or later
        pred = predictor.predict(data)
        p = pred["0.5"] if "0.5" in pred.columns else pred["mean"]
        df = p.rename("prediction").reset_index().rename(columns={"item_id": "Household_ID", "timestamp": "Timestamp"})
        assert df["Timestamp"].min() == T and df["Timestamp"].max() == T + pd.Timedelta(minutes=23 * 60 + 45)
        df["prediction"] = df["prediction"].clip(lower=0.0)    # consumption cannot be negative
        out.append(df)
        if (k + 1) % log_every == 0:
            print(f"  origin {O.date()} ({k + 1}/{len(origins)}), {time.time() - t0:.0f}s elapsed", flush=True)
    return pd.concat(out, ignore_index=True)
