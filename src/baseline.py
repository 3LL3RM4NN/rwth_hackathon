"""Seasonal-naive baselines on a household's daily matrix (days x 96).

Forecast for day T is issued at the end of day T-1, so only days <= T-1 may be read.
All lags are therefore >= 1 day; `check_no_leakage` asserts this.
"""
import numpy as np
import pandas as pd

# lags (in days, relative to the forecast day T) of the source days each model reads
MODEL_LAGS = {
    "naive_1day": [1],                        # same interval, previous day (D)
    "naive_7day": [7],                        # same interval, same weekday last week (T-7)
    "seasonal_mean_4weeks": [7, 14, 21, 28],  # mean over previous 4 same weekdays
}


def check_no_leakage():
    assert all(l >= 1 for lags in MODEL_LAGS.values() for l in lags), "a baseline reads the forecast day or the future"


def forecast_matrix(values: np.ndarray, model: str, cfg) -> np.ndarray:
    """Forecast for every day of `values` (days x 96); NaN where the model cannot predict."""
    n_days = values.shape[0]
    lags = [7 * k for k in range(1, cfg.mean_weeks + 1)] if model == "seasonal_mean_4weeks" else MODEL_LAGS[model]
    stack = np.full((len(lags),) + values.shape, np.nan)
    for i, lag in enumerate(lags):
        if lag < n_days:
            stack[i, lag:] = values[:-lag]            # row T receives values[T - lag]
    if len(lags) == 1:
        return stack[0]
    cnt = np.isfinite(stack).sum(axis=0)
    mean = np.nansum(stack, axis=0) / np.maximum(cnt, 1)
    return np.where(cnt >= cfg.mean_min_occurrences, mean, np.nan)


def predict_household(hh: str, mat: pd.DataFrame, cfg, test_range) -> tuple[pd.DataFrame, list[dict]]:
    """Long prediction frame for the test days + coverage records (days that could not be predicted)."""
    check_no_leakage()
    values = mat.to_numpy()
    days = mat.index
    test_mask = (days >= test_range[0]) & (days <= test_range[1])
    has_actual = np.isfinite(values).any(axis=1)
    enough_hist = np.arange(len(days)) >= cfg.min_history_days
    frames, coverage = [], []
    for m in cfg.models:
        fc = forecast_matrix(values, m, cfg)
        eval_days = test_mask & has_actual
        ok = eval_days & np.isfinite(fc).any(axis=1) & enough_hist
        coverage.append({"Household_ID": hh, "model": m, "test_days_with_actual": int(eval_days.sum()),
                         "predicted_days": int(ok.sum()), "no_prediction_days": int((eval_days & ~ok).sum())})
        if not ok.any():
            continue
        offsets = pd.to_timedelta(np.arange(cfg.intervals_per_day) * 15, unit="min").to_numpy()
        ts = days[ok].to_numpy()[:, None] + offsets[None, :]
        a, p = values[ok], fc[ok]
        keep = np.isfinite(a) & np.isfinite(p)
        frames.append(pd.DataFrame({"Household_ID": hh, "Timestamp": ts[keep],
                                    "actual": a[keep], "prediction": p[keep], "model": m}))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), coverage
