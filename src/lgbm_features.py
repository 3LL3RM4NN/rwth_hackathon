"""Explicit, explainable features for the day-ahead LightGBM model (origin = D-1 11:45, target = the 96 slots of day D).

DIRECT strategy: one global model; every row = (household, target day D, slot s). All features are anchored at the
origin O = D-1 11:45 or are lags of the target slot that are older than O, so no predicted value is ever fed back and
no later slot depends on an earlier forecast.

Target-side (from the daily matrices; day D-1 morning slots 00:00-11:45 come from the RAW series, earlier days from the
completeness-masked series, same rule as the AutoGluon runs):
    lag_1d      value of the same slot on D-1, only if that slot is <= 11:45 (slots 0-47), else NaN   (t-96, never crosses O)
    lag_2d      same slot on D-2 (t-192);   lag_7d  same slot on D-7 (t-672)
    mean_same_slot_4wk   mean of the same slot on D-7, D-14, D-21, D-28 (the seasonal-mean baseline as a feature)
    y_last / y_lag1h     value at the origin (11:45) and 1 h earlier
    roll_mean_{1h,6h,24h,7d}, roll_std_24h   statistics of the window ENDING at the origin (target slot excluded by construction)
Weather (historical only; see src/covariates.py): temperature observations <= D-1 11:00 (latest, lags 1h/6h/24h/7d, means, changes).
Calendar of the target day: slot, hour, quarter, day of week (+sin/cos), weekend, month, German national public holiday.
Household_ID is a categorical feature; pv_flag is optional.
"""
import numpy as np
import pandas as pd

from .covariates import CAL, TEMP, TS_INDEX, Covariates

ORIGIN_SLOT = 47           # slot of 11:45 on day D-1
FEATURES_BASE = ["household", "slot", "hour", "quarter", "dow", "is_weekend", "month", "is_holiday", "slot_sin", "slot_cos",
                 "dow_sin", "dow_cos", "lag_1d", "lag_2d", "lag_7d", "mean_same_slot_4wk", "y_last", "y_lag1h",
                 "roll_mean_1h", "roll_mean_6h", "roll_mean_24h", "roll_mean_7d", "roll_std_24h"] + TEMP


def _cums(a: np.ndarray):
    f = np.isfinite(a)
    z = np.where(f, a, 0.0)
    return (np.concatenate([[0.0], np.cumsum(z)]), np.concatenate([[0.0], np.cumsum(f)]), np.concatenate([[0.0], np.cumsum(z * z)]))


def _window(cm, cr, o, W, k):
    """Sum over the W slots ending at flat index o: last 48 slots (day D-1 morning) from the RAW cumsums, earlier from masked."""
    r_len = min(W, 48)
    s = cr[k][o + 1] - cr[k][np.maximum(o + 1 - r_len, 0)]
    if W > 48:
        s = s + cm[k][o + 1 - 48] - cm[k][np.maximum(o + 1 - W, 0)]
    return s


def build_rows(hh_code: int, mat_m: pd.DataFrame, mat_r: pd.DataFrame, target_days: pd.DatetimeIndex, cov: Covariates,
               station: str, min_valid_days: int = 7, context_days: int = 42):
    """Feature rows for the given target days of one household. Returns (X, y, dates) with y NaN where no actual."""
    M, R = mat_m.to_numpy(dtype="float64"), mat_r.to_numpy(dtype="float64")
    i_all = mat_m.index.get_indexer(target_days)
    valid_day = np.isfinite(M).any(axis=1)
    V = np.concatenate([[0], np.cumsum(valid_day)])
    keep = []
    for i in i_all:
        if i < 1:
            keep.append(False); continue
        a, b = max(0, i - context_days), i - 2            # same eligibility as the AutoGluon runs: >= 7 usable days in D-42..D-2
        keep.append(b >= a and (V[b + 1] - V[a]) >= min_valid_days)
    i_arr = i_all[np.array(keep, dtype=bool)]
    if len(i_arr) == 0:
        return None
    n = len(i_arr)
    Mf, Rf = M.ravel(), R.ravel()
    cm, cr = _cums(Mf), _cums(Rf)
    o = (i_arr - 1) * 96 + ORIGIN_SLOT                     # flat index of D-1 11:45
    origin = {}
    for name, W in (("roll_mean_1h", 4), ("roll_mean_6h", 24), ("roll_mean_24h", 96), ("roll_mean_7d", 672)):
        s, c = _window(cm, cr, o, W, 0), _window(cm, cr, o, W, 1)
        origin[name] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    s, c, s2 = _window(cm, cr, o, 96, 0), _window(cm, cr, o, 96, 1), _window(cm, cr, o, 96, 2)
    mean = s / np.maximum(c, 1)
    origin["roll_std_24h"] = np.where(c > 1, np.sqrt(np.maximum(s2 / np.maximum(c, 1) - mean ** 2, 0)), np.nan)
    origin["y_last"], origin["y_lag1h"] = Rf[o], Rf[o - 4]
    slot = np.arange(96)
    lag_1d = np.full((n, 96), np.nan); lag_1d[:, :48] = R[i_arr - 1, :48]
    def lag(d):
        out = np.full((n, 96), np.nan); ok = i_arr - d >= 0
        out[ok] = M[i_arr[ok] - d]; return out
    st = np.stack([lag(7 * k) for k in range(1, 5)])
    cnt = np.isfinite(st).sum(axis=0)
    m4 = np.where(cnt > 0, np.nansum(st, axis=0) / np.maximum(cnt, 1), np.nan)
    days = mat_m.index[i_arr]
    pos = TS_INDEX.get_indexer(days)                        # row of the day's 00:00 in the covariate tables
    assert (pos >= 0).all()
    X = {"household": np.full(n * 96, hh_code, dtype="int32"), "slot": np.tile(slot, n), "hour": np.tile(slot // 4, n),
         "quarter": np.tile(slot % 4, n), "dow": np.repeat(days.dayofweek.to_numpy(), 96)}
    X["is_weekend"] = (X["dow"] >= 5).astype("int8"); X["month"] = np.repeat(days.month.to_numpy(), 96)
    X["is_holiday"] = np.repeat(cov.cal["is_holiday"][pos], 96)
    X["slot_sin"], X["slot_cos"] = np.sin(2 * np.pi * X["slot"] / 96), np.cos(2 * np.pi * X["slot"] / 96)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * X["dow"] / 7), np.cos(2 * np.pi * X["dow"] / 7)
    X["lag_1d"], X["lag_2d"], X["lag_7d"], X["mean_same_slot_4wk"] = lag_1d.ravel(), lag(2).ravel(), lag(7).ravel(), m4.ravel()
    for k, v in origin.items():
        X[k] = np.repeat(v, 96)
    wt = cov.tables[station][pos]                           # (n, 9) temperature features anchored at D-1 11:00
    for j, name in enumerate(cov.weather_names):
        X[name] = np.repeat(wt[:, j], 96)
    X = pd.DataFrame({k: X[k] for k in FEATURES_BASE})
    y = M[i_arr].ravel()
    return X, y, np.repeat(days.to_numpy(), 96)
