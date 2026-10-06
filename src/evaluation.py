"""Metrics, accumulated from sufficient statistics so predictions can be streamed household by household.

Two scopes are reported:
  all     every (household, timestamp) the model can predict
  common  only timestamps predicted by *all* models (fair like-for-like comparison)
"""
import numpy as np
import pandas as pd


def _stats(df: pd.DataFrame, by: list[str], mape_min: float) -> pd.DataFrame:
    a, p = df["actual"].to_numpy(), df["prediction"].to_numpy()
    err = p - a
    denom = np.abs(a) + np.abs(p)
    t = df[by].copy()
    t["n"] = 1
    t["abs"] = np.abs(err)
    t["sq"] = err ** 2
    t["err"] = err
    t["actual"] = a
    ape_ok = np.abs(a) >= mape_min                       # MAPE only where actual is not ~0
    t["ape"] = np.where(ape_ok, np.abs(err) / np.where(ape_ok, np.abs(a), 1), 0.0)
    t["n_ape"] = ape_ok.astype(int)
    sm_ok = denom > 0                                    # sMAPE: 0/0 intervals are excluded
    t["smape"] = np.where(sm_ok, 2 * np.abs(err) / np.where(sm_ok, denom, 1), 0.0)
    t["n_smape"] = sm_ok.astype(int)
    return t.groupby(by, observed=True, sort=False).sum()


def chunk_stats(df: pd.DataFrame, mape_min: float, n_models: int) -> dict[str, pd.DataFrame]:
    """Sufficient statistics of one household's long prediction frame."""
    df = df.copy()
    df["date"] = df["Timestamp"].dt.normalize()
    df["qh"] = df["Timestamp"].dt.hour * 4 + df["Timestamp"].dt.minute // 15
    df["common"] = df.groupby("Timestamp")["model"].transform("size") == n_models
    out = {k: [] for k in ("household", "qh", "date", "hh_day")}
    for scope, sub in (("all", df), ("common", df[df["common"]])):
        if sub.empty:
            continue
        for key, by in (("household", ["model", "Household_ID"]), ("qh", ["model", "qh"]), ("date", ["model", "date"])):
            s = _stats(sub, by, mape_min).reset_index()
            s["scope"] = scope
            out[key].append(s)
        # per household-day energy totals (complete 96-interval days only)
        d = sub.groupby(["model", "Household_ID", "date"], observed=True).agg(
            n=("actual", "size"), actual=("actual", "sum"), prediction=("prediction", "sum")).reset_index()
        d = d[d["n"] == 96]
        d["scope"] = scope
        out["hh_day"].append(d)
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in out.items()}


def _finalize(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["MAE"] = s["abs"] / s["n"]
    s["RMSE"] = np.sqrt(s["sq"] / s["n"])
    s["bias"] = s["err"] / s["n"]
    s["mean_actual"] = s["actual"] / s["n"]
    s["mean_pred"] = s["mean_actual"] + s["bias"]
    s["WAPE_%"] = 100 * s["abs"] / s["actual"].replace(0, np.nan)
    s["MAPE_%"] = 100 * s["ape"] / s["n_ape"].replace(0, np.nan)
    s["sMAPE_%"] = 100 * s["smape"] / s["n_smape"].replace(0, np.nan)
    return s.drop(columns=["abs", "sq", "err", "actual", "ape", "n_ape", "smape", "n_smape"])


def combine(chunks: list[dict], by_key: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    """Sum the per-household sufficient statistics and turn them into metrics."""
    res = {}
    sum_cols = ["n", "abs", "sq", "err", "actual", "ape", "n_ape", "smape", "n_smape"]
    for key, by in by_key.items():
        allc = pd.concat([c[key] for c in chunks if len(c[key])], ignore_index=True)
        res[key] = _finalize(allc.groupby(["scope"] + by, sort=False)[sum_cols].sum().reset_index()
                             if key != "household" else allc)
    # overall = household stats summed over households
    hh = pd.concat([c["household"] for c in chunks if len(c["household"])], ignore_index=True)
    res["overall"] = _finalize(hh.groupby(["scope", "model"], sort=False)[sum_cols].sum().reset_index())
    # daily energy error
    days = pd.concat([c["hh_day"] for c in chunks if len(c["hh_day"])], ignore_index=True)
    days["abs_daily_err"] = (days["prediction"] - days["actual"]).abs()
    de = days.groupby(["scope", "model"], sort=False).agg(
        household_days=("n", "size"), daily_energy_MAE_kWh=("abs_daily_err", "mean"),
        mean_daily_actual_kWh=("actual", "mean")).reset_index()
    de["daily_energy_MAPE_%"] = 100 * de["daily_energy_MAE_kWh"] / de["mean_daily_actual_kWh"]
    res["overall"] = res["overall"].merge(de, on=["scope", "model"])
    return res
