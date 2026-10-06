"""Shared screening evaluation: stored pooled-AutoGluon predictions (reference A) vs new interval predictions.

Same definitions as scripts/run_weather_calendar.py / run_level2_procurement.py (common timestamps, complete household-days,
error = forecast - actual)."""
import dataclasses

import numpy as np
import pandas as pd

from . import baseline, evaluation


def evaluate(frames: dict, ref_name: str, masked: dict, scored: list, cfg, days, test_range, out):
    models = ["seasonal_mean_4weeks"] + list(frames)
    bcfg = dataclasses.replace(cfg, models=("seasonal_mean_4weeks",))
    by = {m: {h: g[["Household_ID", "Timestamp", "actual", "prediction", "model"]] for h, g in f.groupby("Household_ID")} for m, f in frames.items()}
    chunks, sm_all = [], []
    for hh in scored:
        base, _ = baseline.predict_household(hh, masked[hh], bcfg, test_range)
        if len(base):
            base = base[base["Timestamp"].dt.normalize().isin(days)]; sm_all.append(base)
        fr = ([base] if len(base) else []) + [by[m][hh] for m in frames if hh in by[m]]
        chunks.append(evaluation.chunk_stats(pd.concat(fr, ignore_index=True), cfg.mape_min_actual, len(models)))
    res = evaluation.combine(chunks, {"household": ["model", "Household_ID"], "qh": ["model", "qh"], "date": ["model", "date"]})
    res["household"].to_csv(out / "metrics" / "per_household.csv", index=False)
    res["date"].to_csv(out / "metrics" / "per_date.csv", index=False)
    hm = res["household"].query("scope == 'common'"); ov = res["overall"].query("scope == 'common'").set_index("model")
    key = ["Household_ID", "Timestamp"]
    common = frames[ref_name][key + ["actual"]].copy()
    for m, f in frames.items():
        common = common.merge(f[key + ["prediction"]].rename(columns={"prediction": m}), on=key, how="inner")
    sm = pd.concat(sm_all, ignore_index=True)
    common = common.merge(sm[key + ["prediction"]].rename(columns={"prediction": "seasonal_mean_4weeks"}), on=key, how="inner")
    n_ = common.groupby(["Household_ID", common["Timestamp"].dt.normalize()])["actual"].transform("size"); common = common[n_ == 96]
    rows = []
    for m in models:
        e = common[m] - common["actual"]
        pi = common.assign(e=e).groupby("Timestamp")[["e", "actual"]].sum()
        pdl = common.assign(e=e, d=common["Timestamp"].dt.normalize()).groupby("d")[["e", "actual"]].sum()
        h = hm[hm["model"] == m]
        r = {"model": m, "households": h["Household_ID"].nunique(), "scored_intervals": int(ov.loc[m, "n"]),
             "mean_household_MAE": h["MAE"].mean(), "median_household_MAE": h["MAE"].median(), "overall_MAE": ov.loc[m, "MAE"],
             "RMSE": ov.loc[m, "RMSE"], "sMAPE_%": ov.loc[m, "sMAPE_%"], "MAPE_%": ov.loc[m, "MAPE_%"],
             "household_daily_energy_MAE_kWh": ov.loc[m, "daily_energy_MAE_kWh"], "bias": ov.loc[m, "bias"],
             "portfolio_interval_MAE_kWh": pi["e"].abs().mean(), "portfolio_over_kWh": pi["e"].clip(lower=0).sum(),
             "portfolio_under_kWh": (-pi["e"]).clip(lower=0).sum(), "portfolio_day_MAE_kWh": pdl["e"].abs().mean(),
             "portfolio_day_bias_kWh": pdl["e"].mean(), "portfolio_day_bias_%": 100 * pdl["e"].sum() / pdl["actual"].sum(),
             "portfolio_days_under_%": 100 * (pdl["e"] < 0).mean()}
        for ratio in (1.0, 2.0):
            r[f"cost_per_kWh_ratio_{ratio:g}"] = (ratio * (-pi["e"]).clip(lower=0).sum() + pi["e"].clip(lower=0).sum()) / pi["actual"].sum()
        rows.append(r)
    comp = pd.DataFrame(rows)
    a_row = comp[comp.model == ref_name].iloc[0]
    for col in ("mean_household_MAE", "overall_MAE", "RMSE", "household_daily_energy_MAE_kWh", "portfolio_day_MAE_kWh",
                "portfolio_interval_MAE_kWh", "cost_per_kWh_ratio_1", "cost_per_kWh_ratio_2"):
        comp[f"{col}_vs_ref_%"] = 100 * (comp[col] - a_row[col]) / a_row[col]
    hmc = hm.pivot(index="Household_ID", columns="model", values="MAE")
    for m in frames:
        if m != ref_name:
            d = hmc[m] - hmc[ref_name]
            comp.loc[comp.model == m, "households_better_than_ref"] = int((d < 0).sum())
            comp.loc[comp.model == m, "households_worse_than_ref"] = int((d > 0).sum())
    comp.to_csv(out / "metrics" / "comparison.csv", index=False)
    return comp
