"""Weather + calendar screening: ONE new pooled AutoGluon model vs the stored pooled predictions (D-1 11:45 cutoff).

    uv run python scripts/run_weather_calendar.py [--dry_run] [--max_days N] [--out DIR]

A   = existing pooled predictions (outputs/level1_fast_screening, reused, not retrained)
new = pooled model + calendar (weekend, month, German national holiday) + historical temperature features
Everything else (training window, model set, seed, objective, households, days, cutoff) is identical to A.
Weather is passed as KNOWN covariates anchored to the cutoff of each day (see src/covariates.py): AutoGluon's tabular
models do not support past covariates.
"""
import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from run_autogluon import AG_CFG as BASE_CFG
from run_level1_pv import BASELINES
from src import baseline, evaluation
from src.autogluon_forecast import build_train_frame, fit
from src.config import Config
from src.cov_forecast import rolling_predict_cov, verify_covariate_cutoff, with_covariates
from src.covariates import CAL, TEMP, Covariates, load_station_temps
from src.data import load_all, split_dates
from src.level1_pv import HORIZON, pv_status_table, to_interval_frame, verify_cutoff

VARIANTS = {"calendar": ("B", CAL), "weather": ("C", TEMP), "both": ("D", CAL + TEMP)}
M_A = "pooled_baseline_stored"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["both"], choices=list(VARIANTS))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--time_limit", type=int, default=BASE_CFG["time_limit"])
    ap.add_argument("--train_days", type=int, default=BASE_CFG["train_days"])
    ap.add_argument("--max_days", type=int, default=None, help="debug: only the first N sampled days")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--screen_dir", type=Path, default=None)
    a = ap.parse_args()
    ag = copy.deepcopy(BASE_CFG)
    ag.update(time_limit=a.time_limit, train_days=a.train_days, max_origins=None, prediction_length=HORIZON)
    cfg = Config()
    base_cfg = dataclasses.replace(cfg, models=BASELINES)
    out = a.out or cfg.output_dir / "level2_weather_calendar_screening"
    screen = a.screen_dir or cfg.output_dir / "level1_fast_screening"
    for sub in ("predictions", "metrics"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    masked = {hh: m for hh, m, _ in load_all(cfg)}
    raw = {hh: m for hh, m, _ in load_all(dataclasses.replace(cfg, min_day_completeness=0.0))}
    first = min(m.index.min() for m in masked.values()); last = max(m.index.max() for m in masked.values())
    test_range = split_dates(first, last, cfg)["test"]; train_end = test_range[0] - pd.Timedelta(days=1)
    pv = pv_status_table(cfg).set_index("Household_ID").loc[list(masked)]
    has_test = pd.Series({h: bool(((m.index >= test_range[0]) & np.isfinite(m.to_numpy()).any(axis=1)).any()) for h, m in masked.items()})
    scored = pv.index[(pv.pv_status != "unknown") & has_test].tolist()
    pool_ids = pv.index[pv.pv_status != "unknown"].tolist()               # same 245 training households as the pooled model A
    assert len(scored) == 241 and len(pool_ids) == 245
    sel = json.loads((screen / "selected_weeks.json").read_text())
    days = pd.DatetimeIndex(sel["forecast_days"])
    if a.max_days:
        days = days[: a.max_days]
    hh_meta = pd.read_csv(cfg.data_dir.parent / "smart_meter_meta_data" / "households.csv", sep=";", dtype={"Household_ID": str})
    station_of = dict(zip(hh_meta["Household_ID"], hh_meta["Weather_ID"].astype(str)))
    temps = load_station_temps(cfg.data_dir.parent / "weather_data_hourly")
    print(f"households scored {len(scored)}, training households {len(pool_ids)}, forecast days {len(days)}, "
          f"weather stations {sorted(temps)}; features: calendar {CAL}, temperature {TEMP}")
    print("Config:", json.dumps({**{k: v for k, v in ag.items() if k != 'hyperparameters'}, "variants": a.variants,
                                "train_end": str(train_end.date()), "cutoff": "D-1 11:45", "weather_anchor": "D-1 11:00 (latest hourly obs at cutoff)"}, indent=2), flush=True)

    # ---- checks before any training -----------------------------------------------------------------------------
    verify_cutoff({h: masked[h] for h in scored}, {h: raw[h] for h in scored}, test_range, ag, n_days=4)   # existing target-side test
    verify_covariate_cutoff(temps, station_of, TEMP, days)
    if a.dry_run:
        c = Covariates(temps, station_of, CAL + TEMP)
        idx = pd.MultiIndex.from_product([scored[:3], pd.date_range("2023-06-20", periods=4, freq="15min")])
        print(c.frame(idx).round(2).to_string())
        print("dry run finished"); return
    (out / "run_config.json").write_text(json.dumps({**{k: v for k, v in ag.items()}, "variants": a.variants, "temperature_features": TEMP,
                                                    "calendar_features": CAL, "weather_anchor": "D-1 11:00"}, indent=2, default=str))

    # ---- fit + predict each variant ---------------------------------------------------------------------------------
    info = {}
    all_pred = {}
    for v in a.variants:
        label, names = VARIANTS[v]
        cov = Covariates(temps, station_of, names)
        vcfg = {**ag, "known_covariates": names}
        train = build_train_frame({h: masked[h] for h in pool_ids}, train_end, ag["train_days"], ag["min_train_valid_days"])
        train = with_covariates(train, cov)
        assert train.index.get_level_values("timestamp").max() <= train_end + pd.Timedelta(hours=23, minutes=45)
        print(f"[{label}:{v}] train series {train.num_items}, rows {len(train)}, covariates {names}", flush=True)
        t0 = time.time(); predictor = fit(train, out / f"model_{v}", vcfg); fit_s = time.time() - t0
        lb = predictor.leaderboard(silent=True); lb.to_csv(out / "metrics" / f"leaderboard_{v}.csv", index=False)
        print(lb[["model", "score_val", "fit_time_marginal"]].to_string(), flush=True)
        if v in ("both", "weather"):
            verify_covariate_cutoff(temps, station_of, names, days, n_days=1, predictor=predictor,
                                    masked={h: masked[h] for h in scored}, raw={h: raw[h] for h in scored}, cfg=ag)
        t0 = time.time()
        p = rolling_predict_cov(predictor, {h: masked[h] for h in scored}, {h: raw[h] for h in scored}, cov, vcfg, days)
        p["pv_status"] = p["Household_ID"].map(pv["pv_status"])
        iv = to_interval_frame(p, masked, "pooled_weather_calendar" if v == "both" else f"{label}_pooled_{v}")
        iv.to_parquet(out / "predictions" / f"{v}_interval_predictions.parquet", index=False)
        all_pred[label] = iv
        info[v] = {"label": label, "covariates": names, "fit_seconds": round(fit_s, 1), "predict_seconds": round(time.time() - t0, 1),
                   "models_trained": lb["model"].tolist(), "best_model": predictor.model_best, "series_in_training": int(train.num_items)}
        (out / "metrics" / "run_info.json").write_text(json.dumps(info, indent=2, default=str))

    # ---- evaluation -------------------------------------------------------------------------------------------------------
    A = pd.read_parquet(screen / "predictions" / "pooled_interval_predictions.parquet"); A["model"] = M_A
    frames = {M_A: A, **{iv["model"].iloc[0]: iv for iv in all_pred.values()}}
    models = ["seasonal_mean_4weeks"] + list(frames)
    by = {m: {h: g[["Household_ID", "Timestamp", "actual", "prediction", "model"]] for h, g in f.groupby("Household_ID")} for m, f in frames.items()}
    chunks = {}
    for hh in scored:
        base, _ = baseline.predict_household(hh, masked[hh], dataclasses.replace(cfg, models=("seasonal_mean_4weeks",)), test_range)
        base = base[base["Timestamp"].dt.normalize().isin(days)] if len(base) else base
        fr = ([base] if len(base) else []) + [by[m][hh] for m in frames if hh in by[m]]
        chunks[hh] = evaluation.chunk_stats(pd.concat(fr, ignore_index=True), cfg.mape_min_actual, len(models))
    by_key = {"household": ["model", "Household_ID"], "qh": ["model", "qh"], "date": ["model", "date"]}
    res = evaluation.combine(list(chunks.values()), by_key)
    res["household"].to_csv(out / "metrics" / "per_household.csv", index=False); res["date"].to_csv(out / "metrics" / "per_date.csv", index=False)
    hm = res["household"].query("scope == 'common'"); ov = res["overall"].query("scope == 'common'").set_index("model")
    # portfolio / procurement metrics on identical common keys
    key = ["Household_ID", "Timestamp"]
    common = frames[M_A][key + ["actual"]].copy()
    for m, f in frames.items():
        common = common.merge(f[key + ["prediction"]].rename(columns={"prediction": m}), on=key, how="inner")
    sm = pd.concat([baseline.predict_household(h, masked[h], dataclasses.replace(cfg, models=("seasonal_mean_4weeks",)), test_range)[0] for h in scored], ignore_index=True)
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
             "RMSE": ov.loc[m, "RMSE"], "sMAPE_%": ov.loc[m, "sMAPE_%"], "MAPE_%": ov.loc[m, "MAPE_%"], "household_daily_energy_MAE_kWh": ov.loc[m, "daily_energy_MAE_kWh"], "bias": ov.loc[m, "bias"],
             "portfolio_interval_MAE_kWh": pi["e"].abs().mean(), "portfolio_over_kWh": pi["e"].clip(lower=0).sum(), "portfolio_under_kWh": (-pi["e"]).clip(lower=0).sum(), "portfolio_day_MAE_kWh": pdl["e"].abs().mean(),
             "portfolio_day_bias_kWh": pdl["e"].mean(), "portfolio_day_bias_%": 100 * pdl["e"].sum() / pdl["actual"].sum(),
             "portfolio_days_under_%": 100 * (pdl["e"] < 0).mean()}
        for ratio in (1.0, 2.0):
            ee = pi["e"]; r[f"cost_per_kWh_ratio_{ratio:g}"] = (ratio * (-ee).clip(lower=0).sum() + ee.clip(lower=0).sum()) / pi["actual"].sum()
        rows.append(r)
    comp = pd.DataFrame(rows)
    ref = comp.loc[comp.model == M_A, "mean_household_MAE"].iloc[0]
    comp["mean_hh_MAE_vs_A_%"] = 100 * (comp["mean_household_MAE"] - ref) / ref
    a_row = comp[comp.model == M_A].iloc[0]
    for col in ("overall_MAE", "RMSE", "household_daily_energy_MAE_kWh", "portfolio_day_MAE_kWh", "portfolio_interval_MAE_kWh", "cost_per_kWh_ratio_1", "cost_per_kWh_ratio_2"):
        comp[f"{col}_vs_A_%"] = 100 * (comp[col] - a_row[col]) / a_row[col]
    hmc = hm.pivot(index="Household_ID", columns="model", values="MAE")
    for m in frames:
        if m != M_A:
            d = hmc[m] - hmc[M_A]
            comp.loc[comp.model == m, "households_better_than_A"] = int((d < 0).sum())
            comp.loc[comp.model == m, "households_worse_than_A"] = int((d > 0).sum())
            comp.loc[comp.model == m, "paired_diff_se"] = d.std(ddof=1) / np.sqrt(len(d))
    comp.to_csv(out / "metrics" / "comparison.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(comp.round(4).to_string(index=False)); print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
