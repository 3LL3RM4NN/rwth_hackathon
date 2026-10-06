"""LightGBM objective screening for portfolio procurement (features, cutoff, split, sample, seed identical to run_lightgbm.py).

    uv run python scripts/run_lightgbm_objectives.py [--sample_frac 0.25] [--out DIR]

Models: l1 (MAE, current), l2 (mean regression), quantile 0.60, quantile 0.75.  Early stopping on the validation loss of
each model's own objective. Primary evaluation: portfolio-day totals; household metrics kept. NOTHING is tuned on the test weeks.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_lightgbm import M_A, PARAMS, rows_for, verify_cutoff_features
from src.config import Config
from src.covariates import CAL, TEMP, Covariates, load_station_temps
from src.data import load_all, split_dates
from src.level1_pv import pv_status_table, to_interval_frame
from src.screening_eval import evaluate

OBJECTIVES = {"lgbm_l1": ({"objective": "l1"}, "l1"), "lgbm_l2": ({"objective": "regression"}, "l2"),
              "lgbm_q60": ({"objective": "quantile", "alpha": 0.60}, "quantile"),
              "lgbm_q75": ({"objective": "quantile", "alpha": 0.75}, "quantile")}
RATIOS = [1.0, 1.5, 2.0, 3.0]


def portfolio_tables(frames: dict, seasonal: pd.DataFrame, models: list[str]):
    """Common-key frame (all models + seasonal mean), complete household-days, then daily portfolio totals."""
    key = ["Household_ID", "Timestamp"]
    common = frames[models[0]][key + ["actual"]].copy()
    for m in models:
        common = common.merge(frames[m][key + ["prediction"]].rename(columns={"prediction": m}), on=key, how="inner")
    common = common.merge(seasonal[key + ["prediction"]].rename(columns={"prediction": "seasonal_mean_4weeks"}), on=key, how="inner")
    day = common["Timestamp"].dt.normalize()
    n = common.groupby(["Household_ID", day])["actual"].transform("size")
    common = common[n == 96].copy(); common["date"] = common["Timestamp"].dt.normalize()
    return common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_frac", type=float, default=0.25)
    ap.add_argument("--train_days", type=int, default=365)
    ap.add_argument("--val_days", type=int, default=28)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--screen_dir", type=Path, default=None)
    a = ap.parse_args()
    cfg = Config()
    out = a.out or cfg.output_dir / "lightgbm_objective_screening"
    screen = a.screen_dir or cfg.output_dir / "level1_fast_screening"
    for sub in ("predictions", "metrics", "models"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    masked = {hh: m for hh, m, _ in load_all(cfg)}
    raw = {hh: m for hh, m, _ in load_all(dataclasses.replace(cfg, min_day_completeness=0.0))}
    first = min(m.index.min() for m in masked.values()); last = max(m.index.max() for m in masked.values())
    test_range = split_dates(first, last, cfg)["test"]; train_end = test_range[0] - pd.Timedelta(days=1)
    pv = pv_status_table(cfg).set_index("Household_ID").loc[list(masked)]
    has_test = pd.Series({h: bool(((m.index >= test_range[0]) & np.isfinite(m.to_numpy()).any(axis=1)).any()) for h, m in masked.items()})
    scored = pv.index[(pv.pv_status != "unknown") & has_test].tolist()
    pool_ids = pv.index[pv.pv_status != "unknown"].tolist()
    assert len(scored) == 241 and len(pool_ids) == 245
    days = pd.DatetimeIndex(json.loads((screen / "selected_weeks.json").read_text())["forecast_days"])
    assert len(days) == 84
    meta = pd.read_csv(cfg.data_dir.parent / "smart_meter_meta_data" / "households.csv", sep=";", dtype={"Household_ID": str})
    station_of = dict(zip(meta["Household_ID"], meta["Weather_ID"].astype(str)))
    temps = load_station_temps(cfg.data_dir.parent / "weather_data_hourly")
    cov = Covariates(temps, station_of, CAL + TEMP)
    code = {h: i for i, h in enumerate(sorted(masked))}
    fit_end = train_end - pd.Timedelta(days=a.val_days)
    train_days = pd.date_range(train_end - pd.Timedelta(days=a.train_days - 1), fit_end)
    val_days = pd.date_range(fit_end + pd.Timedelta(days=1), train_end)
    assert val_days.max() <= train_end < test_range[0] and days.min() >= test_range[0]
    print("Config:", json.dumps({"base_params": PARAMS, "objectives": {k: v[0] for k, v in OBJECTIVES.items()}, "sample_frac": a.sample_frac,
          "train_target_days": [str(train_days.min().date()), str(train_days.max().date())], "validation": [str(val_days.min().date()), str(val_days.max().date())],
          "households_scored": len(scored), "forecast_days": len(days)}, indent=2), flush=True)
    verify_cutoff_features(masked, raw, temps, station_of, code, scored, days)       # same feature code, same test

    # ---- identical training / validation / test matrices for all objectives (no pv_flag: model 'lgbm') -----------------------
    rng = np.random.default_rng(0)
    parts = {"fit": [], "val": []}
    for hh in pool_ids:
        for name, dd in (("fit", train_days), ("val", val_days)):
            r = rows_for(hh, dd, masked, raw, cov, station_of, code, None)
            if r is None:
                continue
            X, y, _ = r
            ok = np.isfinite(y)
            if name == "fit":
                ok &= rng.random(len(y)) < a.sample_frac
            parts[name].append((X[ok], y[ok]))
    Xf, yf = pd.concat([p[0] for p in parts["fit"]], ignore_index=True), np.concatenate([p[1] for p in parts["fit"]])
    Xv, yv = pd.concat([p[0] for p in parts["val"]], ignore_index=True), np.concatenate([p[1] for p in parts["val"]])
    test = {hh: r for hh in scored if (r := rows_for(hh, days, masked, raw, cov, station_of, code, None)) is not None}
    print(f"training rows {len(Xf):,}, validation rows {len(Xv):,}, features {Xf.shape[1]}", flush=True)

    frames, info, imps = {}, {}, {}
    for name, (obj, metric) in OBJECTIVES.items():
        params = {**PARAMS, **obj}
        t0 = time.time()
        model = lgb.LGBMRegressor(**params, n_jobs=a.n_jobs)
        model.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric=metric, categorical_feature=["household"],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        fit_s = time.time() - t0
        recs = []
        for hh, (X, y, dates) in test.items():
            p = np.clip(model.predict(X), 0, None)
            ts = pd.to_datetime(dates) + pd.to_timedelta(X["slot"].to_numpy() * 15, unit="min")
            recs.append(pd.DataFrame({"Household_ID": hh, "Timestamp": ts, "prediction": p, "forecast_date": pd.to_datetime(dates)}))
        pr = pd.concat(recs, ignore_index=True)
        pr["forecast_origin"] = pr["forecast_date"] - pd.Timedelta(hours=12, minutes=15)
        pr["pv_status"] = pr["Household_ID"].map(pv["pv_status"])
        frames[name] = to_interval_frame(pr, masked, name)
        frames[name].to_parquet(out / "predictions" / f"{name}_interval_predictions.parquet", index=False)
        imp = pd.Series(model.booster_.feature_importance("gain"), index=Xf.columns); imps[name] = (imp / imp.sum()).sort_values(ascending=False)
        imps[name].to_csv(out / "metrics" / f"feature_importance_{name}.csv", header=["gain_share"])
        model.booster_.save_model(str(out / "models" / f"{name}.txt"))
        info[name] = {"objective": obj, "validation_metric": metric, "fit_seconds": round(fit_s, 1), "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
                      "n_estimators_cap": params["n_estimators"], "val_loss": float(list(model.best_score_["valid_0"].values())[0])}
        print(name, info[name], flush=True)
    # reproducibility: l1 here must equal the earlier LightGBM screening model
    prev = cfg.output_dir / "lightgbm_screening" / "predictions" / "lgbm_interval_predictions.parquet"
    if prev.exists():
        pv_ = pd.read_parquet(prev).set_index(["Household_ID", "Timestamp"])["prediction"]
        cur = frames["lgbm_l1"].set_index(["Household_ID", "Timestamp"])["prediction"].reindex(pv_.index)
        info["lgbm_l1"]["max_abs_diff_vs_previous_lgbm_screening"] = float(np.nanmax(np.abs(cur.to_numpy() - pv_.to_numpy())))
        print("l1 reproduces previous screening, max abs diff:", info["lgbm_l1"]["max_abs_diff_vs_previous_lgbm_screening"])
    (out / "metrics" / "run_info.json").write_text(json.dumps(info, indent=2, default=str))

    # ---- household-level evaluation (shared module), ref = current MAE model ---------------------------------------------------
    A = pd.read_parquet(screen / "predictions" / "pooled_interval_predictions.parquet"); A["model"] = M_A
    allf = {"lgbm_l1": frames["lgbm_l1"], "lgbm_l2": frames["lgbm_l2"], "lgbm_q60": frames["lgbm_q60"], "lgbm_q75": frames["lgbm_q75"], M_A: A}
    comp = evaluate(allf, "lgbm_l1", masked, scored, cfg, days, test_range, out)

    # ---- portfolio-day evaluation ---------------------------------------------------------------------------------------------------
    from src import baseline
    bcfg = dataclasses.replace(cfg, models=("seasonal_mean_4weeks",))
    seasonal = pd.concat([baseline.predict_household(h, masked[h], bcfg, test_range)[0] for h in scored], ignore_index=True)
    models = list(allf)
    common = portfolio_tables(allf, seasonal, models)
    allm = models + ["seasonal_mean_4weeks"]
    daily = []
    for m in allm:
        g = common.groupby("date").agg(actual=("actual", "sum"), prediction=(m, "sum"), n_households=("Household_ID", "nunique")).reset_index()
        g.insert(0, "model", m); daily.append(g)
    daily = pd.concat(daily, ignore_index=True)
    daily["error"] = daily["prediction"] - daily["actual"]
    daily.to_csv(out / "metrics" / "portfolio_daily.csv", index=False)
    rows = []
    for m in allm:
        d = daily[daily.model == m]; e = d["error"].to_numpy(); ac = d["actual"].to_numpy()
        under, over = np.maximum(-e, 0), np.maximum(e, 0)
        ci = common.assign(e=common[m] - common["actual"])
        r = {"model": m, "portfolio_days": len(d), "portfolio_day_MAPE_%": 100 * np.mean(np.abs(e) / ac), "portfolio_day_MAE_kWh": np.abs(e).mean(), "portfolio_day_RMSE_kWh": np.sqrt((e ** 2).mean()),
             "portfolio_bias_kWh_per_day": e.mean(), "portfolio_bias_%": 100 * e.sum() / ac.sum(), "underforecast_MWh": under.sum() / 1e3,
             "overforecast_MWh": over.sum() / 1e3, "net_MWh(over-under)": e.sum() / 1e3, "pct_days_under_%": 100 * (e < 0).mean(),
             "coverage_portfolio_day_actual<=pred_%": 100 * (ac <= d["prediction"].to_numpy()).mean(),
             "coverage_household_interval_actual<=pred_%": 100 * (common["actual"] <= common[m]).mean()}
        for ratio in RATIOS:
            c = ratio * under + over
            r[f"cost_{ratio:g}:1_total_MWh_eq"] = c.sum() / 1e3; r[f"cost_{ratio:g}:1_per_day_kWh_eq"] = c.mean()
        rows.append(r)
    port = pd.DataFrame(rows)
    for col in ("portfolio_day_MAE_kWh", "cost_1:1_total_MWh_eq", "cost_3:1_total_MWh_eq"):
        port[f"{col}_vs_l1_%"] = 100 * (port[col] - port.loc[port.model == "lgbm_l1", col].iloc[0]) / port.loc[port.model == "lgbm_l1", col].iloc[0]
    port.to_csv(out / "metrics" / "portfolio_summary.csv", index=False)
    hh_cols = ["model", "households", "scored_intervals", "mean_household_MAE", "median_household_MAE", "overall_MAE", "RMSE", "MAPE_%", "sMAPE_%",
               "household_daily_energy_MAE_kWh", "bias", "portfolio_interval_MAE_kWh", "portfolio_over_kWh", "portfolio_under_kWh",
               "households_better_than_ref", "households_worse_than_ref"]
    hh = comp[hh_cols].copy(); hh.to_csv(out / "metrics" / "household_summary.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 50)
    print(port.set_index("model").T.round(3).to_string()); print(hh.set_index("model").T.round(4).to_string())
    print({k: v.head(6).round(3).to_dict() for k, v in imps.items()})


if __name__ == "__main__":
    main()
