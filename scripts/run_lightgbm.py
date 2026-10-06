"""Simple global LightGBM day-ahead model (direct strategy, origin D-1 11:45) vs the stored pooled AutoGluon reference.

    uv run python scripts/run_lightgbm.py [--dry_run] [--sample_frac 0.25] [--max_days N] [--out DIR]

Reference A = stored pooled AutoGluon predictions of the 12-week screening (not retrained). No AutoGluon is used here.
Models: lgbm (household id + calendar + target lags/rolling + historical temperature)  and  lgbm_pv (+ PV ownership flag).
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import Config
from src.covariates import CAL, TEMP, Covariates, load_station_temps
from src.data import load_all, split_dates
from src.lgbm_features import FEATURES_BASE, build_rows
from src.level1_pv import pv_status_table, to_interval_frame, origin_of
from src.screening_eval import evaluate

M_A = "pooled_baseline_stored"
PARAMS = dict(objective="l1", n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=0, verbose=-1)


def rows_for(hh, days, masked, raw, cov, station_of, code, pv_flag=None):
    r = build_rows(code[hh], masked[hh], raw[hh], days, cov, station_of[hh])
    if r is None:
        return None
    X, y, dates = r
    X = X.astype({c: "float32" for c in X.columns if c != "household"})
    if pv_flag is not None:
        X["pv_flag"] = np.float32(pv_flag[hh])
    return X, y, dates


def verify_cutoff_features(masked, raw, temps, station_of, code, ids, days, n_hh=8, n_days=4):
    """Overwrite EVERYTHING stamped after D-1 11:45 (targets of D-1 afternoon, all of D, weather) with 9999: features must not change."""
    rng = np.random.default_rng(0)
    real_cov = Covariates(temps, station_of, CAL + TEMP)
    picks = [days[i] for i in rng.choice(len(days), n_days, replace=False)]
    offs = pd.to_timedelta(np.arange(96) * 15, unit="min").to_numpy()
    checked = 0
    for D in picks:
        O = origin_of(D)
        bad_temps = {s: v.where(v.index <= O, 9999.0) for s, v in temps.items()}
        bad_cov = Covariates(bad_temps, station_of, CAL + TEMP)
        for hh in rng.choice(ids, n_hh, replace=False):
            def pert(mat):
                v = mat.to_numpy().copy(); t = mat.index.to_numpy()[:, None] + offs[None, :]
                v[(t > O.to_datetime64()) & np.isfinite(v)] = 9999.0
                return pd.DataFrame(v, index=mat.index)
            a = build_rows(code[hh], masked[hh], raw[hh], pd.DatetimeIndex([D]), real_cov, station_of[hh])
            b = build_rows(code[hh], pert(masked[hh]), pert(raw[hh]), pd.DatetimeIndex([D]), bad_cov, station_of[hh])
            if a is None or b is None:
                continue
            assert np.allclose(a[0].to_numpy(dtype="float64"), b[0].to_numpy(dtype="float64"), equal_nan=True), \
                f"features depend on post-cutoff data (household {hh}, day {D.date()}): {[c for c in a[0].columns if not np.allclose(a[0][c], b[0][c], equal_nan=True)]}"
            assert (a[0]["lag_1d"][a[0]["slot"] >= 48].isna()).all(), "lag_1d crosses the cutoff"
            assert (np.nan_to_num(a[0][TEMP].to_numpy(), nan=0) < 9000).all()
            checked += 1
    print(f"feature cutoff verification passed for {checked} household-days (all targets/weather stamped after D-1 11:45, incl. all of day D, "
          f"set to 9999 without changing any feature; lag_1d is NaN for slots after 11:45)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--sample_frac", type=float, default=0.25, help="random share of training rows (seed 0) to keep")
    ap.add_argument("--train_days", type=int, default=365)
    ap.add_argument("--val_days", type=int, default=28)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--max_days", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--screen_dir", type=Path, default=None)
    a = ap.parse_args()
    cfg = Config()
    out = a.out or cfg.output_dir / "lightgbm_screening"
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
    pool_ids = pv.index[pv.pv_status != "unknown"].tolist()
    assert len(scored) == 241 and len(pool_ids) == 245
    days = pd.DatetimeIndex(json.loads((screen / "selected_weeks.json").read_text())["forecast_days"])
    if a.max_days:
        days = days[: a.max_days]
    meta = pd.read_csv(cfg.data_dir.parent / "smart_meter_meta_data" / "households.csv", sep=";", dtype={"Household_ID": str})
    station_of = dict(zip(meta["Household_ID"], meta["Weather_ID"].astype(str)))
    temps = load_station_temps(cfg.data_dir.parent / "weather_data_hourly")
    cov = Covariates(temps, station_of, CAL + TEMP)
    code = {h: i for i, h in enumerate(sorted(masked))}
    pv_flag = {h: float(pv.loc[h, "pv_status"] == "pv") for h in pool_ids}
    fit_end = train_end - pd.Timedelta(days=a.val_days)
    train_days = pd.date_range(train_end - pd.Timedelta(days=a.train_days - 1), fit_end)
    val_days = pd.date_range(fit_end + pd.Timedelta(days=1), train_end)
    assert val_days.max() <= train_end < test_range[0] and days.min() >= test_range[0]
    config = {"lightgbm": PARAMS, "lightgbm_version": lgb.__version__, "sample_frac": a.sample_frac, "early_stopping_rounds": 50,
              "train_target_days": [str(train_days.min().date()), str(train_days.max().date())],
              "validation_target_days": [str(val_days.min().date()), str(val_days.max().date())],
              "test_days": len(days), "scored_households": len(scored), "training_households": len(pool_ids), "features": FEATURES_BASE,
              "strategy": "direct: one global model, rows = household x target day x slot, all features anchored at D-1 11:45"}
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    print("Config:", json.dumps({k: v for k, v in config.items() if k != "features"}, indent=2), f"\nfeatures ({len(FEATURES_BASE)}): {FEATURES_BASE}", flush=True)

    verify_cutoff_features(masked, raw, temps, station_of, code, scored, days)
    if a.dry_run:
        r = build_rows(code[scored[0]], masked[scored[0]], raw[scored[0]], days[:2], cov, station_of[scored[0]])
        print(r[0].iloc[[0, 50, 95]].T.to_string()); print("dry run finished"); return

    # ---- training rows (sampled) ------------------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    parts = {"fit": [], "val": []}
    t0 = time.time()
    for hh in pool_ids:
        for name, dd in (("fit", train_days), ("val", val_days)):
            r = rows_for(hh, dd, masked, raw, cov, station_of, code, pv_flag)
            if r is None:
                continue
            X, y, _ = r
            ok = np.isfinite(y)
            if name == "fit":
                ok &= rng.random(len(y)) < a.sample_frac
            parts[name].append((X[ok], y[ok]))
    Xf, yf = pd.concat([p[0] for p in parts["fit"]], ignore_index=True), np.concatenate([p[1] for p in parts["fit"]])
    Xv, yv = pd.concat([p[0] for p in parts["val"]], ignore_index=True), np.concatenate([p[1] for p in parts["val"]])
    print(f"training rows {len(Xf):,} (sample {a.sample_frac}), validation rows {len(Xv):,}, built in {time.time() - t0:.0f}s", flush=True)

    # ---- test rows ---------------------------------------------------------------------------------------------------
    test = {}
    for hh in scored:
        r = rows_for(hh, days, masked, raw, cov, station_of, code, pv_flag={**pv_flag, **{h: float(pv.loc[h, "pv_status"] == "pv") for h in scored}})
        if r is not None:
            test[hh] = r
    frames, info, imps = {}, {}, {}
    for name, drop in (("lgbm", ["pv_flag"]), ("lgbm_pv", [])):
        cols = [c for c in Xf.columns if c not in drop]
        t0 = time.time()
        model = lgb.LGBMRegressor(**PARAMS, n_jobs=a.n_jobs)
        model.fit(Xf[cols], yf, eval_set=[(Xv[cols], yv)], eval_metric="l1", categorical_feature=["household"],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        fit_s = time.time() - t0
        t0 = time.time()
        recs = []
        for hh, (X, y, dates) in test.items():
            p = np.clip(model.predict(X[cols]), 0, None)
            ts = pd.to_datetime(dates) + pd.to_timedelta(X["slot"].to_numpy() * 15, unit="min")
            recs.append(pd.DataFrame({"Household_ID": hh, "Timestamp": ts, "prediction": p, "forecast_date": pd.to_datetime(dates)}))
        pred_s = time.time() - t0
        pr = pd.concat(recs, ignore_index=True)
        pr["forecast_origin"] = pr["forecast_date"] - pd.Timedelta(hours=12, minutes=15)
        pr["pv_status"] = pr["Household_ID"].map(pv["pv_status"])
        frames[name] = to_interval_frame(pr, masked, name)
        frames[name].to_parquet(out / "predictions" / f"{name}_interval_predictions.parquet", index=False)
        imp = pd.Series(model.booster_.feature_importance("gain"), index=cols); imps[name] = (imp / imp.sum()).sort_values(ascending=False)
        imps[name].to_csv(out / "metrics" / f"feature_importance_{name}.csv", header=["gain_share"])
        model.booster_.save_model(str(out / f"{name}_model.txt"))
        info[name] = {"fit_seconds": round(fit_s, 1), "predict_seconds": round(pred_s, 1), "best_iteration": int(model.best_iteration_ or PARAMS["n_estimators"]),
                      "val_MAE": float(model.best_score_["valid_0"]["l1"])}
        print(name, info[name], flush=True)
    (out / "metrics" / "run_info.json").write_text(json.dumps(info, indent=2))

    # ---- evaluation vs stored pooled AutoGluon (A) ---------------------------------------------------------------------
    A = pd.read_parquet(screen / "predictions" / "pooled_interval_predictions.parquet"); A["model"] = M_A
    allf = {M_A: A, **frames}
    wc = cfg.output_dir / "level2_weather_calendar_screening" / "predictions" / "both_interval_predictions.parquet"
    if wc.exists():
        w = pd.read_parquet(wc); w["model"] = "ag_pooled_weather_calendar"; allf["ag_pooled_weather_calendar"] = w
    comp = evaluate(allf, M_A, masked, scored, cfg, days, test_range, out)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(comp.round(4).T.to_string()); print({k: v.head(8).round(3).to_dict() for k, v in imps.items()})


if __name__ == "__main__":
    main()
