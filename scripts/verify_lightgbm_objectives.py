"""Independent verification of the LightGBM L1 vs L2 (objective) result.

    uv run python scripts/verify_lightgbm_objectives.py

Rebuilds matrices TWICE (one per model) with the production feature code, proves they are bitwise identical, trains L1 and L2
from scratch, then evaluates with code that does NOT use src.evaluation / screening_eval: actuals come straight from the raw CSVs,
all metrics are recomputed from first principles. A sample of features is additionally recomputed with plain pandas from the raw
CSVs (no cumsum windows) and the leakage perturbation test is run per model. Finally everything is compared with
outputs/lightgbm_objective_screening/ (not modified).
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

from run_lightgbm import PARAMS, rows_for
from src.config import Config
from src.covariates import CAL, TEMP, Covariates, load_station_temps
from src.data import load_all, split_dates
from src.lgbm_features import FEATURES_BASE, build_rows
from src.level1_pv import origin_of, pv_status_table

OBJ = {"L1": ({"objective": "l1"}, "l1"), "L2": ({"objective": "regression"}, "l2")}
OFFS = pd.to_timedelta(np.arange(96) * 15, unit="min").to_numpy()


def build_matrices(cfg, masked, raw, pool_ids, scored, days, train_days, val_days, cov, station_of, code, frac):
    """Exactly the matrix construction of run_lightgbm_objectives.py (same row order, same rng stream)."""
    rng = np.random.default_rng(0)
    parts = {"fit": [], "val": []}
    for hh in pool_ids:
        for name, dd in (("fit", train_days), ("val", val_days)):
            r = rows_for(hh, dd, masked, raw, cov, station_of, code, None)
            if r is None:
                continue
            X, y, dates = r
            ok = np.isfinite(y)
            if name == "fit":
                ok &= rng.random(len(y)) < frac
            parts[name].append((X[ok], y[ok]))
    Xf, yf = pd.concat([p[0] for p in parts["fit"]], ignore_index=True), np.concatenate([p[1] for p in parts["fit"]])
    Xv, yv = pd.concat([p[0] for p in parts["val"]], ignore_index=True), np.concatenate([p[1] for p in parts["val"]])
    test = {hh: r for hh in scored if (r := rows_for(hh, days, masked, raw, cov, station_of, code, None)) is not None}
    return Xf, yf, Xv, yv, test


def same_bits(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return list(a.columns) == list(b.columns) and list(a.dtypes) == list(b.dtypes) and a.shape == b.shape and \
        all(a[c].to_numpy().tobytes() == b[c].to_numpy().tobytes() for c in a.columns)


def raw_series(path, target):
    d = pd.read_csv(path, sep=";", usecols=["Timestamp", target])
    s = pd.Series(d[target].to_numpy(dtype="float64"), index=pd.to_datetime(d["Timestamp"], utc=True).dt.tz_localize(None)).sort_index()
    grid = pd.date_range(s.index.min().normalize(), s.index.max().normalize() + pd.Timedelta(hours=23, minutes=45), freq="15min")
    s = s.reindex(grid)
    cnt = s.notna().groupby(s.index.normalize()).transform("sum")
    return s, s.where(cnt == 96)                       # raw, masked (days with < 96 observed intervals -> NaN)


def independent_feature_check(samples, cfg, temps, station_of, X_by_hh):
    """Recompute features from raw CSVs with plain pandas and compare with the production matrices."""
    worst, n = 0.0, 0
    for hh, D, slot in samples:
        S_raw, S_m = raw_series(cfg.data_dir / f"{hh}.csv", cfg.target)
        O = origin_of(D); d1 = D - pd.Timedelta(days=1)
        C = S_m.copy(); mm = C.index >= d1; C[mm] = S_raw[mm]         # raw for D-1 (morning is all that is used), masked before
        t = D + pd.Timedelta(minutes=15 * slot)
        exp = {"y_last": C.get(O), "y_lag1h": C.get(O - pd.Timedelta(hours=1)),
               "lag_2d": S_m.get(t - pd.Timedelta(days=2)), "lag_7d": S_m.get(t - pd.Timedelta(days=7)),
               "lag_1d": S_raw.get(t - pd.Timedelta(days=1)) if slot <= 47 else np.nan,
               "mean_same_slot_4wk": np.nanmean([S_m.get(t - pd.Timedelta(days=7 * k), np.nan) for k in range(1, 5)])
               if np.isfinite([S_m.get(t - pd.Timedelta(days=7 * k), np.nan) for k in range(1, 5)]).any() else np.nan}
        for name, w in (("roll_mean_1h", 4), ("roll_mean_6h", 24), ("roll_mean_24h", 96), ("roll_mean_7d", 672)):
            win = C[O - pd.Timedelta(minutes=15 * (w - 1)): O]
            exp[name] = win.mean() if win.notna().any() else np.nan
        win = C[O - pd.Timedelta(minutes=15 * 95): O]
        exp["roll_std_24h"] = win.std(ddof=0) if win.notna().sum() > 1 else np.nan
        T = temps[station_of[hh]].ffill(); a = d1 + pd.Timedelta(hours=11)
        exp.update({"temp_latest": T[a], "temp_lag24h": T[a - pd.Timedelta(hours=24)], "temp_lag7d": T[a - pd.Timedelta(hours=168)],
                    "temp_mean24h": T[a - pd.Timedelta(hours=23): a].mean(), "temp_chg6h": T[a] - T[a - pd.Timedelta(hours=6)]})
        X, dates = X_by_hh[hh]
        row = X[(dates == D.to_datetime64()) & (X["slot"].to_numpy() == slot)].iloc[0]
        for k, v in exp.items():
            got = float(row[k]); v = np.nan if v is None else float(v)
            if np.isnan(v) and np.isnan(got):
                continue
            assert np.isfinite(v) and np.isfinite(got) and abs(v - got) <= 1e-4 * max(1.0, abs(v)), f"{k}: independent {v} vs production {got} ({hh}, {D.date()}, slot {slot})"
            worst = max(worst, abs(v - got))
        n += 1
    return n, worst


def leakage_checks(label, samples_hd, masked, raw, temps, station_of, code, cov, models=None, test=None):
    """(1) everything stamped after D-1 11:45 -> 9999 ; (2) only the target day D -> 9999. Features of day D must not change."""
    for hh, D in samples_hd:
        O = origin_of(D)
        for mode in ("after_cutoff", "target_day_only"):
            def pert(mat):
                v = mat.to_numpy().copy(); t = mat.index.to_numpy()[:, None] + OFFS[None, :]
                m = (t > O.to_datetime64()) if mode == "after_cutoff" else ((t >= D.to_datetime64()) & (t < (D + pd.Timedelta(days=1)).to_datetime64()))
                v[m & np.isfinite(v)] = 9999.0
                return pd.DataFrame(v, index=mat.index)
            bad_t = {s: v.where(v.index <= O, 9999.0) for s, v in temps.items()} if mode == "after_cutoff" else temps
            bcov = Covariates(bad_t, station_of, CAL + TEMP) if mode == "after_cutoff" else cov
            a = build_rows(code[hh], masked[hh], raw[hh], pd.DatetimeIndex([D]), cov, station_of[hh])
            b = build_rows(code[hh], pert(masked[hh]), pert(raw[hh]), pd.DatetimeIndex([D]), bcov, station_of[hh])
            assert a is not None and b is not None
            assert same_bits(a[0].astype("float64"), b[0].astype("float64")) or np.allclose(a[0].to_numpy("float64"), b[0].to_numpy("float64"), equal_nan=True), \
                f"[{label}] features changed ({mode}) for {hh} {D.date()}"
            assert (a[0]["lag_1d"][a[0]["slot"] >= 48].isna()).all()
            if models is not None:
                Xa = a[0].astype({c: "float32" for c in a[0].columns if c != "household"}); Xb = b[0].astype({c: "float32" for c in b[0].columns if c != "household"})
                assert np.allclose(models[label].predict(Xa), models[label].predict(Xb)), f"[{label}] predictions changed ({mode})"
    return len(samples_hd)


def pm(a, p, actual_total=None):
    e = p - a
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "bias_kWh": e.mean(), "bias_%": 100 * e.sum() / a.sum(),
            "MAPE_%": 100 * np.mean(np.abs(e) / a), "under_MWh": np.maximum(-e, 0).sum() / 1e3, "over_MWh": np.maximum(e, 0).sum() / 1e3,
            "pct_days_under": 100 * (e < 0).mean(), "n_days": len(a)}


def compute_metrics(D_):
    verified = {}
    for lab, col in (("L1", "prediction_l1"), ("L2", "prediction_l2")):
        pdl = D_.groupby("date").agg(a=("actual", "sum"), p=(col, "sum"))
        hdl = D_.groupby(["Household_ID", "date"]).agg(a=("actual", "sum"), p=(col, "sum"))
        err = D_[col] - D_["actual"]
        hh_mae = err.abs().groupby(D_["Household_ID"]).mean()
        v = {"scored_intervals": len(D_), "overall_MAE": err.abs().mean(), "household_MAE_mean": hh_mae.mean(), "household_MAE_median": hh_mae.median(), "RMSE": np.sqrt((err ** 2).mean()),
             "daily_energy_MAE": (hdl.p - hdl.a).abs().mean(), **{f"portfolio_{k}": x for k, x in pm(pdl.a.to_numpy(), pdl.p.to_numpy()).items()}}
        for ratio in (1.0, 1.5, 2.0, 3.0):
            e = pdl.p - pdl.a; v[f"cost_{ratio:g}:1_MWh_eq"] = (ratio * np.maximum(-e, 0) + np.maximum(e, 0)).sum() / 1e3
        verified[lab] = v
    return verified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n_check", type=int, default=24)
    a = ap.parse_args()
    cfg = Config()
    out = a.out or cfg.output_dir / "lightgbm_objective_verification"
    ex = cfg.output_dir / "lightgbm_objective_screening"
    for sub in ("metrics", "predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    masked = {hh: m for hh, m, _ in load_all(cfg)}
    raw = {hh: m for hh, m, _ in load_all(dataclasses.replace(cfg, min_day_completeness=0.0))}
    first = min(m.index.min() for m in masked.values()); last = max(m.index.max() for m in masked.values())
    test_range = split_dates(first, last, cfg)["test"]; train_end = test_range[0] - pd.Timedelta(days=1)
    pv = pv_status_table(cfg).set_index("Household_ID").loc[list(masked)]
    has_test = pd.Series({h: bool(((m.index >= test_range[0]) & np.isfinite(m.to_numpy()).any(axis=1)).any()) for h, m in masked.items()})
    scored = pv.index[(pv.pv_status != "unknown") & has_test].tolist(); pool_ids = pv.index[pv.pv_status != "unknown"].tolist()
    days = pd.DatetimeIndex(json.loads((cfg.output_dir / "level1_fast_screening" / "selected_weeks.json").read_text())["forecast_days"])
    assert len(scored) == 241 and len(pool_ids) == 245 and len(days) == 84
    meta = pd.read_csv(cfg.data_dir.parent / "smart_meter_meta_data" / "households.csv", sep=";", dtype={"Household_ID": str})
    station_of = dict(zip(meta["Household_ID"], meta["Weather_ID"].astype(str)))
    temps = load_station_temps(cfg.data_dir.parent / "weather_data_hourly")
    cov = Covariates(temps, station_of, CAL + TEMP)
    code = {h: i for i, h in enumerate(sorted(masked))}; inv = {v: k for k, v in code.items()}
    fit_end = train_end - pd.Timedelta(days=28)
    train_days = pd.date_range(train_end - pd.Timedelta(days=364), fit_end); val_days = pd.date_range(fit_end + pd.Timedelta(days=1), train_end)
    assert val_days.max() <= train_end < test_range[0] and days.min() >= test_range[0]
    res = {}

    # ---- 1. two independent matrix builds must be bitwise identical -------------------------------------------------------------
    M = {k: build_matrices(cfg, masked, raw, pool_ids, scored, days, train_days, val_days, cov, station_of, code, 0.25) for k in ("L1", "L2")}
    (Xf1, yf1, Xv1, yv1, te1), (Xf2, yf2, Xv2, yv2, te2) = M["L1"], M["L2"]
    ident = {"Xfit": same_bits(Xf1, Xf2), "yfit": yf1.tobytes() == yf2.tobytes(), "Xval": same_bits(Xv1, Xv2), "yval": yv1.tobytes() == yv2.tobytes(),
             "test_households": list(te1) == list(te2), "Xtest": all(same_bits(te1[h][0], te2[h][0]) and te1[h][1].tobytes() == te2[h][1].tobytes() for h in te1)}
    assert all(ident.values()), ident
    res["matrix_identity"] = ident
    res["data"] = {"training_rows": len(Xf1), "validation_rows": len(Xv1), "n_features": Xf1.shape[1], "features": list(Xf1.columns),
                   "training_households": len(pool_ids), "scored_households": len(scored), "forecast_days": len(days), "seed": PARAMS["random_state"],
                   "train_target_days": [str(train_days.min().date()), str(train_days.max().date())], "validation_target_days": [str(val_days.min().date()), str(val_days.max().date())]}
    print("matrices bitwise identical:", ident, "\n", {k: v for k, v in res["data"].items() if k != "features"}, flush=True)

    # ---- 2. independent feature recomputation + leakage checks -------------------------------------------------------------------------------
    rng = np.random.default_rng(42)
    samp_hd = [(scored[i], days[j]) for i, j in zip(rng.choice(len(scored), a.n_check, replace=False), rng.choice(len(days), a.n_check))]
    X_by_hh = {h: (te1[h][0], te1[h][2]) for h in te1}
    samples = [(h, D, int(s)) for (h, D) in samp_hd for s in rng.choice(96, 3, replace=False) if h in te1]
    n_ind, worst = independent_feature_check(samples, cfg, temps, station_of, X_by_hh)
    res["independent_feature_recompute"] = {"checked_feature_rows": n_ind, "max_abs_diff": worst, "features_checked": ["y_last", "y_lag1h", "lag_1d", "lag_2d", "lag_7d", "mean_same_slot_4wk",
                                            "roll_mean_1h/6h/24h/7d", "roll_std_24h", "temp_latest", "temp_lag24h", "temp_lag7d", "temp_mean24h", "temp_chg6h"]}
    print("independent feature recompute OK:", res["independent_feature_recompute"]["checked_feature_rows"], "rows, max diff", worst, flush=True)

    # ---- 3. train both models from scratch -----------------------------------------------------------------------------------------------------
    models, info = {}, {}
    for label, (obj, metric) in OBJ.items():
        Xf, yf, Xv, yv, _ = M[label]
        t0 = time.time()
        mdl = lgb.LGBMRegressor(**{**PARAMS, **obj}, n_jobs=8)
        mdl.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric=metric, categorical_feature=["household"], callbacks=[lgb.early_stopping(50, verbose=False)])
        models[label] = mdl
        info[label] = {"objective": obj["objective"], "best_iteration": int(mdl.best_iteration_ or 500), "n_estimators_cap": 500, "val_loss": float(list(mdl.best_score_["valid_0"].values())[0]),
                       "fit_seconds": round(time.time() - t0, 1)}
        print(label, info[label], flush=True)
    res["models"] = info
    res["leakage"] = {lab: {"household_days_checked": leakage_checks(lab, samp_hd, masked, raw, temps, station_of, code, cov, models=models),
                            "modes": ["all targets+weather after D-1 11:45 -> 9999", "only target day D -> 9999"], "result": "features and predictions unchanged"} for lab in ("L1", "L2")}
    print("leakage checks passed:", res["leakage"], flush=True)

    # ---- 4. predictions -------------------------------------------------------------------------------------------------------------------------------
    recs = []
    for hh, (X, y, dates) in te1.items():
        r = pd.DataFrame({"Household_ID": hh, "forecast_date": pd.to_datetime(dates), "slot": X["slot"].to_numpy(),
                          "prediction_l1": np.clip(models["L1"].predict(X), 0, None), "prediction_l2": np.clip(models["L2"].predict(X), 0, None)})
        recs.append(r)
    P = pd.concat(recs, ignore_index=True)
    P["Timestamp"] = P["forecast_date"] + pd.to_timedelta(P["slot"] * 15, unit="min")

    # independent actuals straight from the raw CSV files
    acts = []
    for hh in scored:
        s, sm = raw_series(cfg.data_dir / f"{hh}.csv", cfg.target)
        s = sm[(sm.index >= days.min()) & (sm.index < days.max() + pd.Timedelta(days=1)) & sm.index.normalize().isin(days)].dropna()
        acts.append(pd.DataFrame({"Household_ID": hh, "Timestamp": s.index, "actual": s.to_numpy()}))
    Act = pd.concat(acts, ignore_index=True)
    D_ = P.merge(Act, on=["Household_ID", "Timestamp"], how="inner")
    D_["date"] = D_["Timestamp"].dt.normalize()
    n96 = D_.groupby(["Household_ID", "date"])["actual"].transform("size"); D_ = D_[n96 == 96].copy()
    res["evaluation_set"] = {"scored_intervals": len(D_), "household_days": int(D_.groupby(["Household_ID", "date"]).ngroups), "households": int(D_["Household_ID"].nunique())}
    D_.to_parquet(out / "predictions" / "verified_predictions.parquet", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

    # sample CSV of whole household-days
    key_s = pd.DataFrame(samp_hd, columns=["Household_ID", "date"]).drop_duplicates()
    sample = D_.merge(key_s, on=["Household_ID", "date"])[["Household_ID", "forecast_date", "slot", "actual", "prediction_l1", "prediction_l2"]].copy()
    sample["l1_minus_l2"] = sample["prediction_l1"] - sample["prediction_l2"]
    sample.to_csv(out / "metrics" / "sample_household_days_predictions.csv", index=False)
    pstats = pd.DataFrame({c: {"mean": D_[c].mean(), "median": D_[c].median(), "min": D_[c].min(), "max": D_[c].max(), "n_zero": int((D_[c] == 0).sum()),
                               "total_MWh": D_[c].sum() / 1e3} for c in ("actual", "prediction_l1", "prediction_l2")}).T
    pstats.to_csv(out / "metrics" / "prediction_sanity.csv"); print(pstats.round(4).to_string(), flush=True)

    # ---- 5. distribution check (validation + training rows + test rows) ---------------------------------------------------------------------------------
    dist = {}
    for name, X, y in (("validation", Xv1, yv1), ("training_sample", Xf1, yf1)):
        p1, p2 = np.clip(models["L1"].predict(X), 0, None), np.clip(models["L2"].predict(X), 0, None)
        dist[name] = {"rows": len(y), "actual_mean": y.mean(), "actual_median": np.median(y), "L1_mean": p1.mean(), "L1_median": np.median(p1),
                      "L2_mean": p2.mean(), "L2_median": np.median(p2), "actual_share_zero_%": 100 * (y == 0).mean()}
        if name == "validation":
            hhm = pd.DataFrame({"hh": X["household"].map(inv).to_numpy(), "y": y, "p1": p1, "p2": p2})
            lv = hhm.groupby("hh")["y"].mean().sort_values()
            pick = [lv.index[int(q * (len(lv) - 1))] for q in (0.05, 0.25, 0.5, 0.75, 0.95)] + [lv.index[-1]]
            rep = hhm[hhm.hh.isin(pick)].groupby("hh").agg(actual_mean=("y", "mean"), actual_median=("y", "median"), L1_mean=("p1", "mean"), L1_median=("p1", "median"),
                                                           L2_mean=("p2", "mean"), L2_median=("p2", "median"), rows=("y", "size")).loc[pick]
            rep.to_csv(out / "metrics" / "distribution_representative_households_validation.csv")
    dist["test"] = {"rows": len(D_), "actual_mean": D_.actual.mean(), "actual_median": D_.actual.median(), "L1_mean": D_.prediction_l1.mean(), "L1_median": D_.prediction_l1.median(),
                    "L2_mean": D_.prediction_l2.mean(), "L2_median": D_.prediction_l2.median(), "actual_share_zero_%": 100 * (D_.actual == 0).mean()}
    dd = pd.DataFrame(dist).T; dd.to_csv(out / "metrics" / "distribution_check.csv"); print(dd.round(4).to_string()); print(rep.round(4).to_string(), flush=True)

    # ---- 6. independent metrics --------------------------------------------------------------------------------------------------------------------------------
    verified_full = compute_metrics(D_)
    # like-for-like set: the existing screening scores only intervals that the seasonal baseline AND the stored pooled AutoGluon also cover
    from src import baseline
    bcfg = dataclasses.replace(cfg, models=("seasonal_mean_4weeks",))
    sm = pd.concat([baseline.predict_household(h, masked[h], bcfg, test_range)[0] for h in scored], ignore_index=True)
    sm = sm[sm["Timestamp"].dt.normalize().isin(days)][["Household_ID", "Timestamp"]]
    ag = pd.read_parquet(cfg.output_dir / "level1_fast_screening" / "predictions" / "pooled_interval_predictions.parquet")[["Household_ID", "Timestamp"]]
    keys = sm.merge(ag, on=["Household_ID", "Timestamp"])
    D_c = D_.merge(keys, on=["Household_ID", "Timestamp"])
    nn = D_c.groupby(["Household_ID", "date"])["actual"].transform("size"); D_c = D_c[nn == 96].copy()
    extra = D_.merge(D_c[["Household_ID", "Timestamp"]], on=["Household_ID", "Timestamp"], how="left", indicator=True).query("_merge == 'left_only'")
    res["existing_common_set"] = {"intervals": len(D_c), "household_days": int(D_c.groupby(["Household_ID", "date"]).ngroups), "extra_intervals_in_full_set": len(extra),
                                  "extra_household_days": int(extra.groupby(["Household_ID", "date"]).ngroups),
                                  "reason": "household-days without a seasonal_mean_4weeks forecast (<2 same-weekday days in the last 4 weeks) or without stored AutoGluon forecast"}
    verified = compute_metrics(D_c)
    pd.DataFrame(verified_full).T.to_csv(out / "metrics" / "verified_metrics_full_evaluation_set.csv")
    print("evaluation sets:", res["evaluation_set"], res["existing_common_set"], flush=True)
    vdf = pd.DataFrame(verified).T; vdf.to_csv(out / "metrics" / "verified_metrics.csv"); print(vdf.T.round(4).to_string(), flush=True)

    # ---- 7. compare with existing outputs --------------------------------------------------------------------------------------------------------------------------
    ps = pd.read_csv(ex / "metrics" / "portfolio_summary.csv").set_index("model"); hs = pd.read_csv(ex / "metrics" / "household_summary.csv").set_index("model")
    mapping = [("Overall interval MAE", "overall_MAE", hs, "overall_MAE"), ("Mean household MAE", "household_MAE_mean", hs, "mean_household_MAE"),
               ("Median household MAE", "household_MAE_median", hs, "median_household_MAE"), ("Interval RMSE", "RMSE", hs, "RMSE"),
               ("Household daily-energy MAE", "daily_energy_MAE", hs, "household_daily_energy_MAE_kWh"), ("Portfolio-day MAE", "portfolio_MAE", ps, "portfolio_day_MAE_kWh"),
               ("Portfolio-day RMSE", "portfolio_RMSE", ps, "portfolio_day_RMSE_kWh"), ("Portfolio-day MAPE %", "portfolio_MAPE_%", ps, "portfolio_day_MAPE_%"),
               ("Portfolio bias % ", "portfolio_bias_%", ps, "portfolio_bias_%"), ("Portfolio bias kWh/day", "portfolio_bias_kWh", ps, "portfolio_bias_kWh_per_day"),
               ("Underforecast MWh", "portfolio_under_MWh", ps, "underforecast_MWh"), ("Overforecast MWh", "portfolio_over_MWh", ps, "overforecast_MWh"),
               ("% days underforecast", "portfolio_pct_days_under", ps, "pct_days_under_%"), ("Cost 1:1 MWh-eq", "cost_1:1_MWh_eq", ps, "cost_1:1_total_MWh_eq"),
               ("Cost 2:1 MWh-eq", "cost_2:1_MWh_eq", ps, "cost_2:1_total_MWh_eq")]
    rows = []
    for nm, vk, tab, ek in mapping:
        r = {"Metric": nm}
        for lab, em in (("L1", "lgbm_l1"), ("L2", "lgbm_l2")):
            r[f"Existing {lab}"], r[f"Verified {lab}"] = tab.loc[em, ek], verified[lab][vk]
            r[f"Diff {lab}"] = r[f"Verified {lab}"] - r[f"Existing {lab}"]
        rows.append(r)
    cmp_ = pd.DataFrame(rows)[["Metric", "Existing L1", "Verified L1", "Diff L1", "Existing L2", "Verified L2", "Diff L2"]]
    cmp_.to_csv(out / "metrics" / "comparison_vs_existing.csv", index=False); print(cmp_.round(5).to_string(index=False), flush=True)
    pdiff = {}
    for lab, em in (("L1", "lgbm_l1"), ("L2", "lgbm_l2")):
        old = pd.read_parquet(ex / "predictions" / f"{em}_interval_predictions.parquet")[["Household_ID", "Timestamp", "prediction", "actual"]]
        m = D_.merge(old, on=["Household_ID", "Timestamp"], how="outer", indicator=True, suffixes=("", "_old"))
        both = m[m["_merge"] == "both"]
        dif = (both[f"prediction_{lab.lower()}"] - both["prediction"]).abs()
        pdiff[lab] = {"verified_rows": int(D_.shape[0]), "existing_rows": int(old.shape[0]), "rows_only_in_verified": int((m["_merge"] == "left_only").sum()),
                      "rows_only_in_existing": int((m["_merge"] == "right_only").sum()), "compared": len(both), "max_abs_diff": float(dif.max()), "mean_abs_diff": float(dif.mean()),
                      "n_differing_exact": int((dif > 0).sum()), "n_differing_gt_1e-6": int((dif > 1e-6).sum()),
                      "actual_max_abs_diff_raw_csv_vs_pipeline": float((both["actual"] - both["actual_old"]).abs().max()) if "actual_old" in both else float((both["actual"] - both["actual"]).abs().max())}
    res["prediction_comparison"] = pdiff; print(json.dumps(pdiff, indent=2), flush=True)

    # ---- 8. diagnostics: why does L2 help the portfolio? -------------------------------------------------------------------------------------------------------------
    diag, quint = [], []
    hd_act = D_.groupby(["Household_ID", "date"])["actual"].sum()
    level = hd_act.groupby("Household_ID").mean()
    q_of = pd.qcut(level.rank(method="first"), 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    for lab, col in (("L1", "prediction_l1"), ("L2", "prediction_l2")):
        hdl = D_.groupby(["Household_ID", "date"]).agg(a=("actual", "sum"), p=(col, "sum")); e = hdl.p - hdl.a
        pdl = hdl.groupby("date").sum()
        sum_abs = e.abs().groupby("date").sum(); net_abs = e.groupby("date").sum().abs()
        diag.append({"model": lab, "mean_actual_portfolio_day_kWh": pdl.a.mean(), "mean_pred_portfolio_day_kWh": pdl.p.mean(), "pred/actual_ratio": pdl.p.sum() / pdl.a.sum(),
                     "mean_abs_household_daily_error_kWh": e.abs().mean(), "mean_signed_household_daily_error_kWh": e.mean(), "share_household_days_under_%": 100 * (e < 0).mean(),
                     "cancellation_%_(1-|sum e|/sum|e|)": 100 * (1 - net_abs.sum() / sum_abs.sum())})
        qq = pd.DataFrame({"e": e, "a": hdl.a}).reset_index(); qq["quintile"] = qq["Household_ID"].map(q_of)
        g = qq.groupby("quintile", observed=True).agg(mean_actual_daily=("a", "mean"), mean_signed_error=("e", "mean"))
        g["share_of_total_signed_error_%"] = 100 * qq.groupby("quintile", observed=True)["e"].sum() / qq["e"].sum(); g.insert(0, "model", lab); quint.append(g.reset_index())
    dg = pd.DataFrame(diag); dg.to_csv(out / "metrics" / "diagnostics_aggregate_before_loss.csv", index=False)
    qd = pd.concat(quint); qd.to_csv(out / "metrics" / "diagnostics_by_household_level_quintile.csv", index=False)
    D_["bin"] = pd.cut(D_["actual"], [-1e-9, 0.0, 0.05, 0.2, 0.5, 1.0, 100], labels=["0", "(0,0.05]", "(0.05,0.2]", "(0.2,0.5]", "(0.5,1]", ">1"])
    bn = D_.groupby("bin", observed=True).agg(n=("actual", "size"), actual_mean=("actual", "mean"), L1_mean_signed=("prediction_l1", lambda s: (s - D_.loc[s.index, "actual"]).mean()),
                                               L2_mean_signed=("prediction_l2", lambda s: (s - D_.loc[s.index, "actual"]).mean()))
    bn["share_of_intervals_%"] = 100 * bn["n"] / bn["n"].sum(); bn.to_csv(out / "metrics" / "diagnostics_by_actual_level_bin.csv")
    print(dg.round(3).T.to_string()); print(qd.round(3).to_string(index=False)); print(bn.round(4).to_string(), flush=True)

    # ---- verdict --------------------------------------------------------------------------------------------------------------------------------------------------------------
    pred_identical = all(v["n_differing_gt_1e-6"] == 0 and v["rows_only_in_verified"] == 0 and v["rows_only_in_existing"] == 0 for v in pdiff.values())
    metrics_close = bool((cmp_[["Diff L1", "Diff L2"]].abs().max(axis=1) <= 1e-9 * cmp_[["Existing L1", "Existing L2"]].abs().max(axis=1).clip(lower=1)).all())
    res["verdict"] = "VERIFIED" if pred_identical and metrics_close else "CHECK MANUALLY (see comparison)"
    res["pred_identical"], res["metrics_match"] = pred_identical, metrics_close
    (out / "metrics" / "verification_summary.json").write_text(json.dumps(res, indent=2, default=str))
    print("VERDICT:", res["verdict"], "| predictions identical:", pred_identical, "| metrics match:", metrics_close)


if __name__ == "__main__":
    main()
