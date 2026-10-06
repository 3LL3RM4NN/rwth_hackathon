"""Level 1b: PV vs non-PV AutoGluon models with the operational D-1 11:45 cutoff.

    uv run python scripts/run_level1_pv.py [--max_origins N] [--time_limit S] [--train_days D] [--out DIR] [--a_dir DIR]
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
import pyarrow as pa
import pyarrow.parquet as pq

from run_autogluon import AG_CFG as BASE_CFG
from src import baseline, evaluation
from src.autogluon_forecast import build_train_frame, fit
from src.config import Config
from src.data import load_all, split_dates
from src.level1_pv import HORIZON, pv_status_table, rolling_predict_1145, verify_cutoff

M_NEW = "autogluon_pv_split_1145"
M_OLD = "autogluon_pooled_2345_cutoff"            # earlier pooled Model A, D-1 23:45 cutoff (reference only)
BASELINES = ("naive_7day", "seasonal_mean_4weeks")  # lags >= 7 days -> valid at an 11:45 cutoff; naive_1day is NOT
SCHEMA = pa.schema([("Household_ID", pa.string()), ("Timestamp", pa.timestamp("ns")),
                    ("actual", pa.float64()), ("prediction", pa.float64()), ("model", pa.string())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_limit", type=int, default=BASE_CFG["time_limit"])
    ap.add_argument("--max_origins", type=int, default=None)
    ap.add_argument("--train_days", type=int, default=BASE_CFG["train_days"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--a_dir", type=Path, default=None)
    a = ap.parse_args()
    ag = copy.deepcopy(BASE_CFG)
    ag.update(time_limit=a.time_limit, max_origins=a.max_origins, train_days=a.train_days, prediction_length=HORIZON)
    cfg = Config()
    base_cfg = dataclasses.replace(cfg, models=BASELINES)
    out = a.out or cfg.output_dir / "level1_pv_1145"
    a_dir = a.a_dir or cfg.output_dir / "autogluon"
    for sub in ("predictions", "metrics"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    masked = {hh: m for hh, m, _ in load_all(cfg)}
    raw = {hh: m for hh, m, _ in load_all(dataclasses.replace(cfg, min_day_completeness=0.0))}
    first = min(m.index.min() for m in masked.values()); last = max(m.index.max() for m in masked.values())
    test_range = split_dates(first, last, cfg)["test"]
    train_end = test_range[0] - pd.Timedelta(days=1)

    # ---- PV audit ----------------------------------------------------------------------------
    pv = pv_status_table(cfg).set_index("Household_ID")
    pv = pv.loc[[h for h in masked]]                     # the 410 households with 15-min data
    in_test = {hh: bool(((m.index >= test_range[0]) & np.isfinite(m.to_numpy()).any(axis=1)).any()) for hh, m in masked.items()}
    pv["usable_test"] = pd.Series(in_test)
    audit = {"total_households": int(len(pv)), **{k: int(v) for k, v in pv["pv_status"].value_counts().items()},
             "usable_test_households": {k: int(v) for k, v in pv[pv.usable_test].groupby("pv_status").size().items()},
             "unknown_ids_excluded": sorted(pv.index[pv.pv_status == "unknown"].tolist()),
             "pv_by_Group": pd.crosstab(pv["Group"], pv["pv_status"]).to_dict()}
    pv.to_csv(out / "metrics" / "pv_audit_households.csv")
    (out / "metrics" / "pv_audit.json").write_text(json.dumps(audit, indent=2))
    print("PV audit:", json.dumps({k: v for k, v in audit.items() if k != "unknown_ids_excluded"}, indent=2),
          f"\n  ({len(audit['unknown_ids_excluded'])} households with unknown PV status are EXCLUDED; ids in pv_audit.json)")
    groups = {"pv": [h for h in pv.index if pv.loc[h, "pv_status"] == "pv"],
              "non_pv": [h for h in pv.index if pv.loc[h, "pv_status"] == "non_pv"]}
    print("Config:", json.dumps({**ag, "train_end": str(train_end.date()), "test": [str(d.date()) for d in test_range],
                                "forecast_origin": "D-1 11:45", "scored_steps": "D 00:00..D 23:45 (last 96 of 144)"}, indent=2))

    verify_cutoff({h: masked[h] for h in groups["pv"] + groups["non_pv"]}, {h: raw[h] for h in groups["pv"] + groups["non_pv"]},
                  test_range, ag)                                # frame-level, before any training

    # ---- train + predict one model per group ---------------------------------------------------------
    info = {"groups": {}}
    preds = []
    for g, ids in groups.items():
        m_g = {h: masked[h] for h in ids}; r_g = {h: raw[h] for h in ids}
        train = build_train_frame(m_g, train_end, ag["train_days"], ag["min_train_valid_days"])
        assert train.index.get_level_values("timestamp").max() <= train_end + pd.Timedelta(hours=23, minutes=45)
        print(f"[{g}] train series {train.num_items}, rows {len(train)}", flush=True)
        t0 = time.time()
        predictor = fit(train, out / f"model_{g}", ag)
        fit_s = time.time() - t0
        lb = predictor.leaderboard(silent=True); lb.to_csv(out / "metrics" / f"leaderboard_{g}.csv", index=False)
        print(lb[["model", "score_val", "fit_time_marginal"]].to_string(), flush=True)
        verify_cutoff(m_g, r_g, test_range, ag, n_days=2, predictor=predictor)   # predictor-level perturbation test
        t0 = time.time()
        p = rolling_predict_1145(predictor, m_g, r_g, test_range, ag)
        p["pv_status"] = g
        preds.append(p)
        info["groups"][g] = {"series_in_training": int(train.num_items), "fit_seconds": round(fit_s, 1),
                             "predict_seconds": round(time.time() - t0, 1), "models_trained": lb["model"].tolist(),
                             "best_model": predictor.model_best}
    preds = pd.concat(preds, ignore_index=True)

    # ---- attach actuals; build Level-2/3-ready outputs -----------------------------------------------
    parts = []
    for hh, g in preds.groupby("Household_ID"):
        mat = masked[hh]
        ts = (mat.index.to_numpy()[:, None] + pd.to_timedelta(range(0, 1440, 15), unit="min").to_numpy()[None, :]).ravel()
        vals = pd.Series(mat.to_numpy().ravel(), index=ts)
        parts.append(g.assign(actual=g["Timestamp"].map(vals)))
    iv = pd.concat(parts, ignore_index=True).dropna(subset=["actual", "prediction"])
    iv["model"] = M_NEW
    iv["horizon_step"] = ((iv["Timestamp"] - iv["forecast_origin"]) / pd.Timedelta(minutes=15)).astype(int)   # 49..144 from origin
    iv["signed_error"] = iv["prediction"] - iv["actual"]                  # >0 = over-forecast (over-procurement)
    iv["abs_error"] = iv["signed_error"].abs()
    iv = iv[["Household_ID", "pv_status", "forecast_date", "forecast_origin", "Timestamp", "horizon_step",
             "actual", "prediction", "signed_error", "abs_error", "model"]]
    assert (iv["forecast_origin"] + pd.Timedelta(minutes=15) * iv["horizon_step"] == iv["Timestamp"]).all()
    assert (iv["Timestamp"].dt.normalize() == iv["forecast_date"]).all()
    assert (iv["forecast_origin"] == iv["forecast_date"] - pd.Timedelta(hours=12, minutes=15)).all()
    iv.to_parquet(out / "predictions" / "interval_predictions.parquet", index=False)
    day = iv.groupby(["Household_ID", "pv_status", "forecast_date"]).agg(
        n=("actual", "size"), actual_daily_kwh=("actual", "sum"), predicted_daily_kwh=("prediction", "sum")).reset_index()
    day = day[day["n"] == 96].drop(columns="n")
    day["signed_daily_error"] = day["predicted_daily_kwh"] - day["actual_daily_kwh"]
    day["abs_daily_error"] = day["signed_daily_error"].abs()
    day.to_parquet(out / "predictions" / "household_day_predictions.parquet", index=False)

    # ---- evaluation with the shared evaluation module -----------------------------------------------------
    old = pd.read_parquet(a_dir / "predictions" / "predictions.parquet")
    old["model"] = M_OLD
    old_by = {h: g for h, g in old.groupby("Household_ID")}
    new_by = {h: g[["Household_ID", "Timestamp", "actual", "prediction", "model"]] for h, g in iv.groupby("Household_ID")}
    chunks = {}
    for hh in groups["pv"] + groups["non_pv"]:
        base, _ = baseline.predict_household(hh, masked[hh], base_cfg, test_range)
        frames = ([base] if len(base) else []) + [x for x in (old_by.get(hh), new_by.get(hh)) if x is not None]
        if frames:
            chunks[hh] = evaluation.chunk_stats(pd.concat(frames, ignore_index=True), cfg.mape_min_actual, 4)
    models = ["naive_7day", "seasonal_mean_4weeks", M_OLD, M_NEW]
    by_key = {"household": ["model", "Household_ID"], "qh": ["model", "qh"], "date": ["model", "date"]}
    rows, hh_all = [], []
    for name, ids in (("combined", groups["pv"] + groups["non_pv"]), ("pv", groups["pv"]), ("non_pv", groups["non_pv"])):
        sub = [chunks[h] for h in ids if h in chunks]
        res = evaluation.combine(sub, by_key)
        for scope in ("common", "all"):
            hm = res["household"].query("scope == @scope"); ov = res["overall"].query("scope == @scope").set_index("model")
            for m in models:
                h = hm[hm["model"] == m]
                if h.empty:
                    continue
                rows.append({"group": name, "scope": scope, "model": m, "households": h["Household_ID"].nunique(),
                             "scored_intervals": int(ov.loc[m, "n"]), "mean_household_MAE": h["MAE"].mean(),
                             "median_household_MAE": h["MAE"].median(), "min_household_MAE": h["MAE"].min(),
                             "p90_household_MAE": h["MAE"].quantile(0.9), "max_household_MAE": h["MAE"].max(),
                             "overall_MAE": ov.loc[m, "MAE"], "RMSE": ov.loc[m, "RMSE"],
                             "daily_energy_MAE_kWh": ov.loc[m, "daily_energy_MAE_kWh"], "bias": ov.loc[m, "bias"],
                             "mean_actual": ov.loc[m, "mean_actual"], "mean_pred": ov.loc[m, "mean_pred"]})
        if name == "combined":
            h = res["household"].copy(); h["pv_status"] = h["Household_ID"].map(pv["pv_status"])
            h.to_csv(out / "metrics" / "per_household.csv", index=False)
            res["qh"].to_csv(out / "metrics" / "per_quarter_hour.csv", index=False)
            res["date"].to_csv(out / "metrics" / "per_date.csv", index=False)
            hmc = res["household"].query("scope == 'common'").pivot(index="Household_ID", columns="model", values="MAE")
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "metrics" / "comparison.csv", index=False)

    # improved / worsened per household (common scope)
    cmpr = []
    d = hmc.copy(); d["pv_status"] = d.index.map(pv["pv_status"])
    for ref in ("seasonal_mean_4weeks", M_OLD):
        for name, sub in (("combined", d), ("pv", d[d.pv_status == "pv"]), ("non_pv", d[d.pv_status == "non_pv"])):
            delta = sub[M_NEW] - sub[ref]
            cmpr.append({"reference": ref, "group": name, "households": len(sub), "improved": int((delta < 0).sum()),
                         "worsened": int((delta > 0).sum()), "pct_improved": round(100 * (delta < 0).mean(), 1),
                         "mean_rel_change_%": round(100 * (delta / sub[ref]).mean(), 2)})
    pd.DataFrame(cmpr).to_csv(out / "metrics" / "household_improvement.csv", index=False)
    info.update({"autogluon_version": __import__("autogluon.timeseries").timeseries.__version__, "config": ag, "audit": audit,
                 "baselines_valid_at_1145": list(BASELINES), "reference_old_model": M_OLD})
    (out / "metrics" / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    pd.set_option("display.width", 250)
    for gname in ("combined", "pv", "non_pv"):
        print(f"\n== {gname} (common scope) ==")
        print(comp[(comp.group == gname) & (comp.scope == "common")].drop(columns=["group", "scope", "min_household_MAE", "p90_household_MAE", "max_household_MAE", "mean_actual"]).round(4).to_string(index=False))
    print(pd.DataFrame(cmpr).to_string(index=False))
    print(json.dumps(info["groups"], indent=2))


if __name__ == "__main__":
    main()
