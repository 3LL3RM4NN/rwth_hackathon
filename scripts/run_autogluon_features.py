"""AutoGluon + static household features (Model B) vs consumption-only AutoGluon (Model A).

    uv run python scripts/run_autogluon_features.py [--max_origins N] [--time_limit S] [--train_days D] [--out DIR]

Identical to scripts/run_autogluon.py (same target, split, rolling protocol, model set, seed, time limit) except that
`static_features` are attached to every series. Model A predictions are read from outputs/autogluon/ (not re-run).
"""
import argparse
import copy
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
from src.autogluon_forecast import build_train_frame, fit, rolling_predict
from src.config import Config
from src.data import load_all, split_dates
from src.static_features import build_static_features

MODEL_B = "autogluon_household_features"
MODEL_A = "autogluon_timeseries"
SCHEMA = pa.schema([("Household_ID", pa.string()), ("Timestamp", pa.timestamp("ns")),
                    ("actual", pa.float64()), ("prediction", pa.float64()), ("model", pa.string())])
MODEL_ORDER = ["naive_1day", "naive_7day", "seasonal_mean_4weeks", MODEL_A, MODEL_B]


def attach_actuals(df: pd.DataFrame, mat: pd.DataFrame, test_range) -> pd.DataFrame:
    ts = (mat.index.to_numpy()[:, None] + pd.to_timedelta(range(0, 1440, 15), unit="min").to_numpy()[None, :]).ravel()
    vals = pd.Series(mat.to_numpy().ravel(), index=ts)
    df = df.assign(actual=df["Timestamp"].map(vals)).dropna(subset=["actual", "prediction"])
    df = df[(df["Timestamp"] >= test_range[0]) & (df["Timestamp"] < test_range[1] + pd.Timedelta(days=1))]
    return df[["Household_ID", "Timestamp", "actual", "prediction", "model"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_limit", type=int, default=BASE_CFG["time_limit"])
    ap.add_argument("--max_origins", type=int, default=None)
    ap.add_argument("--train_days", type=int, default=BASE_CFG["train_days"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--a_dir", type=Path, default=None, help="dir of Model A outputs (default outputs/autogluon)")
    a = ap.parse_args()
    ag = copy.deepcopy(BASE_CFG)
    ag.update(time_limit=a.time_limit, max_origins=a.max_origins, train_days=a.train_days)
    cfg = Config()
    out = a.out or cfg.output_dir / "autogluon_household_features"
    a_dir = a.a_dir or cfg.output_dir / "autogluon"
    for sub in ("model", "predictions", "metrics"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    mats = {hh: m for hh, m, _ in load_all(cfg)}
    first = min(m.index.min() for m in mats.values())
    last = max(m.index.max() for m in mats.values())
    test_range = split_dates(first, last, cfg)["test"]
    train_end = test_range[0] - pd.Timedelta(days=1)
    static = build_static_features(cfg)
    static.to_csv(out / "metrics" / "static_features_used.csv")
    print("Model B config:", json.dumps({**ag, "static_features": list(static.columns),
                                         "train_end": str(train_end.date()), "test": [str(d.date()) for d in test_range]}, indent=2))

    train = build_train_frame(mats, train_end, ag["train_days"], ag["min_train_valid_days"], static)
    assert train.index.get_level_values("timestamp").max() <= train_end + pd.Timedelta(hours=23, minutes=45)
    print(f"train series: {train.num_items}, rows: {len(train)}, static cols: {list(train.static_features.columns)}")
    t0 = time.time()
    predictor = fit(train, out / "model", ag)
    fit_s = time.time() - t0
    lb = predictor.leaderboard(silent=True)
    print(lb[["model", "score_val", "fit_time_marginal"]].to_string())
    t0 = time.time()
    preds = rolling_predict(predictor, mats, test_range, ag, static=static)
    pred_s = time.time() - t0
    preds["model"] = MODEL_B

    # ---- evaluation: baselines (recomputed) + stored Model A + Model B -> 5 models, 'common' = all five predict
    pa_ = pd.read_parquet(a_dir / "predictions" / "predictions.parquet")
    a_by = {h: g for h, g in pa_.groupby("Household_ID")}
    b_by = {h: g for h, g in preds.groupby("Household_ID")}
    writer = pq.ParquetWriter(out / "predictions" / "predictions.parquet", SCHEMA)
    chunks = []
    for hh, mat in mats.items():
        base, _ = baseline.predict_household(hh, mat, cfg, test_range)
        frames = [base] if len(base) else []
        if hh in a_by:
            frames.append(a_by[hh])
        if hh in b_by:
            b = attach_actuals(b_by[hh], mat, test_range)
            writer.write_table(pa.Table.from_pandas(b, schema=SCHEMA, preserve_index=False))
            frames.append(b)
        if frames:
            chunks.append(evaluation.chunk_stats(pd.concat(frames, ignore_index=True), cfg.mape_min_actual, 5))
    writer.close()
    res = evaluation.combine(chunks, {"household": ["model", "Household_ID"], "qh": ["model", "qh"], "date": ["model", "date"]})
    for k, v in res.items():
        v.to_csv(out / "metrics" / ("overall.csv" if k == "overall" else f"per_{k}.csv"), index=False)
    lb.to_csv(out / "metrics" / "leaderboard.csv", index=False)

    # ---- comparison table (primary metric: mean household MAE) ------------------------------------
    rows = []
    for scope in ("common", "all"):
        hm = res["household"].query("scope == @scope")
        ov = res["overall"].query("scope == @scope").set_index("model")
        for m in MODEL_ORDER:
            h = hm[hm["model"] == m]
            rows.append({"scope": scope, "model": m, "mean_household_MAE": h["MAE"].mean(),
                         "median_household_MAE": h["MAE"].median(), "min_household_MAE": h["MAE"].min(),
                         "p90_household_MAE": h["MAE"].quantile(0.9), "max_household_MAE": h["MAE"].max(),
                         "overall_MAE": ov.loc[m, "MAE"], "RMSE": ov.loc[m, "RMSE"],
                         "daily_energy_MAE_kWh": ov.loc[m, "daily_energy_MAE_kWh"], "bias": ov.loc[m, "bias"],
                         "mean_actual": ov.loc[m, "mean_actual"], "mean_pred": ov.loc[m, "mean_pred"],
                         "households": h["Household_ID"].nunique(), "scored_intervals": int(ov.loc[m, "n"])})
    comp = pd.DataFrame(rows)
    ref = comp[(comp.scope == "common") & (comp.model == MODEL_A)]["mean_household_MAE"].iloc[0]
    comp["mean_hh_MAE_improvement_vs_A_%"] = 100 * (ref - comp["mean_household_MAE"]) / ref
    comp.to_csv(out / "metrics" / "comparison.csv", index=False)

    # ---- per-household B vs A (common scope) ---------------------------------------------------
    hm = res["household"].query("scope == 'common'").pivot(index="Household_ID", columns="model", values="MAE")
    d = pd.DataFrame({"MAE_A": hm[MODEL_A], "MAE_B": hm[MODEL_B]})
    d["delta"] = d["MAE_B"] - d["MAE_A"]
    d["rel_change_%"] = 100 * d["delta"] / d["MAE_A"]
    hh_meta = pd.read_csv(cfg.data_dir.parent / "smart_meter_meta_data" / "households.csv", sep=";", dtype={"Household_ID": str})
    d = d.join(static[["pv_flag", "has_survey", "region"]]).join(hh_meta.set_index("Household_ID")["Group"])
    d.sort_values("rel_change_%").to_csv(out / "metrics" / "per_household_B_vs_A.csv")
    info = {"autogluon_version": __import__("autogluon.timeseries").timeseries.__version__, "config": ag,
            "static_features": list(static.columns), "fit_seconds": round(fit_s, 1), "predict_seconds": round(pred_s, 1),
            "series_in_training": int(train.num_items), "models_trained": lb["model"].tolist(), "best_model": predictor.model_best,
            "households_improved": int((d["delta"] < 0).sum()), "households_worsened": int((d["delta"] > 0).sum()),
            "households_compared": int(len(d))}
    (out / "metrics" / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(comp[comp.scope == "common"].round(4).to_string(index=False))
    print(json.dumps(info, indent=2, default=str))
    print(d.groupby("Group")["rel_change_%"].describe().round(2)); print(d.groupby("pv_flag")["rel_change_%"].describe().round(2))
    print(d.groupby("has_survey")["rel_change_%"].describe().round(2))


if __name__ == "__main__":
    main()
