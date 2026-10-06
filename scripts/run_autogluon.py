"""AutoGluon-TimeSeries benchmark:  uv run python scripts/run_autogluon.py [--max_origins N] [--time_limit S]

Same data loading / split / scoring rules as the baselines; outputs go to outputs/autogluon/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import baseline, evaluation
from src.autogluon_forecast import MODEL_NAME, build_train_frame, fit, rolling_predict
from src.config import Config
from src.data import load_all, split_dates

AG_CFG = {
    "preset": "medium_quality",
    # deliberately small model set; both tabular models are LightGBM trained with an L1 (MAE) objective
    "hyperparameters": {
        "SeasonalNaive": {},
        "RecursiveTabular": {"model_name": "GBM", "model_hyperparameters": {"objective": "regression_l1"}},
        "DirectTabular": {"model_name": "GBM", "model_hyperparameters": {"objective": "regression_l1"}},
    },
    "time_limit": 1200,
    "num_val_windows": 1,
    "enable_ensemble": True,
    "seed": 0,
    "train_days": 365,             # last 365 days of the development period
    "min_train_valid_days": 14,
    "context_days": 42,            # history passed to the predictor at each forecast origin
    "min_context_valid_days": 7,
    "max_origins": None,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_limit", type=int, default=AG_CFG["time_limit"])
    ap.add_argument("--max_origins", type=int, default=None, help="debug: only the first N test days")
    ap.add_argument("--train_days", type=int, default=AG_CFG["train_days"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    AG_CFG.update(time_limit=a.time_limit, max_origins=a.max_origins, train_days=a.train_days)
    cfg = Config()
    out = a.out or cfg.output_dir / "autogluon"
    for sub in ("model", "predictions", "metrics"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    mats = {hh: m for hh, m, _ in load_all(cfg)}
    first = min(m.index.min() for m in mats.values())
    last = max(m.index.max() for m in mats.values())
    split = split_dates(first, last, cfg)
    test_range = split["test"]
    train_end = test_range[0] - pd.Timedelta(days=1)           # 2023-03-15: nothing later is ever used for fitting
    print("AutoGluon config:", json.dumps({**AG_CFG, "train_end": str(train_end.date()),
                                          "test": [str(test_range[0].date()), str(test_range[1].date())]}, indent=2))
    import autogluon.timeseries as agts
    print("autogluon.timeseries", agts.__version__)

    train = build_train_frame(mats, train_end, AG_CFG["train_days"], AG_CFG["min_train_valid_days"])
    assert train.index.get_level_values("timestamp").max() <= train_end + pd.Timedelta(hours=23, minutes=45)
    print(f"train series: {train.num_items}, rows: {len(train)}, "
          f"range {train.index.get_level_values('timestamp').min()} .. {train.index.get_level_values('timestamp').max()}")

    t0 = time.time()
    predictor = fit(train, out / "model", AG_CFG)
    fit_s = time.time() - t0
    lb = predictor.leaderboard(silent=True)
    print(lb[["model", "score_val", "fit_time_marginal", "pred_time_val"]].to_string())

    t0 = time.time()
    preds = rolling_predict(predictor, mats, test_range, AG_CFG)
    pred_s = time.time() - t0
    preds["model"] = MODEL_NAME

    # ---- evaluation: reuse baseline predictions + evaluation module (4 models -> 'common' = all four predict)
    schema = pa.schema([("Household_ID", pa.string()), ("Timestamp", pa.timestamp("ns")),
                        ("actual", pa.float64()), ("prediction", pa.float64()), ("model", pa.string())])
    writer = pq.ParquetWriter(out / "predictions" / "predictions.parquet", schema)
    chunks = []
    by_hh = {h: g for h, g in preds.groupby("Household_ID")}
    for hh, mat in mats.items():
        base, _ = baseline.predict_household(hh, mat, cfg, test_range)
        ag = by_hh.get(hh)
        if ag is None:
            frames = [base] if len(base) else []
        else:
            vals = pd.Series(mat.to_numpy().ravel(),
                             index=(mat.index.to_numpy()[:, None] + pd.to_timedelta(range(0, 1440, 15), unit="min").to_numpy()[None, :]).ravel())
            ag = ag.assign(actual=ag["Timestamp"].map(vals)).dropna(subset=["actual", "prediction"])
            ag = ag[(ag["Timestamp"] >= test_range[0]) & (ag["Timestamp"] < test_range[1] + pd.Timedelta(days=1))]
            ag = ag[["Household_ID", "Timestamp", "actual", "prediction", "model"]]
            writer.write_table(pa.Table.from_pandas(ag, schema=schema, preserve_index=False))
            frames = ([base] if len(base) else []) + [ag]
        if frames:
            chunks.append(evaluation.chunk_stats(pd.concat(frames, ignore_index=True), cfg.mape_min_actual, 4))
    writer.close()
    res = evaluation.combine(chunks, {"household": ["model", "Household_ID"], "qh": ["model", "qh"], "date": ["model", "date"]})
    for k, v in res.items():
        v.to_csv(out / "metrics" / ("overall.csv" if k == "overall" else f"per_{k}.csv"), index=False)
    lb.to_csv(out / "metrics" / "leaderboard.csv", index=False)
    info = {"autogluon_version": agts.__version__, "config": AG_CFG, "fit_seconds": round(fit_s, 1),
            "predict_seconds": round(pred_s, 1), "series_in_training": int(train.num_items),
            "models_trained": lb["model"].tolist(), "best_model": predictor.model_best}
    (out / "metrics" / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    pd.set_option("display.width", 220)
    ov = res["overall"]
    for scope in ("common", "all"):
        print(f"\nOverall [{scope}]:\n", ov[ov["scope"] == scope].drop(columns="scope").round(4).to_string(index=False))
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
