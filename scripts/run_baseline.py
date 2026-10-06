"""Run the seasonal-naive baselines end to end:  python scripts/run_baseline.py [--help]"""
import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import baseline, evaluation, visualization
from src.config import Config
from src.data import load_all, split_dates


def parse_args() -> Config:
    cfg, ap = Config(), argparse.ArgumentParser(description=__doc__)
    for f in fields(Config):
        v = getattr(cfg, f.name)
        if isinstance(v, (float, int)) or f.name == "max_households":
            ap.add_argument(f"--{f.name}", type=type(v) if v is not None else int, default=v)
        elif isinstance(v, Path):
            ap.add_argument(f"--{f.name}", type=Path, default=v)
        elif f.name == "models":
            ap.add_argument("--models", nargs="+", default=list(v), choices=list(baseline.MODEL_LAGS))
    for k, v in vars(ap.parse_args()).items():
        setattr(cfg, k, tuple(v) if k == "models" else v)
    return cfg


def main():
    cfg = parse_args()
    out = cfg.output_dir
    for sub in ("metrics", "plots", "predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- 1. load + clean ---------------------------------------------------------------
    mats, quality = {}, []
    for hh, mat, st in load_all(cfg):
        mats[hh] = mat
        quality.append(st)
    q = pd.DataFrame(quality)
    q.to_csv(out / "metrics" / "data_quality.csv", index=False)
    first, last = pd.Timestamp(q["first_date"].min()), pd.Timestamp(q["last_date"].max())

    # ---- 2. explicit chronological split of the global calendar --------------------------
    split = split_dates(first, last, cfg)
    test_range = split["test"]

    # ---- 3. forecast + accumulate metrics household by household -----------------------------
    schema = pa.schema([("Household_ID", pa.string()), ("Timestamp", pa.timestamp("ns")),
                        ("actual", pa.float64()), ("prediction", pa.float64()), ("model", pa.string())])
    writer = pq.ParquetWriter(out / "predictions" / "predictions.parquet", schema)
    chunks, coverage = [], []
    for hh, mat in mats.items():
        df, cov = baseline.predict_household(hh, mat, cfg, test_range)
        coverage += cov
        if df.empty:
            continue
        writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
        chunks.append(evaluation.chunk_stats(df, cfg.mape_min_actual, len(cfg.models)))
    writer.close()
    by_key = {"household": ["model", "Household_ID"], "qh": ["model", "qh"], "date": ["model", "date"]}
    res = evaluation.combine(chunks, by_key)
    cov = pd.DataFrame(coverage)
    cov_sum = cov.groupby("model")[["test_days_with_actual", "predicted_days", "no_prediction_days"]].sum()
    cov_sum["households_without_prediction"] = cov[cov["predicted_days"] == 0].groupby("model").size()
    cov_sum = cov_sum.fillna(0).astype(int)

    res["overall"].to_csv(out / "metrics" / "overall.csv", index=False)
    res["household"].to_csv(out / "metrics" / "per_household.csv", index=False)
    res["qh"].to_csv(out / "metrics" / "per_quarter_hour.csv", index=False)
    res["date"].to_csv(out / "metrics" / "per_date.csv", index=False)
    cov.to_csv(out / "metrics" / "coverage_per_household.csv", index=False)
    cov_sum.to_csv(out / "metrics" / "coverage_summary.csv")

    # ---- 4. plots ------------------------------------------------------------------------
    preds = pd.read_parquet(out / "predictions" / "predictions.parquet")
    hm = res["household"].query("scope == 'common'")
    visualization.plot_mae_distribution(hm, out / "plots" / "4_mae_distribution.png")
    visualization.plot_mean_profile(res["qh"].query("scope == 'common'"), out / "plots" / "3_mean_profile.png")
    # representative household: full test coverage, MAE closest to the median of naive_7day
    h7 = hm[hm["model"] == "naive_7day"].set_index("Household_ID")
    full = h7[h7["n"] >= h7["n"].quantile(0.9)]
    rep = (full["MAE"] - full["MAE"].median()).abs().idxmin()
    rep_days = pd.DatetimeIndex(sorted(preds.loc[preds["Household_ID"] == rep, "Timestamp"].dt.normalize().unique()))
    mid = rep_days[len(rep_days) // 2]
    mid = rep_days[(rep_days >= mid) & (rep_days.dayofweek == 2)][0]    # a Wednesday mid-test
    c = preds[preds["Household_ID"] == rep]
    c = c[c["Timestamp"].dt.normalize().isin(rep_days)]
    visualization.plot_example_days(c, rep, pd.DatetimeIndex([mid]), out / "plots" / "1_example_day.png",
                                    f"Actual vs forecast, {mid.date()}")
    week = pd.date_range(mid - pd.Timedelta(days=3), periods=7)
    visualization.plot_example_days(c, rep, week, out / "plots" / "2_example_week.png",
                                    f"Actual vs forecast, {week[0].date()} to {week[-1].date()}")

    # ---- 5. summary ----------------------------------------------------------------------
    ov = res["overall"]
    summary = {
        "households": int(len(mats)),
        "date_range": [str(first.date()), str(last.date())],
        "observations_non_missing": int(q["observed_intervals"].sum()),
        "expected_intervals": int(q["expected_intervals"].sum()),
        "missing_intervals": int(q["missing_intervals"].sum()),
        "complete_days": int(q["complete_days"].sum()),
        "incomplete_days": int(q["incomplete_days"].sum()),
        "duplicate_timestamps": int(q["duplicate_timestamps"].sum()),
        "off_grid_timestamps": int(q["off_grid_timestamps"].sum()),
        "negative_values": int(q["negative_values"].sum()),
        "split": {k: [str(v[0].date()), str(v[1].date())] for k, v in split.items()},
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "households_in_test": int(hm["Household_ID"].nunique()),
    }
    (out / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    print("\nCoverage of test days:\n", cov_sum)
    for scope in ("common", "all"):
        print(f"\nOverall metrics [{scope}]:\n",
              ov[ov["scope"] == scope].drop(columns="scope").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
