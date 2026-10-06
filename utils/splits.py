"""Shared evaluation contract: data blocks, portfolio cohort, prediction file format."""
import json
from datetime import date
from pathlib import Path

import polars as pl

from utils.data import ROOT

FIT_END = date(2022, 10, 29)
TUNE = (date(2022, 11, 1), date(2022, 12, 29))
CALIB = (date(2023, 1, 1), date(2023, 2, 27))
TEST = (date(2023, 3, 1), date(2024, 2, 27))
PREDS = ROOT / "results" / "preds"


def cohort(df: pl.DataFrame) -> list[int]:
    """Portfolio members, fixed with information up to 2022-12-29: >= 90 days of complete history
    and data within the 14 days before."""
    hist = df.filter(pl.col("D") <= date(2022, 12, 29), pl.col("kwh").is_not_null()).group_by("Household_ID").agg(
        days=pl.len() / 24, recent=(pl.col("D") >= date(2022, 12, 16)).sum()
    )
    return sorted(hist.filter(pl.col("days") >= 90, pl.col("recent") > 0)["Household_ID"].to_list())


def save_preds(name: str, preds: pl.DataFrame, meta: dict) -> None:
    """Store one model's forecasts: columns Household_ID, hour (UTC), pred [, q* columns].
    `meta` documents track (target-only / W0 / W2_oracle), inputs, point statistic, runtime, device, version."""
    PREDS.mkdir(parents=True, exist_ok=True)
    assert {"Household_ID", "hour", "pred"} <= set(preds.columns)
    preds.write_parquet(PREDS / f"{name}.parquet")
    (PREDS / f"{name}.json").write_text(json.dumps({"model": name, **meta}, indent=2, default=str))


def load_all_preds() -> dict[str, tuple[pl.DataFrame, dict]]:
    out = {}
    for p in sorted(PREDS.glob("*.parquet")):
        meta = json.loads(p.with_suffix(".json").read_text())
        out[p.stem] = (pl.read_parquet(p), meta)
    return out
