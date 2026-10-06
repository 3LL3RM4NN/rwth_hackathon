"""Loading and cleaning of the household smart-meter files.

Every household is turned into a *daily matrix*: one row per calendar day (UTC),
96 columns for the 15-minute intervals of that day. Missing values stay NaN;
nothing is ever interpolated or filled.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def load_household(path: Path, target: str) -> tuple[pd.Series, dict]:
    """Read one CSV -> chronologically sorted, de-duplicated 15-min series (naive UTC) + quality stats."""
    df = pd.read_csv(path, sep=";", usecols=["Household_ID", "Timestamp", target])
    hh = str(df["Household_ID"].iloc[0])
    ts = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(None)  # data is UTC; naive UTC avoids DST issues
    s = pd.Series(df[target].to_numpy(dtype="float64"), index=ts, name=hh).sort_index()

    stats = {"Household_ID": hh, "rows": len(s)}
    stats["duplicate_timestamps"] = int(s.index.duplicated().sum())
    s = s[~s.index.duplicated(keep="first")]
    off_grid = ~((s.index.minute % 15 == 0) & (s.index.second == 0))
    stats["off_grid_timestamps"] = int(off_grid.sum())
    s = s[~off_grid]
    stats["nan_values"] = int(s.isna().sum())
    stats["negative_values"] = int((s < 0).sum())
    return s, stats


def to_daily_matrix(s: pd.Series, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Series -> (days x 96) DataFrame on a full calendar grid, plus completeness stats.

    Days with fewer than `min_day_completeness` observed intervals are treated as
    unavailable: the *whole* day is set to NaN (neither used as history nor evaluated).
    """
    n = cfg.intervals_per_day
    first, last = s.index.min().normalize(), s.index.max().normalize()
    grid = pd.date_range(first, last + pd.Timedelta(days=1) - pd.Timedelta(minutes=15), freq="15min")
    full = s.reindex(grid)                                    # missing timestamps -> NaN (not filled)
    mat = pd.DataFrame(full.to_numpy().reshape(-1, n),
                       index=pd.date_range(first, last, freq="D"))
    observed = mat.notna().sum(axis=1)
    usable = observed >= np.ceil(cfg.min_day_completeness * n)
    mat.loc[~usable.to_numpy()] = np.nan
    stats = {
        "first_date": first.date(), "last_date": last.date(),
        "calendar_days": len(mat),
        "expected_intervals": len(grid),
        "observed_intervals": int(full.notna().sum()),
        "missing_intervals": int(full.isna().sum()),
        "complete_days": int((observed == n).sum()),
        "incomplete_days": int((observed < n).sum()),
        "usable_days": int(usable.sum()),
    }
    return mat, stats


def load_all(cfg: Config):
    """Yield (household_id, daily_matrix, stats) for every household file."""
    files = sorted(cfg.data_dir.glob("*.csv"))
    if cfg.max_households:
        files = files[: cfg.max_households]
    for f in files:
        s, q = load_household(f, cfg.target)
        mat, d = to_daily_matrix(s, cfg)
        yield q["Household_ID"], mat, {**q, **d}


def split_dates(all_first: pd.Timestamp, all_last: pd.Timestamp, cfg: Config) -> dict:
    """Explicit chronological split of the global calendar: train | validation | test."""
    days = pd.date_range(all_first, all_last, freq="D")
    n_test = int(round(len(days) * cfg.test_fraction))
    n_val = int(round(len(days) * cfg.val_fraction))
    n_train = len(days) - n_val - n_test
    return {
        "train": (days[0], days[n_train - 1]),
        "val": (days[n_train], days[n_train + n_val - 1]),
        "test": (days[n_train + n_val], days[-1]),
    }
