"""Independent checks of the saved predictions against the raw CSV files (run after run_baseline.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import Config

cfg = Config()
preds = pd.read_parquet(cfg.output_dir / "predictions" / "predictions.parquet")
rng = np.random.default_rng(0)
households = rng.choice(preds["Household_ID"].unique(), 8, replace=False)
LAGS = {"naive_1day": [1], "naive_7day": [7], "seasonal_mean_4weeks": [7, 14, 21, 28]}
D = pd.Timedelta(days=1)
fails = checked = 0
for hh in households:
    raw = pd.read_csv(cfg.data_dir / f"{hh}.csv", sep=";", usecols=["Timestamp", cfg.target])
    raw.index = pd.to_datetime(raw["Timestamp"], utc=True).dt.tz_localize(None)
    s = raw[cfg.target]
    assert s.index.is_unique and set(s.index.minute) <= {0, 15, 30, 45}
    for m, lags in LAGS.items():
        p = preds[(preds["Household_ID"] == hh) & (preds["model"] == m)]
        for _, r in p.sample(min(300, len(p)), random_state=1).iterrows():
            ts = r["Timestamp"]
            src = [s.get(ts - l * D, np.nan) for l in lags]
            src = [v for v in src if np.isfinite(v)]
            exp = np.mean(src)
            checked += 1
            ok = abs(exp - r["prediction"]) < 1e-9 and abs(s[ts] - r["actual"]) < 1e-12
            # leakage: every source timestamp lies strictly before the forecast day's midnight
            ok &= all((ts - l * D) < ts.normalize() for l in lags)
            if m != "seasonal_mean_4weeks":
                ok &= len(src) == 1
            fails += not ok
    print(f"household {hh}: ok so far, fails={fails}")
print(f"checked {checked} predictions, failures: {fails}")
assert fails == 0
