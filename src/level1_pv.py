"""Level 1b: separate AutoGluon models for PV / non-PV households, operational cutoff D-1 11:45.

Forecast origin for target day D is O = (D-1) 11:45. The predictor receives the last `context_days` days of data
with the LAST observed timestamp exactly O. It forecasts 144 steps (D-1 12:00 .. D 23:45); only the last 96
(D 00:00 .. D 23:45) are scored.

Context construction (the only place where model inputs are built):
  * full days before D-1 : completeness-masked daily matrix (same rule as the baselines; the mask of day X only depends
                           on day X, which is entirely before O)
  * day D-1              : RAW values of slots 00:00..11:45 only (no completeness mask, because a mask would depend on
                           the afternoon slots that are after the cutoff); everything after 11:45 is dropped
"""
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

from .autogluon_forecast import OFFSETS

CUTOFF_SLOTS = 48                      # 11:45 is the 48th quarter-hour of the day (slots 0..47)
HORIZON = 144                          # D-1 12:00 .. D 23:45
SCORED = 96


def origin_of(D: pd.Timestamp) -> pd.Timestamp:
    return D - pd.Timedelta(days=1) + pd.Timedelta(hours=11, minutes=45)


def pv_status_table(cfg) -> pd.DataFrame:
    """Household_ID -> pv_status in {pv, non_pv, unknown} from households.csv (the supplied flag; never inferred)."""
    hh = pd.read_csv(cfg.data_dir.parent / "smart_meter_meta_data" / "households.csv", sep=";", dtype={"Household_ID": str})
    hh["pv_status"] = hh["Installation_HasPVSystem"].map({True: "pv", False: "non_pv"}).fillna("unknown")
    return hh[["Household_ID", "Group", "Installation_HasPVSystem", "pv_status"]]


def context_block(masked: pd.DataFrame, raw: pd.DataFrame, O: pd.Timestamp, context_days: int, min_valid_days: int):
    """(days, values[days,96]) where the last day is D-1 and its slots >= 48 are NaN (dropped later); None if unusable."""
    d1 = O.normalize()
    if d1 not in raw.index:
        return None
    prev = masked.loc[d1 - pd.Timedelta(days=context_days - 1): d1 - pd.Timedelta(days=1)]
    if prev.notna().any(axis=1).sum() < min_valid_days:
        return None
    part = np.full((1, 96), np.nan)
    part[0, :CUTOFF_SLOTS] = raw.loc[d1].to_numpy()[:CUTOFF_SLOTS]
    days = prev.index.append(pd.DatetimeIndex([d1]))
    vals = np.vstack([prev.to_numpy(), part])
    valid = np.isfinite(vals).any(axis=1)
    first = np.argmax(valid)
    return days[first:], vals[first:]


def make_context(blocks: dict, O: pd.Timestamp) -> TimeSeriesDataFrame:
    ids, ts, y = [], [], []
    for hh, (days, vals) in blocks.items():
        t = (days.to_numpy()[:, None] + OFFSETS[None, :]).ravel()
        keep = t <= O.to_datetime64()                        # hard cut at the forecast origin
        ids.append(np.full(keep.sum(), hh, dtype=object)); ts.append(t[keep]); y.append(vals.ravel()[keep])
    idx = pd.MultiIndex.from_arrays([np.concatenate(ids), np.concatenate(ts)], names=["item_id", "timestamp"])
    data = TimeSeriesDataFrame(pd.DataFrame({"target": np.concatenate(y)}, index=idx))
    last = data.index.get_level_values("timestamp")
    assert last.max() == O, f"context must end exactly at the forecast origin {O}, got {last.max()}"
    ends = data.reset_index().groupby("item_id")["timestamp"].max()
    assert (ends == O).all(), "every series must end exactly at the forecast origin"
    return data


def build_contexts(masked: dict, raw: dict, D: pd.Timestamp, cfg: dict):
    """Context for target day D over households that have an actual to score on D."""
    O = origin_of(D)
    blocks = {}
    for hh, m in masked.items():
        if D not in m.index or not np.isfinite(m.loc[D].to_numpy()).any():
            continue                                         # nothing to score (selection only; forecast independent of it)
        b = context_block(m, raw[hh], O, cfg["context_days"], cfg["min_context_valid_days"])
        if b is not None:
            blocks[hh] = b
    return (make_context(blocks, O) if blocks else None), O


def rolling_predict_1145(predictor, masked: dict, raw: dict, test_range, cfg: dict, log_every: int = 20,
                         days: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    import time
    out, t0 = [], time.time()
    if days is None:                                         # default: every test day
        days = pd.date_range(test_range[0], test_range[1])
    if cfg.get("max_origins"):
        days = days[: cfg["max_origins"]]
    for k, D in enumerate(days):
        data, O = build_contexts(masked, raw, D, cfg)
        if data is None:
            continue
        pred = predictor.predict(data)
        p = pred["0.5"] if "0.5" in pred.columns else pred["mean"]
        df = p.rename("prediction").reset_index().rename(columns={"item_id": "Household_ID", "timestamp": "Timestamp"})
        assert df["Timestamp"].min() == O + pd.Timedelta(minutes=15) and len(df) == HORIZON * data.num_items
        df = df[df["Timestamp"] >= D].copy()                 # keep D 00:00 .. D 23:45
        assert df["Timestamp"].min() == D and df["Timestamp"].max() == D + pd.Timedelta(hours=23, minutes=45)
        df["prediction"] = df["prediction"].clip(lower=0.0)
        df["forecast_date"], df["forecast_origin"] = D, O
        out.append(df)
        if (k + 1) % log_every == 0:
            print(f"  day {D.date()} ({k + 1}/{len(days)}), {time.time() - t0:.0f}s elapsed", flush=True)
    return pd.concat(out, ignore_index=True)


def verify_cutoff(masked: dict, raw: dict, test_range, cfg: dict, n_days: int = 6, predictor=None):
    """Perturbation test: overwrite EVERYTHING after the origin with garbage; the model input must not change."""
    rng = np.random.default_rng(0)
    days = pd.date_range(test_range[0], test_range[1])
    picks = [days[i] for i in rng.choice(len(days), n_days, replace=False)] + [days[0], days[-1]]
    checked = 0
    for D in picks:
        O = origin_of(D)
        real, _ = build_contexts(masked, raw, D, cfg)
        bad_m, bad_r = {}, {}
        for hh in masked:
            for src, dst in ((masked, bad_m), (raw, bad_r)):
                m = src[hh].copy()
                t = (m.index.to_numpy()[:, None] + OFFSETS[None, :])
                v = m.to_numpy().copy()
                v[(t > O.to_datetime64()) & np.isfinite(v)] = 9999.0   # NaN pattern kept so the scored set is unchanged
                dst[hh] = pd.DataFrame(v, index=m.index)
        pert, _ = build_contexts(bad_m, bad_r, D, cfg)
        # note: selection of households uses day-D actuals only to decide WHICH series to score; compare common series
        a, b = real.sort_index(), pert.sort_index()
        assert a.index.equals(b.index), "context index changed when future data was perturbed"
        assert np.allclose(a["target"].to_numpy(), b["target"].to_numpy(), equal_nan=True), "context values depend on post-cutoff data"
        assert (a["target"].dropna() < 9000).all() and a.index.get_level_values("timestamp").max() == O
        if predictor is not None:
            pr = predictor.predict(a)["0.5"].to_numpy(); pp = predictor.predict(b)["0.5"].to_numpy()
            assert np.allclose(pr, pp), "predictions changed when post-cutoff data was perturbed"
        checked += 1
    print(f"cutoff verification passed for {checked} forecast days (context ends exactly at D-1 11:45; "
          f"post-cutoff data perturbed to 9999 without effect{'; predictions identical' if predictor is not None else ''})")


def to_interval_frame(preds: pd.DataFrame, masked: dict, model_name: str) -> pd.DataFrame:
    """Raw predictions -> scored interval frame with actuals and signed/absolute errors (Level-2/3 ready)."""
    parts = []
    for hh, g in preds.groupby("Household_ID"):
        mat = masked[hh]
        ts = (mat.index.to_numpy()[:, None] + OFFSETS[None, :]).ravel()
        parts.append(g.assign(actual=g["Timestamp"].map(pd.Series(mat.to_numpy().ravel(), index=ts))))
    iv = pd.concat(parts, ignore_index=True).dropna(subset=["actual", "prediction"])
    iv["model"] = model_name
    iv["horizon_step"] = ((iv["Timestamp"] - iv["forecast_origin"]) / pd.Timedelta(minutes=15)).astype(int)
    iv["signed_error"] = iv["prediction"] - iv["actual"]                  # >0 = over-forecast
    iv["abs_error"] = iv["signed_error"].abs()
    assert (iv["forecast_origin"] + pd.Timedelta(minutes=15) * iv["horizon_step"] == iv["Timestamp"]).all()
    assert (iv["forecast_origin"] == iv["forecast_date"] - pd.Timedelta(hours=12, minutes=15)).all()
    cols = ["Household_ID", "pv_status", "forecast_date", "forecast_origin", "Timestamp", "horizon_step",
            "actual", "prediction", "signed_error", "abs_error", "model"]
    return iv[[c for c in cols if c in iv.columns]]
