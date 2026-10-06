"""Rolling D-1 11:45 forecasts for AutoGluon with known covariates (calendar / lagged temperature)."""
import time

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

from .covariates import ANCHOR_HOUR, TS_INDEX, Covariates
from .level1_pv import HORIZON, build_contexts, origin_of


def with_covariates(data: TimeSeriesDataFrame, cov: Covariates) -> TimeSeriesDataFrame:
    df = pd.DataFrame(data).copy()
    c = cov.frame(df.index)
    for n in c.columns:
        df[n] = c[n].to_numpy()
    return TimeSeriesDataFrame(df)


def future_covariates(data: TimeSeriesDataFrame, O: pd.Timestamp, cov: Covariates) -> TimeSeriesDataFrame:
    """Known covariates for the 144 steps after the origin (D-1 12:00 .. D 23:45) of every series in `data`."""
    items = data.item_ids
    steps = pd.date_range(O + pd.Timedelta(minutes=15), periods=HORIZON, freq="15min")
    idx = pd.MultiIndex.from_arrays([np.repeat(items.to_numpy(), HORIZON), np.tile(steps.to_numpy(), len(items))], names=["item_id", "timestamp"])
    kc = cov.frame(idx)
    assert idx.get_level_values(1).max() == O + pd.Timedelta(hours=36)
    return TimeSeriesDataFrame(kc)


def rolling_predict_cov(predictor, masked, raw, cov: Covariates, cfg: dict, days: pd.DatetimeIndex, log_every: int = 20) -> pd.DataFrame:
    out, t0 = [], time.time()
    for k, D in enumerate(days):
        data, O = build_contexts(masked, raw, D, cfg)
        if data is None:
            continue
        data = with_covariates(data, cov)
        pred = predictor.predict(data, known_covariates=future_covariates(data, O, cov))
        p = pred["0.5"] if "0.5" in pred.columns else pred["mean"]
        df = p.rename("prediction").reset_index().rename(columns={"item_id": "Household_ID", "timestamp": "Timestamp"})
        assert df["Timestamp"].min() == O + pd.Timedelta(minutes=15) and len(df) == HORIZON * data.num_items
        df = df[df["Timestamp"] >= D].copy()
        assert df["Timestamp"].min() == D and df["Timestamp"].max() == D + pd.Timedelta(hours=23, minutes=45)
        df["prediction"] = df["prediction"].clip(lower=0.0)
        df["forecast_date"], df["forecast_origin"] = D, O
        out.append(df)
        if (k + 1) % log_every == 0:
            print(f"  day {D.date()} ({k + 1}/{len(days)}), {time.time() - t0:.0f}s elapsed", flush=True)
    return pd.concat(out, ignore_index=True)


def verify_covariate_cutoff(temps, station_of, names, days, n_days=6, predictor=None, masked=None, raw=None, cfg=None):
    """Perturb every weather observation stamped after the origin (D-1 11:45) to 999: covariates for all t <= origin+36h (context and
    future steps) must be identical; optionally predictions must be identical too."""
    real = Covariates(temps, station_of, names)
    rng = np.random.default_rng(1)
    picks = [days[i] for i in rng.choice(len(days), min(n_days, len(days)), replace=False)] + [days[0], days[-1]]
    for D in picks:
        O = origin_of(D)
        bad = {st: s.where(s.index <= O, 999.0) for st, s in temps.items()}      # every observation stamped after 11:45 -> 999
        pert = Covariates(bad, station_of, names)
        upto = TS_INDEX <= O + pd.Timedelta(hours=36)
        for st in temps:
            if real.weather_names:
                a, b = real.tables[st][upto], pert.tables[st][upto]
                assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)), f"weather covariate depends on post-cutoff data ({st}, {D.date()})"
        if real.weather_names:                                  # the test must be able to fail: later timestamps DO change
            st0 = next(iter(temps)); later = TS_INDEX > O + pd.Timedelta(hours=36)      # day D+1 is anchored at D 11:00 > O, so it must change
            assert not np.allclose(np.nan_to_num(real.tables[st0][later], nan=-1), np.nan_to_num(pert.tables[st0][later], nan=-1)), "perturbation had no effect"
        if predictor is not None:
            data, O2 = build_contexts(masked, raw, D, cfg)
            pr = predictor.predict(with_covariates(data, real), known_covariates=future_covariates(data, O2, real))["0.5"].to_numpy()
            pp = predictor.predict(with_covariates(data, pert), known_covariates=future_covariates(data, O2, pert))["0.5"].to_numpy()
            assert np.allclose(pr, pp), "predictions changed when post-cutoff weather was perturbed"
    extra = "; predictions identical" if predictor is not None else ""
    print(f"covariate cutoff verification passed for {len(picks)} forecast days (all weather stamped after D-1 11:45, incl. all of forecast "
          f"day D, set to 999 without effect on any covariate for t <= D 23:45; anchor = D-1 {ANCHOR_HOUR}:00){extra}", flush=True)
