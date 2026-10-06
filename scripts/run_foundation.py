"""Zero-shot pretrained forecasting models under the shared contract (runs in .venv-fm).

For every delivery day D and portfolio household: context = the 672 hours (28 days) before the cutoff
(end of D-2 local), horizon = 49 hours from the cutoff, of which the hours of local day D are kept.
Tracks: W0 = past temperature + climatology and calendar for the future; W2_oracle = actual temperature
on the horizon. Point forecast = mean integrated from the model's quantile grid (approximation).

Usage: .venv-fm/bin/python -m scripts.run_foundation chronos2 W0 [--days 7]
"""
import argparse
import time
from datetime import timedelta

import holidays
import numpy as np
import polars as pl

from utils.data import PROCESSED, TZ
from utils.features import build_features
from utils.splits import FIT_END, TEST, TUNE, cohort, save_preds

CONTEXT, H = 672, 49


def build_arrays():
    df = build_features(FIT_END)
    members = cohort(df)
    hh = pl.read_parquet(PROCESSED / "households.parquet").filter(pl.col("Household_ID").is_in(members)).sort("Household_ID")
    stations = sorted(hh["Weather_ID"].unique().to_list())

    t0, t1 = df["hour"].min(), df["hour"].max() + timedelta(hours=H + 24)
    T = pl.datetime_range(t0, t1, "1h", time_zone="UTC", eager=True).alias("hour")
    idx = {h: i for i, h in enumerate(T.to_list())}

    # Target matrix (members x timeline), NaN where missing or outside a household's grid
    Y = np.full((len(members), len(T)), np.nan, dtype=np.float32)
    sub = df.filter(pl.col("Household_ID").is_in(members)).select("Household_ID", "hour", "kwh")
    row = {m: i for i, m in enumerate(members)}
    r = np.array([row[m] for m in sub["Household_ID"].to_list()])
    c = np.array([idx[h] for h in sub["hour"].to_list()])
    Y[r, c] = sub["kwh"].fill_null(np.nan).to_numpy()

    # Station weather on the full timeline + fit-period climatology (same definition as features.py)
    w = pl.read_parquet(PROCESSED / "weather.parquet").with_columns(local=pl.col("hour").dt.convert_time_zone(TZ))
    w = w.with_columns(D=pl.col("local").dt.date(), lhour=pl.col("local").dt.hour().cast(pl.Int8), month=pl.col("local").dt.month())
    clim = w.filter(pl.col("D") <= FIT_END).group_by("Weather_ID", "month", "lhour").agg(clim=pl.col("temp").mean())
    grid = T.to_frame().with_columns(local=pl.col("hour").dt.convert_time_zone(TZ))
    grid = grid.with_columns(D=pl.col("local").dt.date(), lhour=pl.col("local").dt.hour().cast(pl.Int8), month=pl.col("local").dt.month())
    TEMP, CLIM = {}, {}
    for s in stations:
        g = grid.join(w.filter(pl.col("Weather_ID") == s).select("hour", "temp"), on="hour", how="left").join(
            clim.filter(pl.col("Weather_ID") == s).drop("Weather_ID"), on=["month", "lhour"], how="left")
        TEMP[s], CLIM[s] = g["temp"].to_numpy().astype(np.float32), g["clim"].to_numpy().astype(np.float32)

    ch = set(holidays.Switzerland(years=range(2019, 2025)).keys())
    # Day off = weekend or Swiss public holiday. Hour-of-day is left to the model (it sees the daily cycle in
    # the target); every extra covariate is an extra variate and costs ~2 s per day of 255 forecasts.
    CAL = {"day_off": ((grid["local"].dt.weekday() >= 6) | grid["D"].is_in(list(ch))).to_numpy().astype(np.float32)}
    station_of = dict(zip(hh["Household_ID"], hh["Weather_ID"]))
    return members, T, idx, Y, TEMP, CLIM, CAL, [station_of[m] for m in members]


def make_inputs(track, Y, TEMP, CLIM, CAL, stations, c):
    """One input dict per household for a cutoff at timeline index c."""
    ctx, fut = slice(c - CONTEXT, c), slice(c, c + H)
    items = []
    for i, s in enumerate(stations):
        past = {k: v[ctx] for k, v in CAL.items()}
        future = {k: v[fut] for k, v in CAL.items()}
        past["temp"] = TEMP[s][ctx]
        if track == "W0":
            past["temp_clim"], future["temp_clim"] = CLIM[s][ctx], CLIM[s][fut]
        elif track == "W2_oracle":  # actual temperature over the horizon
            future["temp"] = TEMP[s][fut]
        else:  # target-only
            past, future = {}, {}
        items.append({"target": Y[i, ctx], "past_covariates": past, "future_covariates": future})
    return items


def quantile_mean(q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """E[Y] ~ integral of the quantile function: trapezoid between levels, flat tails beyond the extremes."""
    inner = ((q[..., 1:] + q[..., :-1]) / 2 * np.diff(levels)).sum(-1)
    return inner + levels[0] * q[..., 0] + (1 - levels[-1]) * q[..., -1]


class Chronos2:
    checkpoint = "amazon/chronos-2"

    def __init__(self):
        import chronos
        from chronos import Chronos2Pipeline
        self.pipe = Chronos2Pipeline.from_pretrained(self.checkpoint, device_map="mps")
        self.levels = np.array(self.pipe.quantiles, dtype=np.float64)
        self.version = f"chronos-forecasting {chronos.__version__}, {self.checkpoint}"

    def predict(self, items):
        out = self.pipe.predict(items, prediction_length=H, batch_size=256, cross_learning=False)
        return np.stack([o[0].float().cpu().numpy() for o in out])  # (B, n_quantiles, H)


class TiRex2:
    checkpoint = "NX-AI/TiRex-2"

    def __init__(self):
        import tirex2
        self.fm = tirex2.load_model(self.checkpoint, device="mps")
        self.levels = np.array(self.fm._quantile_levels(), dtype=np.float64)
        self.version = f"tirex-2 (tirex2 package), {self.checkpoint}"

    def predict(self, items):
        import torch
        from tirex2 import TimeseriesType
        series = []
        for it in items:
            fut_keys = list(it["future_covariates"])
            past_only = [k for k in it["past_covariates"] if k not in fut_keys]
            t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
            series.append(TimeseriesType(
                target=t(it["target"])[None],
                past_covariates=torch.stack([t(it["past_covariates"][k]) for k in past_only]) if past_only else None,
                future_covariates=torch.stack([torch.cat([t(it["past_covariates"][k]), t(it["future_covariates"][k])]) for k in fut_keys]),
            ))
        out = self.fm.forecast(series, prediction_length=H, output_type="numpy", batch_size=256)
        return np.stack([o[0] for o in out])  # (B, n_quantiles, H)


def interp_past(a: np.ndarray) -> np.ndarray:
    """Fill gaps by linear interpolation between observed values (only values already in the array)."""
    a = np.asarray(a, dtype=np.float32)
    ok = ~np.isnan(a)
    if ok.all() or not ok.any():
        return np.nan_to_num(a)
    i = np.arange(len(a))
    return np.interp(i, i[ok], a[ok]).astype(np.float32)


class TimesFM3:
    """Research-only comparison: the downloaded TimesFM-3 weights are licensed non-commercial, non-production."""
    checkpoint = "google/timesfm-3.0-pytorch"

    def __init__(self):
        from timesfm3.mlx import TimesFM3Forecaster
        self.f = TimesFM3Forecaster.from_pretrained(self.checkpoint)
        self.levels = np.arange(1, 10) / 10
        self.version = f"timesfm 3.0.2 (MLX backend), {self.checkpoint}, non-commercial licence"

    def predict(self, items):
        ctx, po, pf = [], [], []
        for it in items:
            fut_keys = list(it["future_covariates"])
            past_only = [k for k in it["past_covariates"] if k not in fut_keys]
            ctx.append(interp_past(it["target"])[None])
            po.append(np.stack([interp_past(it["past_covariates"][k]) for k in past_only]) if past_only else None)
            pf.append(np.stack([np.concatenate([interp_past(it["past_covariates"][k]), interp_past(it["future_covariates"][k])]) for k in fut_keys]))
        outs = self.f.predict_batch(ctx, horizon=H, past_only_covariates=po if any(p is not None for p in po) else None,
                                    past_future_covariates=pf, return_quantiles=True)
        return np.stack([np.asarray(o.quantiles)[0].T for o in outs])  # (B, 9, H)


class Toto2:
    """Target-only: Toto 2.0 has no exogenous-variable support."""
    checkpoint = "Datadog/Toto-2.0-22m"

    def __init__(self):
        import torch
        from toto2 import Toto2Model
        self.torch = torch
        self.model = Toto2Model.from_pretrained(self.checkpoint).to("mps").eval()
        self.levels = np.arange(1, 10) / 10
        self.version = f"toto-models 1.0.0 (toto2), {self.checkpoint}"

    def predict(self, items):
        torch = self.torch
        y = np.stack([np.asarray(it["target"], dtype=np.float32) for it in items])[:, None, :]  # (B, 1, T)
        mask = ~np.isnan(y)
        inputs = {
            "target": torch.as_tensor(np.nan_to_num(y), device="mps"),
            "target_mask": torch.as_tensor(mask, device="mps"),
            "series_ids": torch.zeros((len(items), 1), dtype=torch.long, device="mps"),
        }
        with torch.no_grad():
            q = self.model.forecast(inputs, horizon=H, decode_block_size=768, has_missing_values=True)  # (9, B, 1, H)
        return q[:, :, 0, :].permute(1, 0, 2).float().cpu().numpy()  # (B, 9, H)


ADAPTERS = {"chronos2": Chronos2, "tirex2": TiRex2, "timesfm3": TimesFM3, "toto2": Toto2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["chronos2", "tirex2", "timesfm3", "toto2"])
    ap.add_argument("track", choices=["W0", "W2_oracle", "target-only"])
    ap.add_argument("--days", type=int, default=None, help="only the first N days (timing pilot)")
    args = ap.parse_args()

    members, T, idx, Y, TEMP, CLIM, CAL, stations = build_arrays()
    model = ADAPTERS[args.model]()
    days = pl.date_range(TUNE[0], TEST[1], "1d", eager=True).to_list()[: args.days]
    Tl = T.dt.convert_time_zone(TZ).dt.date().to_numpy()

    out, t0 = [], time.time()
    for k, D in enumerate(days):
        cutoff = pl.Series([D - timedelta(days=1)]).cast(pl.Datetime("us")).dt.replace_time_zone(TZ).dt.convert_time_zone("UTC")[0]
        c = idx[cutoff]
        keep = np.where(Tl[c:c + H] == np.datetime64(D))[0]  # select day D by timestamp (23/24/25 hours)
        assert len(keep) in (23, 24, 25) and keep[-1] < H
        # Households with < 24 observed hours in the context get no forecast (leaderboard falls back to Blend)
        valid = np.where((~np.isnan(Y[:, c - CONTEXT:c])).sum(1) >= 24)[0]
        items = make_inputs(args.track, Y, TEMP, CLIM, CAL, stations, c)
        q = model.predict([items[i] for i in valid])[:, :, keep]  # (B, Q, hours)
        L = model.levels
        mean = np.clip(quantile_mean(np.moveaxis(q, 1, -1), L), 0, None)
        qi = {lv: np.clip(q[:, int(np.argmin(np.abs(L - lv)))], 0, None) for lv in (0.1, 0.5, 0.9)}
        hours = T[c:c + H].gather(keep)
        out.append(pl.DataFrame({
            "Household_ID": np.repeat(np.array(members)[valid], len(keep)),
            "hour": pl.concat([hours] * len(valid)),
            "pred": mean.ravel(), "q0.1": qi[0.1].ravel(), "q0.5": qi[0.5].ravel(), "q0.9": qi[0.9].ravel(),
        }))
        if k % 20 == 0:
            print(f"{D}: {k + 1}/{len(days)} days, {time.time() - t0:.0f}s", flush=True)

    runtime = round(time.time() - t0, 1)
    res = pl.concat(out)
    name = f"{ {'chronos2': 'Chronos2', 'tirex2': 'TiRex2', 'timesfm3': 'TimesFM3', 'toto2': 'Toto2'}[args.model]}_{args.track}"
    if args.days:
        print(f"pilot: {len(days)} days in {runtime}s -> full run ~{runtime / len(days) * 481 / 60:.0f} min")
        return
    save_preds(name, res, {
        "family": "foundation (zero-shot)", "track": args.track,
        "inputs": "672h target only" if args.track == "target-only" else
                  "672h target + temperature + day-off flag; " + ("climatology future temp" if args.track == "W0" else "actual future temp"),
        "point": "quantile-integrated mean (approx.)", "runtime_s": runtime, "device": "mps", "version": model.version,
    })
    print(f"saved {name}: {res.height:,} rows, {runtime}s")


if __name__ == "__main__":
    main()
