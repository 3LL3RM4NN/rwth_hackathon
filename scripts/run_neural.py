"""Global deep models trained on the portfolio households (NeuralForecast, runs in .venv-nf on CPU).

Train on data up to the end of the tune block, with the tune block as validation for early stopping
(max 15 min training). Then, for every delivery day D, forecast 49 hours from the cutoff (end of D-2
local) using the 672 hours before it, and keep the hours of day D. Point forecast = conditional mean
(MSE loss).

Usage: .venv-nf/bin/python -m scripts.run_neural nhits W0 [--days 5]
"""
import argparse
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl

from scripts.run_foundation import CONTEXT, H, build_arrays
from utils.data import TZ
from utils.splits import TEST, TUNE, save_preds

MAX_TIME = "00:00:15:00"  # training budget per model and track


def covariates(T, TEMP, CLIM, CAL, station, sl, track):
    lh = T.dt.convert_time_zone(TZ).dt.hour().to_numpy()[sl]
    c = {"day_off": CAL["day_off"][sl], "hour_sin": np.sin(2 * np.pi * lh / 24), "hour_cos": np.cos(2 * np.pi * lh / 24)}
    if track == "W0":
        c["temp_clim"], c["temp_past"] = CLIM[station][sl], TEMP[station][sl]
    elif track == "W2_oracle":
        c["temp"] = TEMP[station][sl]
    return c


def exog_lists(model, track):
    if model == "patchtst" or track == "target-only":
        return [], []  # PatchTST in NeuralForecast takes no exogenous inputs
    futr = ["day_off", "hour_sin", "hour_cos"] + (["temp_clim"] if track == "W0" else ["temp"])
    hist = ["temp_past"] if track == "W0" and model != "deepar" else []  # DeepAR: no historical exogenous
    return futr, hist


def frame(members, T, Y, TEMP, CLIM, CAL, stations, sl, track, cols):
    parts = []
    ds = T.dt.replace_time_zone(None).to_numpy()[sl]
    for i, m in enumerate(members):
        y = Y[i, sl]
        d = {"unique_id": str(m), "ds": ds, "y": np.nan_to_num(y), "available_mask": (~np.isnan(y)).astype(np.float32)}
        cov = covariates(T, TEMP, CLIM, CAL, stations[i], sl, track)
        d.update({k: np.nan_to_num(cov[k]) for k in cols})
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True)


def build_model(name, futr, hist, max_steps=5000):
    from neuralforecast.losses.pytorch import MAE, MSE
    from neuralforecast.models import NHITS, TFT, DeepAR, PatchTST
    # step_size=24: one training window per day (forecasts are issued daily); small series batches keep the
    # unfolded windows (series length x 721 steps x features) within memory.
    common = dict(h=H, input_size=CONTEXT, max_steps=max_steps, val_check_steps=100, early_stop_patience_steps=5,
                  step_size=24, valid_batch_size=8, inference_windows_batch_size=256,
                  scaler_type="standard", random_seed=0, accelerator="cpu", max_time=MAX_TIME, valid_loss=MAE())
    if name == "nhits":
        return NHITS(futr_exog_list=futr, hist_exog_list=hist, loss=MSE(), batch_size=8, windows_batch_size=256, **common)
    if name == "tft":
        return TFT(futr_exog_list=futr, hist_exog_list=hist, loss=MSE(), hidden_size=64, batch_size=8, windows_batch_size=256, **common)
    if name == "patchtst":
        return PatchTST(loss=MSE(), batch_size=8, windows_batch_size=256, **common)
    if name == "deepar":
        from neuralforecast.losses.pytorch import DistributionLoss
        common["valid_loss"] = DistributionLoss("StudentT")
        return DeepAR(futr_exog_list=futr, loss=DistributionLoss("StudentT"), batch_size=8, windows_batch_size=256, **common)
    raise ValueError(name)


def main() -> None:
    from neuralforecast import NeuralForecast
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["nhits", "tft", "patchtst", "deepar"])
    ap.add_argument("track", choices=["W0", "W2_oracle", "target-only"])
    ap.add_argument("--days", type=int, default=None, help="only the first N forecast days (pilot)")
    args = ap.parse_args()

    members, T, idx, Y, TEMP, CLIM, CAL, stations = build_arrays()
    futr, hist = exog_lists(args.model, args.track)
    cols = futr + hist
    Tl = T.dt.convert_time_zone(TZ).dt.date().to_numpy()

    # Training data: everything up to the end of the tune block; the tune block is the validation set
    end = int(np.where(Tl == np.datetime64(TUNE[1]))[0][-1]) + 1
    val_size = end - int(np.where(Tl == np.datetime64(TUNE[0]))[0][0])
    train = frame(members, T, Y, TEMP, CLIM, CAL, stations, slice(0, end), args.track, cols)
    first = train[train["available_mask"] > 0].groupby("unique_id")["ds"].min()
    train = train[train["ds"] >= train["unique_id"].map(first)]  # start each series at its first observation

    t0 = time.time()
    nf = NeuralForecast(models=[build_model(args.model, futr, hist, max_steps=int(__import__("os").environ.get("NF_STEPS", 50)) if args.days else 5000)], freq="h")
    nf.fit(train, val_size=val_size)
    train_s = time.time() - t0
    print(f"trained {args.model} {args.track} in {train_s:.0f}s", flush=True)

    days = pl.date_range(TUNE[0], TEST[1], "1d", eager=True).to_list()[: args.days]
    out, t1 = [], time.time()
    for k, D in enumerate(days):
        cutoff = pl.Series([D - timedelta(days=1)]).cast(pl.Datetime("us")).dt.replace_time_zone(TZ).dt.convert_time_zone("UTC")[0]
        c = idx[cutoff]
        keep = np.where(Tl[c:c + H] == np.datetime64(D))[0]
        valid = [i for i in range(len(members)) if (~np.isnan(Y[i, c - CONTEXT:c])).sum() >= 24]
        sub = [members[i] for i in valid]
        hist_df = frame(sub, T, Y[valid], TEMP, CLIM, CAL, [stations[i] for i in valid], slice(c - CONTEXT, c), args.track, cols)
        futr_df = None
        if futr:
            futr_df = frame(sub, T, np.zeros((len(valid), len(T)), np.float32), TEMP, CLIM, CAL,
                            [stations[i] for i in valid], slice(c, c + H), args.track, futr).drop(columns=["y", "available_mask"])
        fc = nf.predict(df=hist_df, futr_df=futr_df)
        col = [x for x in fc.columns if x not in ("unique_id", "ds")][0]
        fc = pl.from_pandas(fc[["unique_id", "ds", col]]).with_columns(
            Household_ID=pl.col("unique_id").cast(pl.Int64), hour=pl.col("ds").dt.replace_time_zone("UTC"),
            pred=pl.col(col).clip(lower_bound=0).cast(pl.Float64),
        )
        day_hours = T[c:c + H].gather(keep)
        out.append(fc.filter(pl.col("hour").is_in(day_hours.implode())).select("Household_ID", "hour", "pred"))
        if k % 50 == 0:
            print(f"{D}: {k + 1}/{len(days)} days, {time.time() - t1:.0f}s", flush=True)

    res = pl.concat(out)
    runtime = round(train_s + time.time() - t1, 1)
    names = {"nhits": "NHITS", "tft": "TFT", "patchtst": "PatchTST", "deepar": "DeepAR"}
    name = f"{names[args.model]}_{args.track}"
    if args.days:
        print(f"pilot: {name} train {train_s:.0f}s, {len(days)} days predicted in {time.time() - t1:.0f}s")
        return
    import neuralforecast
    save_preds(name, res, {
        "family": "deep (trained)", "track": args.track, "inputs": f"672h target; futr={futr}; hist={hist}",
        "point": "conditional mean (MSE)" if args.model != "deepar" else "distribution mean (StudentT)",
        "runtime_s": runtime, "train_s": round(train_s, 1), "device": "cpu", "version": f"neuralforecast {neuralforecast.__version__}",
    })
    print(f"saved {name}: {res.height:,} rows, {runtime}s")


if __name__ == "__main__":
    main()
