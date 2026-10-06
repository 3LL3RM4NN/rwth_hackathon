"""Evaluate the trained ProLoaF models in the same terms as the LightGBM
baseline in ``src/forecast.py``: MAE/RMSE/MAPE in kWh on day-ahead forecasts
(origin = midnight, horizon 0-23h) over the held-out test split.

ProLoaF's own ``evaluate.py`` computes its benchmark metrics in *scaled*
([0,1] min-max) space and only plots two sample windows rather than scoring
the full test set, so its numbers aren't directly comparable to the LightGBM
report. This script instead:

1. Rebuilds the same held-out test set evaluate.py does (last
   ``1 - validation_split`` fraction, i.e. the same chronological last-20%
   convention as the LightGBM pipeline).
2. Runs the trained model over *every* sliding window in that test set in a
   single batched forward pass (``TimeSeriesData.make_data_loader`` with
   ``batch_size=None`` -- the same trick ``modelhandler.benchmark`` uses
   internally).
3. Keeps only the windows whose first forecast hour is local midnight, so
   each kept window is a genuine "forecast issued at midnight for the next
   24h" sample -- matching the LightGBM pipeline's origin definition instead
   of ProLoaF's default one-sample-per-hour sliding window.
4. Inverse-transforms the (scaled) median prediction and target back to kWh
   via the model's own fitted per-column MinMaxScaler before computing
   MAE/RMSE/MAPE, so the numbers are directly comparable to the LightGBM
   report's kWh-based table.

Also pulls PICP (prediction-interval coverage) from ProLoaF's native
NllGauss-quantile output -- unit-free, so usable as-is -- as a concrete
example of the uncertainty quantification LightGBM's plain point forecast
doesn't provide.
"""

from __future__ import annotations

import json
import os
from functools import partial

import numpy as np
import pandas as pd
import torch

import proloaf.datahandler as dh
import proloaf.metrics as pmetrics
import proloaf.modelhandler as mh
import proloaf.tensorloader as tl
from proloaf.confighandler import read_config
from src import aggregate

MAIN_PATH = "third_party/proloaf"


def load_test_predictions(name: str):
    par = read_config(model_name=name, main_path=MAIN_PATH)
    infile = os.path.join(MAIN_PATH, par["data_path"])
    df = pd.read_csv(infile, sep=";", index_col=0, parse_dates=True)

    model_path = os.path.join(MAIN_PATH, par["output_path"], f"{par['model_name']}.pkl")
    net = mh.ModelHandler.load_model(model_path, locate="cpu")

    _, test_df = dh.split(df, [par["validation_split"]])
    test_data = tl.TimeSeriesData(
        test_df,
        device="cpu",
        preparation_steps=[
            partial(dh.set_to_hours, freq=par.get("frequency", "1h"), timecolumn=par.get("timecolumn", "Time")),
            partial(dh.fill_if_missing, periodicity=par.get("periodicity", 24)),
            dh.add_cyclical_features,
            net.scalers.transform,
            partial(dh.add_onehot_features, timestep=par["frequency"]),
            partial(dh.add_onehot_daytype, country=par.get("country_code")),
            partial(dh.stack_features, map=par.get("stack_columns")),
            dh.check_continuity,
        ],
        **par,
    )

    loader = test_data.make_data_loader(
        encoder_features=net.encoder_features,
        decoder_features=net.decoder_features,
        batch_size=None,
        shuffle=False,
    )
    inputs_enc, inputs_enc_aux, inputs_dec, inputs_dec_aux, last_value, targets = next(iter(loader))

    with torch.no_grad():
        predictions = net.predict(
            inputs_enc=inputs_enc,
            inputs_enc_aux=inputs_enc_aux,
            inputs_dec=inputs_dec,
            inputs_dec_aux=inputs_dec_aux,
            last_value=last_value,
        )
        quantile_prediction = net.loss_metric.get_quantile_prediction(
            predictions=predictions,
            target=targets,
            inputs_enc=inputs_enc,
            inputs_enc_aux=inputs_enc_aux,
        )
        if isinstance(quantile_prediction, tuple):
            quantile_prediction = quantile_prediction[0]
        median = quantile_prediction.get_quantile(0.5)
        upper = quantile_prediction.select_upper_bound()
        lower = quantile_prediction.select_lower_bound()

        picp = pmetrics.Picp().from_quantiles(
            target=targets,
            quantile_prediction=quantile_prediction,
            avg_over="all",
            inputs_enc=inputs_enc,
            inputs_enc_aux=inputs_enc_aux,
        )

    history_horizon = par["history_horizon"]
    n_windows = targets.shape[0]
    # window i's target timestamps are test_data.data.index[history_horizon+i : +forecast_horizon]
    origin_hours = test_data.data.index[history_horizon : history_horizon + n_windows].hour
    midnight_mask = origin_hours == 0

    target_col = par["target_id"][0]
    scaler = net.scalers.scalers[target_col]

    def inverse(t: torch.Tensor) -> np.ndarray:
        flat = t.detach().numpy().reshape(-1, 1)
        return scaler.inverse_transform(flat).reshape(t.shape)

    y_true = inverse(targets)[midnight_mask]
    y_pred = inverse(median)[midnight_mask]
    # select_upper_bound()/select_lower_bound() keep a trailing singleton
    # quantile-index dimension that get_quantile(0.5) already drops.
    y_upper = inverse(upper.values.squeeze(-1))[midnight_mask]
    y_lower = inverse(lower.values.squeeze(-1))[midnight_mask]

    origins = test_data.data.index[history_horizon : history_horizon + n_windows][midnight_mask]

    return {
        "name": name,
        "n_households": len(aggregate.group_household_ids(name == "pv_group")),
        "n_day_ahead_windows": int(midnight_mask.sum()),
        "origins": origins,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_upper": y_upper,
        "y_lower": y_lower,
        "picp_scaled_benchmark": float(picp.squeeze().item()),
    }


def summarize(result: dict) -> dict:
    y_true = result["y_true"].reshape(-1)
    y_pred = result["y_pred"].reshape(-1)
    y_upper = result["y_upper"].reshape(-1)
    y_lower = result["y_lower"].reshape(-1)

    abs_err = np.abs(y_true - y_pred)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    mape = float((abs_err / np.abs(y_true)).mean())
    picp_kwh = float(((y_true >= y_lower) & (y_true <= y_upper)).mean())

    origins = result["origins"]
    return {
        "name": result["name"],
        "n_households": result["n_households"],
        "n_day_ahead_windows": result["n_day_ahead_windows"],
        "test_origin_range": [str(origins.min()), str(origins.max())],
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mae_per_household": mae / result["n_households"],
        "picp": picp_kwh,
    }


if __name__ == "__main__":
    summary = {}
    for name in ["pv_group", "non_pv_group"]:
        result = load_test_predictions(name)
        metrics = summarize(result)
        summary[name] = metrics
        print(f"\n=== {name} (ProLoaF) ===")
        print(f"households={metrics['n_households']}  day-ahead test windows={metrics['n_day_ahead_windows']}")
        print(f"test period: {metrics['test_origin_range']}")
        print(f"MAE={metrics['mae']:.2f} kWh/h  RMSE={metrics['rmse']:.2f} kWh/h  MAPE={metrics['mape']*100:.1f}%")
        print(f"MAE per household={metrics['mae_per_household']:.4f} kWh/h")
        print(f"95% prediction-interval coverage (PICP)={metrics['picp']*100:.1f}%")

        # Save per-window predictions for later inspection/plotting.
        n_windows, horizon = result["y_true"].shape[:2]
        rows = [
            {
                "origin": str(result["origins"][w]),
                "horizon": h,
                "y": float(result["y_true"][w, h, 0]),
                "pred": float(result["y_pred"][w, h, 0]),
                "pred_upper": float(result["y_upper"][w, h, 0]),
                "pred_lower": float(result["y_lower"][w, h, 0]),
            }
            for w in range(n_windows)
            for h in range(horizon)
        ]
        pd.DataFrame(rows).to_csv(f"reports/{name}_proloaf_test_predictions.csv", index=False)

    with open("reports/proloaf_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
