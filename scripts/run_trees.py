"""LightGBM and CatBoost on the same features and blocks as HGB.

Usage: .venv/bin/python -m scripts.run_trees
"""
import time

import catboost
import lightgbm as lgb
import numpy as np
import polars as pl

from utils.features import BASE_FEATURES, W0_FEATURES, W2_FEATURES, build_features, check_availability
from utils.splits import FIT_END, TUNE, save_preds

TRACKS = {
    "W0": BASE_FEATURES + W0_FEATURES,
    "W2_oracle": list(dict.fromkeys(BASE_FEATURES + W2_FEATURES)),
}


def fit_lightgbm(Xf, yf, Xt, yt, cat_idx):
    m = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0,
    )
    m.fit(Xf, yf, eval_set=[(Xt, yt)], categorical_feature=[cat_idx], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, m.best_iteration_, f"lightgbm {lgb.__version__}"


def fit_catboost(Xf, yf, Xt, yt, cat_idx):
    m = catboost.CatBoostRegressor(
        iterations=3000, learning_rate=0.1, depth=8, loss_function="RMSE", random_seed=0,
        early_stopping_rounds=50, verbose=False, thread_count=-1,
    )
    m.fit(catboost.Pool(Xf, yf, cat_features=[cat_idx]), eval_set=catboost.Pool(Xt, yt, cat_features=[cat_idx]))
    return m, m.get_best_iteration(), f"catboost {catboost.__version__}"


def main() -> None:
    df = build_features(FIT_END)
    check_availability(df)
    fit = df.filter(pl.col("D") <= FIT_END, pl.col("kwh").is_not_null())
    tune = df.filter(pl.col("D").is_between(*TUNE), pl.col("kwh").is_not_null())
    pred = df.filter(pl.col("D") >= TUNE[0])

    for algo, fitter in [("LightGBM", fit_lightgbm), ("CatBoost", fit_catboost)]:
        for track, feats in TRACKS.items():
            # station is categorical; pandas keeps the integer dtype both libraries expect
            to_pd = lambda d: d.select(feats).to_pandas().astype({"station": "int32"})
            t0 = time.time()
            m, iters, version = fitter(to_pd(fit), fit["kwh"].to_numpy(), to_pd(tune), tune["kwh"].to_numpy(), feats.index("station"))
            p = np.clip(m.predict(to_pd(pred)), 0, None)
            runtime = round(time.time() - t0, 1)
            name = f"{algo}_{track}"
            save_preds(name, pred.select("Household_ID", "hour").with_columns(pred=pl.Series(p)), {
                "family": "tree", "track": track, "inputs": "same features as HGB_" + track,
                "point": "conditional mean", "runtime_s": runtime, "device": "cpu", "version": version,
                "best_iteration": iters,
            })
            print(f"{name}: {iters} iterations, {runtime}s")


if __name__ == "__main__":
    main()
