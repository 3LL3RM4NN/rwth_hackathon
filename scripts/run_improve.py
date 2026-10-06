"""Improvement experiment for the tree models (exploratory follow-up: the test year was seen before).

Predeclared protocol:
1. Single-change ablations of the extra feature groups on LightGBM, scored on the TUNE block only
   (portfolio nMAE, same cohort and masks as the leaderboard).
2. The variant with the lowest tune nMAE is selected; LightGBM and CatBoost are refit on it.
3. A CatBoost + Toto 2.0 blend weight (0, 0.1, ..., 1) is chosen on the tune block.
4. The test year is scored once afterwards by scripts.leaderboard.

Usage: .venv/bin/python -m scripts.run_improve
"""
import time

import numpy as np
import polars as pl

from scripts.run_trees import fit_catboost, fit_lightgbm
from utils.evaluate import point_metrics, portfolio
from utils.features import BASE_FEATURES, W0_FEATURES, build_features, check_availability
from utils.features_extra import PEER_X, RECENT_X, WEATHER_X, add_extra_features, check_extra_availability
from utils.splits import FIT_END, TUNE, cohort, load_all_preds, save_preds

VARIANTS = {
    "base": [],
    "+weather": WEATHER_X,
    "+recent": RECENT_X,
    "+peer": PEER_X,
    "+weather+recent": WEATHER_X + RECENT_X,
    "+all": WEATHER_X + RECENT_X + PEER_X,
}


def tune_nmae(frame: pl.DataFrame, members: list[int], col: str) -> float:
    port = portfolio(frame, members, [col])
    return point_metrics(port["y"].to_numpy(), port[col].to_numpy())["nMAE_%"]


def main() -> None:
    df = build_features(FIT_END)
    check_availability(df)
    df = add_extra_features(df, FIT_END)
    check_extra_availability(df)
    members = cohort(df)

    fit = df.filter(pl.col("D") <= FIT_END, pl.col("kwh").is_not_null())
    tune = df.filter(pl.col("D").is_between(*TUNE), pl.col("kwh").is_not_null())
    tune_c = df.filter(pl.col("D").is_between(*TUNE), pl.col("Household_ID").is_in(members))

    def run(fitter, feats, rows):
        to_pd = lambda d: d.select(feats).to_pandas().astype({"station": "int32"})
        m, iters, version = fitter(to_pd(fit), fit["kwh"].to_numpy(), to_pd(tune), tune["kwh"].to_numpy(), feats.index("station"))
        return np.clip(m.predict(to_pd(rows)), 0, None), iters, version

    # 1. Ablations on LightGBM, tune block only
    scores = {}
    for name, extra in VARIANTS.items():
        feats = BASE_FEATURES + W0_FEATURES + extra
        p, iters, _ = run(fit_lightgbm, feats, tune_c)
        scores[name] = tune_nmae(tune_c.with_columns(pred=pl.Series(p)), members, "pred")
        print(f"LightGBM {name:18s} tune portfolio nMAE {scores[name]:.2f}%  ({iters} iterations)", flush=True)
    best = min(scores, key=scores.get)
    feats = BASE_FEATURES + W0_FEATURES + VARIANTS[best]
    print(f"selected: {best}")

    # 2. Refit LightGBM and CatBoost on the selected features, forecast every row from the tune block on
    pred = df.filter(pl.col("D") >= TUNE[0])
    for algo, fitter in [("LightGBM", fit_lightgbm), ("CatBoost", fit_catboost)]:
        t0 = time.time()
        p, iters, version = run(fitter, feats, pred)
        save_preds(f"{algo}_W0_v2", pred.select("Household_ID", "hour").with_columns(pred=pl.Series(p)), {
            "family": "tree (v2, follow-up)", "track": "W0", "inputs": f"HGB_W0 features {best}",
            "point": "conditional mean", "runtime_s": round(time.time() - t0, 1), "device": "cpu", "version": version,
            "best_iteration": iters, "selected_on": "tune block", "tune_scores": scores,
        })
        print(f"saved {algo}_W0_v2 ({iters} iterations)", flush=True)

    # 3. CatBoost v2 + Toto 2.0 blend, weight chosen on the tune block
    preds = load_all_preds()
    cb, toto = preds["CatBoost_W0_v2"][0], preds["Toto2_target-only"][0]
    both = (pred.select("Household_ID", "hour", "D", "lhour", "kwh")
            .join(cb.select("Household_ID", "hour", cb=pl.col("pred")), on=["Household_ID", "hour"], how="left")
            .join(toto.select("Household_ID", "hour", toto=pl.col("pred")), on=["Household_ID", "hour"], how="left")
            .with_columns(pl.col("toto").fill_null(pl.col("cb"))))
    bt = both.filter(pl.col("D").is_between(*TUNE), pl.col("Household_ID").is_in(members))
    wscore = {w: tune_nmae(bt.with_columns(pred=(1 - w) * pl.col("cb") + w * pl.col("toto")), members, "pred")
              for w in np.round(np.arange(0, 1.01, 0.1), 1)}
    w = min(wscore, key=wscore.get)
    print("blend tune nMAE by Toto weight:", {k: round(v, 2) for k, v in wscore.items()}, "-> w =", w)
    save_preds("CatBoostv2_Toto_blend", both.select("Household_ID", "hour", pred=(1 - w) * pl.col("cb") + w * pl.col("toto")), {
        "family": "ensemble (follow-up)", "track": "W0", "inputs": f"{1 - w:.1f} x CatBoost_W0_v2 + {w:.1f} x Toto2 (weight from tune block)",
        "point": "weighted mean", "runtime_s": 0, "device": "-", "version": "-",
    })


if __name__ == "__main__":
    main()
