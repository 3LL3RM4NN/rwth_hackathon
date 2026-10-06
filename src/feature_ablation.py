"""Leave-one-feature-out ablation for the "Grouped, portfolio-wide" MAPE.

For each feature in ``forecast.FEATURE_COLUMNS``, retrains the ``pv_group``
and ``non_pv_group`` LightGBM models with that one feature removed, then
recomputes the combined-bid metric from ``forecast.py``'s "Did grouping
actually help?" analysis (sum predictions + actuals across both groups per
15-min step over their common held-out test window, then score MAPE on the
combined series) and reports how much it changes versus the full-feature
baseline.

The train/test *row set* is fixed once, from the full feature list's
``dropna`` (exactly matching ``forecast.py``'s own split) -- every ablation
round trains on the identical rows, only the model's input columns change.
This keeps the comparison clean: a feature's measured impact is entirely due
to its presence/absence, not to a side effect of removing it also dropping
fewer/more NaN rows.

Runs 2 * (len(FEATURE_COLUMNS) + 1) LightGBM fits (one baseline + one
per-feature-removed, per group) -- a few minutes, not instant.
"""

from __future__ import annotations

import json

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error

from src.forecast import (
    FEATURE_COLUMNS,
    build_supervised_table,
    chronological_split,
    load_group_15min,
)

GROUPS = ["pv_group", "non_pv_group"]


def prepare_group(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_group_15min(name)
    table = build_supervised_table(df)
    usable = table.dropna(subset=FEATURE_COLUMNS + ["y"])
    return chronological_split(usable)


def train_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> pd.Series:
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=0,
        verbosity=-1,
    )
    model.fit(train[columns], train["y"])
    return pd.Series(model.predict(test[columns]), index=test.index)


def portfolio_mape(
    pv_test: pd.DataFrame,
    pv_pred: pd.Series,
    nonpv_test: pd.DataFrame,
    nonpv_pred: pd.Series,
    common_start: pd.Timestamp,
) -> float:
    pv = pv_test[["origin", "target_time", "y"]].copy()
    pv["pred"] = pv_pred
    pv = pv[pv["origin"] >= common_start]

    nonpv = nonpv_test[["origin", "target_time", "y"]].copy()
    nonpv["pred"] = nonpv_pred
    nonpv = nonpv[nonpv["origin"] >= common_start]

    merged = pv.merge(nonpv, on="target_time", suffixes=("_pv", "_nonpv"))
    actual = merged["y_pv"] + merged["y_nonpv"]
    pred = merged["pred_pv"] + merged["pred_nonpv"]
    return float(mean_absolute_percentage_error(actual, pred))


if __name__ == "__main__":
    print("Preparing train/test tables for pv_group and non_pv_group...")
    pv_train, pv_test = prepare_group("pv_group")
    nonpv_train, nonpv_test = prepare_group("non_pv_group")
    common_start = max(pv_test["origin"].min(), nonpv_test["origin"].min())
    print(f"Common test window starts {common_start}")

    print(f"\n=== Baseline (all {len(FEATURE_COLUMNS)} features) ===")
    pv_pred = train_predict(pv_train, pv_test, FEATURE_COLUMNS)
    nonpv_pred = train_predict(nonpv_train, nonpv_test, FEATURE_COLUMNS)
    baseline_mape = portfolio_mape(pv_test, pv_pred, nonpv_test, nonpv_pred, common_start)
    print(f"Baseline portfolio-wide MAPE: {baseline_mape * 100:.3f}%")

    mape_by_feature = {}
    n = len(FEATURE_COLUMNS)
    for i, feature in enumerate(FEATURE_COLUMNS, start=1):
        columns = [c for c in FEATURE_COLUMNS if c != feature]
        print(f"\n[{i}/{n}] Leaving out '{feature}'...")
        pv_pred = train_predict(pv_train, pv_test, columns)
        nonpv_pred = train_predict(nonpv_train, nonpv_test, columns)
        mape = portfolio_mape(pv_test, pv_pred, nonpv_test, nonpv_pred, common_start)
        mape_by_feature[feature] = mape
        print(f"  portfolio-wide MAPE={mape * 100:.3f}%  (delta={(mape - baseline_mape) * 100:+.3f} pp)")

    # Sorted by impact: removing the feature at the top hurt MAPE the most
    # (most important feature); removing the one at the bottom helped MAPE
    # the most (actively unhelpful/noisy feature, if its delta is negative).
    ranked = sorted(mape_by_feature.items(), key=lambda kv: kv[1], reverse=True)

    print("\n=== Summary: leave-one-out impact on portfolio-wide MAPE, vs baseline ===")
    print(f"Baseline (all features): {baseline_mape * 100:.3f}%\n")
    print(f"{'Feature removed':45s} {'MAPE':>8s} {'Delta (pp)':>12s}")
    for feature, mape in ranked:
        delta = (mape - baseline_mape) * 100
        print(f"{feature:45s} {mape * 100:7.3f}% {delta:+11.3f}")

    with open("reports/feature_ablation.json", "w") as f:
        json.dump(
            {
                "common_test_window_start": str(common_start),
                "baseline_mape": baseline_mape,
                "mape_by_feature_removed": {
                    feature: {"mape": mape, "delta_pp": (mape - baseline_mape) * 100}
                    for feature, mape in ranked
                },
            },
            f,
            indent=2,
        )
    print("\nWrote reports/feature_ablation.json")
