"""Feature-group ablation for the day-ahead LightGBM model in ``src/forecast.py``.

For each group series, retrain the model with one *group* of features removed
and compare its error against the full-feature baseline. Answers "does this
feature group actually reduce held-out error?", which the split-count
``feature_importance`` in ``reports/forecast_metrics.json`` can't.

Design choices
--------------
- **Groups, not single features.** Many features here are near-duplicates of
  each other (``horizon`` is exactly ``hour*4 + minute/15``; temperature, dew
  point and humidity are strongly correlated), so removing just one of them
  looks harmless -- the model reads the same information off its neighbours.
  Redundant features are therefore removed together (see ``ABLATIONS``).
- **Selection on a validation slice, not on the test set.** The training days
  from ``forecast.chronological_split`` are split chronologically once more
  (earlier ``1 - VALIDATION_FRACTION`` to fit, the rest to validate). The
  verdict column is derived from this validation delta only. Choosing features
  by test error would make the reported test score optimistic.
- **Test delta as a cross-check only.** A second model per variant is fitted
  on the full training days and scored on the test days, exactly as
  ``forecast.train_and_evaluate`` does, so the baseline row reproduces the MAE
  in ``reports/forecast_metrics.json``. Validation and test cover different
  parts of the year, so a group whose effect flips sign between the two is
  season-dependent rather than reliably useful or useless.
- **Same rows for every variant.** Rows are filtered once on the *full*
  feature set, so a variant never gains extra rows merely because a removed
  lag feature was the one that was missing there.
- **Uncertainty.** The model is deterministic (no row/feature subsampling), so
  there is no seed noise, but a small delta can still be chance. Each delta
  gets a 95% interval from the paired per-day differences in mean absolute
  error (variant minus baseline). Days are autocorrelated, so this interval is
  somewhat too narrow -- treat borderline verdicts as "no clear effect".

Outputs: ``reports/ablation_results.csv`` (all numbers) and
``reports/ablation_table.md`` (one readable table per group).
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from src import forecast

GROUP_NAMES = ["pv_group", "non_pv_group", "all_households_group"]
VALIDATION_FRACTION = 0.2  # share of the *training* days held back for feature selection

TARGET_LAGS_ASOF_CUTOFF = [f"lag_{h}h" for h in forecast.LAG_HOURS_FROM_CUTOFF] + ["rolling_mean_24h_asof_cutoff"]
TARGET_SAME_TIMEOFDAY = ["rolling_mean_same_timeofday_7d"]
TIME_OF_DAY = ["hour", "minute", "horizon"]  # three encodings of the same 15-min-of-day index
DAY_OF_WEEK = ["dow", "is_weekend"]
MONTH = ["month"]


def _weather_columns(feat: str) -> list[str]:
    return [f"{feat}_lag_24h", f"{feat}_rolling_mean_24h_asof_cutoff"]


# Label -> feature columns removed in that variant.
ABLATIONS: dict[str, list[str]] = {
    "all target history": TARGET_LAGS_ASOF_CUTOFF + TARGET_SAME_TIMEOFDAY,
    "target lags as of cutoff": TARGET_LAGS_ASOF_CUTOFF,
    "same-time-of-day 7d mean": TARGET_SAME_TIMEOFDAY,
    "all weather": forecast.WEATHER_FEATURE_COLUMNS,
    **{f"weather: {feat}": _weather_columns(feat) for feat in forecast.WEATHER_FEATURES},
    "all calendar/time": TIME_OF_DAY + DAY_OF_WEEK + MONTH,
    "time of day (hour, minute, horizon)": TIME_OF_DAY,
    "day of week (dow, is_weekend)": DAY_OF_WEEK,
    "month": MONTH,
}

BASELINE_LABEL = "(none: full model)"


def split_fit_validation(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = train["origin"].drop_duplicates().sort_values()
    n_fit = int(len(origins) * (1 - VALIDATION_FRACTION))
    fit_origins = set(origins.iloc[:n_fit])
    in_fit = train["origin"].isin(fit_origins)
    return train[in_fit], train[~in_fit]


def _abs_errors(fit: pd.DataFrame, score: pd.DataFrame, columns: list[str]) -> pd.Series:
    model = LGBMRegressor(**forecast.LGBM_PARAMS)
    model.fit(fit[columns], fit["y"])
    return (score["y"] - model.predict(score[columns])).abs()


def delta_vs_baseline(base_err: pd.Series, err: pd.Series, origins: pd.Series) -> tuple[float, float]:
    """MAE change vs baseline and the half-width of its 95% interval, both in % of baseline MAE."""
    base_mae = base_err.mean()
    daily_diff = (err - base_err).groupby(origins).mean()
    half_width = 1.96 * daily_diff.std(ddof=1) / len(daily_diff) ** 0.5
    return 100 * (err.mean() - base_mae) / base_mae, 100 * half_width / base_mae


def removal_verdict(delta_pct: float, ci_pct: float) -> str:
    if delta_pct - ci_pct > 0:
        return "keep"
    if delta_pct + ci_pct < 0:
        return "drop candidate"
    return "no clear effect"


def run_group(name: str) -> pd.DataFrame:
    print(f"\n=== Ablation for {name} ===")
    table = forecast.build_supervised_table(forecast.load_group_15min(name), name)
    usable = table.dropna(subset=forecast.FEATURE_COLUMNS + ["y"])
    train, test = forecast.chronological_split(usable)
    fit, val = split_fit_validation(train)
    print(f"fit={len(fit)} rows  validation={len(val)} rows  test={len(test)} rows")

    base_val = _abs_errors(fit, val, forecast.FEATURE_COLUMNS)
    base_test = _abs_errors(train, test, forecast.FEATURE_COLUMNS)
    rows = [
        {
            "group": name,
            "removed": BASELINE_LABEL,
            "n_features_removed": 0,
            "val_mae": base_val.mean(),
            "test_mae": base_test.mean(),
        }
    ]
    print(f"baseline: validation MAE={base_val.mean():.3f}  test MAE={base_test.mean():.3f}")

    for label, removed in ABLATIONS.items():
        columns = [c for c in forecast.FEATURE_COLUMNS if c not in removed]
        val_err = _abs_errors(fit, val, columns)
        test_err = _abs_errors(train, test, columns)
        val_delta, val_ci = delta_vs_baseline(base_val, val_err, val["origin"])
        test_delta, test_ci = delta_vs_baseline(base_test, test_err, test["origin"])
        rows.append(
            {
                "group": name,
                "removed": label,
                "n_features_removed": len(removed),
                "val_mae": val_err.mean(),
                "val_delta_pct": val_delta,
                "val_ci_pct": val_ci,
                "test_mae": test_err.mean(),
                "test_delta_pct": test_delta,
                "test_ci_pct": test_ci,
                "verdict": removal_verdict(val_delta, val_ci),
            }
        )
        print(f"without {label}: validation {val_delta:+.1f}% (±{val_ci:.1f})  test {test_delta:+.1f}% (±{test_ci:.1f})")

    return pd.DataFrame(rows)


def _markdown_table(group_rows: pd.DataFrame) -> str:
    baseline = group_rows[group_rows["removed"] == BASELINE_LABEL]
    variants = group_rows[group_rows["removed"] != BASELINE_LABEL].sort_values("val_delta_pct", ascending=False)
    lines = [
        "| Features removed | # | Val MAE | Δ val MAE | Test MAE | Δ test MAE | Verdict (from val) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in baseline.iterrows():
        lines.append(f"| {r['removed']} | 0 | {r['val_mae']:.3f} | | {r['test_mae']:.3f} | | |")
    for _, r in variants.iterrows():
        lines.append(
            f"| {r['removed']} | {r['n_features_removed']} | {r['val_mae']:.3f} | "
            f"{r['val_delta_pct']:+.1f}% ± {r['val_ci_pct']:.1f} | {r['test_mae']:.3f} | "
            f"{r['test_delta_pct']:+.1f}% ± {r['test_ci_pct']:.1f} | {r['verdict']} |"
        )
    return "\n".join(lines)


def write_markdown(results: pd.DataFrame, path: str) -> None:
    parts = [
        "# Feature-group ablation",
        "",
        "Generated by `python3 -m src.ablation`. Each row retrains the day-ahead LightGBM model with one",
        "feature group removed. MAE is in kWh per 15 min for the group sum.",
        "",
        "- **Δ MAE** is the change relative to the full model: positive means the model gets *worse* without",
        "  the group (the group helps), negative means it gets *better* without it.",
        "- **±** is a 95% interval from paired per-day differences; it is somewhat too narrow because",
        "  consecutive days are correlated.",
        f"- **Val** = last {VALIDATION_FRACTION:.0%} of the training days (model fitted on the earlier training days).",
        "  **Test** = the held-out test days (model fitted on all training days). The verdict uses validation",
        "  only; test is a cross-check and must not be used to pick features.",
        "- **Verdict**: `keep` = removing it significantly hurts; `drop candidate` = removing it significantly",
        "  helps; `no clear effect` = the interval includes zero.",
        "- Groups overlap (e.g. `all weather` contains each `weather: ...` row), so deltas do not add up.",
        "",
    ]
    for name in results["group"].drop_duplicates():
        parts += [f"## {name}", "", _markdown_table(results[results["group"] == name]), ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    unknown = {c for removed in ABLATIONS.values() for c in removed} - set(forecast.FEATURE_COLUMNS)
    assert not unknown, f"ablation groups reference non-feature columns: {unknown}"

    results = pd.concat([run_group(name) for name in GROUP_NAMES], ignore_index=True)
    results.to_csv("reports/ablation_results.csv", index=False)
    write_markdown(results, "reports/ablation_table.md")
    print("\nWrote reports/ablation_results.csv and reports/ablation_table.md")
