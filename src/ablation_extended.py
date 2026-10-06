"""Ablation of the candidate feature groups from ``src/features.py``.

Same protocol as ``src/ablation.py`` (same rows for every variant, verdict
from a chronological validation slice of the training days, test days as a
cross-check only, 95% interval from paired per-day differences) -- see that
module's docstring. The questions asked here are different, though:

1. **Add one group to the current model.** Current 23 features + one candidate
   group. Negative delta = the group reduces error on its own.
2. **Remove one group from the extended model.** Current features + *all*
   non-oracle candidate groups, minus one. Positive delta = the group still
   contributes once every other candidate is present; ~0 = redundant.
3. **Oracle weather.** Adds *measured* delivery-day weather. This is not
   available at bid time (see ``src/features.py``); it only bounds what a
   perfect weather forecast could add and is not a day-ahead result.
4. **Target transformation.** Same features, different training target:
   the residual to ``rolling_mean_same_timeofday_7d`` (added back at predict
   time) or ``log1p(y)``. Errors are always measured on the original kWh scale.

Rows are filtered on the *current* feature set only, so the baseline row
reproduces ``reports/forecast_metrics.json``; candidate features may be
missing on some rows (long lookbacks at the start of a series), which LightGBM
handles natively.

Outputs: ``reports/ablation_extended_results.csv`` and
``reports/ablation_extended_table.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src import ablation, features, forecast

RESIDUAL_BASELINE = "rolling_mean_same_timeofday_7d"

SECTION_REFERENCE = "Reference models"
SECTION_ADD = "Add one group to the current model"
SECTION_REMOVE = "Remove one group from the extended model"
SECTION_ORACLE = "Oracle weather (upper bound, not available at bid time)"
SECTION_TARGET = "Target transformation"
SECTION_NOTES = {
    SECTION_REFERENCE: "Δ is relative to the current model.",
    SECTION_ADD: "Δ is relative to the current model. Negative = the group helps.",
    SECTION_REMOVE: "Δ is relative to the extended model. Positive = the group still helps once all others are in.",
    SECTION_ORACLE: "Δ is relative to the current model. Uses measured weather of the delivery day itself.",
    SECTION_TARGET: "Δ is relative to the current model (raw kWh target).",
}


def _abs_errors(fit: pd.DataFrame, score: pd.DataFrame, columns: list[str], target: str = "level") -> pd.Series:
    model = LGBMRegressor(**forecast.MODEL_PARAMS)
    if target == "level":
        model.fit(fit[columns], fit["y"])
        pred = model.predict(score[columns])
    elif target == "residual":
        model.fit(fit[columns], fit["y"] - fit[RESIDUAL_BASELINE])
        pred = model.predict(score[columns]) + score[RESIDUAL_BASELINE]
    elif target == "log1p":
        model.fit(fit[columns], np.log1p(fit["y"]))
        pred = np.expm1(model.predict(score[columns]))
    else:
        raise ValueError(f"unknown target transformation: {target}")
    return (score["y"] - pred).abs()


def addition_verdict(delta_pct: float, ci_pct: float) -> str:
    if delta_pct + ci_pct < 0:
        return "helps"
    if delta_pct - ci_pct > 0:
        return "hurts"
    return "no clear effect"


def run_group(name: str) -> pd.DataFrame:
    print(f"\n=== Extended ablation for {name} ===")
    table, groups = features.build_extended_table(name)
    usable = table.dropna(subset=forecast.FEATURE_COLUMNS + ["y"])
    train, test = forecast.chronological_split(usable)
    fit, val = ablation.split_fit_validation(train)
    print(f"fit={len(fit)} rows  validation={len(val)} rows  test={len(test)} rows")

    current = forecast.FEATURE_COLUMNS
    candidate_groups = {label: cols for label, cols in groups.items() if label != features.ORACLE_GROUP}
    oracle = groups[features.ORACLE_GROUP]
    extended = current + [c for cols in candidate_groups.values() for c in cols]

    def errors(columns: list[str], target: str = "level") -> tuple[pd.Series, pd.Series]:
        return _abs_errors(fit, val, columns, target), _abs_errors(train, test, columns, target)

    rows = []

    def record(section, variant, columns, errs, reference=None, verdict=None) -> None:
        row = {
            "group": name,
            "section": section,
            "variant": variant,
            "n_features": len(columns),
            "val_mae": errs[0].mean(),
            "test_mae": errs[1].mean(),
        }
        if reference is not None:
            row["val_delta_pct"], row["val_ci_pct"] = ablation.delta_vs_baseline(reference[0], errs[0], val["origin"])
            row["test_delta_pct"], row["test_ci_pct"] = ablation.delta_vs_baseline(reference[1], errs[1], test["origin"])
            row["verdict"] = verdict(row["val_delta_pct"], row["val_ci_pct"])
            print(
                f"[{section}] {variant}: validation {row['val_delta_pct']:+.1f}% (±{row['val_ci_pct']:.1f})  "
                f"test {row['test_delta_pct']:+.1f}% (±{row['test_ci_pct']:.1f})"
            )
        rows.append(row)

    current_errs = errors(current)
    extended_errs = errors(extended)
    print(f"current model: validation MAE={current_errs[0].mean():.3f}  test MAE={current_errs[1].mean():.3f}")
    record(SECTION_REFERENCE, "current model", current, current_errs)
    record(SECTION_REFERENCE, "extended model (current + all candidate groups)", extended, extended_errs,
           current_errs, addition_verdict)  # fmt: skip

    for label, cols in candidate_groups.items():
        record(SECTION_ADD, label, current + cols, errors(current + cols), current_errs, addition_verdict)

    for label, cols in candidate_groups.items():
        kept = [c for c in extended if c not in cols]
        record(SECTION_REMOVE, label, kept, errors(kept), extended_errs, ablation.removal_verdict)

    record(SECTION_ORACLE, "current + oracle weather", current + oracle, errors(current + oracle),
           current_errs, addition_verdict)  # fmt: skip
    record(SECTION_ORACLE, "extended + oracle weather", extended + oracle, errors(extended + oracle),
           current_errs, addition_verdict)  # fmt: skip

    for label, columns in (("current", current), ("extended", extended)):
        for target, description in (("residual", f"residual to {RESIDUAL_BASELINE}"), ("log1p", "log1p(y)")):
            record(SECTION_TARGET, f"{label} features, target = {description}", columns, errors(columns, target),
                   current_errs, addition_verdict)  # fmt: skip

    return pd.DataFrame(rows)


def _markdown_table(rows: pd.DataFrame) -> str:
    lines = [
        "| Variant | # features | Val MAE | Δ val MAE | Test MAE | Δ test MAE | Verdict (from val) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in rows.iterrows():
        if pd.isna(r["val_delta_pct"]):
            lines.append(f"| {r['variant']} | {r['n_features']} | {r['val_mae']:.3f} | | {r['test_mae']:.3f} | | |")
            continue
        lines.append(
            f"| {r['variant']} | {r['n_features']} | {r['val_mae']:.3f} | "
            f"{r['val_delta_pct']:+.1f}% ± {r['val_ci_pct']:.1f} | {r['test_mae']:.3f} | "
            f"{r['test_delta_pct']:+.1f}% ± {r['test_ci_pct']:.1f} | {r['verdict']} |"
        )
    return "\n".join(lines)


def write_markdown(results: pd.DataFrame, path: str) -> None:
    parts = [
        "# Ablation of the candidate feature groups",
        "",
        "Generated by `python3 -m src.ablation_extended`; feature definitions and their assumptions are in",
        "`src/features.py`. MAE is in kWh per 15 min for the group sum.",
        "",
        "- **Val** = last 20% of the training days (model fitted on the earlier training days). **Test** = the",
        "  held-out test days (model fitted on all training days). The verdict uses validation only; test is a",
        "  cross-check and must not be used to pick features.",
        "- **±** is a 95% interval from paired per-day differences; it is somewhat too narrow because",
        "  consecutive days are correlated.",
        "- The **oracle** rows use measured weather of the delivery day. That is not known at bid time, so they",
        "  are an upper bound for a perfect weather forecast, not a day-ahead result.",
        "",
    ]
    for name in results["group"].drop_duplicates():
        parts += [f"## {name}", ""]
        group_rows = results[results["group"] == name]
        for section in group_rows["section"].drop_duplicates():
            parts += [f"### {section}", "", SECTION_NOTES[section], "",
                      _markdown_table(group_rows[group_rows["section"] == section]), ""]  # fmt: skip
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    results = pd.concat([run_group(name) for name in features.GROUPS], ignore_index=True)
    results.to_csv("reports/ablation_extended_results.csv", index=False)
    write_markdown(results, "reports/ablation_extended_table.md")
    print("\nWrote reports/ablation_extended_results.csv and reports/ablation_extended_table.md")
