"""Build 15-minute-resolution, group-level aggregated consumption + weather
series for the two Level-1 groups (PV / non-PV).

Group membership: the *surveyed* ``Installation_HasPVSystem`` flag
(`pv_detection.py`'s validation target) where a household was actually
surveyed, falling back to that module's **detector output**
(``reports/pv_detector_scores.csv``, column ``detector_pred_pv``) for the 165
households that were never surveyed. This grows the PV group from 131 to 158
households (+27 detected) and the no-PV group from 114 to 252 (+138
detected) -- every one of the 410 households on disk now lands in one group
or the other, none excluded. The detector's out-of-fold validation numbers
(`pv_detection.py`/`reports/pv_detection_metrics.json`: 0.937 ROC AUC, 89.4%
accuracy) are the honesty check on how much noise this fallback likely adds
versus using the ground-truth flag alone.

Three things have to be handled explicitly, per the task brief:

1. **Partial household coverage.** Households were instrumented at different
   times over 2019-2024, so the number of households reporting at a given
   timestamp ramps up over the series and is not constant -- by mid-2021 only
   ~20-35% of each group's eventual households had even started. Forecasting
   is done on ``kWh_mean_per_active_household`` (the group sum divided by
   however many households are reporting at that timestamp) rather than the
   raw sum specifically *because* that ratio is far more stable across a
   changing household count than the sum is. The window is still trimmed to
   where at least ``MIN_HOUSEHOLDS`` are reporting -- an *absolute* floor, not
   a fraction of the group's eventual size -- since the ratio is still noisy
   when only a handful of households are behind it; this floor is low enough
   to recover roughly a year and a half of additional history (back to
   ~2021) that an earlier, sum-based version of this pipeline discarded
   entirely. The active-household count is kept as a column so downstream
   code (``forecast.py``) can rescale predicted per-household averages back
   to group totals using the *actual* historical count for that timestamp.
2. **Multi-station weather.** A group's households are spread across several
   of the 8 weather stations. Group-level weather features are a
   household-count-weighted average across the stations actually used by that
   group's households, renormalised over whichever stations have a given
   variable populated (e.g. only 5/8 stations report sunshine duration).
3. **Resolution mismatch.** Smart-meter consumption is native 15-minute
   resolution; weather is hourly. Rather than downsampling consumption to
   match weather (as an earlier version of this pipeline did), we keep
   consumption at its native 15-min resolution and upsample the (already
   station-weighted) hourly weather to 15-min via time-based linear
   interpolation (``build_group_weather``). This is a real simplification for
   the two cumulative-style columns (``Precipitation_total_hourly``,
   ``Sunshine_duration_hourly`` are *hourly totals/durations*, not
   instantaneous readings) -- interpolating them linearly implicitly treats
   the hourly total as if it were a smoothly-varying instantaneous rate, which
   isn't physically exact, but no sub-hourly breakdown of those columns exists
   in this dataset, so there's no more-correct disaggregation available either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_loading as dl
from src.pv_features import household_resampled_total

MIN_HOUSEHOLDS = 30  # absolute floor: below this, the per-household average is too
                      # noisy (dominated by a handful of households' idiosyncrasies)
                      # to trust as a representative group-level statistic.
CONSUMPTION_FREQ = "15min"  # native smart-meter resolution; weather is upsampled to match
WEATHER_COLUMNS = [
    "Temperature_avg_hourly",
    "DewPoint_hourly",
    "Humidity_avg_hourly",
    "Precipitation_total_hourly",
    "Sunshine_duration_hourly",
    "WindSpeed_hourly",
]


def household_pv_labels() -> pd.Series:
    """Household_ID -> bool PV label: the surveyed flag where known, else
    ``pv_detection.py``'s detector output. Requires
    ``python3 -m src.pv_detection`` to have been run first (it's the
    pipeline's documented run order)."""
    scores = pd.read_csv("reports/pv_detector_scores.csv", index_col=0, dtype={"Household_ID": str})
    known = scores["Installation_HasPVSystem"].isin([True, False])
    return scores["Installation_HasPVSystem"].where(known, scores["detector_pred_pv"]).astype(bool)


def group_household_ids(pv: bool | None) -> list[str]:
    """pv=True/False selects the PV/non-PV group (surveyed flag, falling back
    to the detector for unsurveyed households -- see module docstring);
    pv=None selects the union of both, i.e. all 410 households -- used to
    build the single-ungrouped-model comparison baseline in the report."""
    labels = household_pv_labels()
    mask = pd.Series(True, index=labels.index) if pv is None else labels == pv
    ids = labels.index[mask].tolist()
    # Only keep households that actually have a 15-min file on disk.
    on_disk = set(dl.household_ids())
    return [h for h in ids if h in on_disk]


def build_group_consumption(household_ids: list[str]) -> pd.DataFrame:
    """Per-15-min summed + per-household-average consumption, and the
    active-household count, for a group."""
    series = {}
    n = len(household_ids)
    print(f"Loading+resampling 15-min data for {n} households...")
    for i, hid in enumerate(household_ids, start=1):
        series[hid] = household_resampled_total(hid, freq=CONSUMPTION_FREQ)
        if i % 50 == 0 or i == n:
            print(f"  {i}/{n} households done")
    wide = pd.DataFrame(series).sort_index()
    active_count = wide.notna().sum(axis=1)
    group_sum = wide.sum(axis=1, min_count=1)
    out = pd.DataFrame(
        {
            "kWh_total_group_sum": group_sum,
            "kWh_mean_per_active_household": group_sum / active_count,
            "active_household_count": active_count,
        }
    )
    return out


def restrict_to_minimum_household_window(df: pd.DataFrame, min_households: int = MIN_HOUSEHOLDS) -> pd.DataFrame:
    stable = df.index[df["active_household_count"] >= min_households]
    if stable.empty:
        raise ValueError("No timestamps meet the minimum household count.")
    start, end = stable.min(), stable.max()
    window = df.loc[start:end].copy()
    # Within the window a handful of individual steps can still dip below the
    # floor (a household briefly offline); keep them but flag.
    window["below_coverage_threshold"] = window["active_household_count"] < min_households
    return window


def build_group_weather(household_ids: list[str]) -> pd.DataFrame:
    """Household-count-weighted average weather across a group's stations,
    upsampled from the stations' native hourly resolution to
    ``CONSUMPTION_FREQ`` (15-min) via time-based linear interpolation."""
    households = dl.load_households()
    station_counts = households.loc[household_ids, "Weather_ID"].value_counts()
    counts_str = ", ".join(f"{wid}={int(n)}" for wid, n in station_counts.items())
    print(f"Building weighted weather from {len(station_counts)} station(s): {counts_str}")

    station_frames = {wid: dl.load_weather(wid).set_index("Timestamp") for wid in station_counts.index}
    all_timestamps = sorted(set().union(*(f.index for f in station_frames.values())))

    out = pd.DataFrame(index=pd.DatetimeIndex(all_timestamps, name="Timestamp"))
    for col in WEATHER_COLUMNS:
        weighted_sum = pd.Series(0.0, index=out.index)
        weight_total = pd.Series(0.0, index=out.index)
        for wid, weight in station_counts.items():
            frame = station_frames[wid]
            if col not in frame.columns:
                continue
            values = frame[col].reindex(out.index)
            present = values.notna()
            weighted_sum = weighted_sum.add(values.fillna(0) * weight * present, fill_value=0)
            weight_total = weight_total.add(weight * present, fill_value=0)
        out[col] = (weighted_sum / weight_total.replace(0, np.nan)).values

    print(f"Interpolating weather from hourly to {CONSUMPTION_FREQ} resolution...")
    return out.resample(CONSUMPTION_FREQ).interpolate(method="time")


def build_group(pv: bool) -> pd.DataFrame:
    ids = group_household_ids(pv)
    print(f"Group has {len(ids)} households with a 15-min file on disk.")
    consumption = build_group_consumption(ids)
    consumption = restrict_to_minimum_household_window(consumption)
    weather = build_group_weather(ids)
    merged = consumption.join(weather, how="left")
    merged.attrs["n_households"] = len(ids)
    merged.attrs["household_ids"] = ids
    return merged


if __name__ == "__main__":
    for name, pv in [("pv_group", True), ("non_pv_group", False), ("all_households_group", None)]:
        print(f"\n=== Building {name} ===")
        df = build_group(pv)
        print(
            f"{name}: n_households={df.attrs['n_households']}, "
            f"rows={len(df)}, range=[{df.index.min()} .. {df.index.max()}], "
            f"below_coverage_steps={int(df['below_coverage_threshold'].sum())}"
        )
        print(df[["kWh_total_group_sum", "kWh_mean_per_active_household", "active_household_count"]].describe())
        df.to_csv(f"reports/{name}_15min.csv")
