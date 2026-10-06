"""Build hourly, group-level aggregated consumption + weather series for the
two Level-1 groups (PV / non-PV), defined by the *surveyed*
``Installation_HasPVSystem`` flag (not the detector from ``pv_detection.py`` --
see that module's docstring for why).

Two things have to be handled explicitly, per the task brief:

1. **Partial household coverage.** Households were instrumented at different
   times over 2019-2024, so the number of households reporting at a given
   hourly timestamp ramps up over the series and is not constant. Summing
   ``kWh_received_Total`` over whatever happens to be reporting would make the
   aggregate's *level* track meter rollout rather than real demand. We instead
   restrict each group to a trailing window where at least
   ``MIN_COVERAGE_FRAC`` of the group's eventual household count is reporting,
   and still record the active-household count per hour so this can be
   inspected/adjusted.
2. **Multi-station weather.** A group's households are spread across several
   of the 8 weather stations. Group-level weather features are a
   household-count-weighted average across the stations actually used by that
   group's households, renormalised over whichever stations have a given
   variable populated (e.g. only 5/8 stations report sunshine duration).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_loading as dl
from src.pv_features import household_hourly_total

MIN_COVERAGE_FRAC = 0.7  # keep hours where >=70% of the group's households report
WEATHER_COLUMNS = [
    "Temperature_avg_hourly",
    "DewPoint_hourly",
    "Humidity_avg_hourly",
    "Precipitation_total_hourly",
    "Sunshine_duration_hourly",
    "WindSpeed_hourly",
]


def group_household_ids(pv: bool | None) -> list[str]:
    """pv=True/False selects the surveyed PV/non-PV group; pv=None selects the
    union of both (every household with a *known* flag) -- used only to build
    the single-ungrouped-model comparison baseline in the report."""
    households = dl.load_households()
    if pv is None:
        mask = households["Installation_HasPVSystem"].isin([True, False])
    else:
        mask = households["Installation_HasPVSystem"] == pv
    ids = households.index[mask].tolist()
    # Only keep households that actually have a 15-min file on disk.
    on_disk = set(dl.household_ids())
    return [h for h in ids if h in on_disk]


def build_group_consumption(household_ids: list[str]) -> pd.DataFrame:
    """Per-hour summed consumption and active-household count for a group."""
    series = {}
    for hid in household_ids:
        series[hid] = household_hourly_total(hid)
    wide = pd.DataFrame(series).sort_index()
    active_count = wide.notna().sum(axis=1)
    group_sum = wide.sum(axis=1, min_count=1)
    out = pd.DataFrame(
        {"kWh_total_group_sum": group_sum, "active_household_count": active_count}
    )
    return out


def restrict_to_stable_window(df: pd.DataFrame, n_households: int) -> pd.DataFrame:
    threshold = MIN_COVERAGE_FRAC * n_households
    stable = df.index[df["active_household_count"] >= threshold]
    if stable.empty:
        raise ValueError("No timestamps meet the minimum coverage threshold.")
    start, end = stable.min(), stable.max()
    window = df.loc[start:end].copy()
    # Within the stable window a handful of individual hours can still dip
    # below threshold (a household briefly offline); keep them but flag.
    window["below_coverage_threshold"] = window["active_household_count"] < threshold
    return window


def build_group_weather(household_ids: list[str]) -> pd.DataFrame:
    households = dl.load_households()
    station_counts = households.loc[household_ids, "Weather_ID"].value_counts()

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
    return out


def build_group(pv: bool) -> pd.DataFrame:
    ids = group_household_ids(pv)
    consumption = build_group_consumption(ids)
    consumption = restrict_to_stable_window(consumption, len(ids))
    weather = build_group_weather(ids)
    merged = consumption.join(weather, how="left")
    merged.attrs["n_households"] = len(ids)
    merged.attrs["household_ids"] = ids
    return merged


if __name__ == "__main__":
    for name, pv in [("pv_group", True), ("non_pv_group", False), ("all_known_group", None)]:
        df = build_group(pv)
        print(
            f"{name}: n_households={df.attrs['n_households']}, "
            f"rows={len(df)}, range=[{df.index.min()} .. {df.index.max()}], "
            f"below_coverage_hours={int(df['below_coverage_threshold'].sum())}"
        )
        print(df[["kWh_total_group_sum", "active_household_count"]].describe())
        df.to_csv(f"reports/{name}_hourly.csv")
