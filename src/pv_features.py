"""Feature engineering for detecting PV ownership from the 15-minute consumption pattern.

Only ``kWh_received_Total`` is usable here: the HeatPump/Other sub-meter columns
are present in the schema but essentially always blank in this dataset (checked
across a random sample of household files), so PV-related behaviour has to be
read out of the net total-consumption signal, not a load breakdown.

The core idea: a household with PV self-consumption shows its net *imported*
energy suppressed around solar noon, more so in summer than in winter, and more
variable day-to-day (sunny vs cloudy) than a heat-pump-only household. We turn
that intuition into a handful of per-household summary features and let a
classifier weigh them, validated against the surveyed ``Installation_HasPVSystem``
flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_loading as dl

# UTC hour windows. Households are in CET/CEST (UTC+1/+2), so UTC 10-14 covers
# local late-morning-to-afternoon (roughly solar noon) across both halves of the
# year; UTC 22-03 is unambiguously night in both offsets.
MIDDAY_HOURS = {10, 11, 12, 13, 14}
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}
SUMMER_MONTHS = {4, 5, 6, 7, 8, 9}
WINTER_MONTHS = {11, 12, 1, 2}

NEAR_ZERO_THRESHOLD_KWH = 0.05  # per-hour import below this counts as "near zero"


def household_hourly_total(household_id: str) -> pd.Series:
    """Hourly-summed kWh_received_Total for one household, UTC-indexed.

    An hour with zero reported 15-min rows is left as NaN (not 0): we do not
    want "no data" to masquerade as "no consumption" in the ratio features below.
    """
    ts = dl.load_household_timeseries(household_id)
    s = ts.set_index("Timestamp")["kWh_received_Total"]
    return s.resample("1h").sum(min_count=1)


def _safe_ratio(numer: float, denom: float) -> float:
    if denom is None or np.isnan(denom) or denom == 0:
        return np.nan
    return numer / denom


def compute_features_for_household(
    household_id: str, weather_id: str | None, sunshine_available: bool
) -> dict:
    hourly = household_hourly_total(household_id)
    if hourly.dropna().empty:
        return {"Household_ID": household_id}

    idx = hourly.index
    month = idx.month
    hour = idx.hour

    summer = pd.Series(month, index=idx).isin(SUMMER_MONTHS)
    winter = pd.Series(month, index=idx).isin(WINTER_MONTHS)
    midday = pd.Series(hour, index=idx).isin(MIDDAY_HOURS)
    night = pd.Series(hour, index=idx).isin(NIGHT_HOURS)

    summer_midday = hourly[summer & midday].dropna()
    summer_night = hourly[summer & night].dropna()
    winter_midday = hourly[winter & midday].dropna()
    winter_night = hourly[winter & night].dropna()

    summer_midday_mean = summer_midday.mean() if len(summer_midday) else np.nan
    summer_night_mean = summer_night.mean() if len(summer_night) else np.nan
    winter_midday_mean = winter_midday.mean() if len(winter_midday) else np.nan
    winter_night_mean = winter_night.mean() if len(winter_night) else np.nan

    ratio_summer = _safe_ratio(summer_midday_mean, summer_night_mean)
    ratio_winter = _safe_ratio(winter_midday_mean, winter_night_mean)
    seasonal_contrast = _safe_ratio(ratio_summer, ratio_winter)

    summer_midday_cv = (
        summer_midday.std() / summer_midday.mean()
        if len(summer_midday) > 1 and summer_midday.mean() not in (0, np.nan)
        else np.nan
    )
    summer_midday_zero_frac = (
        (summer_midday < NEAR_ZERO_THRESHOLD_KWH).mean() if len(summer_midday) else np.nan
    )

    corr_sunshine = np.nan
    if sunshine_available and weather_id is not None:
        weather = dl.load_weather(weather_id).set_index("Timestamp")[
            "Sunshine_duration_hourly"
        ]
        aligned = pd.concat(
            [summer_midday.rename("load"), weather.rename("sun")], axis=1, join="inner"
        ).dropna()
        if len(aligned) >= 30:
            corr_sunshine = aligned["load"].corr(aligned["sun"])

    return {
        "Household_ID": household_id,
        "summer_midday_mean": summer_midday_mean,
        "summer_night_mean": summer_night_mean,
        "winter_midday_mean": winter_midday_mean,
        "winter_night_mean": winter_night_mean,
        "ratio_summer_midday_to_night": ratio_summer,
        "ratio_winter_midday_to_night": ratio_winter,
        "seasonal_contrast": seasonal_contrast,
        "summer_midday_cv": summer_midday_cv,
        "summer_midday_zero_frac": summer_midday_zero_frac,
        "corr_summer_midday_vs_sunshine": corr_sunshine,
        "n_summer_midday_hours": len(summer_midday),
        "n_winter_midday_hours": len(winter_midday),
    }


FEATURE_COLUMNS = [
    "ratio_summer_midday_to_night",
    "ratio_winter_midday_to_night",
    "seasonal_contrast",
    "summer_midday_cv",
    "summer_midday_zero_frac",
    "corr_summer_midday_vs_sunshine",
]


def build_feature_table(household_ids: list[str] | None = None) -> pd.DataFrame:
    households = dl.load_households()
    avail = dl.weather_variable_availability()
    if household_ids is None:
        household_ids = dl.household_ids()

    rows = []
    n = len(household_ids)
    print(f"Computing PV-pattern features for {n} households...")
    for i, hid in enumerate(household_ids, start=1):
        weather_id = households.loc[hid, "Weather_ID"] if hid in households.index else None
        sunshine_ok = bool(
            weather_id is not None
            and weather_id in avail.index
            and avail.loc[weather_id, "Sunshine_duration_hourly"]
        )
        rows.append(compute_features_for_household(hid, weather_id, sunshine_ok))
        if i % 50 == 0 or i == n:
            print(f"  {i}/{n} households done")

    feats = pd.DataFrame(rows).set_index("Household_ID")
    feats = feats.join(households[["Group", "Weather_ID", "Installation_HasPVSystem"]])
    return feats


if __name__ == "__main__":
    out = build_feature_table()
    out.to_csv("reports/pv_features.csv")
    print(out.shape)
    print(out["Installation_HasPVSystem"].value_counts(dropna=False))
