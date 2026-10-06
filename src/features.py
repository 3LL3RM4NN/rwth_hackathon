"""Extended candidate features for the day-ahead model, organised in named
groups so ``src/ablation_extended.py`` can test each group's value.

Three groups (1 calendar, 4 solar geometry x PV share, 7 interactions +
weather-corrected lag) were found worth keeping (see
``reports/beneficial_features.md``) and now live in
``src/forecast.py``'s ``build_supervised_table`` itself, so ``table`` already
has them by the time this module runs; ``PRECOMPUTED_GROUPS`` just lets the
ablation scripts still refer to them by label. This module computes and adds
the other, non-adopted candidate groups on top.

Availability rule
-----------------
Every feature is computable from data known at
the gate-closure cutoff (11:45 UTC on D-1, see ``src/forecast.py``):

- **Target history** is either looked up at the cutoff itself (``*_asof_cutoff``)
  or at the target's own time-of-day at least 2 days back (the same k>=2 rule
  ``forecast.py`` derives), or taken from complete UTC days up to D-2 (the
  "behavioural" daily statistics -- D-1 is only half known at the cutoff).
- **Weather history** is computed on the *hourly* series and looked up at
  ``WEATHER_CUTOFF_LAG_HOURS`` whole hours before the cutoff hour (10:00 UTC on
  D-1). The dataset doesn't say whether an hourly value stamped 11:00 covers
  10:00-11:00 or 11:00-12:00; stopping at the 10:00 stamp is safe either way.
- **Calendar and solar geometry** depend on the target timestamp only and are
  known arbitrarily far ahead.
- **Portfolio composition** uses the set of households reporting *at the
  cutoff*, not at the target time (which households will report tomorrow is
  not known today).

Measured weather of the delivery day itself is never used: it is not available
at bid time.

Assumptions forced by the dataset
---------------------------------
- **Location**: the dataset names no region and has no coordinates. The
  households are assumed to be in Germany, represented by its geographic
  centre (``ASSUMED_LAT``/``ASSUMED_LON``).
- **Local time**: ``Europe/Berlin``.
- **Public holidays**: only the nationwide German ones (New Year, Good Friday,
  Easter Monday, 1 May, Ascension, Whit Monday, 3 October, 25/26 December),
  since the federal state isn't given. School holidays are *not* included for
  the same reason.
- **Solar geometry**: no station coordinates exist, so the one assumed location
  above and textbook declination/hour-angle
  formulas are used instead of pvlib. ``clear_sky_proxy`` is the sine of the
  solar elevation (clipped at 0), a shape proxy for clear-sky irradiance, not
  W/m2.
- **No radiation or cloud cover** is in the dataset; sunshine duration is the
  only sky-condition variable, so the "cloudiness index" is sunshine hours
  relative to astronomical day length.
- **Heat pump / other split** exists only for the minority of households with
  a separate heat-pump meter, so those features describe that subset (per
  submetered household), not the whole group.
- **Temperature sensitivity** is a rolling 28-day regression slope of daily
  load on daily temperature using days up to D-2. That is past-only by
  construction, which a slope fitted once on the training period would also
  be, but a single constant would carry no information for a tree model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import aggregate, forecast
from src import data_loading as dl

GROUPS = {"pv_group": True, "non_pv_group": False, "all_households_group": None}

# Groups 1 (calendar), 4 (solar geometry x PV share) and 7 (interactions +
# weather-corrected lag) were adopted into production (see
# reports/beneficial_features.md) and now live in src/forecast.py's
# build_supervised_table, which this module calls -- so they're already
# present in `table` below rather than recomputed here. PRECOMPUTED_GROUPS
# just records their column names so the ablation scripts can still refer to
# them by label.
PRECOMPUTED_GROUPS: dict[str, list[str]] = {
    "1 calendar (local time, holidays, daylight)": forecast.CALENDAR_FEATURE_COLUMNS,
    "4 PV: solar geometry x PV share": forecast.SOLAR_PV_FEATURE_COLUMNS,
    "7 interactions + weather-corrected lag": forecast.INTERACTION_FEATURE_COLUMNS,
}

SAME_TIMEOFDAY_LAG_DAYS = [2, 3, 7, 14, 21]
SAME_WEEKDAY_WEEKS = 4
ROLLING_WINDOWS_STEPS = {"24h": 96, "3d": 288, "7d": 672}
HEATING_DEGREE_BASES = [12, 15, 17]
DEFROST_TEMP_RANGE = (0.0, 7.0)  # degrees C
DEFROST_MIN_HUMIDITY = 80.0  # percent
SENSITIVITY_WINDOW_DAYS = 28
NIGHT_STEPS = slice(0, 20)  # 00:00-05:00 UTC
DAY_STEPS = slice(32, 80)  # 08:00-20:00 UTC

STEPS_PER_DAY = forecast.STEPS_PER_DAY


def _days(n: int) -> pd.Timedelta:
    return pd.Timedelta(n * 24 * 60, unit="m")


def _extras_path(name: str) -> Path:
    return Path(f"reports/{name}_extras_15min.csv")


def _bool_share(column: pd.Series) -> pd.Series:
    return column.astype(str).map({"True": 1.0, "False": 0.0})


def build_group_extras(name: str) -> pd.DataFrame:
    """Per-15-min heat-pump/other sums (submetered households only) and the
    composition of the households reporting at each timestamp."""
    index = forecast.load_group_15min(name).index
    ids = aggregate.group_household_ids(GROUPS[name])
    households = dl.load_households().loc[ids]
    survey = dl.load_meta_data()
    meta = survey.reindex(ids)
    in_meta = pd.Series(ids, index=ids).isin(survey.index)

    n = len(index)
    active = np.zeros((n, len(ids)))
    after_visit = np.zeros((n, len(ids)))
    hp_sum, other_sum, n_submetered = np.zeros(n), np.zeros(n), np.zeros(n)
    print(f"Loading heat-pump split and visit status for {len(ids)} households...")
    for j, hid in enumerate(ids):
        ts = dl.load_household_timeseries(hid).set_index("Timestamp")
        ts["after"] = (ts["AffectsTimePoint"] == "after visit").astype(float)
        resampled = ts.resample("15min")
        total = resampled["kWh_received_Total"].sum(min_count=1).reindex(index)
        hp = resampled["kWh_received_HeatPump"].sum(min_count=1).reindex(index)
        other = resampled["kWh_received_Other"].sum(min_count=1).reindex(index)
        is_active = total.notna().to_numpy()
        active[:, j] = is_active
        after_visit[:, j] = is_active * resampled["after"].max().reindex(index).fillna(0).to_numpy()
        submetered = (hp.notna() & other.notna()).to_numpy()
        hp_sum += np.where(submetered, hp.fillna(0).to_numpy(), 0.0)
        other_sum += np.where(submetered, other.fillna(0).to_numpy(), 0.0)
        n_submetered += submetered
        if (j + 1) % 50 == 0 or j + 1 == len(ids):
            print(f"  {j + 1}/{len(ids)} households done")

    def active_mean(values: pd.Series) -> np.ndarray:
        """Mean of a per-household attribute over the households reporting at
        each timestamp, ignoring households where the attribute is unknown."""
        v = values.to_numpy(dtype=float)
        known = ~np.isnan(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (active @ np.where(known, v, 0.0)) / (active @ known.astype(float))

    # The three DHW columns only ever contain True or blank: blank is read as
    # "not ticked" for households that answered the survey at all.
    def ticked(column: str) -> pd.Series:
        return meta[column].notna().astype(float).where(in_meta)

    n_active = active.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame(
            {
                "hp_sum": np.where(n_submetered > 0, hp_sum, np.nan),
                "other_sum": np.where(n_submetered > 0, other_sum, np.nan),
                "n_submetered": n_submetered,
                "n_active": n_active,
                "share_after_visit": after_visit.sum(axis=1) / n_active,
                "share_treatment": active_mean((households["Group"] == "treatment").astype(float)),
                "share_pv": active_mean(households["Installation_HasPVSystem"].astype(float)),
                "mean_living_area": active_mean(meta["Survey_Building_LivingArea"]),
                "mean_residents": active_mean(meta["Survey_Building_Residents"]),
                "share_air_source": active_mean(
                    (meta["Survey_HeatPump_Installation_Type"] == "air-source")
                    .astype(float)
                    .where(meta["Survey_HeatPump_Installation_Type"].notna())
                ),
                "share_floor_heating": active_mean(_bool_share(meta["Survey_HeatDistribution_System_FloorHeating"])),
                "share_dhw_by_heatpump": active_mean(ticked("Survey_DHW_Production_ByHeatPump")),
                "share_dhw_electric": active_mean(ticked("Survey_DHW_Production_ByElectricWaterHeater")),
                "share_ev": active_mean(_bool_share(meta["Survey_Installation_HasElectricVehicle"])),
                "share_dryer": active_mean(_bool_share(meta["Survey_Installation_HasDryer"])),
            },
            index=index,
        )
    return out


def load_group_extras(name: str) -> pd.DataFrame:
    path = _extras_path(name)
    if not path.exists():
        build_group_extras(name).to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def build_extended_table(name: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """``forecast.build_supervised_table`` plus all candidate feature groups.

    Returns the table and a ``group label -> feature columns`` mapping.
    """
    df = forecast.load_group_15min(name)
    extras = load_group_extras(name).reindex(df.index)
    table = forecast.build_supervised_table(df, name)

    s = df[forecast.TARGET]
    idx = s.index
    origin = idx.normalize()
    cutoff = origin - forecast.CUTOFF_GAP
    weather_cutoff = cutoff.floor("h") - pd.Timedelta(forecast.WEATHER_CUTOFF_LAG_HOURS * 60, unit="m")
    horizon = idx.hour.to_numpy() * forecast.STEPS_PER_HOUR + idx.minute.to_numpy() // 15

    def at_cutoff(series: pd.Series) -> np.ndarray:
        return series.reindex(cutoff).to_numpy()

    def at_weather_cutoff(series: pd.Series) -> np.ndarray:
        return series.reindex(weather_cutoff).to_numpy()

    def from_day(daily: pd.Series, days_back: int) -> np.ndarray:
        """Value of a per-UTC-day statistic ``days_back`` days before the delivery day."""
        return daily.shift(days_back).reindex(origin).to_numpy()

    groups: dict[str, dict[str, np.ndarray]] = {}

    # --- 2. Consumption history ---------------------------------------------
    lags = {f"lag_same_tod_{k}d": s.shift(STEPS_PER_DAY * k).to_numpy() for k in SAME_TIMEOFDAY_LAG_DAYS}
    same_weekday = pd.concat([s.shift(STEPS_PER_DAY * 7 * w) for w in range(1, SAME_WEEKDAY_WEEKS + 1)], axis=1)
    lags["same_weekday_4w_mean"] = same_weekday.mean(axis=1).to_numpy()
    lags["same_weekday_4w_median"] = same_weekday.median(axis=1).to_numpy()
    lags["same_weekday_4w_std"] = same_weekday.std(axis=1).to_numpy()
    groups["2a history: more lags + same-weekday stats"] = lags

    rolling: dict[str, np.ndarray] = {}
    for label, window in ROLLING_WINDOWS_STEPS.items():
        r = s.rolling(window, min_periods=window // 2)
        for stat in ("mean", "min", "max", "std"):
            if (label, stat) == ("24h", "mean"):
                continue  # already in the baseline as rolling_mean_24h_asof_cutoff
            rolling[f"roll_{stat}_{label}_asof_cutoff"] = at_cutoff(getattr(r, stat)())
    mean_7d = s.rolling(ROLLING_WINDOWS_STEPS["7d"], min_periods=ROLLING_WINDOWS_STEPS["7d"] // 2).mean()
    rolling["trend_7d_vs_prev_7d"] = at_cutoff(mean_7d) / mean_7d.reindex(cutoff - _days(7)).to_numpy()
    rolling["lag_7d_over_roll_mean_7d"] = lags["lag_same_tod_7d"] / rolling["roll_mean_7d_asof_cutoff"]
    groups["2b history: rolling stats + trend as of cutoff"] = rolling

    hp_per_hh = extras["hp_sum"] / extras["n_submetered"]
    other_per_hh = extras["other_sum"] / extras["n_submetered"]
    hp_24h, other_24h = hp_per_hh.rolling(96, min_periods=48).mean(), other_per_hh.rolling(96, min_periods=48).mean()
    groups["2c history: heat pump / other split (submetered households)"] = {
        "hp_share_24h_asof_cutoff": at_cutoff(hp_24h / (hp_24h + other_24h)),
        "hp_per_hh_mean_24h_asof_cutoff": at_cutoff(hp_24h),
        "other_per_hh_mean_24h_asof_cutoff": at_cutoff(other_24h),
        "hp_per_hh_lag_same_tod_2d": hp_per_hh.shift(STEPS_PER_DAY * 2).to_numpy(),
        "hp_per_hh_lag_same_tod_7d": hp_per_hh.shift(STEPS_PER_DAY * 7).to_numpy(),
        "other_per_hh_lag_same_tod_2d": other_per_hh.shift(STEPS_PER_DAY * 2).to_numpy(),
        "other_per_hh_lag_same_tod_7d": other_per_hh.shift(STEPS_PER_DAY * 7).to_numpy(),
    }

    # --- 3. Weather, built on the hourly series -----------------------------
    hourly = df.loc[idx.minute == 0, forecast.WEATHER_FEATURES]
    temp = hourly["Temperature_avg_hourly"]
    humidity = hourly["Humidity_avg_hourly"]
    sunshine = hourly["Sunshine_duration_hourly"]
    precipitation = hourly["Precipitation_total_hourly"]
    wind = hourly["WindSpeed_hourly"]

    temp_mean_24h = at_weather_cutoff(temp.rolling(24, min_periods=12).mean())
    temperature = {
        "wx_temp_asof_cutoff": at_weather_cutoff(temp),
        **{
            f"wx_temp_mean_{h}h": at_weather_cutoff(temp.rolling(h, min_periods=h // 2).mean())
            for h in (6, 12, 48, 168)
        },
        "wx_temp_mean_24h": temp_mean_24h,
        "wx_temp_min_24h": at_weather_cutoff(temp.rolling(24, min_periods=12).min()),
        "wx_temp_max_24h": at_weather_cutoff(temp.rolling(24, min_periods=12).max()),
        "wx_temp_night_min": at_weather_cutoff(temp.rolling(12, min_periods=6).min()),  # 22:00 D-2 .. 10:00 D-1
        "wx_temp_change_24h": at_weather_cutoff(temp - temp.shift(24)),
        **{
            f"wx_heating_degree_hours_{base}_24h": at_weather_cutoff(
                (base - temp).clip(lower=0).rolling(24, min_periods=12).sum()
            )
            for base in HEATING_DEGREE_BASES
        },
        "wx_heating_degree_hours_15_7d": at_weather_cutoff((15 - temp).clip(lower=0).rolling(168, min_periods=84).sum()),
    }
    groups["3a weather as of cutoff: temperature + heating degree"] = temperature

    defrost = (temp.between(*DEFROST_TEMP_RANGE) & (humidity >= DEFROST_MIN_HUMIDITY)).astype(float)
    sunshine_sum_24h = at_weather_cutoff(sunshine.rolling(24, min_periods=12).sum())
    groups["3b weather as of cutoff: defrost, sun, rain, wind"] = {
        "wx_defrost_share_24h": at_weather_cutoff(defrost.rolling(24, min_periods=12).mean()),
        "wx_dewpoint_spread_asof_cutoff": at_weather_cutoff(temp - hourly["DewPoint_hourly"]),
        "wx_sunshine_sum_24h": sunshine_sum_24h,
        "wx_sunshine_sum_3d": at_weather_cutoff(sunshine.rolling(72, min_periods=36).sum()),
        "wx_precip_sum_24h": at_weather_cutoff(precipitation.rolling(24, min_periods=12).sum()),
        "wx_precip_sum_3d": at_weather_cutoff(precipitation.rolling(72, min_periods=36).sum()),
        "wx_wind_max_24h": at_weather_cutoff(wind.rolling(24, min_periods=12).max()),
    }

    # --- 5. Portfolio composition (households reporting at the cutoff) ------
    composition_columns = [
        "n_active", "share_after_visit", "share_treatment", "mean_living_area", "mean_residents",
        "share_air_source", "share_floor_heating", "share_dhw_by_heatpump", "share_dhw_electric",
        "share_ev", "share_dryer",
    ]  # fmt: skip
    groups["5 portfolio composition as of cutoff"] = {
        f"{c}_asof_cutoff": at_cutoff(extras[c]) for c in composition_columns
    }

    # --- 6. Behavioural (complete UTC days up to D-2) ------------------------
    def by_day(series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"day": origin, "step": horizon, "value": series.to_numpy()}).pivot(
            index="day", columns="step", values="value"
        )

    load_by_day = by_day(s)
    daily_load = load_by_day.mean(axis=1)
    daily_temp = df["Temperature_avg_hourly"].groupby(origin).mean()
    peak_step = load_by_day.fillna(-np.inf).idxmax(axis=1).astype(float).where(load_by_day.notna().any(axis=1))
    is_weekend_day = pd.Series((load_by_day.index.dayofweek >= 5).astype(float), index=load_by_day.index)
    weekend_mean = (daily_load * is_weekend_day).rolling(14, min_periods=7).sum() / is_weekend_day.rolling(14).sum()
    weekday_mean = (daily_load * (1 - is_weekend_day)).rolling(14, min_periods=7).sum() / (
        1 - is_weekend_day
    ).rolling(14).sum()
    sensitivity = daily_load.rolling(SENSITIVITY_WINDOW_DAYS, min_periods=14).cov(daily_temp) / daily_temp.rolling(
        SENSITIVITY_WINDOW_DAYS, min_periods=14
    ).var()
    hp_by_day = by_day(hp_per_hh)
    groups["6 behavioural (complete days up to D-2)"] = {
        "peak_step_prev_day": from_day(peak_step, 2),
        "peak_step_mean_7d": from_day(peak_step.rolling(7, min_periods=4).mean(), 2),
        "weekend_weekday_ratio_14d": from_day(weekend_mean / weekday_mean, 2),
        "temp_sensitivity_28d": from_day(sensitivity, 2),
        "night_day_ratio_prev_day": from_day(
            load_by_day.iloc[:, NIGHT_STEPS].mean(axis=1) / load_by_day.iloc[:, DAY_STEPS].mean(axis=1), 2
        ),
        "hp_night_ratio_prev_day": from_day(hp_by_day.iloc[:, NIGHT_STEPS].mean(axis=1) / hp_by_day.mean(axis=1), 2),
    }

    new_columns = pd.DataFrame({col: values for features in groups.values() for col, values in features.items()})
    clash = set(new_columns.columns) & set(table.columns)
    assert not clash, f"candidate features clash with existing table columns: {clash}"
    new_columns = new_columns.replace([np.inf, -np.inf], np.nan)
    table = pd.concat([table, new_columns], axis=1)
    feature_groups = {label: list(cols) for label, cols in groups.items()}
    feature_groups.update(PRECOMPUTED_GROUPS)
    return table, feature_groups


if __name__ == "__main__":
    for group_name in GROUPS:
        print(f"\n=== {group_name} ===")
        build_group_extras(group_name).to_csv(_extras_path(group_name))
        extended, feature_groups = build_extended_table(group_name)
        for label, columns in feature_groups.items():
            missing = extended[columns].isna().mean().mean()
            print(f"{label}: {len(columns)} features, {missing:.1%} missing")
