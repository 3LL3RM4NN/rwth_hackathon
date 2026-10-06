"""Feature table for day-ahead forecasting.

Forecast for local delivery day D is issued at 10:00 on D-1. Meter data is assumed available up to
the end of D-2 (local), called the cutoff. Every lag and aggregate below only uses data before it.
"""
from datetime import date, timedelta

import holidays
import numpy as np
import polars as pl

from utils.data import PROCESSED, TZ

LAG_DAYS = [2, 3, 4, 5, 6, 7, 8, 14]
W0_FEATURES = ["temp_dm2", "temp_3d", "temp_clim"]  # past-only weather (operational)
W2_FEATURES = W0_FEATURES + ["temp", "temp_24h", "temp_72h", "sun"]  # oracle: actual weather on D
BASE_FEATURES = [
    "lhour", "weekday", "is_holiday", "doy_sin", "doy_cos",
    "lag_2", "lag_3", "lag_7", "lag_14", "b3", "hh_mean_7d", "hh_mean_28d",
    "pv", "is_house", "living_area", "residents", "ground_source", "floor_heating", "has_ev",
    "station", "visit_state",
]


def _cutoffs(dates: pl.Series) -> pl.DataFrame:
    """UTC instant of the end of D-2 local (= start of D-1 local) for each delivery date D."""
    return pl.DataFrame({"D": dates.unique()}).with_columns(
        cutoff=(pl.col("D") - pl.duration(days=1)).cast(pl.Datetime("us")).dt.replace_time_zone(TZ).dt.convert_time_zone("UTC")
    )


def build_features(fit_end: date) -> pl.DataFrame:
    hourly = pl.read_parquet(PROCESSED / "hourly.parquet")
    visits = pl.read_parquet(PROCESSED / "visits.parquet")
    hh = pl.read_parquet(PROCESSED / "households.parquet")
    weather = pl.read_parquet(PROCESSED / "weather.parquet")

    df = hourly.select("Household_ID", "hour", "kwh").with_columns(local=pl.col("hour").dt.convert_time_zone(TZ))
    df = df.with_columns(
        D=pl.col("local").dt.date(),
        lhour=pl.col("local").dt.hour().cast(pl.Int8),
        weekday=pl.col("local").dt.weekday().cast(pl.Int8),
    ).drop("local")
    df = df.join(_cutoffs(df["D"]), on="D", how="left")

    # Lags: the hourly grid is gap-free per household, so shift(24k) is exactly t - 24k hours.
    # A lag is masked when its source hour ends after the cutoff (e.g. lag_2 on 25-hour autumn days).
    df = df.with_columns(
        [
            pl.when(pl.col("hour") - pl.duration(hours=24 * k - 1) <= pl.col("cutoff"))
            .then(pl.col("kwh").shift(24 * k).over("Household_ID"))
            .alias(f"lag_{k}")
            for k in LAG_DAYS
        ]
    )
    df = df.with_columns(b3=pl.mean_horizontal([f"lag_{k}" for k in range(2, 9)]))  # same-hour mean, D-8..D-2

    # Household scale: mean hourly kWh over the 7 / 28 days ending D-2 (daily grid is gap-free too).
    daily = (
        df.group_by("Household_ID", "D")
        .agg(day_mean=pl.col("kwh").mean(), n=pl.col("kwh").count())
        .with_columns(day_mean=pl.when(pl.col("n") >= 20).then(pl.col("day_mean")))
        .sort("Household_ID", "D")
        .with_columns(
            hh_mean_7d=pl.col("day_mean").rolling_mean(7, min_samples=3).shift(2).over("Household_ID"),
            hh_mean_28d=pl.col("day_mean").rolling_mean(28, min_samples=7).shift(2).over("Household_ID"),
        )
    )
    df = df.join(daily.select("Household_ID", "D", "hh_mean_7d", "hh_mean_28d"), on=["Household_ID", "D"], how="left")

    # Calendar
    ch = holidays.Switzerland(years=range(2019, 2025))
    doy = pl.col("D").dt.ordinal_day().cast(pl.Float32) * (2 * np.pi / 365.25)
    df = df.with_columns(
        is_holiday=pl.col("D").is_in(list(ch.keys())).cast(pl.Int8), doy_sin=doy.sin(), doy_cos=doy.cos()
    )

    # Static household info; station as integer code
    stations = sorted(hh["Weather_ID"].unique().to_list())
    hh = hh.with_columns(station=pl.col("Weather_ID").replace_strict(stations, list(range(len(stations)))).cast(pl.Int8))
    df = df.join(hh, on="Household_ID", how="left")

    # Visit state, known only once observed by the cutoff: 1 visited, 0 treatment not yet visited, null unknown
    df = df.join(visits, on="Household_ID", how="left").with_columns(
        visit_state=pl.when(pl.col("visit_start").is_not_null() & (pl.col("visit_start") < pl.col("cutoff"))).then(1.0)
        .when((pl.col("group") == "treatment") & (pl.col("has_before") | pl.col("visit_start").is_not_null())).then(0.0)
        .otherwise(None)
    ).drop("group", "visit_start", "has_before")

    # Weather. W2 (oracle): actual hourly values on D. W0: only data up to D-2 plus fit-period climatology.
    w = weather.with_columns(
        temp_24h=pl.col("temp").rolling_mean(24, min_samples=12).over("Weather_ID"),
        temp_72h=pl.col("temp").rolling_mean(72, min_samples=36).over("Weather_ID"),
        local=pl.col("hour").dt.convert_time_zone(TZ),
    ).with_columns(D=pl.col("local").dt.date(), lhour=pl.col("local").dt.hour().cast(pl.Int8), month=pl.col("local").dt.month())
    wd = (
        w.group_by("Weather_ID", "D").agg(t=pl.col("temp").mean()).sort("Weather_ID", "D")
        .with_columns(
            temp_dm2=pl.col("t").shift(2).over("Weather_ID"),
            temp_3d=pl.col("t").rolling_mean(3, min_samples=2).shift(2).over("Weather_ID"),
        )
    )
    clim = (
        w.filter(pl.col("D") <= fit_end)
        .group_by("Weather_ID", "month", "lhour").agg(temp_clim=pl.col("temp").mean())
    )
    df = (
        df.join(w.select("Weather_ID", "hour", "temp", "temp_24h", "temp_72h", "sun"), on=["Weather_ID", "hour"], how="left")
        .join(wd.select("Weather_ID", "D", "temp_dm2", "temp_3d"), on=["Weather_ID", "D"], how="left")
        .with_columns(month=pl.col("D").dt.month())
        .join(clim, on=["Weather_ID", "month", "lhour"], how="left")
        .drop("month", "Weather_ID")
    )

    feats = list(dict.fromkeys(BASE_FEATURES + W2_FEATURES))
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in feats if c not in ("lhour", "weekday", "station")])
    return df.sort("Household_ID", "hour")


def check_availability(df: pl.DataFrame) -> None:
    """Leakage checks: every non-null lag has a source hour ending by the cutoff."""
    for k in LAG_DAYS:
        src_end = pl.col("hour") - pl.duration(hours=24 * k - 1)
        bad = df.filter(pl.col(f"lag_{k}").is_not_null() & (src_end > pl.col("cutoff"))).height
        assert bad == 0, f"lag_{k} uses data after the cutoff in {bad} rows"
    # lag_2 must be masked on the last hour of 25-hour autumn days
    autumn = df.filter((pl.col("D") == date(2023, 10, 29)) & (pl.col("hour") >= pl.col("cutoff") + timedelta(hours=48)))
    assert autumn.height > 0 and autumn["lag_2"].null_count() == autumn.height
