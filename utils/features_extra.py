"""Extra feature groups for the improvement experiment (all past-only: data up to the end of D-2).

WEATHER_X: weather persistence from past observations (no future weather).
RECENT_X:  recent consumption shape.
PEER_X:    lagged mean consumption of similar households (same station; same station and PV flag; all).
"""
from datetime import date

import polars as pl

from utils.data import PROCESSED, TZ

WEATHER_X = ["temp_anom_dm2", "temp_trend", "temp_proxy", "hdh_proxy", "temp_h_lag2", "sun_dm2"]
RECENT_X = ["last6h", "morning_dm2", "evening_dm2", "lag_21", "lag_28", "same_wd_mean", "n_obs_7d", "diff_dm2_dm9"]
PEER_X = ["peer_st_lag2", "peer_st_lag7", "peer_stpv_lag2", "peer_stpv_lag7", "peer_all_lag2", "peer_all_lag7", "peer_st_n"]


def _available(k_days: int) -> pl.Expr:
    """True when the source hour t - 24k ends by the cutoff (end of D-2 local)."""
    return pl.col("hour") - pl.duration(hours=24 * k_days - 1) <= pl.col("cutoff")


def add_extra_features(df: pl.DataFrame, fit_end: date) -> pl.DataFrame:
    hh = pl.read_parquet(PROCESSED / "households.parquet").select("Household_ID", "Weather_ID", "pv")
    weather = pl.read_parquet(PROCESSED / "weather.parquet")
    df = df.join(hh.select("Household_ID", "Weather_ID"), on="Household_ID", how="left")

    # ---- Weather persistence
    w = weather.with_columns(local=pl.col("hour").dt.convert_time_zone(TZ)).with_columns(
        D=pl.col("local").dt.date(), month=pl.col("local").dt.month())
    wd = w.group_by("Weather_ID", "D").agg(
        t=pl.col("temp").mean(), month=pl.col("month").first(),
        sun=pl.when(pl.col("sun").count() > 0).then(pl.col("sun").sum()))  # null only where the station has no sunshine data
    clim_day = wd.filter(pl.col("D") <= fit_end).group_by("Weather_ID", "month").agg(clim_day=pl.col("t").mean())
    wd = (wd.join(clim_day, on=["Weather_ID", "month"], how="left").sort("Weather_ID", "D")
          .with_columns(anom=pl.col("t") - pl.col("clim_day"))
          .with_columns(
              temp_anom_dm2=pl.col("anom").shift(2).over("Weather_ID"),
              temp_trend=(pl.col("t").shift(2) - pl.col("t").shift(3)).over("Weather_ID"),
              sun_dm2=pl.col("sun").shift(2).over("Weather_ID"),
          ))
    df = df.join(wd.select("Weather_ID", "D", "temp_anom_dm2", "temp_trend", "sun_dm2"), on=["Weather_ID", "D"], how="left")
    df = df.with_columns(temp_proxy=pl.col("temp_clim") + pl.col("temp_anom_dm2"))
    df = df.with_columns(hdh_proxy=(18 - pl.col("temp_proxy")).clip(lower_bound=0))
    wh = weather.select("Weather_ID", src=pl.col("hour"), temp_src=pl.col("temp"))
    df = (df.with_columns(src=pl.col("hour") - pl.duration(hours=48))
          .join(wh, on=["Weather_ID", "src"], how="left")
          .with_columns(temp_h_lag2=pl.when(_available(2)).then(pl.col("temp_src")))
          .drop("src", "temp_src"))

    # ---- Recent consumption shape (the hourly grid is gap-free per household, so shift(n) = t - n hours)
    df = df.sort("Household_ID", "hour").with_columns(
        [pl.when(_available(k)).then(pl.col("kwh").shift(24 * k).over("Household_ID")).alias(f"lag_{k}") for k in (21, 28)]
        + [pl.col("kwh").rolling_mean(6, min_samples=3).over("Household_ID").alias("_r6")]
    )
    last6 = df.select("Household_ID", key=pl.col("hour") + pl.duration(hours=1), last6h=pl.col("_r6"))
    df = df.with_columns(key=pl.col("cutoff")).join(last6, on=["Household_ID", "key"], how="left").drop("key", "_r6")
    lh = pl.col("lhour")
    daily = (df.group_by("Household_ID", "D").agg(
                day=pl.col("kwh").mean(), n=pl.col("kwh").count(),
                morning=pl.col("kwh").filter(lh.is_between(6, 11)).mean(),
                evening=pl.col("kwh").filter(lh.is_between(17, 21)).mean())
             .sort("Household_ID", "D")
             .with_columns(
                morning_dm2=pl.col("morning").shift(2).over("Household_ID"),
                evening_dm2=pl.col("evening").shift(2).over("Household_ID"),
                n_obs_7d=pl.col("n").rolling_sum(7, min_samples=1).shift(2).over("Household_ID"),
                diff_dm2_dm9=(pl.col("day").shift(2) - pl.col("day").shift(9)).over("Household_ID")))
    df = df.join(daily.select("Household_ID", "D", "morning_dm2", "evening_dm2", "n_obs_7d", "diff_dm2_dm9"),
                 on=["Household_ID", "D"], how="left")
    df = df.with_columns(same_wd_mean=pl.mean_horizontal("lag_7", "lag_14", "lag_21", "lag_28"))

    # ---- Peer groups: mean kWh per reporting household, by hour, then lagged like the household's own lags
    g = df.select("Household_ID", "hour", "kwh").join(hh, on="Household_ID", how="left").with_columns(
        stpv=pl.col("Weather_ID") + "_" + pl.col("pv").cast(pl.Utf8).fill_null("unknown"))
    groups = {
        "st": g.group_by("Weather_ID", "hour").agg(m=pl.col("kwh").mean(), n=pl.col("kwh").count()),
        "stpv": g.group_by("stpv", "hour").agg(m=pl.col("kwh").mean()),
        "all": g.group_by("hour").agg(m=pl.col("kwh").mean()),
    }
    df = df.join(hh.select("Household_ID", stpv=pl.col("Weather_ID") + "_" + pl.col("pv").cast(pl.Utf8).fill_null("unknown")),
                 on="Household_ID", how="left")
    for name, key in [("st", ["Weather_ID"]), ("stpv", ["stpv"]), ("all", [])]:
        for k in (2, 7):
            src = groups[name].rename({"hour": "src", "m": f"peer_{name}_lag{k}"})
            cols = key + ["src", f"peer_{name}_lag{k}"] + (["n"] if name == "st" and k == 2 else [])
            df = (df.with_columns(src=pl.col("hour") - pl.duration(hours=24 * k))
                  .join(src.select(cols), on=key + ["src"], how="left")
                  .with_columns(pl.when(_available(k)).then(pl.col(f"peer_{name}_lag{k}")).alias(f"peer_{name}_lag{k}"))
                  .drop("src"))
    df = df.rename({"n": "peer_st_n"}).drop("stpv", "Weather_ID")

    extra = WEATHER_X + RECENT_X + PEER_X
    return df.with_columns([pl.col(c).cast(pl.Float32) for c in extra])


def check_extra_availability(df: pl.DataFrame) -> None:
    """Every 2-day hourly lag of the new groups is null wherever its source hour ends after the cutoff
    (the last hour of 25-hour autumn days), and the check is not vacuous."""
    late = df.filter(~_available(2))
    assert late.height > 0
    for c in ["temp_h_lag2", "peer_st_lag2", "peer_stpv_lag2", "peer_all_lag2"]:
        assert late[c].null_count() == late.height, f"{c} leaks past the cutoff"
    for k, c in [(21, "lag_21"), (28, "lag_28")]:
        assert df.filter(~_available(k))[c].null_count() == df.filter(~_available(k)).height
