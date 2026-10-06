"""Load raw meter, weather and household data into clean hourly tables."""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
TZ = "Europe/Zurich"
SPIKE_KWH_15MIN = 10.0  # >40 kW in one quarter-hour: flagged as suspicious, excluded

METER_COLS = ["Household_ID", "Group", "AffectsTimePoint", "Timestamp", "kWh_received_Total"]


def _scan_meter(path: Path) -> pl.LazyFrame:
    # Column order differs between files (control vs treatment), so select by name.
    return (
        pl.scan_csv(path, separator=";", schema_overrides={"Timestamp": pl.Utf8, "kWh_received_Total": pl.Float64})
        .select(METER_COLS)
        .with_columns(ts=pl.col("Timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%z", time_zone="UTC"))
        .drop("Timestamp")
    )


def load_hourly_meter() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hourly kWh per household on a gap-free UTC grid, plus per-household visit info.

    An hour's kWh is the sum of its 4 quarter-hours and is null unless all 4 are observed.
    """
    lf = pl.concat([_scan_meter(p) for p in sorted((DATA / "15min").glob("*.csv"))])

    visits = (
        lf.group_by("Household_ID")
        .agg(
            group=pl.col("Group").first(),
            visit_start=pl.col("ts").filter(pl.col("AffectsTimePoint").is_in(["during visit", "after visit"])).min(),
            has_before=(pl.col("AffectsTimePoint") == "before visit").any(),
        )
        .collect()
    )

    q = lf.select("Household_ID", "ts", kwh=pl.col("kWh_received_Total"))
    q = q.with_columns(spike=pl.col("kwh") > SPIKE_KWH_15MIN)
    q = q.with_columns(kwh=pl.when(pl.col("spike")).then(None).otherwise(pl.col("kwh")))

    hourly = (
        q.group_by("Household_ID", hour=pl.col("ts").dt.truncate("1h"))
        .agg(n_q=pl.col("kwh").is_not_null().sum(), kwh=pl.col("kwh").sum(), spike=pl.col("spike").any())
        .with_columns(kwh=pl.when(pl.col("n_q") == 4).then(pl.col("kwh")).otherwise(None))
        .collect()
    )
    # Keep households that have at least one complete hour; reindex each to a gap-free hourly grid.
    span = hourly.filter(pl.col("kwh").is_not_null()).group_by("Household_ID").agg(
        first=pl.col("hour").min(), last=pl.col("hour").max()
    )
    grid = (
        span.with_columns(hour=pl.datetime_ranges("first", "last", "1h"))
        .explode("hour")
        .select("Household_ID", "hour")
    )
    hourly = grid.join(hourly, on=["Household_ID", "hour"], how="left").with_columns(
        pl.col("n_q").fill_null(0), pl.col("spike").fill_null(False)
    )
    return hourly.sort("Household_ID", "hour"), visits


def load_households() -> pl.DataFrame:
    hh = pl.read_csv(DATA / "smart_meter_meta_data" / "households.csv", separator=";")
    md = pl.read_csv(DATA / "smart_meter_meta_data" / "meta_data.csv", separator=";", infer_schema_length=10000)
    md = md.select(
        "Household_ID",
        is_house=(pl.col("Survey_Building_Type") == "house").cast(pl.Float32),
        living_area=pl.col("Survey_Building_LivingArea").cast(pl.Float32),
        residents=pl.col("Survey_Building_Residents").cast(pl.Float32),
        ground_source=(pl.col("Survey_HeatPump_Installation_Type") == "ground-source").cast(pl.Float32),
        floor_heating=pl.col("Survey_HeatDistribution_System_FloorHeating").cast(pl.Float32),
        has_ev=pl.col("Survey_Installation_HasElectricVehicle").cast(pl.Float32),
    )
    # Left join keeps the 17 households without a survey row (their features stay null).
    return hh.select(
        "Household_ID", "Weather_ID", pv=pl.col("Installation_HasPVSystem").cast(pl.Float32)
    ).join(md, on="Household_ID", how="left")


def load_weather() -> pl.DataFrame:
    """Hourly weather keyed by the START of the hour it describes.

    Source timestamps mark the END of the hourly interval (MeteoSwiss convention; checked with
    the sunshine centroid), so the row labelled h+1 describes meter hour [h, h+1).
    """
    frames = []
    for p in sorted((DATA / "weather_data_hourly").glob("*.csv")):
        w = pl.read_csv(p, separator=";", infer_schema_length=50000)
        frames.append(
            w.select(
                "Weather_ID",
                hour=pl.col("Timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S%z", time_zone="UTC") - pl.duration(hours=1),
                temp=pl.col("Temperature_avg_hourly").cast(pl.Float32),
                sun=pl.col("Sunshine_duration_hourly").cast(pl.Float32) if "Sunshine_duration_hourly" in w.columns else pl.lit(None, pl.Float32),
            )
        )
    return pl.concat(frames, how="vertical_relaxed").sort("Weather_ID", "hour")


def build_processed() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    hourly, visits = load_hourly_meter()

    # Loader checks
    assert hourly.select(pl.struct("Household_ID", "hour").is_duplicated().any()).item() is False
    assert hourly.filter(pl.col("kwh").is_not_null() & (pl.col("n_q") != 4)).height == 0
    step = hourly.select(pl.col("hour").diff().over("Household_ID").drop_nulls().unique())
    assert step.height == 1 and step.item().total_seconds() == 3600, "hourly grid must be gap-free"

    hourly.write_parquet(PROCESSED / "hourly.parquet")
    visits.write_parquet(PROCESSED / "visits.parquet")
    load_households().write_parquet(PROCESSED / "households.parquet")
    load_weather().write_parquet(PROCESSED / "weather.parquet")
    print(
        f"hourly rows {hourly.height:,} | households {hourly['Household_ID'].n_unique()} | "
        f"complete hours {hourly['kwh'].is_not_null().sum():,} | spike hours {hourly['spike'].sum()}"
    )


if __name__ == "__main__":
    build_processed()
