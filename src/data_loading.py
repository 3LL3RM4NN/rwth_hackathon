"""Loading and basic parsing of the raw E.ON challenge data.

All on-disk CSVs are semicolon-separated with UTC timestamps. Column order in
``data/15min/*.csv`` is ``Household_ID;AffectsTimePoint;Group;Timestamp;...`` —
this is the real header order and differs from the table in README.md, so code
here always selects by name, never by position.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HOUSEHOLD_15MIN_DIR = DATA_DIR / "15min"
WEATHER_DIR = DATA_DIR / "weather_data_hourly"
META_DIR = DATA_DIR / "smart_meter_meta_data"


def household_ids() -> list[str]:
    """All household IDs that have a 15-minute data file, as they appear on disk."""
    return sorted(p.stem for p in HOUSEHOLD_15MIN_DIR.glob("*.csv"))


def load_households() -> pd.DataFrame:
    """Join hub: Household_ID -> Group, Weather_ID, Installation_HasPVSystem, ..."""
    df = pd.read_csv(META_DIR / "households.csv", sep=";", dtype={"Household_ID": str})
    df["Installation_HasPVSystem"] = df["Installation_HasPVSystem"].map(
        {True: True, False: False, "True": True, "False": False}
    )
    return df.set_index("Household_ID")


def load_meta_data() -> pd.DataFrame:
    """Survey answers, keyed by Household_ID. Only covers 393/410 households."""
    df = pd.read_csv(META_DIR / "meta_data.csv", sep=";", dtype={"Household_ID": str})
    return df.set_index("Household_ID")


def load_overview() -> pd.DataFrame:
    """Per-household data-coverage stats (earliest/latest timestamp, day counts)."""
    df = pd.read_csv(
        META_DIR / "smart_meter_data_15min_overview.csv",
        sep=";",
        dtype={"Household_ID": str},
    )
    for col in (
        "SMD_15min_TimeAvailable_EarliestTimestamp",
        "SMD_15min_TimeAvailable_LatestTimestamp",
    ):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.set_index("Household_ID")


def load_household_timeseries(household_id: str) -> pd.DataFrame:
    """One household's raw 15-minute series, sorted by timestamp."""
    path = HOUSEHOLD_15MIN_DIR / f"{household_id}.csv"
    df = pd.read_csv(path, sep=";", dtype={"Household_ID": str})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    return df.sort_values("Timestamp").reset_index(drop=True)


def load_weather(weather_id: str) -> pd.DataFrame:
    """One weather station's hourly series, sorted by timestamp."""
    path = WEATHER_DIR / f"{weather_id}.csv"
    df = pd.read_csv(path, sep=";")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    return df.sort_values("Timestamp").reset_index(drop=True)


def weather_variable_availability() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "weather_data_overview" / "weather_variables_availability.csv",
        sep=";",
    )
    return df.set_index("Weather_ID")
