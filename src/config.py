"""Central configuration for the baseline experiment."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    data_dir: Path = ROOT / "data" / "15min"
    output_dir: Path = ROOT / "outputs"
    target: str = "kWh_received_Total"
    intervals_per_day: int = 96          # forecast horizon: 96 x 15 min = next day
    # chronological split on the *global* calendar (shared by all households)
    test_fraction: float = 0.20          # final 20 % of calendar dates
    val_fraction: float = 0.10           # the 10 % before that (no fitting in a baseline, reported only)
    min_history_days: int = 1            # a household needs this many days of history before a forecast day
    min_day_completeness: float = 1.0    # share of 96 intervals that must be observed for a day to be usable
    models: tuple = ("naive_1day", "naive_7day", "seasonal_mean_4weeks")
    mean_weeks: int = 4
    mean_min_occurrences: int = 2        # min. non-missing same-weekday days needed for seasonal_mean_4weeks
    mape_min_actual: float = 0.05        # kWh; MAPE ignores intervals with actual below this
    max_households: int | None = None    # for quick debugging
