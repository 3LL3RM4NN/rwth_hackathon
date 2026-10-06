"""Known-covariate tables (calendar + historical temperature) that respect the D-1 11:45 cutoff.

AutoGluon's tabular models accept only KNOWN covariates, i.e. values that are a function of the target timestamp t.
Weather is therefore anchored to the forecast cutoff of the day t belongs to:

    for every timestamp t of UTC day D':  anchor = (D'-1) 11:00   (latest hourly value complete at the cutoff D'-1 11:45)
    features (all from observations with timestamp <= anchor, never later):
        latest, lag 1 h, 6 h, 24 h, 7 d, mean of the 6 / 24 h ending at the anchor, change over 6 / 24 h.

So for a forecast issued at D-1 11:45 the target day D uses exactly the latest observation available at the cutoff; the
12:00-23:45 steps of day D-1 use the (older) cutoff of D-2. The value does not depend on the forecast origin, which is what
a known covariate requires. No fixed lag is imposed beyond the cutoff itself. Hourly values are labelled at the END of the
hour (sunshine/temperature pattern, see report), so the value stamped 11:00 is complete at 11:00 <= 11:45.
Missing hourly values are forward-filled (backward-looking only). There is no weather from the forecast day anywhere.
Calendar: weekend flag, month, German NATIONAL public holidays (household state unknown -> no state/school holidays).
"""
import holidays
import numpy as np
import pandas as pd

ANCHOR_HOUR = 11          # hour-of-day (UTC) of the anchor observation on the day before the target day
TEMP = ["temp_latest", "temp_lag1h", "temp_lag6h", "temp_lag24h", "temp_lag7d",
        "temp_mean6h", "temp_mean24h", "temp_chg6h", "temp_chg24h"]
CAL = ["is_weekend", "is_holiday", "month"]
TS_INDEX = pd.date_range("2019-03-01", "2024-03-01", freq="15min")


def hourly_features(temp: pd.Series) -> pd.DataFrame:
    t = temp.ffill()
    return pd.DataFrame({
        "temp_latest": t, "temp_lag1h": t.shift(1), "temp_lag6h": t.shift(6), "temp_lag24h": t.shift(24),
        "temp_lag7d": t.shift(168), "temp_mean6h": t.rolling(6).mean(), "temp_mean24h": t.rolling(24).mean(),
        "temp_chg6h": t - t.shift(6), "temp_chg24h": t - t.shift(24)})


def load_station_temps(weather_dir) -> dict[str, pd.Series]:
    out = {}
    for f in sorted(weather_dir.glob("*.csv")):
        d = pd.read_csv(f, sep=";", usecols=["Weather_ID", "Timestamp", "Temperature_avg_hourly"])
        idx = pd.to_datetime(d["Timestamp"], utc=True).dt.tz_localize(None)
        s = pd.Series(d["Temperature_avg_hourly"].to_numpy(dtype="float64"), index=idx).sort_index()
        out[str(d["Weather_ID"].iloc[0])] = s.asfreq("h")
    return out


class Covariates:
    def __init__(self, temps: dict[str, pd.Series], station_of: dict[str, str], names: list[str]):
        self.names, self.station_of = list(names), station_of
        self.weather_names = [n for n in names if n in TEMP]
        a = TS_INDEX.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=ANCHOR_HOUR)     # anchor of each timestamp's day
        self.tables = {st: hourly_features(s).reindex(a)[self.weather_names].to_numpy(dtype="float32")
                       for st, s in temps.items()} if self.weather_names else {}
        hol = holidays.Germany(years=range(2019, 2025))
        days = TS_INDEX.normalize()
        self.cal = {"is_weekend": (days.dayofweek >= 5).astype("float32"), "month": days.month.to_numpy().astype("float32"),
                    "is_holiday": np.array([d.date() in hol for d in days], dtype="float32")}

    def frame(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Covariate values for every (item_id, timestamp) of `index`."""
        items = index.get_level_values(0).to_numpy(); ts = index.get_level_values(1)
        pos = TS_INDEX.get_indexer(ts)
        assert (pos >= 0).all(), "timestamps outside covariate table"
        cols = {}
        if self.weather_names:
            arr = np.full((len(index), len(self.weather_names)), np.nan, dtype="float32")
            station_per_row = pd.Series(items).map(self.station_of).to_numpy()
            for st in np.unique(station_per_row):
                m = station_per_row == st
                arr[m] = self.tables[st][pos[m]]
            assert np.isfinite(arr).all(), "NaN weather covariate (check station coverage / forward fill)"
            for j, n in enumerate(self.weather_names):
                cols[n] = arr[:, j]
        for n in self.names:
            if n in CAL:
                cols[n] = self.cal[n][pos]
        return pd.DataFrame({n: cols[n] for n in self.names}, index=index)
