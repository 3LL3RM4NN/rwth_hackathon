"""Static household features for the AutoGluon 'household features' experiment.

Used (known independently of the forecast horizon):
  * Installation_HasPVSystem  -> pv_flag in {true,false,unknown}   (ownership flag only, NOT PV generation)
  * Weather_ID                -> region (station id used as a location label; no weather values)
  * all Survey_* columns of meta_data.csv; has_survey indicator
Excluded on purpose: Group, AffectsTimePoint, Protocols_*, MetaData_Available, SmartMeterData_Available_*
(experiment-design / data-availability labels, see analysis/audit_static.py), HeatPump/Other consumption.

Missing values are never imputed: categoricals get an explicit "missing" level, numerics stay NaN
(LightGBM handles NaN natively).
"""
import pandas as pd

from .config import Config

NUMERIC = ["Survey_Building_LivingArea", "Survey_Building_Residents"]


def _tri(s: pd.Series) -> pd.Series:
    return s.map({True: "true", False: "false"}).fillna("missing")


def build_static_features(cfg: Config) -> pd.DataFrame:
    meta_dir = cfg.data_dir.parent / "smart_meter_meta_data"
    hh = pd.read_csv(meta_dir / "households.csv", sep=";", dtype={"Household_ID": str})
    sv = pd.read_csv(meta_dir / "meta_data.csv", sep=";", dtype={"Household_ID": str})
    df = hh[["Household_ID", "Weather_ID", "Installation_HasPVSystem"]].merge(sv, on="Household_ID", how="left")
    out = pd.DataFrame(index=pd.Index(df["Household_ID"], name="Household_ID"))
    out["region"] = df["Weather_ID"].to_numpy()
    out["pv_flag"] = _tri(df["Installation_HasPVSystem"]).to_numpy()
    out["has_survey"] = df["Household_ID"].isin(sv["Household_ID"]).map({True: "true", False: "false"}).to_numpy()
    for c in sv.columns[1:]:
        col = df[c]
        out[c] = col.to_numpy(dtype="float64") if c in NUMERIC else (
            _tri(col.astype("object").map({True: True, False: False})) if col.dtype == bool or col.dropna().isin([True, False]).all()
            else col.fillna("missing")).to_numpy()
    for c in out.columns:
        if c not in NUMERIC:
            out[c] = out[c].astype("category")
    return out
