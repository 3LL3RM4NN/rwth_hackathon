"""Export the group hourly aggregates from ``src/aggregate.py`` into the raw
CSV format + config files ProLoaF's own ``src/preprocess.py``/``train.py``/
``evaluate.py`` (vendored as the ``third_party/proloaf`` submodule) expect.

Why a raw-CSV export instead of feeding ``reports/<name>_hourly.csv`` straight
in: ProLoaF's ``load_raw_data_csv`` parses timestamps with a fixed
``"%Y-%m-%d %H:%M:%S"`` format (no UTC-offset suffix), so the offset-suffixed
strings our aggregation pipeline writes (``...+00:00``) have to be stripped
first. Everything else -- continuity gaps (including the 2023-10-29
full-day outage, see ``reports/level1_report.md`` §3), the housekeeping
columns, and feature selection -- is handled by ProLoaF's own
``preprocess.py``/``train.py`` pipeline (``set_to_hours`` + ``fill_if_missing``
+ ``add_cyclical_features`` + ``add_onehot_features``), not duplicated here.

``third_party/proloaf``'s scripts resolve ``targets/<station>/config.json``
and all paths inside it relative to the submodule's own root (derived from
``__file__``, not cwd), so every path written into the generated
preprocessing.json/config.json below is relative to
``third_party/proloaf/`` and points back out to ``targets/<name>/`` at this
repo's root -- keeping all ProLoaF inputs/outputs inside *this* repo rather
than inside the submodule's own tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS_DIR = REPO_ROOT / "targets"
# third_party/proloaf/src/{train,preprocess,evaluate}.py each compute
# MAIN_PATH = dirname(dirname(__file__)) = third_party/proloaf/ -- two levels
# up from there is this repo's root.
FROM_SUBMODULE_ROOT = "../.."

WEATHER_FEATURES = [
    "Temperature_avg_hourly",
    "DewPoint_hourly",
    "Humidity_avg_hourly",
    "Precipitation_total_hourly",
    "Sunshine_duration_hourly",
    "WindSpeed_hourly",
]
TARGET = "kWh_total_group_sum"
AUX_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "mnth_sin", "mnth_cos"]


def export_raw_csv(name: str) -> None:
    df = pd.read_csv(f"reports/{name}_hourly.csv", index_col=0, parse_dates=True)
    df = df[[TARGET] + WEATHER_FEATURES].copy()
    df.index.name = "Timestamp"
    # Strip the UTC offset suffix: ProLoaF's load_raw_data_csv parses with a
    # fixed "%Y-%m-%d %H:%M:%S" format and no %z.
    df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")

    raw_dir = TARGETS_DIR / name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / f"{name}_raw.csv", sep=";")


def write_preprocessing_json(name: str) -> None:
    config = {
        "data_path": f"{FROM_SUBMODULE_ROOT}/targets/{name}/data/{name}.csv",
        "raw_path": f"{FROM_SUBMODULE_ROOT}/targets/{name}/raw/",
        "local": True,
        "add_aux_features": True,
        "csv_files": [
            {
                "file_name": f"{name}_raw.csv",
                "date_column": "Timestamp",
                "time_zone": "UTC",
                "dayfirst": False,
                "sep": ";",
                "combine": False,
                "use_columns": ["Timestamp", TARGET] + WEATHER_FEATURES,
            }
        ],
    }
    (TARGETS_DIR / name / "preprocessing.json").write_text(json.dumps(config, indent=4) + "\n")


def write_config_json(name: str) -> None:
    config = {
        "data_path": f"{FROM_SUBMODULE_ROOT}/targets/{name}/data/{name}.csv",
        "output_path": f"{FROM_SUBMODULE_ROOT}/targets/{name}/oracles/",
        "exploration_path": None,
        "evaluation_path": f"{FROM_SUBMODULE_ROOT}/targets/{name}/oracles/eval/",
        "log_path": f"{FROM_SUBMODULE_ROOT}/targets/{name}/logs/",
        "model_name": name,
        "target_id": [TARGET],
        "start_date": None,
        "history_horizon": 168,
        "forecast_horizon": 24,
        "frequency": "1h",
        "cap_limit": 1,
        "train_split": 0.6,
        "validation_split": 0.8,
        "periodicity": 24,
        "optimizer_name": "adam",
        "exploration": False,
        "cuda_id": None,
        "stack_columns": {},
        "feature_groups": [
            {
                "name": "main",
                "scaler": ["minmax", 0.0, 1.0],
                "features": [TARGET] + WEATHER_FEATURES,
            },
            {
                "name": "aux",
                "scaler": None,
                "features": AUX_FEATURES,
            },
        ],
        "encoder_features": [TARGET] + WEATHER_FEATURES,
        "decoder_features": WEATHER_FEATURES,
        "aux_features": AUX_FEATURES,
        "max_epochs": 40,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "early_stopping_patience": 7,
        "early_stopping_margin": 0.0,
        "model_class": "recurrent",
        "model_parameters": {
            "recurrent": {
                "core_net": "torch.nn.LSTM",
                "core_layers": 1,
                "dropout_fc": 0.2,
                "dropout_core": 0.2,
                "rel_linear_hidden_size": 1.0,
                "rel_core_hidden_size": 1.0,
                "relu_leak": 0.1,
            }
        },
    }
    (TARGETS_DIR / name / "config.json").write_text(json.dumps(config, indent=4) + "\n")


def ensure_submodule_symlink(name: str) -> None:
    """ProLoaF's preprocess.py hardcodes targets/<station>/preprocessing.json
    *relative to its own submodule root* (no --config override for that one
    script, unlike train.py/evaluate.py). Since files placed inside the
    submodule's own working tree aren't tracked by this repo, we instead keep
    targets/<name>/ here (git-tracked) and symlink it into the submodule --
    recreate this after a fresh `git submodule update --init`.
    """
    submodule_targets = REPO_ROOT / "third_party" / "proloaf" / "targets"
    submodule_targets.mkdir(parents=True, exist_ok=True)
    link = submodule_targets / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(f"../../../targets/{name}")


if __name__ == "__main__":
    for name in ["pv_group", "non_pv_group"]:
        export_raw_csv(name)
        write_preprocessing_json(name)
        write_config_json(name)
        ensure_submodule_symlink(name)
        print(f"wrote targets/{name}/raw, preprocessing.json, config.json; linked into submodule")
