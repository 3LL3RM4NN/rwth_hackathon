"""Equal-weight ensemble of the 3 best models, selected on the tune block (never the test year).

Chosen among strict-weather (W0) and consumption-only models.
TimesFM-3 is excluded (non-commercial weights). Missing member forecasts fall back to Blend.

Usage: .venv/bin/python -m scripts.ensemble   (then rerun scripts.leaderboard)
"""
import polars as pl

from utils.evaluate import point_metrics, portfolio
from utils.features import build_features
from utils.splits import FIT_END, TUNE, cohort, load_all_preds, save_preds

POOLS = {"Ensemble_W0": ("target-only", "W0")}
EXCLUDE = ("TimesFM3", "Ensemble")


def main() -> None:
    df = build_features(FIT_END)
    members = cohort(df)
    preds = {k: v for k, v in load_all_preds().items() if not k.startswith(EXCLUDE)}
    frame = df.filter(pl.col("D") >= TUNE[0], pl.col("Household_ID").is_in(members)).select("Household_ID", "hour", "D", "lhour", "kwh")
    for name, (p, _) in preds.items():
        frame = frame.join(p.select("Household_ID", "hour", pl.col("pred").alias(name)), on=["Household_ID", "hour"], how="left")
    frame = frame.with_columns([pl.col(m).fill_null(pl.col("Blend")) for m in preds])

    tune = portfolio(frame.filter(pl.col("D").is_between(*TUNE)), members, list(preds))
    score = {m: point_metrics(tune["y"].to_numpy(), tune[m].to_numpy())["nMAE_%"] for m in preds}

    for ens, tracks in POOLS.items():
        pool = sorted((m for m in preds if preds[m][1]["track"] in tracks), key=score.get)
        top = pool[:3]
        print(f"{ens}: tune-block portfolio nMAE " + ", ".join(f"{m} {score[m]:.2f}%" for m in pool) + f"\n  -> members {top}")
        out = frame.select("Household_ID", "hour", pred=pl.mean_horizontal(top))
        save_preds(ens, out, {
            "family": "ensemble", "track": "W0" if ens.endswith("W0") else "W2_oracle",
            "inputs": f"equal-weight mean of {top} (selected on tune block)", "point": "mean of member means",
            "runtime_s": sum(preds[m][1]["runtime_s"] for m in top), "device": "-", "version": "-",
        })


if __name__ == "__main__":
    main()
