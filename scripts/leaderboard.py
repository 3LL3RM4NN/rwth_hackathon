"""One leaderboard for every model in results/preds/, scored on the shared contract.

Cohort: the fixed 255-household portfolio. Test: 2023-03-01..2024-02-27, hourly kWh.
Missing forecasts are filled with the Blend baseline and the fallback rate is reported.

Usage: .venv/bin/python -m scripts.leaderboard
"""
import polars as pl

from utils.data import ROOT
from utils.evaluate import C_OVER, C_UNDER, block_bootstrap_diff, point_metrics, portfolio, regret_per_mwh, rolling_quantile_bids
from utils.features import build_features
from utils.splits import CALIB, FIT_END, TEST, cohort, load_all_preds

TAU_STAR = C_UNDER / (C_UNDER + C_OVER)
REFERENCE = "HGB_W0"


def main() -> None:
    df = build_features(FIT_END)
    members = cohort(df)
    frame = df.filter(pl.col("D") >= CALIB[0], pl.col("Household_ID").is_in(members)).select("Household_ID", "hour", "D", "lhour", "kwh")

    preds = load_all_preds()
    models = list(preds)
    for name, (p, _) in preds.items():
        frame = frame.join(p.select("Household_ID", "hour", pl.col("pred").alias(name)), on=["Household_ID", "hour"], how="left")
    scored = frame.filter(pl.col("D").is_between(*TEST), pl.col("kwh").is_not_null())
    fallback = {m: scored[m].null_count() / scored.height for m in models}
    frame = frame.with_columns([pl.col(m).fill_null(pl.col("Blend")) for m in models])

    test = frame.filter(pl.col("D").is_between(*TEST), pl.col("kwh").is_not_null())
    port = portfolio(frame, members, models)
    pt = port.filter(pl.col("D").is_between(*TEST))

    midnight = pl.col("D").cast(pl.Datetime("us")).dt.replace_time_zone("Europe/Zurich")
    daily = (
        pt.group_by("D").agg(hours=pl.len(), y=pl.col("y").sum(), *[pl.col(m).sum() for m in models])
        .with_columns(expected=(midnight.dt.offset_by("1d") - midnight).dt.total_hours())
        .filter(pl.col("hours") == pl.col("expected"))
    )

    rows = []
    for m in models:
        meta = preds[m][1]
        hh = point_metrics(test["kwh"].to_numpy(), test[m].to_numpy())
        pm = point_metrics(pt["y"].to_numpy(), pt[m].to_numpy())
        b = rolling_quantile_bids(port, m, [0.1, 0.9, TAU_STAR]).filter(pl.col("D").is_between(*TEST))
        y, f = pl.col("y"), pl.col(m)
        lo, hi = block_bootstrap_diff(pt, (f - y).abs(), (pl.col(REFERENCE) - y).abs(), y)
        rows.append({
            "model": m, "family": meta["family"], "track": meta["track"],
            "port_MAE_kWh": pm["MAE"], "port_nMAE_%": pm["nMAE_%"],
            "port_nMAE_vs_HGB_W0_pp_CI": f"[{100 * lo:+.2f}, {100 * hi:+.2f}]",
            "port_bias_kWh": pm["bias"],
            "daily_nMAE_%": 100 * (daily[m] - daily["y"]).abs().mean() / daily["y"].mean(),
            "hh_nMAE_%": hh["nMAE_%"],
            "regret_bid_forecast": regret_per_mwh(b["y"].to_numpy(), b[m].to_numpy()),
            "regret_bid_q_tau_star": regret_per_mwh(b["y"].to_numpy(), b[f"bid_{TAU_STAR}"].to_numpy()),
            "coverage_10_90": ((b["y"] > b["bid_0.1"]) & (b["y"] <= b["bid_0.9"])).mean(),
            "point": meta["point"], "fallback_%": 100 * fallback[m], "runtime_s": meta["runtime_s"], "device": meta["device"],
        })
    board = pl.DataFrame(rows).sort("track", "port_nMAE_%")
    board.write_csv(ROOT / "results" / "leaderboard.csv")

    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_float_precision(2); pl.Config.set_tbl_width_chars(250)
    print(f"cohort {len(members)} | test hours {pt.height} | complete days {daily.height}")
    print(board.drop("point", "device"))


if __name__ == "__main__":
    main()
