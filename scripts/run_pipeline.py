"""End-to-end run: features -> baselines + HGB -> household & portfolio evaluation -> quantile bids.

Usage: .venv/bin/python -m scripts.run_pipeline
"""
import time
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor

from utils.data import PROCESSED, ROOT, build_processed
from utils.evaluate import C_OVER, C_UNDER, bootstrap_regret_diff, point_metrics, portfolio, regret_per_mwh, rolling_quantile_bids
from utils.features import BASE_FEATURES, W0_FEATURES, W2_FEATURES, build_features, check_availability
from utils.splits import CALIB, FIT_END, TEST, TUNE, cohort as get_cohort, save_preds

TAU_STAR = C_UNDER / (C_UNDER + C_OVER)
RATIOS = [(1, 4), (1, 2), (1, 1), (5, 4), (2, 1), (4, 1)]  # c_under : c_over sweep
OUT = ROOT / "results"


def fit_hgb(df: pl.DataFrame, feats: list[str]) -> HistGradientBoostingRegressor:
    fit = df.filter(pl.col("D") <= FIT_END, pl.col("kwh").is_not_null())
    tune = df.filter(pl.col("D").is_between(*TUNE), pl.col("kwh").is_not_null())
    model = HistGradientBoostingRegressor(
        max_iter=1000, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, n_iter_no_change=30, categorical_features=[feats.index("station")], random_state=0,
    )
    model.fit(fit.select(feats).to_numpy(), fit["kwh"].to_numpy(), X_val=tune.select(feats).to_numpy(), y_val=tune["kwh"].to_numpy())
    print(f"  HGB {len(feats)} features: {model.n_iter_} iterations, {fit.height:,} fit rows")
    return model


def main() -> None:
    OUT.mkdir(exist_ok=True)
    if not (PROCESSED / "hourly.parquet").exists():
        build_processed()
    df = build_features(FIT_END)
    check_availability(df)

    # Baselines with a fixed fallback chain so every row has a forecast
    global_mean = df.filter(pl.col("D") <= FIT_END)["kwh"].mean()
    fallback = [pl.col("b3"), pl.col("hh_mean_7d"), pl.col("hh_mean_28d"), pl.lit(global_mean)]
    df = df.with_columns(
        B1_lastweek=pl.coalesce(pl.col("lag_7"), *fallback),
        B2_twodays=pl.coalesce(pl.col("lag_2"), *fallback),
        B3_weekmean=pl.coalesce(*fallback),
    )
    base = ["B1_lastweek", "B2_twodays", "B3_weekmean"]
    tune = df.filter(pl.col("D").is_between(*TUNE), pl.col("kwh").is_not_null())
    w, _ = nnls(tune.select(base).to_numpy(), tune["kwh"].to_numpy())
    print("blend weights", dict(zip(base, w.round(3))))
    df = df.with_columns(Blend=pl.sum_horizontal([pl.col(b) * wi for b, wi in zip(base, w)]))

    # Global HGB: W0 = past-only weather (operational), W2 = actual weather on D (oracle diagnostic)
    feats_w0, feats_w2 = BASE_FEATURES + W0_FEATURES, list(dict.fromkeys(BASE_FEATURES + W2_FEATURES))
    pred = df.filter(pl.col("D") >= TUNE[0])
    runtime = {}
    for name, feats in [("HGB_W0", feats_w0), ("HGB_W2_oracle", feats_w2)]:
        t0 = time.time()
        m = fit_hgb(df, feats)
        pred = pred.with_columns(pl.Series(name, np.clip(m.predict(pred.select(feats).to_numpy()), 0, None)))
        runtime[name] = round(time.time() - t0, 1)
    models = base + ["Blend", "HGB_W0", "HGB_W2_oracle"]

    # Store every model in the common format used by scripts/leaderboard.py
    meta = {
        "B1_lastweek": ("baseline", "target-only", "same hour D-7"),
        "B2_twodays": ("baseline", "target-only", "same hour D-2"),
        "B3_weekmean": ("baseline", "target-only", "mean of same hour D-8..D-2"),
        "Blend": ("baseline", "target-only", "NNLS blend of B1-B3, weights fit on tune block"),
        "HGB_W0": ("tree", "W0", "lags, calendar, household info, past-only weather"),
        "HGB_W2_oracle": ("tree", "W2_oracle", "as W0 plus actual weather on D"),
    }
    for m in models:
        family, track, inputs = meta[m]
        save_preds(m, pred.select("Household_ID", "hour", pred=pl.col(m)), {
            "family": family, "track": track, "inputs": inputs, "point": "conditional mean",
            "runtime_s": runtime.get(m, 0), "device": "cpu", "version": "scikit-learn HistGradientBoostingRegressor" if m.startswith("HGB") else "-",
        })

    # ---- Household level (test, identical mask for all models). nMAE is pooled over household-hours;
    # median_hh_nMAE_% is the median of per-household nMAEs.
    test = pred.filter(pl.col("D").is_between(*TEST), pl.col("kwh").is_not_null())

    def hh_metrics(t: pl.DataFrame, m: str) -> dict:
        per_hh = t.group_by("Household_ID").agg(n=100 * (pl.col(m) - pl.col("kwh")).abs().mean() / pl.col("kwh").mean())
        return {**point_metrics(t["kwh"].to_numpy(), t[m].to_numpy()), "median_hh_nMAE_%": per_hh["n"].median()}

    hh_rows = [{"model": m, **hh_metrics(test, m)} for m in models]
    for pv, label in [(1.0, "PV"), (0.0, "no PV")]:
        t = test.filter(pl.col("pv") == pv)
        hh_rows += [{"model": m, "subgroup": label, **hh_metrics(t, m)} for m in ["Blend", "HGB_W0"]]
    hh_tab = pl.DataFrame(hh_rows).with_columns(pl.col("subgroup").fill_null("all")).rename({"nMAE_%": "pooled_nMAE_%"})
    hh_tab.write_csv(OUT / "household_metrics.csv")

    # ---- Portfolio over reporting members (cohort fixed with information up to 2022-12-29)
    cohort = get_cohort(df)
    port = portfolio(pred.filter(pl.col("D") >= CALIB[0]), cohort, models)
    pt = port.filter(pl.col("D").is_between(*TEST))
    assert pt["D"].n_unique() == 364, "test must cover 2023-03-01..2024-02-27"
    print(f"cohort {len(cohort)} households | test hours scored {pt.height} | mean coverage {pt['coverage'].mean():.1%}")

    # Daily metrics only on complete local days (every expected 23/24/25 hours scored)
    midnight = pl.col("D").cast(pl.Datetime("us")).dt.replace_time_zone("Europe/Zurich")
    daily = (
        pt.group_by("D").agg(hours=pl.len(), y=pl.col("y").sum(), *[pl.col(m).sum() for m in models])
        .with_columns(expected=((midnight.dt.offset_by("1d") - midnight).dt.total_hours()))
        .filter(pl.col("hours") == pl.col("expected"))
    )
    print(f"complete days for daily metrics: {daily.height} of {pt['D'].n_unique()}")
    port_tab = pl.DataFrame([
        {"model": m, **point_metrics(pt["y"].to_numpy(), pt[m].to_numpy()),
         "daily_nMAE_%": 100 * (daily[m] - daily["y"]).abs().mean() / daily["y"].mean(),
         "regret_EUR_per_MWh_bid_forecast": regret_per_mwh(pt["y"].to_numpy(), pt[m].to_numpy())}
        for m in models
    ])
    port_tab.write_csv(OUT / "portfolio_metrics.csv")

    # ---- Level 3: quantile bids from recent portfolio errors (per local hour, 56 days ending D-2)
    taus = sorted({0.1, 0.5, 0.9, TAU_STAR} | {cu / (cu + co) for cu, co in RATIOS})
    bids = {m: rolling_quantile_bids(port, m, taus).filter(pl.col("D").is_between(*TEST)) for m in ["Blend", "HGB_W0"]}

    cov_rows, sweep_rows = [], []
    for m, b in bids.items():
        cov_rows.append({"model": m, **{f"P(y<=q{t:.3g})": (b["y"] <= b[f"bid_{t}"]).mean() for t in [0.1, 0.5, TAU_STAR, 0.9]},
                         "coverage_10_90_nominal_80": ((b["y"] > b["bid_0.1"]) & (b["y"] <= b["bid_0.9"])).mean()})
        for cu, co in RATIOS:
            t = cu / (cu + co)
            cu_e, co_e = 90 * cu / (cu + co), 90 * co / (cu + co)  # keep c_under + c_over = 90 €/MWh
            sweep_rows.append({
                "model": m, "c_under:c_over": f"{cu}:{co}", "tau": round(t, 3),
                "regret_bid_forecast": regret_per_mwh(b["y"].to_numpy(), b[m].to_numpy(), cu_e, co_e),
                "regret_bid_quantile": regret_per_mwh(b["y"].to_numpy(), b[f"bid_{t}"].to_numpy(), cu_e, co_e),
            })
    cov_tab, sweep_tab = pl.DataFrame(cov_rows), pl.DataFrame(sweep_rows)
    cov_tab.write_csv(OUT / "quantile_coverage.csv")
    sweep_tab.write_csv(OUT / "regret_sweep.csv")

    b = bids["HGB_W0"]
    lo, hi = bootstrap_regret_diff(b, f"bid_{TAU_STAR}", "HGB_W0")
    by_cov = b.with_columns(rep=pl.col("coverage").cut([0.95, 0.98], labels=["<95%", "95-98%", ">=98%"])).group_by("rep").agg(
        hours=pl.len(), hit_rate_tau_star=(pl.col("y") <= pl.col(f"bid_{TAU_STAR}")).mean()
    ).sort("rep")

    # ---- Figures
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharey=False)
    for ax, start in zip(axes, [date(2023, 12, 4), date(2023, 6, 5)]):
        wk = b.filter(pl.col("D").is_between(start, date.fromordinal(start.toordinal() + 6)))
        x = wk["hour"].dt.convert_time_zone("Europe/Zurich").to_list()
        ax.fill_between(x, wk["bid_0.1"], wk["bid_0.9"], alpha=0.25, label="10–90% range")
        ax.plot(x, wk["y"], color="black", lw=1.2, label="actual")
        ax.plot(x, wk["HGB_W0"], lw=1.2, label="forecast (HGB, past-only weather)")
        ax.set_ylabel("kWh per hour")
        ax.set_title(f"Portfolio over reporting members, week of {start}")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "portfolio_weeks.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4))
    s = sweep_tab.filter(pl.col("model") == "HGB_W0")
    ax.plot(s["tau"], s["regret_bid_forecast"], "o-", label="bid the forecast")
    ax.plot(s["tau"], s["regret_bid_quantile"], "o-", label="bid the estimated τ-quantile")
    ax.set_xlabel("τ = c_under / (c_under + c_over)")
    ax.set_ylabel("regret, € per MWh served")
    ax.set_title("Retrospective masked regret (assumed c_under + c_over = 90 €/MWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "regret_sweep.png", dpi=130)

    pred.select("Household_ID", "hour", "D", "lhour", "kwh", *models).write_parquet(OUT / "predictions.parquet")

    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_float_precision(3); pl.Config.set_tbl_width_chars(220)
    print("\nHousehold level, test (kWh per hour)\n", hh_tab)
    print("\nPortfolio over reporting members, test (kWh per hour)\n", port_tab)
    print("\nQuantile coverage, test\n", cov_tab)
    print(f"\nRegret sweep\n", sweep_tab)
    print(f"\nHGB_W0: τ*={TAU_STAR:.3f} bid minus forecast bid, retrospective masked regret diff, 95% CI (7-day block bootstrap): [{lo:.2f}, {hi:.2f}] €/MWh")
    print("\nτ* hit rate by reporting fraction\n", by_cov)


if __name__ == "__main__":
    main()
