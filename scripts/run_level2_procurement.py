"""Level 2: procurement analysis of EXISTING predictions (no model is trained).

    uv run python scripts/run_level2_procurement.py

Inputs : outputs/level1_fast_screening/predictions/pooled_interval_predictions.parquet (pooled AutoGluon, D-1 11:45)
         seasonal_mean_4weeks recomputed deterministically from the raw data (a lookup, not a trained model)
Sample : the 12 sampled weeks / 84 days / 241 households of the fast-screening run
Sign   : error = forecast - actual ; >0 over-forecast (over-procurement), <0 under-forecast (under-procurement)
"""
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import baseline
from src.config import Config
from src.data import load_all, split_dates

POOLED, BASE = "autogluon_pooled", "seasonal_mean_4weeks"
RATIOS = [1.0, 1.5, 2.0, 3.0]                       # under-forecast cost : over-forecast cost
SEASONS = {m: s for s, ms in {"spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11), "winter": (12, 1, 2)}.items() for m in ms}
C_POOLED, C_BASE, C_ACT, C_OVER, C_UNDER = "#2a78d6", "#52514e", "#0b0b0b", "#eb6834", "#2a78d6"   # over=orange, under=blue
EPS = Config().mape_min_actual                       # same 'safe MAPE' threshold as the evaluation module (0.05 kWh)


# ---------------------------------------------------------------------------------------------- metrics
def interval_metrics(a: np.ndarray, p: np.ndarray) -> dict:
    e = p - a
    ok = np.abs(a) >= EPS
    den = np.abs(a) + np.abs(p)
    sm = den > 0
    return {"n": len(a), "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)), "median_AE": np.median(np.abs(e)),
            "bias": np.mean(e), "MAPE_%": 100 * np.mean(np.abs(e[ok]) / np.abs(a[ok])) if ok.any() else np.nan,
            "MAPE_excluded_n": int((~ok).sum()), "MAPE_excluded_%": 100 * (~ok).mean(), "actual_zero_%": 100 * (a == 0).mean(),
            "sMAPE_%": 100 * np.mean(2 * np.abs(e[sm]) / den[sm]), "over_kWh": np.maximum(e, 0).sum(),
            "under_kWh": np.maximum(-e, 0).sum(), "actual_kWh": a.sum()}


def daily_metrics(a: np.ndarray, p: np.ndarray) -> dict:
    e = p - a
    return {"n_days": len(a), "daily_energy_MAE_kWh": np.mean(np.abs(e)), "daily_RMSE_kWh": np.sqrt(np.mean(e ** 2)),
            "daily_median_AE_kWh": np.median(np.abs(e)), "daily_bias_kWh": e.mean(), "daily_median_signed_kWh": np.median(e),
            "daily_over_kWh_total": np.maximum(e, 0).sum(), "daily_under_kWh_total": np.maximum(-e, 0).sum(),
            "daily_over_kWh_mean": np.maximum(e, 0).mean(), "daily_under_kWh_mean": np.maximum(-e, 0).mean(),
            "pct_days_over_%": 100 * (e > 0).mean(), "pct_days_under_%": 100 * (e < 0).mean(),
            "cumulative_signed_kWh": e.sum(), "daily_MAPE_%": 100 * np.mean(np.abs(e) / np.maximum(a, 1e-9))}


def block(df: pd.DataFrame, pcol: str, group: str, model: str) -> list[dict]:
    """Four levels for one model / one household group."""
    rows = []
    hi = interval_metrics(df["actual"].to_numpy(), df[pcol].to_numpy())
    rows.append({"model": model, "group": group, "level": "household_interval", **hi})
    pi = df.groupby("Timestamp")[["actual", pcol]].sum()
    rows.append({"model": model, "group": group, "level": "portfolio_interval", **interval_metrics(pi["actual"].to_numpy(), pi[pcol].to_numpy())})
    hd = df.groupby(["Household_ID", "forecast_date"]).agg(n=("actual", "size"), a=("actual", "sum"), p=(pcol, "sum")).reset_index()
    hd = hd[hd["n"] == 96]
    rows.append({"model": model, "group": group, "level": "household_day", **daily_metrics(hd["a"].to_numpy(), hd["p"].to_numpy())})
    pdd = hd.groupby("forecast_date")[["a", "p"]].sum()
    rows.append({"model": model, "group": group, "level": "portfolio_day", **daily_metrics(pdd["a"].to_numpy(), pdd["p"].to_numpy())})
    return rows


def main():
    cfg = Config()
    src = cfg.output_dir / "level1_fast_screening"
    out = cfg.output_dir / "level2_procurement"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    sel = json.loads((src / "selected_weeks.json").read_text())
    days = pd.DatetimeIndex(sel["forecast_days"])

    # ---- data: pooled predictions + recomputed seasonal-mean baseline on identical keys ----------------------------
    pooled = pd.read_parquet(src / "predictions" / "pooled_interval_predictions.parquet")
    assert pooled["forecast_date"].nunique() == 84 and pooled["Household_ID"].nunique() == 241
    masked = {hh: m for hh, m, _ in load_all(cfg)}
    first = min(m.index.min() for m in masked.values()); last = max(m.index.max() for m in masked.values())
    test_range = split_dates(first, last, cfg)["test"]
    bcfg = dataclasses.replace(cfg, models=(BASE,))
    parts = [baseline.predict_household(h, masked[h], bcfg, test_range)[0] for h in pooled["Household_ID"].unique()]
    b = pd.concat([x for x in parts if len(x)], ignore_index=True)
    b = b[b["Timestamp"].dt.normalize().isin(days)][["Household_ID", "Timestamp", "actual", "prediction"]].rename(
        columns={"prediction": "pred_base", "actual": "actual_b"})
    df = pooled.rename(columns={"prediction": "pred_pooled"}).merge(b, on=["Household_ID", "Timestamp"], how="inner")
    assert np.allclose(df["actual"], df["actual_b"]), "actuals differ between sources"
    df = df.drop(columns="actual_b")
    # keep only complete household-days (all 96 intervals scored by both models)
    n = df.groupby(["Household_ID", "forecast_date"])["actual"].transform("size")
    df = df[n == 96].copy()
    df["hour"] = df["Timestamp"].dt.hour; df["slot"] = df["hour"] * 4 + df["Timestamp"].dt.minute // 15
    df["daytype"] = np.where(df["Timestamp"].dt.dayofweek >= 5, "weekend", "weekday")
    df["season"] = df["forecast_date"].dt.month.map(SEASONS)
    df["week_monday"] = df["forecast_date"] - pd.to_timedelta(df["forecast_date"].dt.dayofweek, unit="D")
    print(f"common sample: {df['Household_ID'].nunique()} households, {df['forecast_date'].nunique()} days, "
          f"{df['actual'].size} intervals, {df.groupby(['Household_ID','forecast_date']).ngroups} household-days", flush=True)
    models = {POOLED: "pred_pooled", BASE: "pred_base"}

    # ---- main metrics (all households + descriptive PV split) ------------------------------------------------------
    rows = []
    for m, pc in models.items():
        rows += block(df, pc, "all", m)
        for g in ("pv", "non_pv"):
            rows += block(df[df["pv_status"] == g], pc, g, m)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    # sanity: pooled MAE on ALL pooled intervals must equal the fast-screening 'all'-scope value
    ref = pd.read_csv(src / "metrics" / "comparison.csv").query("group == 'combined' and scope == 'all'").set_index("model")["overall_MAE"]
    assert abs(pooled["abs_error"].mean() - ref["autogluon_pooled_1145"]) < 1e-4, "pooled MAE does not match fast-screening metrics"
    print(f"pooled MAE on all pooled intervals {pooled['abs_error'].mean():.4f} == fast-screening {ref['autogluon_pooled_1145']:.4f}; "
          f"analysis uses pooled ∩ seasonal_mean_4weeks ({len(df)} intervals; the 4-model common set of the screening run had 1,777,920 because naive_7day covers fewer)", flush=True)

    # ---- daily tables -------------------------------------------------------------------------------------------------
    hd_rows, pd_rows = [], []
    for m, pc in models.items():
        hd = df.groupby(["Household_ID", "pv_status", "forecast_date"]).agg(actual_kWh=("actual", "sum"), predicted_kWh=(pc, "sum")).reset_index()
        hd["signed_error_kWh"] = hd["predicted_kWh"] - hd["actual_kWh"]; hd["abs_error_kWh"] = hd["signed_error_kWh"].abs(); hd.insert(0, "model", m)
        hd_rows.append(hd)
        iv = df.assign(e=df[pc] - df["actual"])
        pint = iv.groupby(["forecast_date", "Timestamp"])["e"].sum().reset_index()
        pint["o"], pint["u"] = pint["e"].clip(lower=0), (-pint["e"]).clip(lower=0)
        iv_over = pint.groupby("forecast_date")[["o", "u"]].sum().rename(columns={"o": "interval_over_kWh", "u": "interval_under_kWh"})
        day = df.groupby("forecast_date").agg(n_households=("Household_ID", "nunique"), actual_kWh=("actual", "sum"), predicted_kWh=(pc, "sum"))
        day["signed_error_kWh"] = day["predicted_kWh"] - day["actual_kWh"]; day["abs_error_kWh"] = day["signed_error_kWh"].abs()
        day["daily_over_kWh"], day["daily_under_kWh"] = day["signed_error_kWh"].clip(lower=0), (-day["signed_error_kWh"]).clip(lower=0)
        day["cumulative_signed_kWh"] = day["signed_error_kWh"].cumsum()
        day = day.join(iv_over).reset_index(); day.insert(0, "model", m); pd_rows.append(day)
    hdf, pdf = pd.concat(hd_rows, ignore_index=True), pd.concat(pd_rows, ignore_index=True)
    hdf.to_csv(out / "household_daily_metrics.csv", index=False); pdf.to_csv(out / "portfolio_daily_metrics.csv", index=False)

    # ---- time patterns -------------------------------------------------------------------------------------------------
    tp = []
    for dim in ("slot", "hour", "daytype", "season", "week_monday"):
        for m, pc in models.items():
            g = df.assign(e=df[pc] - df["actual"]); g["ae"] = g["e"].abs()
            port = g.groupby([dim, "Timestamp"])["e"].sum().reset_index()
            port["ae"] = port["e"].abs()
            hh = g.groupby(dim).agg(household_MAE=("ae", "mean"), household_bias=("e", "mean"), mean_actual=("actual", "mean"))
            pp = port.groupby(dim).agg(portfolio_MAE_kWh=("ae", "mean"), portfolio_bias_kWh=("e", "mean"))
            pp["pct_portfolio_intervals_under_%"] = port.groupby(dim)["e"].apply(lambda s: 100 * (s < 0).mean())
            t = hh.join(pp).reset_index().rename(columns={dim: "value"})
            t.insert(0, "dimension", dim); t.insert(0, "model", m); t["value"] = t["value"].astype(str); tp.append(t)
    tpdf = pd.concat(tp, ignore_index=True); tpdf.to_csv(out / "time_of_day_metrics.csv", index=False)

    # ---- parametric cost sensitivity ---------------------------------------------------------------------------------
    # cost = r * under_kWh + 1 * over_kWh (over-forecast cost normalised to 1); reported per kWh of actual consumption.
    def levels(pcol, scale=1.0):
        d = df.assign(p=df[pcol] * scale)
        pi = d.groupby("Timestamp")[["actual", "p"]].sum(); hd = d.groupby(["Household_ID", "forecast_date"])[["actual", "p"]].sum()
        pdl = d.groupby("forecast_date")[["actual", "p"]].sum()
        out_ = {}
        for name, x in (("portfolio_interval", pi), ("household_day", hd), ("portfolio_day", pdl)):
            e = x["p"] - x["actual"]; out_[name] = (np.maximum(e, 0).sum(), np.maximum(-e, 0).sum(), x["actual"].sum())
        hi = d["p"] - d["actual"]; out_["household_interval"] = (np.maximum(hi, 0).sum(), np.maximum(-hi, 0).sum(), d["actual"].sum())
        return out_
    cost_rows = []
    for r in RATIOS:
        # hindsight uplift: scalar on the pooled forecast minimising portfolio-interval cost IN-SAMPLE (illustration of the bias, not a method)
        grid = np.arange(0.90, 1.60, 0.01)
        best = min(grid, key=lambda s: (lambda o: o["portfolio_interval"][1] * r + o["portfolio_interval"][0])(levels("pred_pooled", s)))
        strategies = {POOLED: levels("pred_pooled"), BASE: levels("pred_base"), f"{POOLED}_hindsight_uplift": levels("pred_pooled", best)}
        for sname, lv in strategies.items():
            for lvl, (ov, un, act) in lv.items():
                c = r * un + ov
                cost_rows.append({"uplift_factor": best if "uplift" in sname else 1.0, "under_to_over_cost_ratio": r, "implied_optimal_quantile": r / (1 + r), "strategy": sname, "level": lvl,
                                  "over_kWh": ov, "under_kWh": un, "cost_units": c, "cost_per_kWh_consumed": c / act})
    cost = pd.DataFrame(cost_rows)
    base_c = cost[cost.strategy == BASE].set_index(["under_to_over_cost_ratio", "level"])["cost_units"]
    cost["cost_index_vs_baseline"] = [c / base_c[(r, l)] for c, r, l in zip(cost.cost_units, cost.under_to_over_cost_ratio, cost.level)]
    cost.to_csv(out / "cost_sensitivity.csv", index=False)

    # ---- plots --------------------------------------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    P = pdf[pdf.model == POOLED].set_index("forecast_date"); B = pdf[pdf.model == BASE].set_index("forecast_date")
    x = np.arange(len(P))
    gaps = np.where(np.diff(P.index.values.astype("datetime64[D]").astype(int)) > 1)[0] + 1
    def ax_fmt(ax):
        for g in gaps: ax.axvline(g - 0.5, color="#c3c2b7", lw=0.8)
        wk = list(range(0, len(P), 7)); ax.set_xticks(wk); ax.set_xticklabels([P.index[i].strftime("%b %d") for i in wk], rotation=60, fontsize=8)
    fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(x, P["actual_kWh"], color=C_ACT, lw=1.8, label="actual")
    ax.plot(x, P["predicted_kWh"], color=C_POOLED, lw=1.4, label="pooled AutoGluon"); ax.plot(x, B["predicted_kWh"], color=C_BASE, lw=1.1, ls="--", label="seasonal mean 4w")
    ax_fmt(ax); ax.set(ylabel="portfolio energy per day [kWh]", title="Daily portfolio energy: actual vs forecast (84 sampled days, 12 weeks; vertical lines = week gaps)"); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "plots" / "1_daily_portfolio_actual_vs_forecast.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (title, series) in zip(axs, (("Household-day", {m: hdf[hdf.model == m]["signed_error_kWh"] for m in models}),
                                          ("Portfolio-day", {m: pdf[pdf.model == m]["signed_error_kWh"] for m in models}))):
        lo, hi = np.percentile(np.concatenate([s.to_numpy() for s in series.values()]), [0.5, 99.5]); bins = np.linspace(lo, hi, 45)
        for m, c in ((POOLED, C_POOLED), (BASE, C_BASE)): ax.hist(series[m].clip(lo, hi), bins=bins, alpha=0.5, color=c, label=f"{m} (median {series[m].median():.1f})")
        ax.axvline(0, color=C_ACT, lw=1); ax.set(title=f"{title} signed error", xlabel="forecast - actual [kWh/day]  (<0 under-forecast)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "plots" / "2_daily_signed_error_distribution.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.8)); e = P["signed_error_kWh"]
    ax.bar(x, e, color=np.where(e >= 0, C_OVER, C_UNDER), width=0.8); ax.axhline(0, color=C_ACT, lw=0.8); ax_fmt(ax)
    ax.set(ylabel="forecast - actual [kWh/day]", title="Pooled model: portfolio daily signed error (orange = over-forecast, blue = under-forecast)")
    fig.tight_layout(); fig.savefig(out / "plots" / "3_portfolio_daily_signed_error.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.8)); s = tpdf[(tpdf.dimension == "slot")].copy(); s["v"] = s["value"].astype(int)
    for m, c, ls in ((POOLED, C_POOLED, "-"), (BASE, C_BASE, "--")):
        t = s[s.model == m].sort_values("v"); ax.plot(t["v"] / 4, t["portfolio_bias_kWh"], color=c, ls=ls, lw=1.6, label=m)
    ax.axhline(0, color=C_ACT, lw=0.8); ax.set(xlim=(0, 24), xlabel="hour of day (UTC)", ylabel="portfolio mean error [kWh per 15 min]", title="Mean portfolio forecast error by time of day (<0 = under-forecast)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "plots" / "4_mean_error_by_time_of_day.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, lvl, title in zip(axs, ("portfolio_interval", "household_interval"), ("Portfolio-interval (summed over 241 households)", "Household-interval")):
        mm = metrics[(metrics.group == "all") & (metrics.level == lvl)].set_index("model"); xs = np.arange(2)
        ax.bar(xs - 0.2, [mm.loc[m, "over_kWh"] / 1e3 for m in models], 0.38, color=C_OVER, label="overforecast"); ax.bar(xs + 0.2, [mm.loc[m, "under_kWh"] / 1e3 for m in models], 0.38, color=C_UNDER, label="underforecast")
        ax.set_xticks(xs); ax.set_xticklabels(["pooled", "seasonal mean 4w"]); ax.set(title=title, ylabel="energy [MWh] over 84 days"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "plots" / "5_over_vs_under_energy.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4)); c = cost[cost.level == "portfolio_interval"]
    for sname, col, ls in ((POOLED, C_POOLED, "-"), (BASE, C_BASE, "--"), ([k for k in c.strategy.unique() if "hindsight" in k][0], "#1baf7a", ":")):
        t = c[c.strategy == sname] if "hindsight" not in sname else c[c.strategy.str.contains("hindsight")]
        t = t.groupby("under_to_over_cost_ratio")["cost_per_kWh_consumed"].mean(); ax.plot(t.index, t.values, color=col, ls=ls, marker="o", label="pooled x best uplift (hindsight, in-sample)" if "hindsight" in sname else sname)
    ax.set(xlabel="under-forecast cost : over-forecast cost", ylabel="cost per kWh consumed (over-forecast cost = 1)", title="Portfolio-interval cost vs cost ratio"); ax.set_xticks(RATIOS); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "plots" / "6_cost_ratio_sensitivity.png", dpi=130); plt.close(fig)

    # ---- concise summary table -------------------------------------------------------------------------------------------------
    def g(m, lvl, col, grp="all"): return metrics[(metrics.model == m) & (metrics.level == lvl) & (metrics.group == grp)][col].iloc[0]
    summ = pd.DataFrame({m: {
        "Household-interval MAE [kWh/15min]": g(m, "household_interval", "MAE"), "Household-interval RMSE": g(m, "household_interval", "RMSE"),
        "MAPE % (actual>=0.05 kWh)": g(m, "household_interval", "MAPE_%"), "sMAPE %": g(m, "household_interval", "sMAPE_%"),
        "Household daily-energy MAE [kWh/day]": g(m, "household_day", "daily_energy_MAE_kWh"), "Household daily bias [kWh/day]": g(m, "household_day", "daily_bias_kWh"),
        "Portfolio-interval MAE [kWh/15min, 241 hh]": g(m, "portfolio_interval", "MAE"), "Portfolio-day MAE [kWh/day]": g(m, "portfolio_day", "daily_energy_MAE_kWh"),
        "Portfolio-day bias [kWh/day]": g(m, "portfolio_day", "daily_bias_kWh"), "Portfolio over-forecast total [MWh]": g(m, "portfolio_interval", "over_kWh") / 1e3,
        "Portfolio under-forecast total [MWh]": g(m, "portfolio_interval", "under_kWh") / 1e3, "Portfolio days under-forecast %": g(m, "portfolio_day", "pct_days_under_%"),
        "Portfolio days over-forecast %": g(m, "portfolio_day", "pct_days_over_%")} for m in models})
    summ.to_csv(out / "summary_table.csv")
    (out / "run_metadata.json").write_text(json.dumps({"source": str(src), "models": list(models), "households": 241, "forecast_days": 84,
        "selected_weeks": sel["selected_weeks_monday"], "error_definition": "forecast - actual", "mape_min_actual_kWh": EPS,
        "cost_ratios_under_to_over": RATIOS, "common_intervals": int(len(df)), "no_models_trained": True}, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summ.round(4).to_string())
    print(metrics[metrics.group == "all"].set_index(["model", "level"])[["MAE", "RMSE", "bias", "median_AE", "MAPE_%", "MAPE_excluded_%", "actual_zero_%", "sMAPE_%", "over_kWh", "under_kWh"]].round(4).to_string())
    print(metrics[metrics.level.isin(["household_day", "portfolio_day"]) & (metrics.group == "all")].set_index(["model", "level"]).drop(columns=["n", "MAE", "RMSE", "median_AE", "bias", "MAPE_%", "MAPE_excluded_n", "MAPE_excluded_%", "actual_zero_%", "sMAPE_%", "over_kWh", "under_kWh", "actual_kWh"]).round(3).T.to_string())
    print(metrics[metrics.group != "all"].query("level in ['household_interval','household_day']").pivot_table(index=["group", "model"], columns="level", values=["MAE", "bias", "daily_energy_MAE_kWh", "daily_bias_kWh", "pct_days_under_%", "pct_days_over_%"]).round(4).to_string())
    print(cost[cost.level.isin(["portfolio_interval", "household_day"])].pivot_table(index=["level", "under_to_over_cost_ratio"], columns="strategy", values="cost_per_kWh_consumed").round(4).to_string())
    print(cost[cost.level == "portfolio_interval"].pivot_table(index="under_to_over_cost_ratio", columns="strategy", values="cost_index_vs_baseline").round(3).to_string())
    for dim in ("daytype", "season"):
        print(tpdf[tpdf.dimension == dim].pivot_table(index="value", columns="model", values=["household_MAE", "portfolio_bias_kWh", "household_bias"]).round(4).to_string())
    print(tpdf[(tpdf.dimension == "hour") & (tpdf.model == POOLED)][["value", "household_MAE", "household_bias", "portfolio_bias_kWh", "pct_portfolio_intervals_under_%"]].round(4).to_string(index=False))
    wk = tpdf[(tpdf.dimension == "week_monday") & (tpdf.model == POOLED)][["value", "household_MAE", "portfolio_bias_kWh", "pct_portfolio_intervals_under_%"]]; print(wk.round(4).to_string(index=False))
    print("portfolio mean households/day:", pdf.n_households.mean().round(1), "| portfolio mean actual kWh/day:", P.actual_kWh.mean().round(1))
    h = metrics[(metrics.group == "all") & (metrics.model == POOLED)].set_index("level")
    print("cancellation: mean household abs daily error", h.loc["household_day", "daily_energy_MAE_kWh"], " | portfolio abs daily error / n_hh:", (P.abs_error_kWh / P.n_households).mean())


if __name__ == "__main__":
    main()
