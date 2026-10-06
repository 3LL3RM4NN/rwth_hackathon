"""Metrics, portfolio aggregation, procurement regret and quantile bids."""
import numpy as np
import polars as pl

C_UNDER, C_OVER = 50.0, 40.0  # €/MWh, assumed incremental costs of buying too little / too much


def point_metrics(y: np.ndarray, f: np.ndarray) -> dict:
    e = f - y
    return {
        "MAE": np.mean(np.abs(e)),
        "RMSE": np.sqrt(np.mean(e**2)),
        "bias": np.mean(e),
        "nMAE_%": 100 * np.mean(np.abs(e)) / np.mean(y),
    }


def regret_per_mwh(y_kwh: np.ndarray, q_kwh: np.ndarray, c_under=C_UNDER, c_over=C_OVER) -> float:
    """Simulated procurement regret vs perfect day-ahead purchase, in € per actual MWh served."""
    under = np.clip(y_kwh - q_kwh, 0, None)
    over = np.clip(q_kwh - y_kwh, 0, None)
    return (c_under * under.sum() + c_over * over.sum()) / y_kwh.sum()


def portfolio(df: pl.DataFrame, cohort: list[int], models: list[str]) -> pl.DataFrame:
    """Portfolio over reporting members: per hour, sum actual and forecasts over cohort members with an
    observed complete hour (mask M_t). The mask depends only on the target, never on the forecasts."""
    sub = df.filter(pl.col("Household_ID").is_in(cohort) & pl.col("kwh").is_not_null())
    assert sub.select([pl.col(m).null_count() for m in models]).sum_horizontal().item() == 0, "missing forecasts"
    return (
        sub.group_by("hour")
        .agg(pl.col("D").first(), pl.col("lhour").first(), n=pl.len(), y=pl.col("kwh").sum(), *[pl.col(m).sum() for m in models])
        .with_columns(coverage=pl.col("n") / len(cohort))
        .sort("hour")
    )


def rolling_quantile_bids(port: pl.DataFrame, model: str, taus: list[float], window_days=56, lag_days=2) -> pl.DataFrame:
    """Bid = forecast × (1 + q_tau of relative portfolio errors), with q_tau estimated per local hour from
    the `window_days` days ending D-2 (outcomes observable at the forecast origin)."""
    p = port.with_columns(r=(pl.col("y") - pl.col(model)) / pl.col(model))
    days = pl.date_range(p["D"].min(), p["D"].max(), "1d", eager=True)
    R = (
        pl.DataFrame({"D": days}).join(
            p.group_by("D", "lhour").agg(r=pl.col("r").mean()).pivot(on="lhour", index="D", values="r", sort_columns=True),
            on="D", how="left",
        )
    )
    hours = [c for c in R.columns if c != "D"]
    M = R.select(hours).to_numpy().astype(float)
    out = []
    for i, d in enumerate(days):
        lo, hi = i - lag_days - window_days + 1, i - lag_days + 1
        if lo < 0:
            continue
        Q = np.nanquantile(M[lo:hi], taus, axis=0)  # (n_taus, n_hours)
        for j, h in enumerate(hours):
            out.append([d, int(h), *Q[:, j]])
    qt = pl.DataFrame(out, schema=["D", "lhour", *[f"q{t}" for t in taus]], orient="row").with_columns(pl.col("lhour").cast(pl.Int8))
    p = p.join(qt, on=["D", "lhour"], how="inner")
    return p.with_columns([(pl.col(model) * (1 + pl.col(f"q{t}"))).alias(f"bid_{t}") for t in taus])


def hourly_cost(y, q, c_under=C_UNDER, c_over=C_OVER) -> pl.Expr:
    return c_under * (y - q).clip(lower_bound=0) + c_over * (q - y).clip(lower_bound=0)


def bootstrap_regret_diff(port: pl.DataFrame, bid_a: str, bid_b: str, block_days=7, n_boot=2000, seed=0) -> tuple[float, float]:
    """95% interval of regret(a) - regret(b) in €/MWh. Paired moving-block bootstrap over contiguous
    `block_days`-day blocks, since weather and the rolling calibration make neighbouring days dependent."""
    d = (
        port.group_by("D")
        .agg(
            ca=hourly_cost(pl.col("y"), pl.col(bid_a)).sum(),
            cb=hourly_cost(pl.col("y"), pl.col(bid_b)).sum(),
            y=pl.col("y").sum(),
        )
        .sort("D")
    )
    ca, cb, y = (d[c].to_numpy() for c in ("ca", "cb", "y"))
    n = len(y)
    starts = np.random.default_rng(seed).integers(0, n - block_days + 1, (n_boot, -(-n // block_days)))
    idx = (starts[:, :, None] + np.arange(block_days)).reshape(n_boot, -1)[:, :n]
    diff = (ca[idx].sum(1) - cb[idx].sum(1)) / y[idx].sum(1)
    return tuple(np.percentile(diff, [2.5, 97.5]))
