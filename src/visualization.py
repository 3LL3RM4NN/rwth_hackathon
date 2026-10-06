"""Four plots, kept deliberately small."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"actual": "#222222", "naive_1day": "#1f77b4", "naive_7day": "#d95f02", "seasonal_mean_4weeks": "#1b9e77"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_example_days(df: pd.DataFrame, hh: str, days: pd.DatetimeIndex, path, title):
    """Actual vs predictions of one household over the given consecutive days."""
    d = df[(df["Household_ID"] == hh) & (df["Timestamp"].dt.normalize().isin(days))]
    fig, ax = plt.subplots(figsize=(11 if len(days) > 1 else 8, 4))
    act = d[d["model"] == d["model"].iloc[0]].set_index("Timestamp")["actual"]
    ax.plot(act.index, act.values, color=COLORS["actual"], lw=1.8, label="actual")
    for m, g in d.groupby("model"):
        ax.plot(g["Timestamp"], g["prediction"], color=COLORS.get(m), lw=1, label=m)
    ax.set(title=f"{title} (household {hh})", ylabel="kWh per 15 min", xlabel="Timestamp (UTC)")
    ax.legend(ncol=4, fontsize=8)
    _save(fig, path)


def plot_mean_profile(qh: pd.DataFrame, path):
    """Mean actual vs mean forecast per quarter-hour (qh = per-quarter-hour metrics, one scope)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    first = qh[qh["model"] == qh["model"].iloc[0]].sort_values("qh")
    ax.plot(first["qh"] / 4, first["mean_actual"], color=COLORS["actual"], lw=2, label="actual")
    for m, g in qh.groupby("model"):
        g = g.sort_values("qh")
        ax.plot(g["qh"] / 4, g["mean_pred"], color=COLORS.get(m), lw=1.2, label=m)
    ax.set(title="Mean daily load profile, test period, all households",
           xlabel="Hour of day (UTC)", ylabel="kWh per 15 min", xlim=(0, 24))
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_mae_distribution(hh_metrics: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    hi = hh_metrics["MAE"].quantile(0.99)
    for m, g in hh_metrics.groupby("model"):
        ax.hist(g["MAE"].clip(upper=hi), bins=40, alpha=0.45, color=COLORS.get(m), label=f"{m} (median {g['MAE'].median():.3f})")
    ax.set(title="Per-household MAE, test period (clipped at 99th pct)", xlabel="MAE [kWh per 15 min]", ylabel="households")
    ax.legend(fontsize=8)
    _save(fig, path)
