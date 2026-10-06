"""One-week plots of the portfolio forecast against the actual consumption.

Four panels: strict weather / actual weather (upper bound) x without / with Chronos-2.
The week is chosen without cherry-picking: the winter test week (Mon-Sun, complete days) whose
LightGBM v2 portfolio error is closest to the median over all winter weeks.

Usage: .venv/bin/python -m scripts.plot_week [--week 2024-01-01]
(--week picks a specific Monday; the title then states how that week ranks among all test weeks.)
"""
import argparse
from datetime import date, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from utils.data import ROOT, TZ
from utils.evaluate import portfolio, rolling_quantile_bids
from utils.features import build_features
from utils.splits import CALIB, FIT_END, PREDS, TEST, cohort

# Fixed identity per model (validated categorical slots 1-3; actual and baseline are neutral ink)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#8a8984", "#e4e3df", "#fcfcfb"
STYLE = {
    "actual": dict(color=INK, lw=1.9, label="Actual"),
    "B1_lastweek": dict(color=MUTED, lw=1.5, label="Same hour last week"),
    "Toto2_target-only": dict(color="#eb6834", lw=1.6, label="Toto 2.0"),
    "Chronos2": dict(color="#1baf7a", lw=1.6, label="Chronos-2"),
    "main": dict(color="#2a78d6", lw=1.9),
}
PANELS = [
    ("Strict weather", "LightGBM_W0_v2", "LightGBM v2", "Chronos2_W0", False),
    ("Strict weather", "LightGBM_W0_v2", "LightGBM v2", "Chronos2_W0", True),
    ("Actual weather", "LightGBM_W2_oracle", "LightGBM", "Chronos2_W2_oracle", False),
    ("Actual weather", "LightGBM_W2_oracle", "LightGBM", "Chronos2_W2_oracle", True),
]
OUT = ROOT / "results"


def load_portfolio() -> pl.DataFrame:
    df = build_features(FIT_END)
    members = cohort(df)
    frame = df.filter(pl.col("D") >= CALIB[0], pl.col("Household_ID").is_in(members)).select("Household_ID", "hour", "D", "lhour", "kwh")
    models = ["Blend", "B1_lastweek", "Toto2_target-only", "LightGBM_W0_v2", "LightGBM_W2_oracle", "Chronos2_W0", "Chronos2_W2_oracle"]
    for m in models:
        p = pl.read_parquet(PREDS / f"{m}.parquet").select("Household_ID", "hour", pl.col("pred").alias(m))
        frame = frame.join(p, on=["Household_ID", "hour"], how="left")
    frame = frame.with_columns([pl.col(m).fill_null(pl.col("Blend")) for m in models])  # same fallback as the leaderboard
    port = portfolio(frame, members, models)
    for main in ("LightGBM_W0_v2", "LightGBM_W2_oracle"):
        b = rolling_quantile_bids(port, main, [0.1, 0.9]).select("hour", pl.col("bid_0.1").alias(f"{main}_lo"), pl.col("bid_0.9").alias(f"{main}_hi"))
        port = port.join(b, on="hour", how="left")
    return port.filter(pl.col("D").is_between(*TEST))


def pick_week(port: pl.DataFrame) -> date:
    midnight = pl.col("D").cast(pl.Datetime("us")).dt.replace_time_zone(TZ)
    days = (port.group_by("D").agg(n=pl.len(), err=(pl.col("LightGBM_W0_v2") - pl.col("y")).abs().sum(), y=pl.col("y").sum())
            .with_columns(expected=(midnight.dt.offset_by("1d") - midnight).dt.total_hours(),
                          monday=pl.col("D") - pl.duration(days=1) * (pl.col("D").dt.weekday() - 1)))
    weeks = (days.filter(pl.col("D").dt.month().is_in([12, 1, 2]))
             .group_by("monday").agg(complete=(pl.col("n") == pl.col("expected")).all(), days=pl.len(),
                                     nmae=100 * pl.col("err").sum() / pl.col("y").sum())
             .filter(pl.col("complete"), pl.col("days") == 7).sort("monday"))
    med = weeks["nmae"].median()
    chosen = weeks.with_columns(gap=(pl.col("nmae") - med).abs()).sort("gap")["monday"][0]
    print(f"{weeks.height} complete winter weeks; median LightGBM v2 week nMAE {med:.1f}%; chosen week starts {chosen}")
    return chosen


def draw(ax, wk: pl.DataFrame, title: str, main: str, main_label: str, chronos: str, with_chronos: bool,
         fs: float = 10.5, compact: bool = False) -> None:
    x = wk["hour"].dt.convert_time_zone(TZ).dt.replace_time_zone(None).to_list()
    y = wk["y"].to_numpy()
    err = lambda col: 100 * np.abs(wk[col].to_numpy() - y).sum() / y.sum()
    ax.set_facecolor(SURFACE)
    ax.fill_between(x, wk[f"{main}_lo"], wk[f"{main}_hi"], color=STYLE["main"]["color"], alpha=0.12, lw=0,
                    label="10–90% range")
    ax.plot(x, wk["B1_lastweek"], **{**STYLE["B1_lastweek"], "label": f"Same hour last week – error {err('B1_lastweek'):.1f}%"})
    ax.plot(x, wk["Toto2_target-only"], **{**STYLE["Toto2_target-only"], "label": f"Toto 2.0 – error {err('Toto2_target-only'):.1f}%"})
    if with_chronos:
        ax.plot(x, wk[chronos], **{**STYLE["Chronos2"], "label": f"Chronos-2 – error {err(chronos):.1f}%"})
    ax.plot(x, wk[main], color=STYLE["main"]["color"], lw=STYLE["main"]["lw"], label=f"{main_label} – error {err(main):.1f}%")
    ax.plot(x, y, **STYLE["actual"])
    ax.set_title(title, loc="left", fontsize=11.5 if compact else 14, color=INK, pad=5 if compact else 52)
    ax.set_ylabel("kWh per hour", fontsize=fs if compact else 11, color="#52514e")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors="#52514e", labelsize=fs, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    for line in ax.get_lines():
        line.set_solid_capstyle("round"); line.set_solid_joinstyle("round")
    # Legend above the plot area (never over the data), in reading order: truth, our model, comparisons
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (0 if labels[i] == "Actual" else 1 if labels[i].startswith(main_label + " –")
                                                      else 2 if "range" in labels[i] else 3 if "Chronos" in labels[i]
                                                      else 4 if "Toto" in labels[i] else 5))
    place = (dict(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1) if compact      # right side: adds no height
             else dict(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3))                  # above the plot
    leg = ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=fs, frameon=False,
                    handlelength=1.8, columnspacing=1.4, **place)
    for t in leg.get_texts():
        t.set_color(INK)


def week_rank(port: pl.DataFrame, start: date) -> str:
    """Share of complete test weeks in which LightGBM v2 did worse than in the chosen week."""
    midnight = pl.col("D").cast(pl.Datetime("us")).dt.replace_time_zone(TZ)
    days = (port.group_by("D").agg(n=pl.len(), err=(pl.col("LightGBM_W0_v2") - pl.col("y")).abs().sum(), y=pl.col("y").sum())
            .with_columns(expected=(midnight.dt.offset_by("1d") - midnight).dt.total_hours(),
                          monday=pl.col("D") - pl.duration(days=1) * (pl.col("D").dt.weekday() - 1)))
    weeks = (days.group_by("monday").agg(complete=(pl.col("n") == pl.col("expected")).all(), nd=pl.len(),
                                         nmae=pl.col("err").sum() / pl.col("y").sum())
             .filter(pl.col("complete"), pl.col("nd") == 7))
    this = weeks.filter(pl.col("monday") == start)["nmae"][0]
    return f"better than {100 * (weeks['nmae'] > this).mean():.0f}% of the {weeks.height} test weeks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=date.fromisoformat, default=None, help="Monday of the week to plot")
    args = ap.parse_args()
    port = load_portfolio()
    start = args.week or pick_week(port)
    rank = week_rank(port, start)
    print(f"week of {start}: LightGBM v2 {rank}")
    wk = port.filter(pl.col("D").is_between(start, start + timedelta(days=6))).sort("hour")
    span = f"{start:%-d}–{start + timedelta(days=6):%-d %b %Y}"
    for compact in (False, True):
        render(wk, span, compact)


def render(wk: pl.DataFrame, span: str, compact: bool) -> None:
    """All 7 figures. compact=True: about half the height, legend on the right (for one-page layouts)."""
    sfx, fs = ("_compact", 9) if compact else ("", 10.5)
    title = "Model comparison"
    names = ["week_strict", "week_strict_chronos", "week_actualweather", "week_actualweather_chronos"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 5.6) if compact else (18, 10.5), sharey=True, facecolor=SURFACE)
    for ax, panel in zip(axes.ravel(), PANELS):
        draw(ax, wk, *panel, fs=8.5 if compact else 9.5, compact=compact)
    if compact:  # no separate title line: the date goes into the top panel titles
        for ax in axes[0]:
            ax.set_title(f"{title} · {ax.get_title(loc='left').lower()}", loc="left", fontsize=11.5, color=INK, pad=5)
        fig.tight_layout(h_pad=1.2)
    else:
        fig.suptitle(title, x=0.01, y=0.98, ha="left", fontsize=17, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
    fig.savefig(OUT / f"week_2x2{sfx}.png", dpi=300, facecolor=SURFACE)
    ylim = axes[0, 0].get_ylim()  # same y-scale everywhere

    # Stacked: strict weather on top, actual weather below (without / with Chronos-2)
    for name, (top, bottom) in [("week_stacked", (PANELS[0], PANELS[2])), ("week_stacked_chronos", (PANELS[1], PANELS[3]))]:
        f, (a1, a2) = plt.subplots(2, 1, figsize=(14, 5.3) if compact else (13, 10.5), sharey=True, facecolor=SURFACE)
        draw(a1, wk, *top, fs=fs, compact=compact)
        draw(a2, wk, *bottom, fs=fs, compact=compact)
        a1.set_ylim(*ylim)
        if compact:
            a1.set_title(f"{title} · {a1.get_title(loc='left').lower()}", loc="left", fontsize=11.5, color=INK, pad=5)
            f.tight_layout(h_pad=1.0)
        else:
            f.suptitle(title, x=0.01, y=0.98, ha="left", fontsize=16, color=INK)
            f.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
        f.savefig(OUT / f"{name}{sfx}.png", dpi=300, facecolor=SURFACE)

    for name, panel in zip(names, PANELS):
        f, ax = plt.subplots(figsize=(14, 2.9) if compact else (13, 5.6), facecolor=SURFACE)
        draw(ax, wk, *panel, fs=fs, compact=compact)
        ax.set_ylim(*ylim)
        single = title if panel[0] == "Strict weather" else f"{title} · actual weather"
        ax.set_title(single, loc="left", fontsize=11.5 if compact else 14, color=INK,
                     pad=5 if compact else 52)
        f.tight_layout()
        f.savefig(OUT / f"{name}{sfx}.png", dpi=300, facecolor=SURFACE)
    print("saved:", ", ".join(f"results/{n}{sfx}.png" for n in ["week_2x2", "week_stacked", "week_stacked_chronos"] + names))


if __name__ == "__main__":
    main()
