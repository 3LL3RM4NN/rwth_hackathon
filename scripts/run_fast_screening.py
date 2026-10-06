"""Fast screening: PV/non-PV split vs one pooled model on a deterministic sample of complete test weeks.

    uv run python scripts/run_fast_screening.py [--dry_run] [--time_limit S] [--out DIR]

* split approach  : the already fitted PV / non-PV models of outputs/level1_pv_1145 (not re-trained); their stored
                    predictions are filtered to the sampled weeks (a 1-day re-prediction check confirms they reproduce)
* pooled approach : ONE new model with the identical AutoGluon configuration, trained on the same households and the
                    same development window as the two split models together; predicts only the sampled weeks
* same D-1 11:45 cutoff / 144-step horizon / last-96 scoring as scripts/run_level1_pv.py
"""
import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from run_autogluon import AG_CFG as BASE_CFG
from run_level1_pv import BASELINES, M_NEW as M_SPLIT
from src import baseline, evaluation
from src.autogluon_forecast import build_train_frame, fit
from src.config import Config
from src.data import load_all, split_dates
from src.level1_pv import HORIZON, pv_status_table, rolling_predict_1145, to_interval_frame, verify_cutoff

M_POOLED = "autogluon_pooled_1145"
SEASONS = {"spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11), "winter": (12, 1, 2)}
WEEKS_PER_SEASON = 3
MIN_SCOREABLE_SHARE = 0.5          # a day must be scoreable for >= 50% of the 241 households (data availability only)


def select_weeks(test_range, scoreable: pd.Series):
    """Deterministic: Monday-Sunday weeks fully inside the test period and one meteorological season, with every day
    scoreable for >= 50% of households (excludes e.g. the 2023-10-29 all-missing day); per season the weeks at the
    1/6, 3/6, 5/6 positions of the eligible list. Uses data availability only, never model errors."""
    mondays = pd.date_range(test_range[0], test_range[1], freq="W-MON")
    eligible = {s: [] for s in SEASONS}
    rejected = []
    for mon in mondays:
        days = pd.date_range(mon, periods=7)
        if days[-1] > test_range[1]:
            continue
        seas = {s for s, ms in SEASONS.items() if all(d.month in ms for d in days)}
        if len(seas) != 1:
            continue                                          # straddles two seasons
        if (scoreable.reindex(days).fillna(0) < MIN_SCOREABLE_SHARE * scoreable.max()).any():
            rejected.append(str(mon.date()))
            continue
        eligible[seas.pop()].append(mon)
    sel = {}
    for s, ws in eligible.items():
        idx = sorted({int((2 * k + 1) * len(ws) / (2 * WEEKS_PER_SEASON)) for k in range(WEEKS_PER_SEASON)})
        sel[s] = [ws[i] for i in idx]
    return sel, {s: [str(w.date()) for w in ws] for s, ws in eligible.items()}, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true", help="selection + checks only, no training")
    ap.add_argument("--time_limit", type=int, default=BASE_CFG["time_limit"])
    ap.add_argument("--train_days", type=int, default=BASE_CFG["train_days"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--split_dir", type=Path, default=None, help="Level 1b outputs (default outputs/level1_pv_1145)")
    a = ap.parse_args()
    ag = copy.deepcopy(BASE_CFG)
    ag.update(time_limit=a.time_limit, train_days=a.train_days, max_origins=None, prediction_length=HORIZON)
    cfg = Config()
    base_cfg = dataclasses.replace(cfg, models=BASELINES)
    out = a.out or cfg.output_dir / "level1_fast_screening"
    split_dir = a.split_dir or cfg.output_dir / "level1_pv_1145"
    for sub in ("predictions", "metrics"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    masked = {hh: m for hh, m, _ in load_all(cfg)}
    raw = {hh: m for hh, m, _ in load_all(dataclasses.replace(cfg, min_day_completeness=0.0))}
    first = min(m.index.min() for m in masked.values()); last = max(m.index.max() for m in masked.values())
    test_range = split_dates(first, last, cfg)["test"]
    train_end = test_range[0] - pd.Timedelta(days=1)

    # ---- households: exactly the Level 1b scored set (known PV status AND >=1 usable test day) ------------------
    pv = pv_status_table(cfg).set_index("Household_ID").loc[list(masked)]
    has_test = pd.Series({h: bool(((m.index >= test_range[0]) & np.isfinite(m.to_numpy()).any(axis=1)).any()) for h, m in masked.items()})
    pv["usable_test"] = has_test
    train_groups = {g: pv.index[pv.pv_status == g].tolist() for g in ("pv", "non_pv")}    # what the split models trained on
    scored = pv.index[(pv.pv_status != "unknown") & pv.usable_test].tolist()
    n_pv, n_non = int(((pv.loc[scored].pv_status) == "pv").sum()), int(((pv.loc[scored].pv_status) == "non_pv").sum())
    assert (len(scored), n_pv, n_non) == (241, 131, 110), (len(scored), n_pv, n_non)
    print(f"scored households: {len(scored)} (PV {n_pv}, non-PV {n_non}); excluded unknown PV status: "
          f"{int((pv.pv_status == 'unknown').sum())}; known-status without test days: {int(((pv.pv_status != 'unknown') & ~pv.usable_test).sum())}")

    # ---- representative weeks ---------------------------------------------------------------------------
    scoreable = pd.Series(0, index=pd.date_range(test_range[0], test_range[1]))
    for h in scored:
        m = masked[h]; ok = m.index[np.isfinite(m.to_numpy()).any(axis=1)]
        scoreable.loc[scoreable.index.intersection(ok)] += 1
    sel, eligible, rejected = select_weeks(test_range, scoreable)
    days = pd.DatetimeIndex(sorted(d for ws in sel.values() for w in ws for d in pd.date_range(w, periods=7)))
    assert len(days) == 7 * sum(len(v) for v in sel.values()) and days.min() >= test_range[0] and days.max() <= test_range[1]
    assert (scoreable.reindex(days) >= MIN_SCOREABLE_SHARE * len(scored)).all()
    exp_hh_days = int(sum(1 for h in scored for d in days if d in masked[h].index and np.isfinite(masked[h].loc[d].to_numpy()).any()))
    sel_json = {"weeks_per_season": WEEKS_PER_SEASON, "rule": select_weeks.__doc__,
                "selected_weeks_monday": {s: [str(w.date()) for w in ws] for s, ws in sel.items()},
                "eligible_weeks": eligible, "rejected_for_data_availability": rejected,
                "forecast_days": [str(d.date()) for d in days], "n_days": len(days),
                "scoreable_households_per_day_min": int(scoreable.reindex(days).min())}
    (out / "selected_weeks.json").write_text(json.dumps(sel_json, indent=2))
    print("selected weeks (Mon):", json.dumps(sel_json["selected_weeks_monday"]), "\nrejected (data availability):", rejected)
    print(f"EXPECTED: households={len(scored)}  forecast days={len(days)}  household-days<={exp_hh_days}  "
          f"intervals<={exp_hh_days * 96}  (full-year run had 241 households x 364 days)")
    print("pooled training households:", len(train_groups['pv']) + len(train_groups['non_pv']), "(same as PV + non-PV models)")

    verify_cutoff({h: masked[h] for h in scored}, {h: raw[h] for h in scored}, test_range, ag, n_days=6)
    if a.dry_run:
        print("dry run finished: checks passed, nothing trained")
        return

    print("Config:", json.dumps({**ag, "train_end": str(train_end.date()), "forecast_origin": "D-1 11:45"}, indent=2), flush=True)
    (out / "run_config.json").write_text(json.dumps({**ag, "train_end": str(train_end.date()), "pooled_model": M_POOLED,
                                                     "split_model": M_SPLIT, "split_models_dir": str(split_dir)}, indent=2, default=str))

    # ---- reuse split predictions; confirm they reproduce ---------------------------------------------------
    from autogluon.timeseries import TimeSeriesPredictor
    split_all = pd.read_parquet(split_dir / "predictions" / "interval_predictions.parquet")
    split_iv = split_all[split_all["forecast_date"].isin(days)].copy()
    chk_day = days[len(days) // 2]
    for g in ("pv", "non_pv"):
        ids = [h for h in scored if pv.loc[h, "pv_status"] == g]
        pr = TimeSeriesPredictor.load(str(split_dir / f"model_{g}"))
        re = rolling_predict_1145(pr, {h: masked[h] for h in ids}, {h: raw[h] for h in ids}, test_range, ag, days=pd.DatetimeIndex([chk_day]))
        st = split_iv[(split_iv.pv_status == g) & (split_iv.forecast_date == chk_day)].set_index(["Household_ID", "Timestamp"])["prediction"]
        re = re.set_index(["Household_ID", "Timestamp"])["prediction"].reindex(st.index)
        assert np.allclose(re.to_numpy(), st.to_numpy(), atol=1e-6, equal_nan=True), f"stored {g} predictions do not reproduce"
        print(f"stored {g} predictions reproduce on {chk_day.date()} (max abs diff {np.nanmax(np.abs(re.to_numpy() - st.to_numpy())):.2e})")

    # ---- pooled model ------------------------------------------------------------------------------------------
    pool_ids = train_groups["pv"] + train_groups["non_pv"]
    train = build_train_frame({h: masked[h] for h in pool_ids}, train_end, ag["train_days"], ag["min_train_valid_days"])
    assert train.index.get_level_values("timestamp").max() <= train_end + pd.Timedelta(hours=23, minutes=45)
    print(f"[pooled] train series {train.num_items}, rows {len(train)}", flush=True)
    t0 = time.time()
    predictor = fit(train, out / "model_pooled", ag)
    fit_s = time.time() - t0
    lb = predictor.leaderboard(silent=True); lb.to_csv(out / "metrics" / "leaderboard_pooled.csv", index=False)
    print(lb[["model", "score_val", "fit_time_marginal"]].to_string(), flush=True)
    verify_cutoff({h: masked[h] for h in scored}, {h: raw[h] for h in scored}, test_range, ag, n_days=2, predictor=predictor)
    t0 = time.time()
    p = rolling_predict_1145(predictor, {h: masked[h] for h in scored}, {h: raw[h] for h in scored}, test_range, ag, days=days)
    pred_s = time.time() - t0
    p["pv_status"] = p["Household_ID"].map(pv["pv_status"])
    pooled_iv = to_interval_frame(p, masked, M_POOLED)
    pooled_iv.to_parquet(out / "predictions" / "pooled_interval_predictions.parquet", index=False)
    split_iv.to_parquet(out / "predictions" / "split_interval_predictions.parquet", index=False)

    # ---- evaluation (shared evaluation module; 'common' scope = all four models predict) --------------------------
    new_by = {h: g[["Household_ID", "Timestamp", "actual", "prediction", "model"]] for h, g in pd.concat([split_iv, pooled_iv]).groupby("Household_ID")}
    chunks = {}
    daymask = lambda df: df[df["Timestamp"].dt.normalize().isin(days)]
    for hh in scored:
        base, _ = baseline.predict_household(hh, masked[hh], base_cfg, test_range)
        frames = ([daymask(base)] if len(base) else []) + ([new_by[hh]] if hh in new_by else [])
        if frames:
            chunks[hh] = evaluation.chunk_stats(pd.concat(frames, ignore_index=True), cfg.mape_min_actual, 4)
    models = ["naive_7day", "seasonal_mean_4weeks", M_POOLED, M_SPLIT]
    by_key = {"household": ["model", "Household_ID"], "qh": ["model", "qh"], "date": ["model", "date"]}
    groups = {"combined": scored, "pv": [h for h in scored if pv.loc[h, "pv_status"] == "pv"],
              "non_pv": [h for h in scored if pv.loc[h, "pv_status"] == "non_pv"]}
    rows, hmc = [], None
    for name, ids in groups.items():
        res = evaluation.combine([chunks[h] for h in ids if h in chunks], by_key)
        for scope in ("common", "all"):
            hm = res["household"].query("scope == @scope"); ov = res["overall"].query("scope == @scope").set_index("model")
            for m in models:
                h = hm[hm["model"] == m]
                rows.append({"group": name, "scope": scope, "model": m, "households": h["Household_ID"].nunique(),
                             "scored_intervals": int(ov.loc[m, "n"]), "mean_household_MAE": h["MAE"].mean(),
                             "median_household_MAE": h["MAE"].median(), "overall_MAE": ov.loc[m, "MAE"], "RMSE": ov.loc[m, "RMSE"],
                             "daily_energy_MAE_kWh": ov.loc[m, "daily_energy_MAE_kWh"], "bias": ov.loc[m, "bias"],
                             "mean_actual": ov.loc[m, "mean_actual"], "mean_pred": ov.loc[m, "mean_pred"]})
        if name == "combined":
            h = res["household"].copy(); h["pv_status"] = h["Household_ID"].map(pv["pv_status"]); h.to_csv(out / "metrics" / "per_household.csv", index=False)
            res["date"].to_csv(out / "metrics" / "per_date.csv", index=False)
            res["qh"].to_csv(out / "metrics" / "per_quarter_hour.csv", index=False)
            hmc = res["household"].query("scope == 'common'").pivot(index="Household_ID", columns="model", values="MAE")
    comp = pd.DataFrame(rows); comp.to_csv(out / "metrics" / "comparison.csv", index=False)

    # ---- split vs pooled: differences, win/loss, paired uncertainty over households --------------------------------
    c = comp[comp.scope == "common"].set_index(["group", "model"])
    diff = []
    d = hmc.copy(); d["pv_status"] = d.index.map(pv["pv_status"])
    for name in groups:
        sub = d if name == "combined" else d[d.pv_status == name]
        delta = sub[M_SPLIT] - sub[M_POOLED]                          # <0: split better
        se = delta.std(ddof=1) / np.sqrt(len(delta))
        r = {"group": name, "households": len(sub)}
        for k in ("mean_household_MAE", "overall_MAE", "RMSE", "daily_energy_MAE_kWh", "bias"):
            s_, p_ = c.loc[(name, M_SPLIT), k], c.loc[(name, M_POOLED), k]
            r[f"{k}_split"], r[f"{k}_pooled"], r[f"{k}_diff(split-pooled)"] = s_, p_, s_ - p_
            r[f"{k}_diff_%"] = 100 * (s_ - p_) / abs(p_)
        r.update({"split_better_households": int((delta < 0).sum()), "split_worse_households": int((delta > 0).sum()),
                  "pct_split_better": round(100 * (delta < 0).mean(), 1), "paired_mean_diff_hh_MAE": delta.mean(),
                  "paired_se": se, "paired_95ci_low": delta.mean() - 1.96 * se, "paired_95ci_high": delta.mean() + 1.96 * se})
        diff.append(r)
    pd.DataFrame(diff).to_csv(out / "metrics" / "split_vs_pooled.csv", index=False)
    both = pd.concat([split_iv, pooled_iv])
    both["week_monday"] = both["forecast_date"] - pd.to_timedelta(both["forecast_date"].dt.dayofweek, unit="D")
    both["season"] = both["forecast_date"].dt.month.map({m: s for s, ms in SEASONS.items() for m in ms})
    wk = both.groupby(["season", "week_monday", "model"]).agg(MAE=("abs_error", "mean"), bias=("signed_error", "mean"), n=("abs_error", "size")).reset_index()
    wk.to_csv(out / "metrics" / "per_week.csv", index=False)
    info = {"selected": sel_json["selected_weeks_monday"], "fit_seconds_pooled": round(fit_s, 1), "predict_seconds_pooled": round(pred_s, 1),
            "pooled_models_trained": lb["model"].tolist(), "pooled_best_model": predictor.model_best,
            "pooled_series_in_training": int(train.num_items), "households_scored": len(scored), "forecast_days": len(days)}
    (out / "metrics" / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    pd.set_option("display.width", 250)
    for gname in groups:
        print(f"\n== {gname} (common scope) ==")
        print(comp[(comp.group == gname) & (comp.scope == "common")].drop(columns=["group", "scope", "mean_actual"]).round(4).to_string(index=False))
    print(pd.DataFrame(diff).T.to_string()); print(json.dumps(info, indent=2, default=str))
    print(wk.pivot_table(index=["season", "week_monday"], columns="model", values="MAE").round(4).to_string())


if __name__ == "__main__":
    main()
