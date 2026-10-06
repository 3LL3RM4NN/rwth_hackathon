# Fast screening: PV/non-PV split vs pooled AutoGluon (D-1 11:45 cutoff)

**Question:** on a representative seasonal sample, does splitting households into PV/non-PV models improve forecasting enough to justify the complexity?

**Answer: no.** The split is ~1% better on mean household MAE, but the gain is concentrated in two spring weeks, the split is worse in 8 of 12 weeks, worse on daily-energy MAE, and confounded with the treatment/control split.

## Setup
- 12 complete Mon-Sun test weeks (84 days; spring 2023-03-27, 04-24, 05-15; summer 06-19, 07-17, 08-14; autumn 09-11, 10-09, 11-13; winter 2023-12-18, 2024-01-15, 02-12). Rule in `selected_weeks.json` (data availability only; week 2023-10-23 rejected: contains the all-missing 2023-10-29).
- 241 scored households (131 PV, 110 non-PV); 1,777,920 scored intervals (all four models, common scope).
- Split = the existing Level 1b PV / non-PV models (not retrained; stored predictions filtered to the sample; 1-day re-prediction reproduced them exactly). Pooled = one new model, same config/seed/preset/model list/ensemble, trained on the same 245 households and window as the two split models together. No static features / PV flag in the pooled model.
- Cutoff D-1 11:45 verified by perturbation test (before training and on the trained predictor).

## Results (common scope)
| Model | Mean hh MAE | Median hh MAE | Overall MAE | RMSE | Daily-energy MAE (kWh) | Bias |
|---|--:|--:|--:|--:|--:|--:|
| naive_7day | 0.2000 | 0.1814 | 0.1991 | 0.3841 | 7.10 | +0.013 |
| seasonal_mean_4weeks | 0.1828 | 0.1673 | 0.1820 | 0.3185 | 7.13 | +0.014 |
| AutoGluon pooled | 0.1564 | 0.1441 | 0.1555 | 0.2989 | 7.08 | -0.047 |
| AutoGluon PV/non-PV split | 0.1548 | 0.1427 | 0.1538 | 0.2992 | 7.26 | -0.048 |

## Split vs pooled
| Group | hh | Mean hh MAE diff (split-pooled) | % | Overall MAE % | Daily-energy % | Households split better |
|---|--:|--:|--:|--:|--:|--:|
| combined | 241 | -0.0015 | -0.97% | -1.09% | +2.5% | 167 (69%) |
| PV | 131 | +0.0012 | +0.80% | +0.57% | +8.4% | 59 (45%) |
| non-PV | 110 | -0.0047 | -2.86% | -2.81% | -4.2% | 108 (98%) |

Paired 95% CI of the combined mean household MAE difference (over households only; ignores training/seed variance): [-0.0022, -0.0009].

By week, split is better in 4/12 weeks (2023-03-27 and 04-24 by 6-11%, 11-13 and 2024-02-12 marginally) and slightly worse in the other 8 (summer all three).

## Caveats
- All non-PV households are treatment households; the non-PV "gain" may reflect the treatment/control split or smaller specialised training set, not PV status.
- One seed, one run per approach, 12 sampled weeks: a ~1% difference is within what training variance could produce.
- Target is net received energy; there is no PV production data.

## Recommendation
Do not spend more runtime on PV segmentation; continue with the pooled model.
