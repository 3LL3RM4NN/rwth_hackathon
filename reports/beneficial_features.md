# Beneficial new features

Three of the ten candidate feature groups from `src/features.py` are worth keeping: solar geometry,
the local calendar, and the interactions. Together they add 21 features to the current 23. The other
seven groups showed no benefit or made the forecast worse; see `ablation_extended_table.md` for all
results.

All 21 features are computable at the day-ahead gate closure (11:45 UTC the day before delivery).
None uses anything measured after it.

## Evidence

Change in MAE when one group is added to the current model, as validation / test. Negative means the
group helps. Bold means the 95% interval excludes zero.

| Group | PV group | Non-PV group | All known |
|---|---|---|---|
| 4 solar geometry × PV share | **-6.0%** / -1.2% | **-2.7%** / -0.1% | **-6.0%** / +1.1% |
| 1 local calendar, holidays, daylight | -4.3% / -1.3% | -1.2% / -2.5% | -6.3% / -1.9% |
| 7 interactions + weather-corrected lag | -0.6% / -1.4% | **-3.5%** / -2.0% | -1.3% / -1.2% |

- **Solar geometry** is the only group that significantly reduces validation error in all three
  series. On the test days it is neutral.
- **Local calendar** points in the helpful direction in all six comparisons, though each one alone is
  within noise. It is also the only group the all-known model clearly misses when removed from the
  model with all groups (+5.2% ± 3.3).
- **Interactions** also point in the helpful direction in all six comparisons, but the effect is
  small and significant only for non-PV validation.

The evidence is modest. The three groups were each tested alone and not yet together, so their gains
may not add up. That combination should be run before `forecast.FEATURE_COLUMNS` is changed.

## Group 4: solar geometry × PV share (6 features)

The dataset has no coordinates, so sun positions are calculated for one assumed location (47.4°N,
8.5°E) using standard astronomical formulas.

| Feature | What it is |
|---|---|
| `solar_elevation_deg` | Angle of the sun above the horizon at the target time, in degrees. Negative at night. |
| `clear_sky_proxy` | Sine of the solar elevation, clipped at 0. Runs from 0 at night to about 0.9 at midsummer noon. It follows the shape of clear-sky irradiance but is not in W/m². |
| `pv_share_asof_cutoff` | Share of the households reporting at the cutoff that have PV. |
| `sunshine_fraction_prev_24h` | Sunshine hours over the 24 hours up to 10:00 the day before, divided by the astronomical day length. Near 0 for an overcast day, near 1 for a clear one. |
| `clear_sky_x_pv_share` | `clear_sky_proxy` × PV share: how much PV could suppress grid demand at this time if the sky were clear. |
| `expected_sun_x_pv_share` | The same, multiplied by yesterday's sunshine fraction. It assumes tomorrow is about as sunny as yesterday. |

The PV share is constant at 1 in the PV group and 0 in the non-PV group; it only varies in the
all-known series. In the non-PV group both product features are therefore always 0, and the gain
there came from sun position and yesterday's sunshine alone, most likely as a daylight and season
signal rather than a PV effect.

## Group 1: local calendar, holidays, daylight (11 features)

The existing calendar features are in UTC. These are in local clock time (`Europe/Zurich`, the same
clock as Germany), which is what household routines follow.

| Feature | What it is |
|---|---|
| `quarter_of_day_local` | 15-minute slot of the local day, 0 to 95. Offset from the UTC `horizon` by 4 slots in winter and 8 in summer. |
| `weekday_local` | Day of the week by local date, 0 = Monday. |
| `is_weekend_local` | 1 on local Saturday and Sunday. |
| `day_of_year` | 1 to 366, a finer season signal than `month`. |
| `is_public_holiday` | 1 on New Year, Good Friday, Easter Monday, 1 May, Ascension, Whit Monday, 25 and 26 December. |
| `is_bridge_day` | 1 on a Friday after a Thursday holiday or a Monday before a Tuesday holiday. |
| `is_dst` | 1 while summer time is in effect. |
| `is_dst_change_day` | 1 on the two days a year with 23 or 25 local hours. |
| `is_christmas_period` | 1 from 24 December to 2 January. |
| `is_easter_week` | 1 from the Monday before Easter through Easter Monday. |
| `day_length_hours` | Hours between sunrise and sunset at the assumed location, roughly 8.4 in December to 15.9 in June. |

The holiday list is limited to days that are holidays both across Germany and in the canton of
Zurich, because the dataset does not name the region. School holidays are left out for the same
reason. The series cover only 15 to 23 months, so each holiday appears once or twice and the model
has very few examples to learn those flags from.

## Group 7: interactions + weather-corrected lag (4 features)

These build on two inputs:

- **Heating degree hours:** the sum of `max(0, 15 °C − temperature)` over the 24 hourly readings up
  to 10:00 the day before. It is 0 in warm weather and grows as it gets colder.
- **Temperature sensitivity:** the slope of daily mean load against daily mean temperature over the
  last 28 complete days, ending two days before delivery. It is in kWh per 15 minutes per °C and is
  normally negative, since colder days mean more load.

| Feature | What it is |
|---|---|
| `heating_degree_x_quarter` | Heating degree hours × the UTC quarter-of-day index (0 to 95). Lets the cold-weather effect differ by time of day. |
| `heating_degree_x_weekend` | Heating degree hours × the local weekend flag. Lets the cold-weather effect differ between weekends and weekdays. |
| `temp_delta_vs_lag_7d` | Mean temperature of the 24 hours up to 10:00 the day before, minus the mean temperature of the day one week before delivery. Positive means it is warmer now than a week ago. |
| `lag_7d_weather_corrected` | Load at the same time one week ago, plus sensitivity × `temp_delta_vs_lag_7d`. Shifts last week's load to what it would have been at the current temperature. |

Two limitations:

- **"Now" is yesterday's temperature.** The delivery day's own temperature is not known at bid time,
  so the correction uses the latest 24-hour mean as a stand-in. With a real weather forecast this
  feature would be much stronger.
- **The quarter-of-day product is crude.** Multiplying by an index from 0 to 95 makes the feature 0
  at midnight and largest late in the evening, which has no physical meaning.
