# Bringing the Heat

# Challenge for E.ON
Increasing numbers of prosumers, who are both producers and consumers of energy, are changing the energy landscape.
With the high volatility of energy production for renewable energy sources, the energy consumption patterns are changing as well.
This makes it more difficult to predict the energy demand and supply, which is crucial for the energy market and grid management.
In addition, heavily energy consuming technologies like electric vehicles and heat pumps are becoming more and more common, which further increases the volatility of energy consumption.

As an energy distribution system operator (DSO) and energy supplier, E.ON is facing the challenge of adapting to these changes and finding ways to optimise the energy supply and demand.
One of the key challenges is to purchase the right amount of energy at the right time, which requires accurate predictions of energy consumption and production.

## How does energy procurement work?
Energy procurement is a highly complex topic and is divided into different stages and steps. To not overwhelm you with the complexity, we will give you a simplified overview of the relevant process steps and the role of forecasting in this process.

Since energy can not be stored in large scale by the electricity grid itself, it needs to be produced at the same time as it is consumed. This means that energy producers and consumers need to be matched in real-time, which is done through the energy market.
We will focus on three stages how this matching process works:
1. **Long-term procurement**: This is the stage where energy is procured for a longer period of time, e.g., for the next year. This is done through long-term contracts with energy producers, which are usually signed several months in advance. The energy procured here usually covers the baseload (Grundlast) of the energy consumption.
2. **Day-ahead procurement**: This is the stage where energy is procured for the next day. This is done through the day-ahead market, where energy producers and consumers submit their bids for the next day. The day-ahead procurement tries to cover the middle load (Mittellast) of the energy consumtion.
3. **Intraday procurement**: This is the stage where energy is procured for the same day. This is done through the intraday market, where energy producers and consumers submit their bids for the next hours. The intraday procurement tries to cover the remaining peak loads (Spitzenlasten) of the energy consumption, caused by unexpected changes in energy consumption and production or errors in the forecast.

As you can imagine, having long term contracts, e.g. with a long term procurement contract, is much cheaper than having to buy the energy on the intra-day market. Hence forecasting the right amount of energy needed is crucial.

For this challenge, we will focus on the day-ahead procurement stage.
The goal for E.ON is to have a good forecast of the energy consumption and production for the next day, which can be used to procure the right amount of energy on the day-ahead market.

# Data Sheet
## Household Data
The dataset comprises consumption data from **410 households**, who own a heat-pump and in some cases also a photovoltaic system (PV system).
All of these households own a heat-pump; coverage length varies per household (**379** of the 410 have more than one year of data) — see `data/smart_meter_meta_data/smart_meter_data_15min_overview.csv` for the exact per-household date range and datapoint counts.
The resolution of the data is on a **15-minute level**, which means that for each day there are up to **96 datapoints** for the energy consumption of the household.
The energy consumption is measured in kWh (kilowatt-hour).
The dataset covers the period from **2019-05-20 to 2024-02-27**, with different coverages for different households.

Each 15-minute file (`data/15min/<Household_ID>.csv`) is **semicolon-separated** and contains the following columns:

| Column | Description |
|---|---|
| `Household_ID` | Unique identifier of the household |
| `Group` | Experiment group (e.g. `treatment`) |
| `AffectsTimePoint` | Whether the row is `before visit` or `after visit` relative to the heat-pump optimisation |
| `Timestamp` | Timestamp of the interval in UTC (`YYYY-MM-DD HH:MM:SS+00:00`) |
| `kWh_received_Total` | Total energy consumed in the interval |
| `kWh_received_HeatPump` | Energy consumed by the heat pump |
| `kWh_received_Other` | Energy consumed by other loads |

> **Note:** The 15-minute files do **not** contain reactive-power (`kvarh_*`), grid-return (`kWh_returned_Total`) or PV-production columns — this dataset has no such signals. Whether a household owns a PV system is instead recorded as a flag in `data/smart_meter_meta_data/households.csv` (`Installation_HasPVSystem`; missing for households where this was not surveyed).

All customers have an optimised heat-pump, where a technical expert visited the household and optimised the heat pump settings to reduce the energy consumption, hence the features `AffectsTimePoint` and `Group`.

In addition, some meta data for the households are provided in `data/smart_meter_meta_data/`:
- `meta_data.csv` — survey answers (building type, living area, heat-pump type, appliances, ...), described by `meta_data_variables.csv`. Covers **393** of the 410 households (**17** have no meta-data row in the source file).
- `households.csv` — one row per household with `Group`, the `Installation_HasPVSystem` flag, and `Weather_ID`, which links each household to its nearest weather station in `data/weather_data_hourly/<Weather_ID>.csv`.
- `smart_meter_data_15min_overview.csv` — per-household data availability (earliest/latest timestamp, number of days before/after the visit, etc.).

# Weather Data
Additionally, data from **8 weather stations** are provided. The data can be found in `data/weather_data_hourly`.
The resolution here is on an **hourly level**. The provided signals for the weather stations can be found in `data/weather_data_overview`.

Since the data does not have the same resolution, you will have to find methods to align the data, e.g., by resampling the weather data to a 15-minute level (upsampling/interpolation), by aggregating the 15-minute energy consumption to an hourly level, or by something completely different.

# Use Cases
The challenge is to build a forecasting engine for the day-ahead procurement stage for prosumers and/or PV system owners.
The goal is to have a good forecast of the energy consumption and production for the next day, which can be used to procure the right amount of energy on the day-ahead market.

**Important Note**
Please make sure to only use data from the past to predict the future.
This is a common mistake in time series forecasting and can lead to overly optimistic results.
E.g.: When splitting the data into train/test, make sure to not split randomly but to consider the temporal order of the data.
The test set should be the most recent data, e.g., the last 20% of the data, and the train set should be the older data, e.g., the first 80% of the data.

When working on this case, you can choose the depth and the broadness of your solution.
You can focus on a specific aspect of the problem, e.g., on the forecasting model, on the data preprocessing, on the evaluation metrics or on the uncertainty estimation.
Or you can try to tackle all aspects of the problem. The choice is yours. The evaluation will not depend on the number of levels you have completed, but on the quality of your solution for the levels you have completed.
Meaning that you can work on the level of your choice and go in depth on that topic, or you can show a broad range of solutions for different levels. The choice is yours.

Complexity levels:
- Level 0
Build a baseline forecasting model for the day-ahead procurement stage for prosumers and/or PV system owners.
You can use any method you like, e.g. linear regression, random forest, neural network, ARIMA, Holt-Winters or something completely different.
The goal is to have a good forecast of the energy consumption and production for the next day, which can be used to procure the right amount of energy on the day-ahead market.

- Level 1
There are some households, that have a PV system in addition to the heat pump.
This means that their energy consumption is probably more volatile and completely different from the households without a PV system.

For level 1 either:
**a)** Try to identify groups of households with similar energy consumption patterns or meta data and build a forecasting model for each group.
**b)** Try to identify PV owners by their energy consumption patterns and build a forecasting model for households with PV systems and without PV systems.

- Level 2
Compare your both models and try to identify metrics that reflects the performance of the models in a way that is relevant for the day-ahead procurement stage.
Note: If you want to compare the models from a business perspective, you are allowed to assume costs and prices, where necessary (e.g. for purchasing energy on the day-ahead market vs. having to buy/sell on the intra-day market).

- Level 3
Enrich your model with uncertainties, where is your model more uncertain and where is it more certain? How can you use this information to make better decisions for the day-ahead procurement stage?
Think about the business interest and the costs of over- and under-estimating the energy consumption and production for the next day. How can you use the uncertainty information to make better decisions for the day-ahead procurement stage?

# References
Here are some base courses you can have a look at if you are looking for some inspiration:

## Statistics

- [A Simple Guide to Data Distribution in Statistics and Data Science](https://medium.com/@datasciencewizards/a-simple-guide-to-data-distribution-in-statistics-and-data-science-39bc835dcb72)
- [How to Handle Skewed Data: A Guide for Data Scientists](https://medium.com/gopenai/how-to-handle-skewed-data-a-guide-for-data-scientists-84187ba7f805)

## How-To Guides
- [How to implement Random Forest from scratch with Python](https://www.youtube.com/watch?v=kFwe2ZZU7yw)
- [Getting Started Predicting Time Series Data with Facebook Prophet](https://medium.com/data-science/getting-started-predicting-time-series-data-with-facebook-prophet-c74ad3040525)

## Models

- [How to implement Random Forest from scratch with Python](https://www.youtube.com/watch?v=kFwe2ZZU7yw)
- [Decision Tree Regressor, Explained: A Visual Guide with Code Examples](https://medium.com/data-science/decision-tree-regressor-explained-a-visual-guide-with-code-examples-fbd2836c3bef)
- [Regression Trees, Clearly Explained!!!](https://www.youtube.com/watch?v=g9c66TUylZ4)
- [Support Vector Machines Part 1 (of 3): Main Ideas!!!](https://www.youtube.com/watch?v=efR1C6CvhmE)

## Metrics
- [How to evaluate ML models | Evaluation metrics for machine learning](https://www.youtube.com/watch?v=LbX4X71-TFI)
- [Pearson's Correlation, Clearly Explained!!!](https://www.youtube.com/watch?v=xZ_z8KWkhXE)
- [R-squared, Clearly Explained!!!](https://www.youtube.com/watch?v=bMccdk8EdGo)

## Data Distribution

- [Data Distributions Explained | What are the different types of distribution](https://leanscape.io/data-distributions-explained-what-are-the-different-types-of-distribution/)
- [7 Types of Statistical Distributions with Practical Examples](https://datasciencedojo.com/blog/types-of-statistical-distributions-in-ml/)

## Library Docs

- [The Data Visualisation Catalogue](https://datavizcatalogue.com/index.html)
- [Visualizing distributions of data in Seaborn](https://seaborn.pydata.org/tutorial/distributions.html)

# Our expectations

We expect participants to stay in touch with the mentors throughout the day and reach out early if you're stuck rather than saving your questions for the end.
We are open to your questions and hope to provide as much support as possible to you given your motivation and dedication to the challenge topic.
Given the one-day format, pick a scope you can realistically finish rather than one you can only start — a small, complete solution beats a big, unfinished one.
Have fun and happy hacking!
