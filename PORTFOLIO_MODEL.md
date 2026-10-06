    # Portfolio-level forecasting

1. **Forecast target:** This model predicts **each household’s** next-day consumption; the alternative model predicts **only the total portfolio**.

2. **Training objective:** This model learns household-level errors; the portfolio model directly minimizes:

$$
\min_\theta \sum_{d,h}(Y_{d,h}-\hat Y_{d,h})^2
$$

3. **Main trade-off:** Household-level modeling provides individual forecasts and flexibility; direct portfolio modeling focuses exclusively on **overall procurement accuracy**.
