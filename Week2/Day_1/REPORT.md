# Time Series as a Regression Problem — Report

**Dataset:** AirPassengers (monthly international airline passengers, 1949–1960)
**Deliverables:** `air_passengers_analysis.ipynb`, `app.py`, `tabpfn_air_passengers_model.pkl`, `scaler.pkl`, `air_passengers_supervised.csv`

## 1. Approach

Instead of a specialised sequential model (ARIMA, exponential smoothing), the
time series is reframed as a **supervised regression** problem using a
**sliding-window / lag-feature** transformation:

- For each month `t`, the previous `WINDOW_SIZE = 6` months become the input
  features `[lag_6, lag_5, …, lag_1]` (oldest → newest) and the value at month
  `t` is the target.
- Rows with `NaN` lags (the first 6 observations) are dropped, leaving 138
  training examples.
- A **chronological 80/20 split** is used (no shuffling) so the test set is
  strictly in the future relative to training — the correct protocol for time
  series.
- Features are standardised with `StandardScaler` (fit on train only), then a
  **`TabPFNRegressor`** is trained on the scaled features.

This preserves the temporal structure while making any regression algorithm
usable, and it makes lag importance directly interpretable.

## 2. EDA Findings

- **Strong upward trend** with **multiplicative seasonality** (peaks each summer,
  amplitude growing over time) — confirmed by seasonal decomposition.
- The **target is right-skewed** and variance increases over time
  (heteroscedasticity), a natural candidate for a log transform.
- **High autocorrelation:** `lag_1` correlates most strongly with the target,
  which justifies the lag-feature formulation.

## 3. Model Performance

| Metric | Train | Test |
|--------|-------|------|
| MSE    | 67.39 | 1,102.43 |
| RMSE   | 8.21  | 33.20 |
| MAE    | 6.00  | 24.82 |
| R²     | 0.9921 | 0.8213 |

The model explains **82%** of the variance on unseen future months, with a
typical error of **±25 passengers**. Train/test performance are consistent
enough to rule out severe overfitting; the higher test error is expected
because the test period contains the largest passenger volumes the model never
saw during training (the trend keeps rising).

## 4. Interpretation & Limitations

- **Most important feature:** the most recent month (`lag_1`), consistent with
  the autocorrelation observed in EDA.
- **Limitations:**
  - The window drops the first 6 observations.
  - The series is non-stationary (trend), so extrapolating beyond the observed
    range under-predicts — visible as growing residuals in the later test months.
  - One-step-ahead only; multi-step forecasting would require recursive feeding
    of predictions.
  - Seasonality is captured only implicitly through the lags.
- **Improvements:** log-transform the target to stabilise variance, add explicit
  calendar/seasonal features (month, rolling means), and retrain regularly as
  new data arrives.

## 5. REST API (`app.py`)

A Flask service loads the saved model + scaler once at startup and exposes:

| Method & path | Purpose |
|---------------|---------|
| `GET /`       | Service description |
| `GET /health` | Health / readiness check |
| `POST /predict` | Predict next month from the last 6 months |

**Request** — 6 values, oldest → newest:

```bash
curl -X POST http://localhost:5001/predict \
     -H "Content-Type: application/json" \
     -d '{"passengers": [417, 391, 419, 461, 472, 535]}'
```

**Response:**

```json
{"input_passengers": [417.0, 391.0, 419.0, 461.0, 472.0, 535.0],
 "predicted_next_month": 536.4, "window_size": 6}
```

**Validation & error handling** (all return JSON):
- Non-JSON body / wrong content type → `400`
- Missing `passengers` field → `400`
- Not exactly 6 values → `400`
- Non-numeric, non-finite (`NaN`/`inf`), or negative values → `400`
- Unknown route / wrong method → `404` / `405`
- Inference failure → `500`

**Run:**

```bash
cd Week2
uv add flask                       # one-time (adds Flask to the env)
PORT=5001 ./.venv/bin/python Day_1/app.py
```

> Note: the app defaults to **port 5001** because macOS AirPlay occupies port
> 5000. Override with the `PORT` environment variable.
