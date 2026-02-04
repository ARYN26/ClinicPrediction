# Interview Preparation Guide: Time Series Forecasting Project

This document covers key concepts and common interview questions related to this clinic revenue prediction project.

---

## 1. Why Time Series is Different from Regular ML

### Key Differences

| Aspect | Regular ML | Time Series |
|--------|-----------|-------------|
| **Data Independence** | Samples are i.i.d. (independent) | Observations are temporally dependent |
| **Train/Test Split** | Random split is fine | Must use temporal split (no future leakage) |
| **Cross-Validation** | K-fold works | Need TimeSeriesSplit |
| **Features** | Static features | Lag features, rolling stats, seasonality |
| **Evaluation** | Standard metrics | Same metrics but on held-out future data |

### Why This Matters

In time series, you cannot randomly shuffle data because:
1. **Temporal dependency**: Today's value depends on yesterday's
2. **Data leakage**: Using future information to predict the past is cheating
3. **Seasonality**: Patterns repeat at fixed intervals (monthly, yearly)

---

## 2. Prophet Model Explained

### The Model Equation

```
y(t) = g(t) + s(t) + h(t) + ε(t)
```

Where:
- **g(t)**: Trend component (linear or logistic growth)
- **s(t)**: Seasonality (weekly, yearly patterns using Fourier series)
- **h(t)**: Holiday effects (optional)
- **ε(t)**: Error term (noise)

### Why Prophet for This Project

1. **Works with ~30 data points**: Unlike deep learning models that need thousands
2. **Automatic seasonality**: Detects yearly patterns without manual feature engineering
3. **Handles missing data**: Robust to gaps in the time series
4. **Interpretable**: Can explain trend and seasonality to stakeholders
5. **Uncertainty quantification**: Built-in confidence intervals

### Prophet Settings Used

```python
Prophet(
    yearly_seasonality=True,      # Capture yearly patterns
    weekly_seasonality=False,     # Monthly data, no weekly signal
    seasonality_mode='multiplicative',  # Revenue scales with trend
    interval_width=0.95           # 95% confidence intervals
)
```

**Why multiplicative seasonality?** Business revenue typically has proportional seasonal effects - a 10% December boost on $100k revenue is different from $50k.

---

## 3. ARIMA/SARIMA Explained

### Components Breakdown

**ARIMA(p, d, q)**:
- **AR (p)**: AutoRegressive - uses past values
  - `y(t) = c + φ₁y(t-1) + φ₂y(t-2) + ... + ε(t)`
- **I (d)**: Integrated - differencing to make stationary
  - `d=1` means: `y'(t) = y(t) - y(t-1)`
- **MA (q)**: Moving Average - uses past errors
  - `y(t) = c + θ₁ε(t-1) + θ₂ε(t-2) + ... + ε(t)`

**SARIMA(p,d,q)(P,D,Q,s)** adds seasonal terms:
- **(P,D,Q)**: Seasonal AR, differencing, MA
- **s**: Seasonal period (12 for monthly data)

### Model Used: SARIMA(1,1,1)(1,1,1,12)

```
- (1,1,1): One AR term, one difference, one MA term
- (1,1,1,12): Same for seasonal component with period 12
```

### Stationarity Test (ADF)

The Augmented Dickey-Fuller test checks if the series is stationary:
- **H0**: Series has a unit root (non-stationary)
- **H1**: Series is stationary
- **If p > 0.05**: Need differencing (d ≥ 1)

---

## 4. Why Prophet Over ARIMA for This Project

| Factor | Prophet | ARIMA |
|--------|---------|-------|
| **Data size** | Works with 30+ points | Needs more for reliable seasonal estimation |
| **Seasonality** | Automatic via Fourier | Manual seasonal order selection |
| **Interpretability** | Component plots | Parameter interpretation complex |
| **Uncertainty** | Built-in intervals | Requires additional calculation |
| **Robustness** | Handles missing data | Sensitive to gaps |
| **Ease of use** | Minimal tuning | Requires p,d,q selection |

**Conclusion**: For a small business with 34 months of data, Prophet provides better out-of-box performance with less tuning.

---

## 5. Evaluation Metrics Explained

### MAE (Mean Absolute Error)

```
MAE = (1/n) × Σ|actual - predicted|
```

- **Interpretation**: Average absolute deviation in dollars
- **Pros**: Easy to interpret, same units as target
- **Cons**: Treats all errors equally

### RMSE (Root Mean Square Error)

```
RMSE = √[(1/n) × Σ(actual - predicted)²]
```

- **Interpretation**: Standard deviation of residuals
- **Pros**: Penalizes large errors more heavily
- **Cons**: Harder to interpret, sensitive to outliers

### MAPE (Mean Absolute Percentage Error)

```
MAPE = (100/n) × Σ|actual - predicted| / actual
```

- **Interpretation**: Average percentage error
- **Pros**: Scale-independent, easy to communicate
- **Cons**: Undefined when actual = 0, asymmetric

### Which to Use?

- **MAE**: When you want interpretable dollar amounts
- **RMSE**: When large errors are especially costly
- **MAPE**: When comparing across different revenue scales

---

## 6. Common Interview Questions & Answers

### Q1: "Walk me through this project"

**Answer Framework:**
1. **Problem**: Predict clinic revenue for 12 months to aid budget planning
2. **Data**: 34 months of revenue across 3 departments (GP, Internal Medicine, Dental)
3. **Approach**: Compared Prophet, SARIMA, and XGBoost; selected Prophet
4. **Results**: Generated forecast with 95% confidence intervals
5. **Value**: Enables proactive staffing and resource allocation

### Q2: "Why did you choose Prophet?"

**Answer:**
"Prophet was ideal because:
1. It handles small datasets well (we only had 34 months)
2. Automatic seasonality detection saved feature engineering time
3. Built-in uncertainty quantification for risk assessment
4. The component decomposition makes it easy to explain to business stakeholders"

### Q3: "How did you validate your model?"

**Answer:**
"I used TimeSeriesSplit cross-validation with 3 folds. This ensures:
1. Training always precedes test data temporally
2. No future information leaks into training
3. Model performance is evaluated on unseen future periods"

### Q4: "What would you do with more data?"

**Answer:**
"With more data, I would:
1. Add external regressors (holidays, economic indicators)
2. Try deep learning approaches (LSTM, Temporal Fusion Transformers)
3. Build separate models per department for more granular forecasts
4. Implement automated retraining as new data arrives"

### Q5: "How would you deploy this model?"

**Answer:**
"For production deployment:
1. Containerize with Docker for consistent environments
2. Set up monthly retraining pipeline (cron job or Airflow)
3. Store predictions in a database with confidence intervals
4. Create a dashboard for stakeholders (Streamlit or Tableau)
5. Add alerting if actual revenue deviates significantly from forecast"

### Q6: "What's the biggest limitation?"

**Answer:**
"The main limitation is data size - 34 months is enough for trend and seasonality detection, but:
1. Rare events (like COVID) can skew the model
2. We can't capture longer-term cycles (5-7 year economic cycles)
3. External shocks aren't predictable

To mitigate: I included wide confidence intervals and recommend quarterly model review."

---

## 7. Feature Engineering Highlights

### Cyclical Encoding

```python
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

**Why?** Regular month numbers (1-12) imply December is "far" from January. Sine/cosine encoding preserves the circular nature of months.

### Lag Features

```python
revenue_lag_1   # Last month
revenue_lag_12  # Same month last year
```

**Why?** Captures autocorrelation and year-over-year patterns.

### Rolling Statistics

```python
revenue_ma_3  # 3-month moving average
revenue_ma_6  # 6-month moving average
```

**Why?** Smooths noise and captures underlying trends.

---

## 8. Key Takeaways for Interviews

1. **Know your data**: 34 months, 3 departments, insurance + cash revenue
2. **Justify model choice**: Prophet for small data + automatic seasonality
3. **Explain validation**: TimeSeriesSplit prevents data leakage
4. **Discuss limitations**: Data size, external factors, uncertainty
5. **Show business value**: Budget planning, staffing decisions, risk assessment

---

## Quick Reference

| Metric | Value | Meaning |
|--------|-------|---------|
| Data points | 34 months | Jan 2021 - Oct 2023 |
| Forecast horizon | 12 months | Nov 2023 - Oct 2024 |
| Primary model | Prophet | Best for this dataset |
| Confidence level | 95% | Prediction intervals |
| Key seasonality | Monthly/Yearly | Revenue peaks in fall |

---

*This project demonstrates skills in: time series analysis, model selection, feature engineering, cross-validation, and business communication.*
