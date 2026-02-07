# Clinic Revenue Prediction # Patient Data taken out for security reasons

A machine learning project to predict clinic revenue for the next 12 months using historical patient and revenue data from 2021-2023.

## Project Overview

This project uses time series forecasting to predict monthly revenue for a multi-department clinic. Three modeling approaches are compared:

1. **Prophet** (Primary) - Facebook's time series library, ideal for business data
2. **SARIMA** - Classical statistical approach
3. **XGBoost** - Machine learning with engineered features

## Data

- **Source**: `PatientData/TOTAL COUNT 2021-2023.xlsx`
- **Period**: January 2021 - October 2023 (34 months)
- **Departments**: General Practice, Internal Medicine, Dental
- **Revenue Sources**: Insurance payments + Cash payments

## Project Structure

```
ClinicPrediction/
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Load and clean raw data
│   ├── 02_feature_engineering.ipynb # Create ML features
│   ├── 03_model_development.ipynb   # Train and evaluate models
│   └── 04_final_forecast.ipynb      # Generate 12-month forecast
├── data/
│   └── processed/
│       ├── cleaned_revenue_data.csv
│       └── features_revenue_data.csv
├── outputs/
│   └── figures/
│       ├── revenue_forecast.png     # Main forecast visualization
│       ├── model_comparison.png
│       ├── seasonality_pattern.png
│       └── prophet_components.png
├── docs/
│   └── interview_prep.md            # Interview Q&A guide
├── PatientData/                     # Raw data
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd ClinicPrediction

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook
```

## Key Results

- **Forecast Horizon**: 12 months (2024)
- **Primary Model**: Prophet with multiplicative seasonality
- **Confidence Interval**: 95%

## Features Engineered

- **Temporal**: Cyclical month encoding (sin/cos)
- **Lag Features**: Previous month, same month last year
- **Rolling Statistics**: 3-month and 6-month moving averages
- **Business Metrics**: Revenue per patient, insurance ratio

## Requirements

- Python 3.9+
- pandas, numpy, openpyxl
- prophet, statsmodels, scikit-learn, xgboost
- matplotlib, seaborn
- jupyter

## License

MIT
