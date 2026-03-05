# German Energy Market Dashboard & Forecasting

## Overview

This project provides an interactive dashboard for analyzing and forecasting the German electricity market.

The application combines historical load, day-ahead price, and weather data with machine learning models to enable:

- Exploratory market analysis  
- Short-term load forecasting (24h horizon)  
- Short-term price forecasting (6h horizon)  
- Model diagnostics and interpretability (SHAP-based analysis)  

Two XGBoost models are trained on engineered time-series features, including lag structures, rolling statistics, and seasonal encodings.

The primary objective of this project is to demonstrate an end-to-end data science workflow. Starting from feature engineering and model development to interactive visualization and deployment.

---

### Live Dashboard

The application is fully deployed as a multi-page Streamlit dashboard.

**Access the live dashboard here:**  
👉 https://project-load-and-price-forecast.streamlit.app

---

### Notes

Forecast performance is evaluated against simple baseline models (naive and moving average).  
A more detailed research-oriented analysis of model selection and validation will be added in the notebook section.

---

## Data Source

The data is sourced from:

**Open Power System Data – Time Series Dataset**  
https://data.open-power-system-data.org/time_series/

**Open Power System Data – Weather Dataset**  
https://data.open-power-system-data.org/weather_data/

The following variables are used:

- German electricity load (ENTSO-E transparency platform)  
- Day-ahead electricity price (DE-LU bidding zone)  
- Weather variables (temperature, radiation)  
- Renewable generation expectations (wind, solar)  

All timestamps are handled in UTC and converted to naive datetime objects to ensure modeling consistency.  
The data has an hourly frequency.

The load data spans from 01.01.2015 to 31.12.2019.  
Day-ahead price data is available from 14.10.2018 to 31.12.2019.

---

## Project Structure

```python
├── app/ #Streamlit
│
├── pages/
│ ├── 1_Market_Overview.py
│ ├── 2_Forecast_Load.py
│ ├── 3_Forecast_Price.py
│ └── 4_Modell_Diagnostic.py
├── data/
│ ├── processed/# Engineered feature datasets
│ └── raw/ # raw datasets
├── models/
│ ├── load_model.pkl
│ ├── price_model.pkl
│ └── scalers/
├── notebooks/ # (reserved for future research notebook)
├── src/
│ ├── feature.py # feature engineering
│ └── modeling/
│  ├── predict.py 
│  └── train.py
├── enviroment.txt
└── README.md
```

The `notebooks/` directory is reserved for a future research notebook containing deeper exploratory and modeling analysis.

---

## Feature Engineering

### Load Model Features

The load forecasting model uses:

- Calendar features (hour, weekday, month)  
- Cyclical encodings (hour_sin, hour_cos)  
- Seasonal indicators (is_winter, is_summer, is_holiday)  
- Weather inputs  
- Lag features (1h to 336h)  
- Rolling statistics (mean, std over 6h–24h)  
- Load deltas  

This allows the model to capture:

- Intraday seasonality  
- Weekly structure  
- Medium-term dependencies  
- Volatility dynamics  

---

### Price Model Features

The price forecasting model includes:

- Day-ahead price lags (1h–336h)  
- Rolling price statistics  
- Negative price indicators  
- Residual load proxy  
- Weather inputs  
- Renewable generation expectations  
- Calendar & cyclical features  
- Load lag interaction terms  

This reflects the structural coupling between residual load and price formation.

---

## Models

Two separate models are trained:

### 1. Load Forecast Model

- Algorithm: XGBoost (multi-output structure)  
- Input horizon: 24 hours
- Forecast horizon: 24 hours  

**Baselines:**
- Naive (last observed value)  
- Moving average (24h)  

---

### 2. Price Forecast Model

- Algorithm: XGBoost  
- Input horizon: 24 hours
- Forecast horizon: 6 hours  

**Baselines:**
- Naive (last observed price)  
- Moving average (24h)  

---

Both models use:

- Feature scaling  
- Stored preprocessing pipelines  
- Persisted scalers and model bundles via joblib  

---

## Dashboard Components

### 1. Market Overview

- Historical load and price time series  
- KPI metrics:
  - Average load  
  - Average price  
  - Share of negative prices  
  - Load–price correlation  
- Hour × weekday heatmap  
- Rolling correlation regime detection (168h window)  

Regimes are classified into:

- High coupling  
- Mid regime  
- Low coupling  

---

### 2. Load Forecast Page

- 24h historical context  
- 24h model forecast  
- Baseline comparisons  
- Realized future values (if available)  
- MAE evaluation  
- Peak comparison  

---

### 3. Price Forecast Page

- 24h historical context  
- 6h model forecast  
- Baseline comparisons  
- Realized future values (if available)  
- MAE evaluation  
- Peak comparison  

---

### 4. Model Diagnostics

- Feature importance (XGBoost)  
- SHAP summary plot (TreeExplainer)  
- Diagnostics based on recent valid timestamps  

This allows interpretation of:

- Dominant lag structures  
- Weather influence  
- Seasonal effects  
- Structural drivers  

---

## Evaluation

Model performance is evaluated using:

- Mean Absolute Error (MAE)  
- Comparison against baseline models (naive and moving average)  
- Visual inspection of forecast trajectories  

The dashboard enables dynamic evaluation at arbitrary historical timestamps.  
These timestamps are strictly out-of-sample and were not included in the training dataset, ensuring an unbiased assessment of model performance.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/NiklasWe1999/Project-Load-and-Price-forecast.git
cd Project-Load-and-Price-forecast
```
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the dashboard:
```bash
streamlit run app/main.py
```

## Future work:

Planned extensions include:

- Dedicated research notebook with deeper statistical analysis

- Hyperparameter optimization comparison

- Probabilistic forecasting / uncertainty bands

- The notebook section will provide a more research-oriented exploration of the modeling pipeline.

## Limitations

- Deterministic forecasts only (no probabilistic intervals)

- No exogenous forecast uncertainty modeling

- Feature engineering is static and not dynamically updated

- Structural market regime changes are not explicitly modeled

## Techstak

- Python

- Streamlit

- XGBoost

- SHAP

- Plotly

- Pandas / NumPy

- Scikit-learn