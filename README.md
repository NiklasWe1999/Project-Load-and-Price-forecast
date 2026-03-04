# German Energy Market Dashboard & Forecasting

## Overview

This project develops an interactive dashboard and forecasting framework for the German electricity market. It combines historical load and price data with machine learning models to provide:

- Exploratory market analysis  
- Short-term load forecasting (24h horizon)  
- Short-term price forecasting (6h horizon)  
- Model diagnostics and interpretability (SHAP)  

Two XGBoost-based models are trained on engineered time-series features derived from historical German load and day-ahead price data and weather data.

The project is implemented as a multi-page Streamlit dashboard.

Acess the dashboard and try it out: LINK

---

## Data Source

The data originates from:

**Open Power System Data – Time Series Dataset**  
https://data.open-power-system-data.org/time_series/

**Open Power System Data – weather_data**  
https://data.open-power-system-data.org/weather_data/

Relevant variables include:

- German electricity load (ENTSO-E transparency)  
- Day-ahead electricity price (DE-LU)  
- Weather variables (temperature, radiation)  
- Renewable expectations (wind, solar)  

All timestamps are handled in UTC and converted to naive datetime objects for modeling consistency.

The data span over the time from 01.01.2015 to 31.12.2019, but the prices where only available from 14.10.2018 to 31.12.2019

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
- 6h forward prediction  
- Baseline comparison  
- Realized values  
- MAE evaluation  

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
- Baseline comparison  
- Visual forecast inspection  
- Regime-aware correlation analysis  

The dashboard allows dynamic evaluation at arbitrary historical timestamps.

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
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

- Scenario simulation (load shocks, renewable penetration shifts)

- Residual diagnostics & stability tests

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