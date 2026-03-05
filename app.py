import streamlit as st


st.title("German Energy Market – Forecasting & Analytics Dashboard")

st.markdown("""
This platform provides an end-to-end view of the German electricity market,
combining exploratory analysis, short-term forecasting, and model diagnostics.

---

### Dashboard Structure

**1️⃣ Market Overview**  
Exploratory analysis of historical load and price data.  
Includes key metrics, rolling correlations, regime detection, and structural patterns.

**2️⃣ Load Forecast (24h Horizon)**  
Short-term electricity load forecasting using XGBoost.  
Includes baseline comparison and MAE evaluation.

**3️⃣ Price Forecast (6h Horizon)**  
Short-term day-ahead price forecasting.  
Includes baseline comparison and MAE evaluation.

**4️⃣ Model Diagnostics**  
Feature importance analysis and SHAP-based interpretability.  
Provides insight into dominant drivers and model behavior.

---

Use the navigation menu on the left to explore each section.
""")
