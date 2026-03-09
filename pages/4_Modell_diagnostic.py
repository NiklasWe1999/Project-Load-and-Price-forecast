import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px


# =========================
# Konfiguration & Pfade
# =========================

# LOAD
MODEL_PATH_LOAD = "models/load_model.pkl"
DATA_LOAD_TEST = "data/processed/features_load_processed.csv"
SCALER_X_LOAD = "models/scalers/scaler_X_load.pkl"
SCALER_Y_LOAD = "models/scalers/scaler_y_load.pkl"

# PRICE
MODEL_PATH_PRICE = "models/price_model.pkl"
DATA_PRICE_TEST = "data/processed/features_price_processed.csv"
SCALER_X_PRICE = "models/scalers/scaler_X_price.pkl"
SCALER_Y_PRICE = "models/scalers/scaler_y_price.pkl"

st.set_page_config(page_title="Model Diagnostics", layout="wide")

TARGET_COL_load = "DE_load_actual_entsoe_transparency"
TARGET_COL_price = "DE_LU_price_day_ahead"

SEQ_LEN = 24


# =========================
# Hilfsfunktion Sequenzen
# =========================


def get_processed_sequence(df, model_features, dt):

    seq = []

    for f in model_features:
        if "_t-" in f:
            base_name, lag_str = f.split("_t-")
            lag = int(lag_str)
        else:
            base_name = f
            lag = 0

        timestamp = dt - pd.Timedelta(hours=lag)

        if base_name in df.columns:
            hist_val = df.loc[df["utc_timestamp"] == timestamp, base_name]
            val = hist_val.iloc[0] if len(hist_val) > 0 else 50.0
        else:
            val = 50.0

        seq.append(val)

    return np.array(seq).reshape(1, -1)


# =========================
# LOAD MODEL
# =========================


@st.cache_resource
def prepare_load_data():

    bundle = joblib.load(MODEL_PATH_LOAD)
    model = bundle["model"]
    features = bundle["features"]

    scaler_X = joblib.load(SCALER_X_LOAD)
    # scaler_y = joblib.load(SCALER_Y_LOAD)

    df = pd.read_csv(DATA_LOAD_TEST, parse_dates=["utc_timestamp"])

    timestamps = df["utc_timestamp"].sort_values().unique()

    valid_timestamps = []

    for dt in timestamps:
        min_time = dt - pd.Timedelta(hours=SEQ_LEN - 1)
        if min_time >= timestamps[0]:
            valid_timestamps.append(dt)

    valid_timestamps = valid_timestamps[-100:]

    X_list = []
    y_true_list = []

    for dt in valid_timestamps:
        seq = get_processed_sequence(df, features, dt)

        if not np.isnan(seq).any():
            X_list.append(seq.flatten())

            actual = df.loc[df["utc_timestamp"] == dt, TARGET_COL_load]
            y_true_list.append(actual.iloc[0])

    X_final = np.array(X_list)
    # y_true = np.array(y_true_list)

    X_scaled = scaler_X.transform(X_final)

    # y_pred_scaled = model.predict(X_scaled)
    # y_pred = scaler_y.inverse_transform(y_pred_scaled)

    explainer = shap.TreeExplainer(model.estimators_[0])
    shap_values = explainer.shap_values(X_scaled)

    return model, features, shap_values, X_scaled


# =========================
# PRICE MODEL
# =========================


@st.cache_resource
def prepare_price_data():

    bundle = joblib.load(MODEL_PATH_PRICE)
    model = bundle["model"]
    features = bundle["features"]

    scaler_X = joblib.load(SCALER_X_PRICE)
    # scaler_y = joblib.load(SCALER_Y_PRICE)

    df = pd.read_csv(DATA_PRICE_TEST, parse_dates=["utc_timestamp"])

    timestamps = df["utc_timestamp"].sort_values().unique()

    valid_timestamps = []

    for dt in timestamps:
        min_time = dt - pd.Timedelta(hours=SEQ_LEN - 1)
        if min_time >= timestamps[0]:
            valid_timestamps.append(dt)

    valid_timestamps = valid_timestamps[-100:]

    X_list = []
    y_true_list = []

    for dt in valid_timestamps:
        seq = get_processed_sequence(df, features, dt)

        if not np.isnan(seq).any():
            X_list.append(seq.flatten())

            actual = df.loc[df["utc_timestamp"] == dt, TARGET_COL_price]
            y_true_list.append(actual.iloc[0])

    X_final = np.array(X_list)

    X_scaled = scaler_X.transform(X_final)

    explainer = shap.TreeExplainer(model.estimators_[0])
    shap_values = explainer.shap_values(X_scaled)

    return model, features, shap_values, X_scaled


# =========================
# Daten laden
# =========================

model_load, features_load, shap_load, X_load = prepare_load_data()
model_price, features_price, shap_price, X_price = prepare_price_data()


# =========================
# UI
# =========================

st.title("Model Diagnostics")

tab1, tab2 = st.tabs(["Load Model", "Price Model"])


# -------------------------
# TAB 1 LOAD
# -------------------------

with tab1:
    st.header("Load Model Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance")

        importance_df = pd.DataFrame(
            {
                "Feature": features_load,
                "Importance": model_load.estimators_[0].feature_importances_,
            }
        ).sort_values(by="Importance", ascending=False)

        st.plotly_chart(
            px.bar(
                importance_df.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
            ),
            use_container_width=True,
        )

    with col2:
        st.subheader("SHAP Beeswarm")

        fig, ax = plt.subplots(figsize=(6, 6))

        shap.summary_plot(
            shap_load,
            X_load,
            feature_names=features_load,
            show=False,
        )

        st.pyplot(fig)

        plt.close()


# -------------------------
# TAB 2 PRICE
# -------------------------

with tab2:
    st.header("Price Model Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance")

        importance_df = pd.DataFrame(
            {
                "Feature": features_price,
                "Importance": model_price.estimators_[0].feature_importances_,
            }
        ).sort_values(by="Importance", ascending=False)

        st.plotly_chart(
            px.bar(
                importance_df.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
            ),
            use_container_width=True,
        )

    with col2:
        st.subheader("SHAP Beeswarm")

        fig, ax = plt.subplots(figsize=(6, 6))

        shap.summary_plot(
            shap_price,
            X_price,
            feature_names=features_price,
            show=False,
        )

        st.pyplot(fig)

        plt.close()
