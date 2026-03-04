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
MODEL_PATH_LOAD = "models/load_model.pkl"
DATA_LOAD_TEST = "data/processed/features_load_processed.csv"
SCALER_X_LOAD = "models/scalers/scaler_X_load.pkl"
SCALER_Y_LOAD = "models/scalers/scaler_y_load.pkl"

st.set_page_config(page_title="Model Diagnostics", layout="wide")

TARGET_COL = "DE_load_actual_entsoe_transparency"
SEQ_LEN = 24  # Muss zu deinem Modell passen


# =========================
# Hilfsfunktion für Sequenzen
# =========================
def get_processed_sequence(df, model_features, dt):
    """
    Erstellt eine flache Sequenz (1, n_features) für einen Zeitstempel dt.
    Fehlende Werte werden mit 50.0 aufgefüllt.
    """
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
# Daten & Modelle laden
# =========================
@st.cache_resource
def prepare_diagnostic_data():
    # 1. Assets laden
    bundle = joblib.load(MODEL_PATH_LOAD)
    model = bundle["model"]  # MultiOutputRegressor
    features = bundle["features"]
    scaler_X = joblib.load(SCALER_X_LOAD)
    scaler_y = joblib.load(SCALER_Y_LOAD)

    df = pd.read_csv(DATA_LOAD_TEST, parse_dates=["utc_timestamp"])
    if df["utc_timestamp"].dt.tz is not None:
        df["utc_timestamp"] = df["utc_timestamp"].dt.tz_convert(None)

    # 2. Zeitstempel validieren
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
            actual = df.loc[df["utc_timestamp"] == dt, TARGET_COL]
            y_true_list.append(actual.iloc[0])

    X_final = np.array(X_list)
    y_true = np.array(y_true_list)

    # 3. Skalieren & Vorhersage
    X_scaled = scaler_X.transform(X_final)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # 4. SHAP nur für den ersten Output
    explainer = shap.TreeExplainer(model.estimators_[0])
    shap_values = explainer.shap_values(X_scaled)

    return (
        model,
        features,
        explainer,
        shap_values,
        X_scaled,
        y_true,
        y_pred,
        valid_timestamps,
    )


model, features, explainer, shap_values, X_scaled, y_true, y_pred, timestamps = (
    prepare_diagnostic_data()
)

# =========================
# Nur erste Stunde für Tabs 2 & 3
# =========================
y_true_1h = y_true  # y_true ist schon 1D
y_pred_1h = y_pred[:, 0]  # erste Vorhersagestunde

# =========================
# UI Layout
# =========================
st.title("Model Diagnostics load model ")

tab1 = st.tabs(["Feature Importance & SHAP"])

# -------------------------
# Tab 1: Feature Importance & SHAP
# -------------------------

col1, col2 = st.columns(2)
with col1:
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.estimators_[0].feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)
    st.plotly_chart(
        px.bar(importance_df.head(15), x="Importance", y="Feature", orientation="h"),
        use_container_width=True,
    )

with col2:
    st.subheader("SHAP Beeswarm Plot")
    fig, ax = plt.subplots(figsize=(6, 6))
    shap.summary_plot(shap_values, X_scaled, feature_names=features, show=False)
    st.pyplot(fig)
    plt.close()
