import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =========================
# Konfiguration & Pfade
# =========================
DATA_LOAD = "data/processed/features_load_processed.csv"
DATA_PRICE = "data/processed/features_price_processed.csv"
MODEL_PATH_LOAD = "models/load_model.pkl"
MODEL_PATH_PRICE = "models/price_model.pkl"
SCALER_X_LOAD = "models/scalers/scaler_X_load.pkl"
SCALER_Y_LOAD = "models/scalers/scaler_y_load.pkl"
SCALER_X_PRICE = "models/scalers/scaler_X_price.pkl"
SCALER_Y_PRICE = "models/scalers/scaler_y_price.pkl"


# =========================
# Hilfsfunktionen
# =========================
@st.cache_data
def load_data():
    df_l = pd.read_csv(DATA_LOAD, parse_dates=["utc_timestamp"])
    df_p = pd.read_csv(DATA_PRICE, parse_dates=["utc_timestamp"])
    df_l["utc_timestamp"] = df_l["utc_timestamp"].dt.tz_convert(None)
    df_p["utc_timestamp"] = df_p["utc_timestamp"].dt.tz_convert(None)
    return df_l, df_p


def get_processed_sequence(
    df, model_features, dt, scenario_dict=None, load_multiplier=1.0
):
    seq = []
    for f in model_features:
        base_name = f.split("_t-")[0] if "_t-" in f else f
        lag = int(f.split("_t-")[1]) if "_t-" in f else 0

        if scenario_dict and base_name in scenario_dict:
            val = scenario_dict[base_name]
        else:
            timestamp = dt - pd.Timedelta(hours=lag)
            hist_val = df.loc[df["utc_timestamp"] == timestamp, base_name]
            val = hist_val.values[0] if len(hist_val) > 0 else 50.0

        # Last-Szenario-Logik
        if "load" in base_name:
            val *= load_multiplier

        seq.append(val)
    return np.array(seq, dtype=float).reshape(1, -1)


# =========================
# App
# =========================
st.title("Energy Live Forecast")

df_load, df_price = load_data()
horizon = 24
load_increase = 1


# Modelle laden
@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH_LOAD), joblib.load(MODEL_PATH_PRICE)


model_load_bundle, model_price_bundle = load_models()
scaler_X_load = joblib.load(SCALER_X_LOAD)
scaler_y_load = joblib.load(SCALER_Y_LOAD)

# ----------------------------
# Sidebar / Einstellungen
# ----------------------------
with st.sidebar:
    st.header("Settings")
    sel_date = st.date_input("Date", value=pd.to_datetime("2019-10-01"))
    sel_hour = st.slider("hour", 0, 23, 12)
    # horizon = st.selectbox("Forecast-Horizont", [6, 24], index=1)
    # st.markdown("---")
    # st.subheader("Szenario-Modifier")
    # load_increase = st.slider("Zusätzliche Last (%)", -20, 50, 0) / 100 + 1.0

dt = pd.Timestamp(sel_date.year, sel_date.month, sel_date.day, sel_hour)

# ----------------------------
# Historische Daten
# ----------------------------
hist_data = df_load[df_load["utc_timestamp"] < dt].tail(24)
last_val = hist_data["DE_load_actual_entsoe_transparency"].iloc[-1]
moving_avg = hist_data["DE_load_actual_entsoe_transparency"].mean()


# ----------------------------
# XGBoost Forecast
# ----------------------------
X_seq = get_processed_sequence(
    df_load, model_load_bundle["features"], dt, load_multiplier=load_increase
)
X_scaled = scaler_X_load.transform(X_seq)
y_xgb = scaler_y_load.inverse_transform(
    model_load_bundle["model"].predict(X_scaled)
).flatten()
y_xgb = y_xgb[:horizon]

# Baselines
y_naive = np.full(horizon, last_val)
y_ma = np.full(horizon, moving_avg)

# Realer Verlauf nach dt
real_future = df_load[df_load["utc_timestamp"] > dt].head(horizon)
y_real = real_future["DE_load_actual_entsoe_transparency"].values
real_x = list(range(1, len(y_real) + 1))

# ----------------------------
# MAE berechnen
# ----------------------------
if len(y_real) > 0:  # Sicherstellen, dass reale Werte vorhanden sind
    mae = np.mean(np.abs(y_xgb[: len(y_real)] - y_real))
else:
    mae = None
# ----------------------------
# Plot mit Plotly (Dark Mode freundlich)
# ----------------------------
fig = go.Figure()

# Historie
hist_x = list(range(-23, 1))
fig.add_trace(
    go.Scatter(
        x=hist_x,
        y=hist_data["DE_load_actual_entsoe_transparency"],
        mode="lines+markers",
        line=dict(color="white", width=2),
        marker=dict(color="white"),
        name="History (24h)",
    )
)

# Forecast
forecast_x = list(range(1, horizon + 1))
fig.add_trace(
    go.Scatter(
        x=forecast_x,
        y=y_xgb,
        mode="lines+markers",
        line=dict(color="cyan", width=2),
        marker=dict(color="cyan"),
        name="XGBoost Forecast",
    )
)

# Naive Baseline
fig.add_trace(
    go.Scatter(
        x=forecast_x,
        y=y_naive,
        mode="lines",
        line=dict(color="orange", dash="dash"),
        name="Naive (Last Value)",
    )
)

# Moving Average
fig.add_trace(
    go.Scatter(
        x=forecast_x,
        y=y_ma,
        mode="lines",
        line=dict(color="lime", dash="dash"),
        name="Moving Average (24h)",
    )
)

# # Konfidenzintervall +/- 5% XGBoost
# fig.add_trace(
#     go.Scatter(
#         x=forecast_x + forecast_x[::-1],  # Hin + Rück
#         y=list(y_xgb * 1.05) + list((y_xgb * 0.95)[::-1]),
#         fill="toself",
#         fillcolor="rgba(0,255,255,0.2)",  # hellblau transparent
#         line=dict(color="rgba(255,255,255,0)"),
#         hoverinfo="skip",
#         showlegend=True,
#         name="Konfidenzintervall (5%)",
#     )
# )

fig.add_trace(
    go.Scatter(
        x=real_x,
        y=y_real,
        mode="lines+markers",
        line=dict(color="magenta", width=2),
        marker=dict(color="magenta"),
        name="Real history",
    )
)

# Trennlinie Heute/Morgen
fig.add_vline(x=0, line=dict(color="gray", dash="dash"))

fig.update_layout(
    title=f"Load Forecast from {dt}",
    xaxis_title="Hours relative to the selected time",
    yaxis_title="MW",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_dark",  # Dark Mode
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Kennzahlen
# ----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last value", f"{last_val:.2f} MW")
c2.metric(
    "Forecast peak", f"{y_xgb.max():.2f} MW", delta=f"{y_xgb.max() - last_val:.2f}"
)
if len(y_real) > 0:
    real_peak = y_real.max()
    c3.metric("Real Peak", f"{real_peak:.2f} MW", delta=f"{real_peak - last_val:.2f}")
else:
    c3.metric("Real Peak", "N/A", delta="N/A")

c4.metric("Mean-Absolute-Error Forecast", f"{mae:.2f} MW" if mae is not None else "N/A")
