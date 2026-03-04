import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =========================
# Konfiguration & Pfade
# =========================
DATA_PRICE = "data/processed/features_price_processed.csv"
MODEL_PATH_PRICE = "models/price_model.pkl"
SCALER_X_PRICE = "models/scalers/scaler_X_price.pkl"
SCALER_Y_PRICE = "models/scalers/scaler_y_price.pkl"

HORIZON = 6
SEQ_LEN = 24


# =========================
# Hilfsfunktionen
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PRICE, parse_dates=["utc_timestamp"])
    df["utc_timestamp"] = df["utc_timestamp"].dt.tz_convert(None)
    return df


def get_processed_sequence(df, model_features, dt):
    seq = []
    for f in model_features:
        base_name = f.split("_t-")[0] if "_t-" in f else f
        lag = int(f.split("_t-")[1]) if "_t-" in f else 0

        timestamp = dt - pd.Timedelta(hours=lag)
        hist_val = df.loc[df["utc_timestamp"] == timestamp, base_name]
        val = hist_val.values[0] if len(hist_val) > 0 else 50.0

        seq.append(val)

    return np.array(seq, dtype=float).reshape(1, -1)


# =========================
# App
# =========================
st.title("Electricity Price Forecast (24h → 6h)")

df_price = load_data()

min_ts = pd.Timestamp("2019-10-31 12:00:00")
max_ts = df_price["utc_timestamp"].max()
min_valid_ts = min_ts
max_valid_ts = max_ts - pd.Timedelta(hours=HORIZON)


# =========================
# Modell laden
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH_PRICE)


model_bundle = load_model()
model = model_bundle["model"]
model_features = model_bundle["features"]

scaler_X = joblib.load(SCALER_X_PRICE)
scaler_y = joblib.load(SCALER_Y_PRICE)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Settings")

    sel_date = st.date_input(
        "Date",
        value=min_valid_ts.date(),
        min_value=min_valid_ts.date(),
        max_value=max_valid_ts.date(),
    )

    sel_hour = st.slider("Hour", 0, 23, 12)

dt = pd.Timestamp(sel_date.year, sel_date.month, sel_date.day, sel_hour)


# =========================
# Historische Preisdaten (letzte 24h)
# =========================
hist_data = df_price[df_price["utc_timestamp"] < dt].tail(SEQ_LEN)

last_val = hist_data["DE_LU_price_day_ahead"].iloc[-1]
moving_avg = hist_data["DE_LU_price_day_ahead"].mean()


# =========================
# Forecast
# =========================
X_seq = get_processed_sequence(df_price, model_features, dt)
X_scaled = scaler_X.transform(X_seq)

y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_pred = y_pred[:HORIZON]


# =========================
# Baselines
# =========================
y_naive = np.full(HORIZON, last_val)
y_ma = np.full(HORIZON, moving_avg)


# =========================
# Realer Verlauf nach dt
# =========================
real_future = df_price[df_price["utc_timestamp"] > dt].head(HORIZON)
y_real = real_future["DE_LU_price_day_ahead"].values
real_x = list(range(1, len(y_real) + 1))


# =========================
# MAE
# =========================
if len(y_real) > 0:
    mae = np.mean(np.abs(y_pred[: len(y_real)] - y_real))
else:
    mae = None


# =========================
# Plot
# =========================
fig = go.Figure()

# Historie (24h)
hist_x = list(range(-23, 1))
fig.add_trace(
    go.Scatter(
        x=hist_x,
        y=hist_data["DE_LU_price_day_ahead"],
        mode="lines+markers",
        line=dict(color="white", width=2),
        marker=dict(color="white"),
        name="History (24h)",
    )
)

# Forecast
forecast_x = list(range(1, HORIZON + 1))
fig.add_trace(
    go.Scatter(
        x=forecast_x,
        y=y_pred,
        mode="lines+markers",
        line=dict(color="cyan", width=2),
        marker=dict(color="cyan"),
        name="Model Forecast",
    )
)

# Naive
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

# Real
fig.add_trace(
    go.Scatter(
        x=real_x,
        y=y_real,
        mode="lines+markers",
        line=dict(color="magenta", width=2),
        marker=dict(color="magenta"),
        name="Real Price",
    )
)

fig.add_vline(x=0, line=dict(color="gray", dash="dash"))

fig.update_layout(
    title=f"Price Forecast from {dt}",
    xaxis_title="Hours relative to selected time",
    yaxis_title="€/MWh",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_dark",
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# KPIs
# =========================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Last price", f"{last_val:.2f} €/MWh")
c2.metric(
    "Forecast peak",
    f"{y_pred.max():.2f} €/MWh",
    delta=f"{y_pred.max() - last_val:.2f}",
)

if len(y_real) > 0:
    real_peak = y_real.max()
    c3.metric(
        "Real Peak",
        f"{real_peak:.2f} €/MWh",
        delta=f"{real_peak - last_val:.2f}",
    )
else:
    c3.metric("Real Peak", "N/A", delta="N/A")

c4.metric(
    "Mean Absolute Error",
    f"{mae:.2f} €/MWh" if mae is not None else "N/A",
)
