import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(layout="wide")
st.title("German Energy Market – System Dashboard")

# ----------------------------
# DATA LOADING
# ----------------------------
df_load = pd.read_csv("data/processed/features_load_processed.csv")
df_price = pd.read_csv("data/processed/features_price_processed.csv")

df = df_load.merge(df_price, on="utc_timestamp", how="inner")
df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"]).dt.tz_localize(None)
df = df.sort_values("utc_timestamp")

# Doppelte Spalten bereinigen
for col in df.columns:
    if col.endswith("_x"):
        base = col[:-2]
        if base + "_y" in df.columns:
            df[base] = df[col]
            df.drop(columns=[col, base + "_y"], inplace=True)

# ----------------------------
# SIDEBAR FILTER
# ----------------------------
st.sidebar.header("Filter")

min_date = df["utc_timestamp"].min()
max_date = df["utc_timestamp"].max()

date_range = st.sidebar.date_input(
    "Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
)

df = df[
    (df["utc_timestamp"] >= pd.to_datetime(date_range[0]))
    & (df["utc_timestamp"] <= pd.to_datetime(date_range[1]))
]

# ----------------------------
# KPI SECTION
# ----------------------------
st.subheader(
    f"Key Metrics ({date_range[0].strftime('%Y-%m-%d')} – {date_range[1].strftime('%Y-%m-%d')})"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Load (MW)", round(df["DE_load_actual_entsoe_transparency"].mean(), 0))

col2.metric("Avg Price (€/MWh)", round(df["DE_LU_price_day_ahead"].mean(), 2))

col3.metric("Negative Price Share (%)", round(100 * df["Is_price_negative"].mean(), 2))

col4.metric(
    "Load–Price Correlation",
    round(
        df["DE_load_actual_entsoe_transparency"].corr(df["DE_LU_price_day_ahead"]), 3
    ),
)

# ==========================================================
# 1) HISTORICAL TIME SERIES
# ==========================================================
st.subheader("Historical Load and Price")

fig_ts = go.Figure()

fig_ts.add_trace(
    go.Scatter(
        x=df["utc_timestamp"],
        y=df["DE_load_actual_entsoe_transparency"],
        name="Load (MW)",
        yaxis="y1",
    )
)

fig_ts.add_trace(
    go.Scatter(
        x=df["utc_timestamp"],
        y=df["DE_LU_price_day_ahead"],
        name="Price (€/MWh)",
        yaxis="y2",
    )
)

fig_ts.update_layout(
    xaxis=dict(title="Time"),
    yaxis=dict(title="Load (MW)"),
    yaxis2=dict(title="Price (€/MWh)", overlaying="y", side="right"),
    legend=dict(x=0.01, y=0.99),
)

st.plotly_chart(fig_ts, use_container_width=True)

# ==========================================================
# 2) HEATMAP (Hour × Weekday)
# ==========================================================
st.subheader("Load Heatmap (Hour × Weekday)")

pivot_load = df.pivot_table(
    values="DE_load_actual_entsoe_transparency",
    index="weekday",
    columns="hour",
    aggfunc="mean",
)

fig_heat = px.imshow(
    pivot_load,
    aspect="auto",
    color_continuous_scale="Blues",
    labels=dict(x="Hour", y="Weekday", color="Avg Load (MW)"),
)

st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================================
# 3) ROLLING MEAN & VOLATILITY
# ==========================================================
# st.subheader("Rolling Mean and Volatility")

# fig_roll = go.Figure()

# fig_roll.add_trace(
#    go.Scatter(
#        x=df["utc_timestamp"],
#        y=df["price_roll_mean_24h"],
#        name="Price Rolling Mean (24h)",
#    )
# )
#
# fig_roll.add_trace(
#    go.Scatter(
#        x=df["utc_timestamp"],
#        y=df["price_roll_std_24h"],
#        name="Price Rolling Std (24h)",
#    )
# )

# fig_roll.update_layout(xaxis_title="Time", yaxis_title="Value")

# st.plotly_chart(fig_roll, use_container_width=True)


# # Optional Rolling Correlation
# df["rolling_corr_168h"] = (
#     df["DE_load_actual_entsoe_transparency"]
#     .rolling(72)
#     .corr(df["DE_LU_price_day_ahead"])
# )

# st.subheader("Rolling Correlation (7 Days)")

# fig_rc = px.line(
#     df,
#     x="utc_timestamp",
#     y="rolling_corr_168h",
#     labels={"rolling_corr_168h": "Correlation"},
# )

# st.plotly_chart(fig_rc, use_container_width=True)


# ==========================================================
# 3) Rolling Correlation (168h) + Quantile Regime
# ==========================================================

st.subheader("Rolling Correlation (168h) with Quantile Regime")

# Rolling Correlation
df["rolling_corr"] = (
    df["DE_load_actual_entsoe_transparency"]
    .rolling(168)
    .corr(df["DE_LU_price_day_ahead"])
)

corr_clean = df["rolling_corr"].dropna()
upper = corr_clean.quantile(0.75)
lower = corr_clean.quantile(0.25)

# Regime definieren
df["regime"] = "Mid"
df.loc[df["rolling_corr"] >= upper, "regime"] = "High Coupling"
df.loc[df["rolling_corr"] <= lower, "regime"] = "Low Coupling"


regime_colors = {"High Coupling": "green", "Mid": "grey", "Low Coupling": "red"}


fig_rc = go.Figure()
start_idx = 0
legend_used = {r: False for r in regime_colors.keys()}

for i in range(1, len(df)):
    if df["regime"].iloc[i] != df["regime"].iloc[start_idx] or i == len(df) - 1:
        reg = df["regime"].iloc[start_idx]
        show_legend = not legend_used[reg]
        fig_rc.add_trace(
            go.Scatter(
                x=df["utc_timestamp"].iloc[start_idx : i + 1],
                y=df["rolling_corr"].iloc[start_idx : i + 1],
                mode="lines",
                line=dict(color=regime_colors[reg], width=2),
                name=reg if show_legend else None,
                showlegend=show_legend,
            )
        )
        legend_used[reg] = True
        start_idx = i

# Schwellenlinien
fig_rc.add_hrect(y0=upper, y1=1, fillcolor="green", opacity=0.08, line_width=0)
fig_rc.add_hrect(y0=-0, y1=lower, fillcolor="red", opacity=0.08, line_width=0)
fig_rc.add_hline(y=upper, line_dash="dash")
fig_rc.add_hline(y=lower, line_dash="dash")

fig_rc.update_layout(
    xaxis_title="Time",
    yaxis_title="Correlation",
    yaxis=dict(range=[0, 1]),
    template="plotly_dark",
)

st.plotly_chart(fig_rc, use_container_width=True)

# ==========================================================
# 4) CORRELATION LOAD vs PRICE
# ==========================================================
st.subheader("Load vs Price Correlation")

fig_corr = px.scatter(
    df.sample(min(len(df), 5000)),
    x="DE_load_actual_entsoe_transparency",
    y="DE_LU_price_day_ahead",
    trendline="ols",
    labels={
        "DE_load_actual_entsoe_transparency": "Load (MW)",
        "DE_LU_price_day_ahead": "Price (€/MWh)",
    },
)

st.plotly_chart(fig_corr, use_container_width=True)
