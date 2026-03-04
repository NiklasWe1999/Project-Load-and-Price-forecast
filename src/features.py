# src/features.py
import pandas as pd
import numpy as np
import os
import holidays

# =========================
# Funktionen für Feature Engineering
# =========================


def merge_load_weather(df_load, df_weather):
    """Merge Load- und Wetterdaten auf UTC Timestamp"""
    df_load["utc_timestamp"] = pd.to_datetime(df_load["utc_timestamp"])
    df_weather["utc_timestamp"] = pd.to_datetime(df_weather["utc_timestamp"])
    df = df_load.merge(df_weather, on="utc_timestamp", how="inner")
    return df


def add_time_features(df):
    """Fügt Stunde, Wochentag, Monat und zyklische Hour Features hinzu"""
    df["hour"] = df["utc_timestamp"].dt.hour
    df["weekday"] = df["utc_timestamp"].dt.weekday
    df["month"] = df["utc_timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def compute_residual_load(df):
    """Residual Load = Forecast - Wind - Solar"""
    df["residual_load"] = (
        df["DE_LU_load_forecast_entsoe_transparency"]
        - df["DE_LU_wind_generation_actual"]
        - df["DE_LU_solar_generation_actual"]
    )
    df["residual_load_lag_24h"] = df["residual_load"].shift(24)
    return df


def add_holiday_features(df, years=[2015, 2021]):
    """Markiert Feiertage, Winter- und Sommermonate"""
    de_holidays = holidays.Germany(years=years)
    df["is_holiday"] = df["utc_timestamp"].dt.date.apply(
        lambda x: 1 if x in de_holidays else 0
    )
    df = df.set_index("utc_timestamp")
    df["is_winter"] = df.index.month.isin([12, 1, 2]).astype(int)
    df["is_summer"] = df.index.month.isin([6, 7, 8]).astype(int)
    df = df.reset_index()
    return df


def interpolate_missing(df, col):
    df[col] = df[col].interpolate(method="time")
    return df


def create_load_features(df):
    """Lag-, Rolling- und Ramp Features für Load"""
    df["load_lag_1h"] = df["DE_load_actual_entsoe_transparency"].shift(1)
    df["load_lag_2h"] = df["DE_load_actual_entsoe_transparency"].shift(2)
    df["load_lag_3h"] = df["DE_load_actual_entsoe_transparency"].shift(3)
    df["load_lag_24h"] = df["DE_load_actual_entsoe_transparency"].shift(24)
    df["load_lag_48h"] = df["DE_load_actual_entsoe_transparency"].shift(48)
    df["load_lag_72h"] = df["DE_load_actual_entsoe_transparency"].shift(72)
    df["load_lag_168h"] = df["DE_load_actual_entsoe_transparency"].shift(168)
    df["load_lag_336h"] = df["DE_load_actual_entsoe_transparency"].shift(336)

    # Rolling Statistics
    df["load_roll_mean_6h"] = df["DE_load_actual_entsoe_transparency"].rolling(6).mean()
    df["load_roll_mean_12h"] = (
        df["DE_load_actual_entsoe_transparency"].rolling(12).mean()
    )
    df["load_roll_mean_24h"] = (
        df["DE_load_actual_entsoe_transparency"].rolling(24).mean()
    )
    df["load_roll_std_24h"] = df["DE_load_actual_entsoe_transparency"].rolling(24).std()
    df["load_roll_std_6h"] = df["DE_load_actual_entsoe_transparency"].rolling(6).std()

    # Ramp Feature
    df["load_delta_1h"] = df["DE_load_actual_entsoe_transparency"].diff(1)

    return df


def create_price_features(df):
    """Lag-, Rolling- und Ramp Features für Preis"""
    df["price_lag_1h"] = df["DE_LU_price_day_ahead"].shift(1)
    df["price_lag_2h"] = df["DE_LU_price_day_ahead"].shift(2)
    df["price_lag_3h"] = df["DE_LU_price_day_ahead"].shift(3)
    df["price_lag_12h"] = df["DE_LU_price_day_ahead"].shift(12)
    df["price_lag_24h"] = df["DE_LU_price_day_ahead"].shift(24)
    df["price_lag_48h"] = df["DE_LU_price_day_ahead"].shift(48)
    df["price_lag_72h"] = df["DE_LU_price_day_ahead"].shift(72)
    df["price_lag_168h"] = df["DE_LU_price_day_ahead"].shift(168)
    df["price_lag_336h"] = df["DE_LU_price_day_ahead"].shift(336)

    # Rolling Volatility
    df["price_roll_mean_6h"] = df["DE_LU_price_day_ahead"].rolling(6).mean()
    df["price_roll_mean_12h"] = df["DE_LU_price_day_ahead"].rolling(12).mean()
    df["price_roll_mean_24h"] = df["DE_LU_price_day_ahead"].rolling(24).mean()
    df["price_roll_std_6h"] = df["DE_LU_price_day_ahead"].rolling(6).std()
    df["price_roll_std_24h"] = df["DE_LU_price_day_ahead"].rolling(24).std()

    df["price_delta_1h"] = df["DE_LU_price_day_ahead"].diff(1)
    df["wind_expected"] = df["DE_wind_profile"] * df["DE_wind_capacity"]
    df["solar_expected"] = df["DE_solar_profile"] * df["DE_solar_capacity"]
    df["Is_price_negative"] = (df["DE_LU_price_day_ahead"] < 0).astype(int)
    df["was_negative_1h_ago"] = df["Is_price_negative"].shift(1)

    return df


def select_load_columns(df):
    cols_load = [
        "utc_timestamp",
        "DE_temperature",
        "DE_radiation_direct_horizontal",
        "DE_radiation_diffuse_horizontal",
        "DE_load_actual_entsoe_transparency",  # Target
        "hour",
        "weekday",
        "month",
        "hour_sin",
        "hour_cos",
        "is_holiday",
        "is_winter",
        "is_summer",
        "load_delta_1h",
        "load_roll_std_24h",
        "load_roll_std_6h",
        "load_roll_mean_6h",
        "load_roll_mean_12h",
        "load_roll_mean_24h",
        "load_lag_168h",
        "load_lag_24h",
        "load_lag_1h",
        "load_lag_2h",
        "load_lag_3h",
        "load_lag_48h",
        "load_lag_72h",
        "load_lag_336h",
    ]
    return df[cols_load].dropna()


def select_price_columns(df):
    cols_price = [
        "utc_timestamp",
        "DE_LU_price_day_ahead",  # Target
        "Is_price_negative",
        "was_negative_1h_ago",
        "wind_expected",
        "solar_expected",
        "DE_temperature",
        "DE_radiation_direct_horizontal",
        "DE_radiation_diffuse_horizontal",
        "hour",
        "weekday",
        "month",
        "hour_sin",
        "hour_cos",
        "residual_load",
        "is_holiday",
        "is_winter",
        "is_summer",
        "price_lag_1h",
        "price_lag_2h",
        "price_lag_3h",
        "price_lag_12h",
        "price_lag_24h",
        "price_lag_48h",
        "price_lag_72h",
        "price_lag_168h",
        "price_lag_336h",
        "price_roll_mean_6h",
        "price_roll_mean_12h",
        "price_roll_mean_24h",
        "price_roll_std_6h",
        "price_roll_std_24h",
        "price_delta_1h",
        "load_lag_24h",
        "DE_load_forecast_entsoe_transparency",
    ]
    return df[cols_price].dropna()


# =========================
# Pipeline: von raw -> interim -> processed
# =========================


def create_features_pipeline(
    load_csv,
    weather_csv,
    interim_path="data/interim/",
    processed_path="data/processed/",
):
    os.makedirs(interim_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)

    # Daten laden
    df_load = pd.read_csv(load_csv)
    df_weather = pd.read_csv(weather_csv)

    # Merge
    df = merge_load_weather(df_load, df_weather)

    # Features
    df = add_time_features(df)
    df = compute_residual_load(df)
    df = add_holiday_features(df)
    # df = interpolate_missing(df, "DE_load_forecast_entsoe_transparency")
    df = create_load_features(df)
    df = create_price_features(df)

    # Interim speichern
    df.to_csv(os.path.join(interim_path, "features_interim.csv"), index=False)

    # Processed Daten
    df_load_model = select_load_columns(df)
    df_load_model.to_csv(
        os.path.join(processed_path, "features_load_processed.csv"), index=False
    )

    df_price_model = select_price_columns(df)
    df_price_model.to_csv(
        os.path.join(processed_path, "features_price_processed.csv"), index=False
    )

    print("Feature Pipeline abgeschlossen!")
    return df_load_model, df_price_model


# =========================
# Beispielaufruf
# =========================

if __name__ == "__main__":
    csv_path = "data/raw/time_series_60min_singleindex_filtered.csv"
    weather_data = "data/raw/weather_data_filtered.csv"
    df_load_model, df_price_model = create_features_pipeline(csv_path, weather_data)
