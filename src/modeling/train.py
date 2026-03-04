import os
import pandas as pd
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

# =========================
# Projektpfade
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_PATH_price = os.path.join(
    PROJECT_ROOT, "data", "processed", "features_price_processed.csv"
)
DATA_PATH_load = os.path.join(
    PROJECT_ROOT, "data", "processed", "features_load_processed.csv"
)

MODEL_PATH_price = os.path.join(PROJECT_ROOT, "models", "price_model.pkl")
MODEL_PATH_load = os.path.join(PROJECT_ROOT, "models", "load_model.pkl")

SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scalers")


# =========================
# Funktionen
# =========================
def load_data(data_path):
    df = pd.read_csv(data_path)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
    df = df.sort_values("utc_timestamp")
    df = df.dropna().reset_index(drop=True)
    return df


def create_sequences(df, target_col, seq_len, horizon):
    features_df = df.drop(columns=[target_col, "utc_timestamp"])
    target_series = df[target_col]

    base_feature_names = features_df.columns.tolist()
    X_raw = features_df.values
    y_raw = target_series.values

    X_out, y_out = [], []

    for i in range(len(X_raw) - seq_len - horizon + 1):
        X_window = X_raw[i : i + seq_len].flatten()
        y_window = y_raw[i + seq_len : i + seq_len + horizon]
        X_out.append(X_window)
        y_out.append(y_window)

    seq_feature_names = [
        f"{feat}_t-{seq_len - h}" for h in range(seq_len) for feat in base_feature_names
    ]

    return np.array(X_out), np.array(y_out), seq_feature_names


def train_model(X_train, y_train):
    base_model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test_scaled, scaler_y):
    predictions_scaled = model.predict(X_test)

    # Rückskalieren
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test = scaler_y.inverse_transform(y_test_scaled)

    mae = mean_absolute_error(y_test.flatten(), predictions.flatten())
    rmse = mean_squared_error(y_test.flatten(), predictions.flatten()) ** 0.5

    print("Model Performance (Real Scale):")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")


def save_model(model, feature_names, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {"model": model, "features": feature_names}
    joblib.dump(bundle, model_path)
    print(f"Model saved to: {model_path}")


def save_scaler(scaler, name):
    os.makedirs(SCALER_PATH, exist_ok=True)
    joblib.dump(scaler, os.path.join(SCALER_PATH, f"{name}.pkl"))
    print(f"Scaler saved: {name}")


# =========================
# Hauptfunktion
# =========================
def main():
    print("Loading data...")
    df_price = load_data(DATA_PATH_price)
    df_load = load_data(DATA_PATH_load)

    SEQ_LEN = 24
    HORIZON_LOAD = 24
    HORIZON_PRICE = 6

    print("Creating sequences...")
    X_load_seq, y_load_seq, load_feature_names = create_sequences(
        df_load, "DE_load_actual_entsoe_transparency", SEQ_LEN, HORIZON_LOAD
    )
    X_price_seq, y_price_seq, price_feature_names = create_sequences(
        df_price, "DE_LU_price_day_ahead", SEQ_LEN, HORIZON_PRICE
    )

    # Train/Test Split
    split_load = int(len(X_load_seq) * 0.8)
    split_price = int(len(X_price_seq) * 0.8)

    X_load_train = X_load_seq[:split_load]
    X_load_test = X_load_seq[split_load:]
    y_load_train = y_load_seq[:split_load]
    y_load_test = y_load_seq[split_load:]

    X_price_train = X_price_seq[:split_price]
    X_price_test = X_price_seq[split_price:]
    y_price_train = y_price_seq[:split_price]
    y_price_test = y_price_seq[split_price:]

    # =========================
    # Scaler anwenden
    # =========================
    print("Scaling data...")
    scaler_X_load = StandardScaler()
    scaler_y_load = StandardScaler()
    X_load_train_scaled = scaler_X_load.fit_transform(X_load_train)
    X_load_test_scaled = scaler_X_load.transform(X_load_test)
    y_load_train_scaled = scaler_y_load.fit_transform(y_load_train)
    y_load_test_scaled = scaler_y_load.transform(y_load_test)

    scaler_X_price = StandardScaler()
    scaler_y_price = StandardScaler()
    X_price_train_scaled = scaler_X_price.fit_transform(X_price_train)
    X_price_test_scaled = scaler_X_price.transform(X_price_test)
    y_price_train_scaled = scaler_y_price.fit_transform(y_price_train)
    y_price_test_scaled = scaler_y_price.transform(y_price_test)

    # =========================
    # Modelle trainieren
    # =========================
    print("Training models...")
    model_load = train_model(X_load_train_scaled, y_load_train_scaled)
    model_price = train_model(X_price_train_scaled, y_price_train_scaled)

    # =========================
    # Evaluieren
    # =========================
    print("Evaluating models...")
    evaluate_model(model_load, X_load_test_scaled, y_load_test_scaled, scaler_y_load)
    evaluate_model(
        model_price, X_price_test_scaled, y_price_test_scaled, scaler_y_price
    )

    # =========================
    # Modelle speichern
    # =========================
    save_model(model_load, load_feature_names, MODEL_PATH_load)
    save_model(model_price, price_feature_names, MODEL_PATH_price)

    # =========================
    # Scaler speichern
    # =========================
    save_scaler(scaler_X_load, "scaler_X_load")
    save_scaler(scaler_y_load, "scaler_y_load")
    save_scaler(scaler_X_price, "scaler_X_price")
    save_scaler(scaler_y_price, "scaler_y_price")

    print("Training complete.")


if __name__ == "__main__":
    main()
