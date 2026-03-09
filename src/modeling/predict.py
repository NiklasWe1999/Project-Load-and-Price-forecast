import pandas as pd
import joblib

# Lade Modelle & Scaler
load_model_bundle = joblib.load("models/load_model.pkl")
load_model = load_model_bundle["model"]
load_features = load_model_bundle["features"]

price_model_bundle = joblib.load("models/price_model.pkl")
price_model = price_model_bundle["model"]
price_features = price_model_bundle["features"]


scaler_X_load = joblib.load("models/scaler_X_load.pkl")
scaler_y_load = joblib.load("models/scaler_y_load.pkl")
scaler_X_price = joblib.load("models/scaler_X_price.pkl")
scaler_y_price = joblib.load("models/scaler_y_price.pkl")

SEQ_LEN = 24
HORIZON_LOAD = 24
HORIZON_PRICE = 6


def prepare_input_flattened(inputs, feature_list, seq_len):

    df = pd.DataFrame([inputs])

    df_seq = pd.concat([df] * seq_len, ignore_index=True)

    df_seq = df_seq[feature_list]

    X_flat = df_seq.values.flatten().reshape(1, -1)

    return X_flat


def predict_load(inputs, model=load_model, scaler_X=None, scaler_y=None):
    X_flat = prepare_input_flattened(inputs, load_features, SEQ_LEN)

    if scaler_X is not None:
        X_flat = scaler_X.transform(X_flat)

    y_pred_scaled = model.predict(X_flat)

    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    else:
        y_pred = y_pred_scaled

    return y_pred.flatten()  # 1D-Array mit HORIZON_LOAD Werten


def predict_price(inputs, model=price_model, scaler_X=None, scaler_y=None):
    X_flat = prepare_input_flattened(inputs, price_features, SEQ_LEN)

    if scaler_X is not None:
        X_flat = scaler_X.transform(X_flat)

    y_pred_scaled = model.predict(X_flat)

    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    else:
        y_pred = y_pred_scaled

    return y_pred.flatten()  # 1D-Array mit HORIZON_PRICE Werten
