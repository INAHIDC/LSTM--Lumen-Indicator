import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from indicator.technical_indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
)


def load_data(file_path):

    try:
        data = pd.read_csv(file_path)
        data = data.drop("Date", axis=1)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def add_technical_indicators(data):

    data["SMA_20"] = calculate_sma(data["Close"], 20)
    data["EMA_20"] = calculate_ema(data["Close"], 20)
    data["RSI_14"] = calculate_rsi(data["Close"], 14)
    data["MACD"] = calculate_macd(data["Close"])
    return data


def normalize_data(data):
    try:

        features = data.drop("Close", axis=1)
        target = data[["Close"]]

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        scaled_features = feature_scaler.fit_transform(features)
        scaled_target = target_scaler.fit_transform(target)

        return scaled_features, scaled_target, feature_scaler, target_scaler
    except Exception as e:
        print(f"Error normalizing data: {e}")
        return None, None, None, None


def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)
