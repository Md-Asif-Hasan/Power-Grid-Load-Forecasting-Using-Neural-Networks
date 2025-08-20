import numpy as np
import pandas as pd
from config import PROCESSED_DIR, INPUT_WINDOW, FORECAST_HORIZON

def add_time_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df

def add_lags_and_rolls(df, target="load"):
    df = df.copy()
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    for l in lags:
        df[f"{target}_lag_{l}"] = df[target].shift(l)
    for w in [3, 6, 12, 24, 48, 168]:
        df[f"{target}_rollmean_{w}"] = df[target].shift(1).rolling(w).mean()
        df[f"{target}_rollstd_{w}"] = df[target].shift(1).rolling(w).std()
    return df

def make_sequences(df, feature_cols, target_col="load", input_window=INPUT_WINDOW, horizon=FORECAST_HORIZON):
    X, y = [], []
    values = df[feature_cols + [target_col]].dropna().values
    n = len(values); feat_dim = len(feature_cols)
    for i in range(input_window, n - horizon + 1):
        X.append(values[i - input_window:i, :feat_dim])
        y.append(values[i:i + horizon, -1])
    return np.array(X), np.array(y)

def run_feature_pipeline():
    df = pd.read_csv(PROCESSED_DIR / "prepared.csv", parse_dates=["datetime"], index_col="datetime")
    df = add_time_features(df)
    df = add_lags_and_rolls(df, target="load")
    df.dropna(inplace=True)
    df.to_csv(PROCESSED_DIR / "features.csv")
    return df

if __name__ == "__main__":
    run_feature_pipeline()
