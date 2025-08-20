import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

from config import (PROCESSED_DIR, MODEL_TYPE, INPUT_WINDOW, FORECAST_HORIZON,
                    EPOCHS, BATCH_SIZE, LEARNING_RATE, HIDDEN_SIZE, NUM_LAYERS,
                    DROPOUT, VAL_SPLIT_DATE, TEST_SPLIT_DATE, METRICS, MODELS_DIR, SEED)
from features import make_sequences
from models import build_model

def split_by_time(df):
    train = df[df.index < VAL_SPLIT_DATE]
    val = df[(df.index >= VAL_SPLIT_DATE) & (df.index < TEST_SPLIT_DATE)]
    test = df[df.index >= TEST_SPLIT_DATE]
    return train, val, test

def train_main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["datetime"], index_col="datetime")
    target = "load"
    feature_cols = [c for c in df.columns if c != target]

    train_df, val_df, test_df = split_by_time(df)
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    X_train, y_train = make_sequences(train_df, feature_cols, target_col=target, input_window=INPUT_WINDOW, horizon=FORECAST_HORIZON)
    X_val, y_val = make_sequences(val_df, feature_cols, target_col=target, input_window=INPUT_WINDOW, horizon=FORECAST_HORIZON)
    X_test, y_test = make_sequences(test_df, feature_cols, target_col=target, input_window=INPUT_WINDOW, horizon=FORECAST_HORIZON)

    if MODEL_TYPE == "MLP":
        X_train_t = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32)
        X_val_t = torch.tensor(X_val.reshape(X_val.shape, -1), dtype=torch.float32)
        X_test_t = torch.tensor(X_test.reshape(X_test.shape, -1), dtype=torch.float32)
        input_size = X_train_t.shape[1]
    else:
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        input_size = X_train.shape[2]

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(MODEL_TYPE, input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, FORECAST_HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val = np.inf
    best_path = MODELS_DIR / f"{MODEL_TYPE}_best.pt"

    for epoch in range(EPOCHS):
        model.train(); losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        print(f"Epoch {epoch+1}/{EPOCHS}  Train MSE: {np.mean(losses):.4f}  Val MSE: {val_loss:.4f}")

    model.load_state_dict(torch.load(best_path))
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).numpy()

    y_true = y_test
    mae = mean_absolute_error(y_true.flatten(), test_preds.flatten())
    rmse = mean_squared_error(y_true.flatten(), test_preds.flatten(), squared=False)
    mape = np.mean(np.abs((y_true.flatten() - test_preds.flatten()) / np.clip(np.abs(y_true.flatten()), 1e-6, None))) * 100.0

    results = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    print("Test metrics:", results)

    import joblib, json, numpy as np
    joblib.dump(scaler, "artifacts/scaler.joblib")
    np.save("artifacts/feature_cols.npy", np.array(feature_cols, dtype=object))
    with open("artifacts/results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    train_main()
