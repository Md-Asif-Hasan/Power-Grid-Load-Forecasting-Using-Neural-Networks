from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import joblib
from config import API_HOST, API_PORT, MODEL_TYPE, FORECAST_HORIZON, INPUT_WINDOW
from models import build_model

app = Flask(__name__)

scaler = joblib.load("artifacts/scaler.joblib")
feature_cols = np.load("artifacts/feature_cols.npy", allow_pickle=True).tolist()
model = build_model(MODEL_TYPE, len(feature_cols), 128, 2, 0.2, FORECAST_HORIZON)
model.load_state_dict(torch.load(f"models/{MODEL_TYPE}_best.pt"))
model.eval()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/forecast", methods=["POST"])
def forecast():
    """
    Expects last INPUT_WINDOW rows of engineered features (load-only features).
    {
      "data": [
        {"datetime":"YYYY-MM-DD HH:MM:SS", "hour":0,...,"load_lag_1":..., ...},
        ...
      ]
    }
    """
    payload = request.get_json()
    df = pd.DataFrame(payload["data"]).set_index("datetime")
    X = scaler.transform(df[feature_cols].values)
    if MODEL_TYPE == "MLP":
        X = X.reshape(1, -1)
    else:
        X = X.reshape(1, -1, len(feature_cols))
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        yhat = model(X_t).numpy().flatten().tolist()
    return jsonify({"horizon_hours": FORECAST_HORIZON, "forecast": yhat})

if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT, debug=False)
