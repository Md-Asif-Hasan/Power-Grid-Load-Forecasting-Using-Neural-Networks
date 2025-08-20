import pandas as pd
from config import DATASET, FREQUENCY, PROCESSED_DIR
from data_download import load_raw

def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def preprocess():
    df = load_raw()
    if DATASET == "uci_eld":
        df = df.sort_values("datetime").set_index("datetime")
        value_cols = [c for c in df.columns if c != "datetime"]
        df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
        # Aggregate client loads to a system-level series
        s = df[value_cols].sum(axis=1)
        # Resample from 15-min to hourly mean
        hourly = s.resample(FREQUENCY).mean().to_frame(name="load")
        # Forward/back fill any occasional gaps
        hourly = hourly.ffill().bfill()
        hourly.to_csv(PROCESSED_DIR / "prepared.csv")
        return hourly
    raise ValueError("Unsupported dataset.")

if __name__ == "__main__":
    ensure_dirs()
    preprocess()
