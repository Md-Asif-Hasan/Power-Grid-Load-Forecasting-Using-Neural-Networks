from pathlib import Path
import pandas as pd
from config import RAW_DIR, DATASET

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def instructions():
    if DATASET == "uci_eld":
        print("Download UCI ElectricityLoadDiagrams20112014")
        print("Option A: huggingface.co/datasets/tulipa762/electricity_load_diagrams")
        print("Option B: UCI zip -> LD2011_2014.txt; save as data/raw/LD2011_2014.txt")
    else:
        print("Set DATASET=uci_eld for load-only pipeline.")

def load_raw():
    if DATASET == "uci_eld":
        path = RAW_DIR / "LD2011_2014.txt"
        df = pd.read_csv(path, delimiter=";", decimal=",")
        dt_col = df.columns[0]
        df = df.rename(columns={dt_col: "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    raise ValueError("Unsupported dataset for load-only pipeline.")

if __name__ == "__main__":
    ensure_dirs()
    instructions()
