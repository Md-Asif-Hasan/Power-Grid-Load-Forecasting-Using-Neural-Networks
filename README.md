# Power Grid Load Forecasting (Load-Only)

- Target: Next-24h hourly system load, using only historical load.
- Frequency: Hourly (resampled from 15-min for UCI ELDiagrams).

Data (manual download):
- UCI ElectricityLoadDiagrams20112014 (LD2011_2014.txt). Save to data/raw/LD2011_2014.txt.

Run:
- python data_download.py   # follow instructions to place file
- python preprocess.py
- python features.py
- python train.py
- python forecast.py
- python app.py

Notes:
- No weather, holidays, or external regressors used.
- Features: time encodings (hour, dayofweek, month, weekend), load lags (1–168h), rolling stats.
- Models: MLP baseline or seq2seq RNN/LSTM/GRU with direct multi-step output.
