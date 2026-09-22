# Bitcoin BTC/USD Analytics Pipeline

This repository contains a complete Jupyter Notebook pipeline for analyzing Bitcoin (BTC/USD) 1-minute OHLCV data. It processes the raw historical data into cleaned minute data, daily features, statistical analyses, machine-learning evaluations, time-series forecasts, anomaly/regime analysis, and an executive dashboard.

## Dataset

Download the Kaggle Bitcoin Historical Data dataset:

<https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data>

The pipeline expects the raw file to be named `btcusd_1-min_data.csv`.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Jupyter or open the notebook in VS Code:

```bash
jupyter notebook Bitcoin_Analytics_Full_Pipeline.ipynb
```

## Running the pipeline

Run the notebook cells from top to bottom. The stages are:

1. Load and audit the raw minute data.
2. Build daily, monthly, and yearly datasets and engineer features.
3. Produce exploratory data analysis charts and summaries.
4. Run statistical tests and GARCH volatility analysis.
5. Detect anomalies and segment market regimes.
6. Evaluate walk-forward machine-learning models.
7. Decompose the time series and benchmark forecasts.
8. Create the correlation heatmap and executive dashboard.

## Outputs

- Parquet: `btc_1min_clean.parquet`, `btc_daily.parquet`, `btc_monthly.parquet`, `btc_yearly.parquet`, `daily_regimes.parquet`
- CSV: `hourly_pattern.csv`
- JSON: `audit_01.json`, `summary_daily.json`, `extra_eda.json`, `stats_04.json`, `anomaly_seg_05.json`, `ml_06.json`, `ml_final_06.json`, `ts_07.json`
- Charts: `c1_log_close.png` through `c17_dashboard.png` (17 PNG files)
