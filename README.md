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

## Expected folders

The notebook currently uses the following working layout:

```text
archive/
|-- btcusd_1-min_data.csv
`-- out/
```

Create the output directory before running the notebook. Update the absolute paths in the notebook if your dataset is stored elsewhere.

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

Results are written to `out/`, including cleaned and aggregated Parquet files, JSON summaries, and charts `c1` through `c17`. The notebook's final markdown also references `REPORT.md` and `PraveenKumarC_ProjectReport.docx` as optional report deliverables generated from these outputs.

## Notes

- The source CSV is large, so sufficient disk space and memory are required.
- This project is for historical analysis and research. It is not financial advice.
- The notebook is currently unexecuted; run it in order to generate the outputs.