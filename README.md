# EGX AI Paper Trading (Phases 1-4, 7, 8)

    pip install -r requirements.txt
    pytest
    python run_backtest.py --csv data/RAYA.csv --symbol RAYA --start 2022-01-01 --end 2024-12-31
    python run_backtest.py --demo        # SYNTHETIC data, smoke test only

CSV columns: `date,open,high,low,close,volume`.
Outputs: `reports/decisions.csv` (all agent reports per day), `trades.csv`, `equity.csv`.

Rules: decisions use data up to day t close; fills at t+1 open with slippage + fees.
Edit fees in `config/settings.yaml` (placeholders) before trusting any result.
Fundamental/News agents plug into `MasterAgent(extra_agents=[...])`:
`analyze(symbol, df) -> {"agent":..., "blocking":bool, "exit":bool, "reason":str}`.

## GUI
Double-click `run_gui.bat` (or `py -m streamlit run app.py`). Opens in your browser.
Sidebar: data (download/upload/saved), period + train/test split, costs and risk settings, Run.
