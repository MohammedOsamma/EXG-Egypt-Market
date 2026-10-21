"""EGX stock list, price cache, company info (Yahoo via yfinance; coverage for EGX can be incomplete)."""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
from egx.data.loader import load_csv, load_yfinance

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
STOCK_LIST = ROOT / "config" / "egx_stocks.csv"


def load_stock_list() -> pd.DataFrame:
    df = pd.read_csv(STOCK_LIST)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    return df.drop_duplicates("symbol").reset_index(drop=True)


def ensure_prices(symbol: str, ticker: str, max_age_hours: float = 12, start: str = "2020-01-01"):
    """Returns (df, note). note is '' when data is fresh. Never invents data."""
    DATA.mkdir(exist_ok=True)
    p = DATA / f"{symbol}.csv"
    if p.exists() and time.time() - p.stat().st_mtime < max_age_hours * 3600:
        return load_csv(str(p)), ""
    try:
        d = load_yfinance(ticker, start)
        d.to_csv(p, index_label="date")
        return d, ""
    except Exception as e:
        if p.exists():
            return load_csv(str(p)), f"Download failed ({e}); showing older saved data."
        raise


def get_info(ticker: str):
    import yfinance as yf
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        div = t.dividends
        if getattr(div.index, "tz", None) is not None:
            div.index = div.index.tz_localize(None)
    except Exception:
        div = pd.Series(dtype=float)
    return info, div
