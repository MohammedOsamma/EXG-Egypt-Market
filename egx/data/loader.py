"""Load + validate OHLCV. Never fills missing data with invented values."""
from __future__ import annotations
import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df[COLS].sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate dates")
    if df.isna().any().any():
        raise ValueError("NaN values present; fix source data, do not fill silently")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive prices")
    if (df["high"] < df["low"]).any():
        raise ValueError("high < low")
    if (df["volume"] < 0).any():
        raise ValueError("negative volume")
    return df


def load_csv(path: str) -> pd.DataFrame:
    """CSV columns: date,open,high,low,close,volume"""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return validate(df.set_index("date"))


def load_yfinance(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """e.g. 'RAYA.CA'. Coverage/quality for EGX must be verified manually.
    auto_adjust=True adjusts for dividends/splits (historical, not live)."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"no data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return validate(df)
