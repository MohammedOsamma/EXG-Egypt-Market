"""Delayed live quotes for EGX from TradingView's public scanner endpoint (UNOFFICIAL, undocumented).
- Free TradingView EGX data is typically delayed ~15 min; the 'update_mode' field reports it when available.
- TradingView's terms restrict automated access. Personal, low-volume use only; it may break or be blocked.
- Never used for trading signals; display only."""
from __future__ import annotations
import pandas as pd
import requests

URL = "https://scanner.tradingview.com/egypt/scan"
BASE_COLS = ["name", "description", "close", "open", "high", "low", "volume", "change"]
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}


def _payload(cols):
    return {"filter": [], "options": {"lang": "en"}, "markets": ["egypt"],
            "symbols": {"query": {"types": []}, "tickers": []}, "columns": cols,
            "sort": {"sortBy": "volume", "sortOrder": "desc"}, "range": [0, 600]}


def fetch_quotes(timeout: float = 15) -> pd.DataFrame:
    cols = BASE_COLS + ["update_mode"]
    r = requests.post(URL, json=_payload(cols), headers=HEADERS, timeout=timeout)
    if r.status_code == 400:                       # column not accepted: retry without it
        cols = BASE_COLS
        r = requests.post(URL, json=_payload(cols), headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    rows = []
    for it in r.json().get("data", []):
        exch, _, sym = it["s"].partition(":")
        if exch != "EGX":
            continue
        row = dict(zip(cols, it["d"]))
        if row.get("close") is None or not row["close"] > 0:
            continue
        row["symbol"] = sym
        rows.append(row)
    if not rows:
        raise ValueError("TradingView returned no EGX quotes")
    df = pd.DataFrame(rows)
    if "update_mode" not in df:
        df["update_mode"] = None
    df["fetched_at"] = pd.Timestamp.now(tz="Africa/Cairo")
    return df


def delay_label(mode) -> str:
    if mode is None or (isinstance(mode, float) and pd.isna(mode)):
        return "delay not reported (EGX is typically 15 min delayed on free TradingView)"
    m = str(mode)
    if m.startswith("delayed_streaming_"):
        try:
            return f"delayed {int(m.rsplit('_', 1)[1]) // 60} min"
        except ValueError:
            pass
    return "real-time" if m == "streaming" else m
