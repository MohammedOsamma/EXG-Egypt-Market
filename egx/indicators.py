"""Indicator columns + a descriptive Buy/Sell/Neutral reading per indicator (causal: no lookahead)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from egx.agents.technical import rsi, atr


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    d["sma20"], d["sma50"], d["sma200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    d["rsi"] = rsi(c, 14)
    d["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    ll, hh = d["low"].rolling(14).min(), d["high"].rolling(14).max()
    d["stoch_k"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_up"], d["bb_lo"] = mid + 2 * sd, mid - 2 * sd
    d["atr"] = atr(d, 14)
    return d


def indicator_table(d: pd.DataFrame):
    """Returns (table, score, overall, counts). Conventional readings; descriptive only,
    NOT the system's trading decision (that comes from Master + Risk)."""
    r = d.iloc[-1]
    px = r["close"]
    rows = []

    def add(name, value, reading):
        rows.append({"Indicator": name, "Value": "n/a" if pd.isna(value) else round(float(value), 2), "Reading": reading})

    def vs(name, ma):
        add(name, ma, "n/a" if pd.isna(ma) else ("Buy" if px > ma else "Sell"))

    vs("Price vs SMA20", r["sma20"])
    vs("Price vs SMA50", r["sma50"])
    vs("Price vs SMA200", r["sma200"])
    cross = r["sma20"] - r["sma50"]
    add("SMA20 vs SMA50", cross, "n/a" if pd.isna(cross) else ("Buy" if cross > 0 else "Sell"))
    x = r["rsi"]
    add("RSI(14)", x, "n/a" if pd.isna(x) else ("Buy" if x < 30 else "Sell" if x > 70 else "Neutral"))
    add("MACD vs signal", r["macd"] - r["macd_signal"], "Buy" if r["macd"] > r["macd_signal"] else "Sell")
    k = r["stoch_k"]
    add("Stochastic %K", k, "n/a" if pd.isna(k) else ("Buy" if k < 20 else "Sell" if k > 80 else "Neutral"))
    if pd.isna(r["bb_lo"]):
        add("Bollinger(20,2)", np.nan, "n/a")
    else:
        add("Bollinger(20,2)", px, "Buy" if px < r["bb_lo"] else "Sell" if px > r["bb_up"] else "Neutral")
    t = pd.DataFrame(rows)
    counts = t["Reading"].value_counts().to_dict()
    b, s = counts.get("Buy", 0), counts.get("Sell", 0)
    score = b - s
    overall = "Buy" if score >= 2 else "Sell" if score <= -2 else "Neutral"
    return t, score, overall, {"Buy": b, "Sell": s, "Neutral": counts.get("Neutral", 0)}
