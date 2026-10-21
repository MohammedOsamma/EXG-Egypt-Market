"""Technical Agent: deterministic, uses only rows present in df (caller must pass data <= decision date)."""
from __future__ import annotations
import pandas as pd


def rsi(close, n):
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd_hist(close, fast=12, slow=26, sig=9):
    m = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return m - m.ewm(span=sig, adjust=False).mean()


class TechnicalAgent:
    def __init__(self, cfg):
        self.c = cfg["technical"]

    def analyze(self, df: pd.DataFrame) -> dict:
        c = self.c
        if len(df) < c["min_bars"]:
            return {"agent": "technical", "status": "insufficient_data", "bars": len(df)}
        close = df["close"]
        sma_f = close.rolling(c["sma_fast"]).mean().iloc[-1]
        sma_s = close.rolling(c["sma_slow"]).mean().iloc[-1]
        r = float(rsi(close, c["rsi_period"]).iloc[-1])
        mh = float(macd_hist(close).iloc[-1])
        a = float(atr(df, c["atr_period"]).iloc[-1])
        lb = c["sr_lookback"]
        prior = df.iloc[-lb - 1:-1]
        resistance, support = float(prior["high"].max()), float(prior["low"].min())
        px = float(close.iloc[-1])

        bull = [px > sma_s, sma_f > sma_s, mh > 0, c["rsi_buy_min"] <= r <= c["rsi_buy_max"]]
        bear = [px < sma_s, sma_f < sma_s, mh < 0, r < 50]
        if all(bull):
            signal, conf = "bullish", 0.75 + 0.25 * (px > resistance)
        elif px < sma_s and mh < 0:
            signal, conf = "bearish", sum(bear) / 4
        else:
            signal, conf = "neutral", sum(bull) / 4
        near_res = resistance * (1 - c["resistance_buffer"]) <= px <= resistance
        return {
            "agent": "technical", "status": "ok", "date": str(df.index[-1].date()),
            "signal": signal, "confidence": round(float(conf), 2),
            "trend": "up" if sma_f > sma_s else "down",
            "close": px, "sma_fast": float(sma_f), "sma_slow": float(sma_s),
            "rsi": r, "macd_hist": mh, "atr": a,
            "support": support, "resistance": resistance,
            "near_resistance": bool(near_res), "breakout": bool(px > resistance),
            "invalidation": f"close below SMA{c['sma_slow']} ({sma_s:.2f}) or below support {support:.2f}",
            "note": "daily-close signal; a possible setup, not a confirmed forecast",
        }
