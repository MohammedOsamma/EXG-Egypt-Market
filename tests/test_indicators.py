import pandas as pd
from egx.config import load_config
from egx.data.synthetic import make_synthetic
from egx.indicators import add_indicators, indicator_table
from egx.scanner import scan_row

cfg = load_config()


def test_indicators_causal():
    df = make_synthetic()
    a = add_indicators(df.iloc[:300]).iloc[-1]
    df2 = df.copy()
    df2.iloc[300:, :4] *= 7
    b = add_indicators(df2.iloc[:300]).iloc[-1]
    assert a.equals(b)


def test_indicator_table_and_scan():
    df = make_synthetic()
    t, score, overall, counts = indicator_table(add_indicators(df))
    assert overall in ("Buy", "Sell", "Neutral") and sum(counts.values()) <= len(t)
    short = indicator_table(add_indicators(df.iloc[:30]))      # not enough history: must not crash
    assert short[2] in ("Buy", "Sell", "Neutral")
    row = scan_row("X", "Test", "S", df, cfg)
    assert row["System"] in ("BUY", "SELL", "HOLD", "WAIT")


def test_stock_list():
    from egx.data.stocks import load_stock_list
    s = load_stock_list()
    assert s["symbol"].is_unique and {"symbol", "name", "ticker", "sector"} <= set(s.columns)
