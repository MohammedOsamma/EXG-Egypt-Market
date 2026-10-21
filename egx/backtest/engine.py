"""Event loop. Decision at close of day t (data <= t) -> fill at open of t+1. No lookahead."""
from __future__ import annotations
import pandas as pd
from egx.portfolio.portfolio import Portfolio, OrderRejected
from egx.backtest.metrics import compute_metrics


def run_backtest(symbol, df, cfg, master):
    pf = Portfolio(cfg)
    slip = cfg["slippage_pct"]
    pending, equity, decisions, rejected = None, [], [], []

    for i in range(len(df)):
        date, row = df.index[i], df.iloc[i]

        # 1) fill pending order at today's open
        if pending:
            try:
                if pending["action"] == "BUY":
                    pf.buy(date, symbol, pending["shares"], float(row["open"]) * (1 + slip),
                           stop=pending["stop"], target=pending["target"])
                elif pending["action"] == "SELL":
                    pf.sell(date, symbol, pending["shares"], float(row["open"]) * (1 - slip))
            except OrderRejected as e:
                rejected.append({"date": str(date.date()), "order": pending["action"], "reason": str(e)})
            pending = None

        # 2) intraday stop check (gap-aware)
        pos = pf.positions.get(symbol)
        if pos and pos.stop is not None and row["low"] <= pos.stop:
            fill = min(float(row["open"]), pos.stop) * (1 - slip)
            pf.sell(date, symbol, pos.shares, fill)

        # 3) mark to market
        equity.append(pf.equity({symbol: float(row["close"])}))

        # 4) decide for next session
        if i < len(df) - 1:
            d = master.decide(symbol, df.iloc[: i + 1], pf)
            decisions.append(d)
            if d["action"] in ("BUY", "SELL"):
                pending = d

    eq = pd.Series(equity, index=df.index)
    return {"portfolio": pf, "equity": eq, "decisions": decisions, "rejected": rejected,
            "metrics": compute_metrics(eq, pf.closed_trades(), pf.initial_capital)}


def buy_and_hold(symbol, df, cfg, start_idx):
    """Baseline: all-in at open of bar start_idx+1 (same start as strategy), same fees/slippage."""
    pf = Portfolio(cfg)
    j = start_idx + 1
    px = float(df["open"].iloc[j]) * (1 + cfg["slippage_pct"])
    n = int(pf.cash // px)
    while n > 0 and n * px + pf.fees(n * px) > pf.cash:
        n -= 1
    if n > 0:
        pf.buy(df.index[j], symbol, n, px)
    eq = pd.Series([pf.equity({symbol: float(c)}) for c in df["close"].iloc[j:]], index=df.index[j:])
    return {"equity": eq, "metrics": compute_metrics(eq, [], pf.initial_capital)}
