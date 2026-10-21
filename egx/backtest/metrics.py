import pandas as pd


def compute_metrics(equity: pd.Series, closed_trades: list, initial: float) -> dict:
    pnls = [t.realized_pnl for t in closed_trades]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p <= 0]
    dd = (equity / equity.cummax() - 1).min() * 100
    return {
        "return_pct": round(float((equity.iloc[-1] - initial) / initial * 100), 2),
        "max_drawdown_pct": round(float(dd), 2),
        "closed_trades": len(pnls),
        "win_rate_pct": round(len(wins) / len(pnls) * 100, 1) if pnls else None,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "net_pnl": round(sum(pnls), 2),
    }
