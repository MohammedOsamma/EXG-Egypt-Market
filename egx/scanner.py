import pandas as pd
from egx.agents.technical import TechnicalAgent
from egx.agents.risk import RiskAgent
from egx.master.master import MasterAgent
from egx.portfolio.portfolio import Portfolio
from egx.indicators import add_indicators, indicator_table


def scan_row(sym, name, sector, df, cfg, note=""):
    master = MasterAgent(cfg, TechnicalAgent(cfg), RiskAgent(cfg))
    dec = master.decide(sym, df, Portfolio(cfg))          # assumes no open position
    tech = dec["reports"]["technical"]
    _, score, overall, _ = indicator_table(add_indicators(df))
    c = df["close"]
    age = (pd.Timestamp.today().normalize() - df.index[-1]).days
    return {
        "Symbol": sym, "Name": name, "Sector": sector, "Last date": str(df.index[-1].date()),
        "Close": round(float(c.iloc[-1]), 2),
        "1D %": round(float((c.iloc[-1] / c.iloc[-2] - 1) * 100), 2) if len(c) > 1 else None,
        "1M %": round(float((c.iloc[-1] / c.iloc[-22] - 1) * 100), 2) if len(c) > 22 else None,
        "Trend": tech.get("trend", "n/a"), "RSI": round(tech["rsi"], 1) if "rsi" in tech else None,
        "Tech signal": tech.get("signal", "n/a"), "Indicators": overall, "System": dec["action"],
        "Avg value 20d": int((c * df["volume"]).tail(20).mean()),
        "Status": "Stale" if age > 7 or note else "OK",
    }
