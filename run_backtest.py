import argparse, json
import pandas as pd
from egx.config import load_config
from egx.data.loader import load_csv
from egx.data.synthetic import make_synthetic
from egx.agents.technical import TechnicalAgent
from egx.agents.risk import RiskAgent
from egx.master.master import MasterAgent
from egx.backtest.engine import run_backtest, buy_and_hold

p = argparse.ArgumentParser()
p.add_argument("--csv"); p.add_argument("--symbol", default="RAYA")
p.add_argument("--start"); p.add_argument("--end")
p.add_argument("--demo", action="store_true", help="SYNTHETIC data, smoke test only")
p.add_argument("--config")
a = p.parse_args()

cfg = load_config(a.config)
df = make_synthetic() if a.demo else load_csv(a.csv)
if a.demo:
    print("WARNING: synthetic data, results are meaningless\n")
df = df.loc[a.start:a.end]

master = MasterAgent(cfg, TechnicalAgent(cfg), RiskAgent(cfg))
res = run_backtest(a.symbol, df, cfg, master)
base = buy_and_hold(a.symbol, df, cfg, cfg["technical"]["min_bars"])

print("System  :", res["metrics"])
print("Buy&Hold:", base["metrics"])
print("Rejected orders:", len(res["rejected"]))

pd.DataFrame([{**{k: v for k, v in d.items() if k != "reports"}, "reasons": "; ".join(d["reasons"]),
               "reports": json.dumps(d["reports"], default=str)} for d in res["decisions"]]).to_csv("reports/decisions.csv", index=False)
pd.DataFrame([t.__dict__ for t in res["portfolio"].trades]).to_csv("reports/trades.csv", index=False)
res["equity"].to_csv("reports/equity.csv", header=["equity"])
