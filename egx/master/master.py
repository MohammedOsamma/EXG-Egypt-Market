"""Master Agent: weighs evidence (not vote counting). Risk veto and blocking news always win."""
from __future__ import annotations


class MasterAgent:
    def __init__(self, cfg, technical, risk, extra_agents=None):
        """extra_agents: objects with analyze(symbol, df) -> dict; optional keys:
        'blocking' (bool, blocks new entries), 'exit' (bool, forces exit), 'reason' (str)."""
        self.tech, self.risk, self.extra = technical, risk, extra_agents or []

    def decide(self, symbol, df, pf) -> dict:
        date = str(df.index[-1].date())
        close = float(df["close"].iloc[-1])
        tech = self.tech.analyze(df)
        reports = {"technical": tech}
        out = {"date": date, "symbol": symbol, "action": "WAIT", "shares": 0,
               "stop": None, "target": None, "reasons": [], "invalidation": None, "reports": reports}
        pos = pf.positions.get(symbol)
        held = pos is not None and pos.shares > 0

        if tech["status"] != "ok":
            out["action"] = "HOLD" if held else "WAIT"
            out["reasons"].append("insufficient data for technical analysis")
            return out

        extras = []
        for a in self.extra:
            r = a.analyze(symbol, df)
            reports[r.get("agent", a.__class__.__name__)] = r
            extras.append(r)
        out["invalidation"] = tech["invalidation"]

        if held:
            if pos.stop is not None and close <= pos.stop:
                out.update(action="SELL", shares=pos.shares)
                out["reasons"].append(f"close {close:.2f} <= stop {pos.stop:.2f}")
            elif any(e.get("exit") for e in extras):
                out.update(action="SELL", shares=pos.shares)
                out["reasons"] += [e.get("reason", "exit flag") for e in extras if e.get("exit")]
            elif tech["signal"] == "bearish":
                out.update(action="SELL", shares=pos.shares)
                out["reasons"].append("technical trend turned bearish")
            else:
                out["action"] = "HOLD"
                out["reasons"].append("no exit condition met")
            return out

        blockers = [e.get("reason", "blocked") for e in extras if e.get("blocking")]
        if tech["signal"] != "bullish":
            out["reasons"].append(f"technical signal is {tech['signal']}")
        elif tech["near_resistance"]:
            out["reasons"].append("price just below resistance; wait for breakout or pullback")
        elif blockers:
            out["reasons"] += blockers
        else:
            r = self.risk.evaluate_entry(df, tech, pf.cash, pf.cash, pf.fees)
            reports["risk"] = r
            if r["approved"]:
                out.update(action="BUY", shares=r["shares"], stop=r["stop"], target=r["target"])
                out["reasons"].append(f"bullish setup, risk approved (R/R {r['risk_reward']})")
            else:
                out["reasons"] += ["risk veto: " + v for v in r["veto_reasons"]]
        return out
