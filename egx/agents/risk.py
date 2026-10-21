"""Risk Agent: sizes trades and can veto regardless of other agents."""
from __future__ import annotations
import math
import pandas as pd


class RiskAgent:
    def __init__(self, cfg):
        self.c = cfg["risk"]

    def evaluate_entry(self, df: pd.DataFrame, tech: dict, equity: float, cash: float, fees_fn) -> dict:
        c = self.c
        px, a = tech["close"], tech["atr"]
        reasons = []
        avg_vol = float(df["volume"].tail(20).mean())
        avg_val = float((df["close"] * df["volume"]).tail(20).mean())
        stop = px - c["atr_stop_mult"] * a
        target = tech["resistance"] if tech["resistance"] > px * 1.01 else px + 3 * a
        per_share_risk = px - stop
        rr = (target - px) / per_share_risk if per_share_risk > 0 else 0.0
        eff_px = px * (1 + c["size_buffer_pct"])

        if avg_val < c["min_avg_daily_value"]:
            reasons.append(f"illiquid: avg daily value {avg_val:,.0f} < {c['min_avg_daily_value']:,.0f}")
        if rr < c["min_rr"]:
            reasons.append(f"risk/reward {rr:.2f} < {c['min_rr']}")

        shares = min(
            math.floor(equity * c["risk_per_trade_pct"] / per_share_risk) if per_share_risk > 0 else 0,
            math.floor(equity * c["max_position_pct"] / eff_px),
            math.floor(avg_vol * c["max_volume_participation"]),
        )
        while shares > 0 and shares * eff_px + fees_fn(shares * eff_px) > cash:
            shares -= 1
        if shares < 1:
            reasons.append("position size < 1 share under risk/cash/liquidity limits")
        else:
            rt_fees = 2 * fees_fn(shares * px)
            reward = (target - px) * shares
            if reward < c["min_edge_over_fees"] * rt_fees:
                reasons.append(f"expected reward {reward:.2f} < {c['min_edge_over_fees']}x round-trip fees {rt_fees:.2f}")
        return {
            "agent": "risk", "approved": not reasons, "veto_reasons": reasons,
            "shares": max(shares, 0), "entry_ref": px, "stop": stop, "target": target,
            "risk_reward": round(rr, 2), "avg_daily_value": avg_val,
        }
