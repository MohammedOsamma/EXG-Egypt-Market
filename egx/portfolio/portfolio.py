"""Shared virtual portfolio + execution validation. Agents never touch this directly."""
from __future__ import annotations
import math
from dataclasses import dataclass


class OrderRejected(Exception):
    pass


@dataclass
class Position:
    shares: int = 0
    avg_cost: float = 0.0          # per share, includes buy fees
    stop: float | None = None
    target: float | None = None


@dataclass
class Trade:
    date: object
    symbol: str
    side: str
    shares: int
    price: float
    fees: float
    realized_pnl: float = 0.0      # net of fees, sells only


class Portfolio:
    def __init__(self, cfg):
        self.initial_capital = float(cfg["capital"])
        self.cash = self.initial_capital
        self.fee_cfg = cfg["fees"]
        self.lot = int(cfg.get("lot_size", 1))
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []

    def fees(self, notional: float) -> float:
        if notional <= 0:
            return 0.0
        f = self.fee_cfg
        return max(f["min_commission"], notional * f["commission_pct"]) + notional * f["exchange_pct"]

    def _check(self, shares, price):
        if not isinstance(shares, int) or shares <= 0:
            raise OrderRejected(f"invalid quantity {shares!r}")
        if shares % self.lot:
            raise OrderRejected(f"quantity {shares} not a multiple of lot size {self.lot}")
        if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            raise OrderRejected(f"invalid price {price!r}")

    def buy(self, date, symbol, shares, price, stop=None, target=None) -> Trade:
        self._check(shares, price)
        notional = shares * price
        fee = self.fees(notional)
        if notional + fee > self.cash + 1e-9:
            raise OrderRejected(f"insufficient cash: need {notional + fee:.2f}, have {self.cash:.2f}")
        self.cash -= notional + fee
        pos = self.positions.setdefault(symbol, Position())
        total_cost = pos.avg_cost * pos.shares + notional + fee
        pos.shares += shares
        pos.avg_cost = total_cost / pos.shares
        pos.stop, pos.target = stop, target
        t = Trade(date, symbol, "BUY", shares, price, fee)
        self.trades.append(t)
        return t

    def sell(self, date, symbol, shares, price) -> Trade:
        self._check(shares, price)
        pos = self.positions.get(symbol)
        if pos is None or pos.shares < shares:
            raise OrderRejected(f"cannot sell {shares} {symbol}: own {pos.shares if pos else 0}")
        notional = shares * price
        fee = self.fees(notional)
        pnl = notional - fee - shares * pos.avg_cost
        self.cash += notional - fee
        pos.shares -= shares
        if pos.shares == 0:
            del self.positions[symbol]
        t = Trade(date, symbol, "SELL", shares, price, fee, pnl)
        self.trades.append(t)
        return t

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            if sym not in prices or not math.isfinite(prices[sym]):
                raise ValueError(f"missing price for held symbol {sym}")
            total += pos.shares * prices[sym]
        return total

    def closed_trades(self):
        return [t for t in self.trades if t.side == "SELL"]
