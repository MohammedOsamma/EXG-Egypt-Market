import pytest
from egx.config import load_config
from egx.portfolio.portfolio import Portfolio, OrderRejected
from egx.data.synthetic import make_synthetic
from egx.agents.technical import TechnicalAgent
from egx.agents.risk import RiskAgent
from egx.master.master import MasterAgent
from egx.backtest.engine import run_backtest, buy_and_hold

cfg = load_config()


def test_no_overspend():
    pf = Portfolio(cfg)
    with pytest.raises(OrderRejected):
        pf.buy("d", "X", 1000, 10.0)
    assert pf.cash == 1000


def test_no_short_selling():
    pf = Portfolio(cfg)
    with pytest.raises(OrderRejected):
        pf.sell("d", "X", 1, 10.0)
    pf.buy("d", "X", 10, 10.0)
    with pytest.raises(OrderRejected):
        pf.sell("d", "X", 11, 10.0)


def test_invalid_inputs_rejected():
    pf = Portfolio(cfg)
    for sh, px in [(0, 10), (-1, 10), (1.5, 10), (1, 0), (1, float("nan"))]:
        with pytest.raises(OrderRejected):
            pf.buy("d", "X", sh, px)


def test_fees_and_pnl_roundtrip():
    pf = Portfolio(cfg)
    pf.buy("d", "X", 10, 10.0)
    t = pf.sell("d", "X", 10, 10.0)
    assert t.realized_pnl < 0            # flat price loses the fees
    assert pf.cash == pytest.approx(1000 + t.realized_pnl)


def test_equity_requires_price():
    pf = Portfolio(cfg)
    pf.buy("d", "X", 10, 10.0)
    with pytest.raises(ValueError):
        pf.equity({})


def test_technical_no_lookahead():
    df = make_synthetic()
    t = TechnicalAgent(cfg)
    a = t.analyze(df.iloc[:300])
    df2 = df.copy()
    df2.iloc[300:, :4] *= 5               # corrupt the future
    b = t.analyze(df2.iloc[:300])
    assert a == b


def test_risk_veto_overrides_bullish():
    df = make_synthetic()
    class Bull(TechnicalAgent):
        def analyze(self, d):
            r = super().analyze(d)
            r.update(signal="bullish", near_resistance=False)
            return r
    cfg2 = load_config()
    cfg2["risk"]["min_avg_daily_value"] = 1e12      # force illiquidity veto
    m = MasterAgent(cfg2, Bull(cfg2), RiskAgent(cfg2))
    d = m.decide("X", df.iloc[:200], Portfolio(cfg2))
    assert d["action"] == "WAIT" and any("illiquid" in r for r in d["reasons"])


def test_blocking_news_prevents_buy():
    df = make_synthetic()
    class Bull(TechnicalAgent):
        def analyze(self, d):
            r = super().analyze(d)
            r.update(signal="bullish", near_resistance=False)
            return r
    class News:
        def analyze(self, s, d):
            return {"agent": "news", "blocking": True, "reason": "unverified rumor"}
    m = MasterAgent(cfg, Bull(cfg), RiskAgent(cfg), [News()])
    assert m.decide("X", df.iloc[:200], Portfolio(cfg))["action"] == "WAIT"


def test_backtest_deterministic_and_consistent():
    df = make_synthetic()
    def run():
        m = MasterAgent(cfg, TechnicalAgent(cfg), RiskAgent(cfg))
        return run_backtest("X", df, cfg, m)
    a, b = run(), run()
    assert a["metrics"] == b["metrics"]
    assert len(a["equity"]) == len(df)
    assert a["portfolio"].cash >= 0
    base = buy_and_hold("X", df, cfg, cfg["technical"]["min_bars"])
    assert "return_pct" in base["metrics"]
