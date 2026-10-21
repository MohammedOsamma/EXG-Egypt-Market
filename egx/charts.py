"""Plotly figure builders."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GREEN, RED, GREY = "#26a69a", "#ef5350", "#9e9e9e"


def _breaks(idx):
    full = pd.date_range(idx[0], idx[-1], freq="D")
    return [x.strftime("%Y-%m-%d") for x in full.difference(idx)]


def price_chart(d: pd.DataFrame, trades: pd.DataFrame | None = None, bars: int | None = 250) -> go.Figure:
    """d must come from add_indicators(). trades: columns date, side, price."""
    if bars:
        d = d.iloc[-bars:]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.42, 0.11, 0.14, 0.17, 0.16])
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="Price", increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
    for col, color in [("sma20", "#ffa726"), ("sma50", "#42a5f5"), ("sma200", "#ab47bc")]:
        if d[col].notna().any():
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col.upper(), line=dict(color=color, width=1.2)), row=1, col=1)
    for col in ("bb_up", "bb_lo"):
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name="Bollinger", legendgroup="bb", showlegend=(col == "bb_up"),
                                 line=dict(color=GREY, width=0.8, dash="dot")), row=1, col=1)
    if trades is not None and len(trades):
        t = trades.copy()
        t["date"] = pd.to_datetime(t["date"])
        t = t[(t["date"] >= d.index[0]) & (t["date"] <= d.index[-1])]
        for side, sym, color in (("BUY", "triangle-up", "#00c853"), ("SELL", "triangle-down", "#d50000")):
            s = t[t["side"] == side]
            if len(s):
                fig.add_trace(go.Scatter(x=s["date"], y=s["price"], mode="markers", name=f"System {side}",
                                         marker=dict(symbol=sym, size=13, color=color, line=dict(color="white", width=1))),
                              row=1, col=1)
    up = d["close"] >= d["open"]
    fig.add_trace(go.Bar(x=d.index, y=d["volume"], name="Volume", marker_color=np.where(up, GREEN, RED), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI(14)", line=dict(color="#7e57c2")), row=3, col=1)
    for y in (30, 70):
        fig.add_hline(y=y, line_dash="dot", line_color=GREY, row=3, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["macd_hist"], name="MACD hist", marker_color=np.where(d["macd_hist"] >= 0, GREEN, RED), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["macd"], name="MACD", line=dict(color="#42a5f5", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["macd_signal"], name="Signal", line=dict(color="#ffa726", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["stoch_k"], name="Stoch %K", line=dict(color="#26c6da", width=1.2)), row=5, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["stoch_d"], name="Stoch %D", line=dict(color="#ff7043", width=1.2)), row=5, col=1)
    for y in (20, 80):
        fig.add_hline(y=y, line_dash="dot", line_color=GREY, row=5, col=1)
    for r, label in ((2, "Volume"), (3, "RSI"), (4, "MACD"), (5, "Stochastic")):
        fig.update_yaxes(title_text=label, row=r, col=1)
    fig.update_xaxes(rangebreaks=[dict(values=_breaks(d.index))])
    fig.update_layout(height=950, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.03))
    return fig


def equity_fig(system: pd.Series, baseline: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=system.index, y=system, name="System", line=dict(color="#42a5f5", width=2)))
    if baseline is not None:
        fig.add_trace(go.Scatter(x=baseline.index, y=baseline, name="Buy & Hold", line=dict(color=GREY, width=2)))
    fig.update_layout(title="Portfolio value (EGP)", height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


def drawdown_fig(system: pd.Series, baseline: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=system.index, y=(system / system.cummax() - 1) * 100, name="System", fill="tozeroy",
                             line=dict(color=RED, width=1)))
    if baseline is not None:
        fig.add_trace(go.Scatter(x=baseline.index, y=(baseline / baseline.cummax() - 1) * 100, name="Buy & Hold",
                                 line=dict(color=GREY, width=1.5)))
    fig.update_layout(title="Drawdown (%)", height=300, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


def trades_fig(trades: pd.DataFrame) -> go.Figure:
    s = trades[trades["side"] == "SELL"]
    fig = go.Figure(go.Bar(x=pd.to_datetime(s["date"]), y=s["realized_pnl"],
                           marker_color=np.where(s["realized_pnl"] >= 0, GREEN, RED)))
    fig.update_layout(title="Profit / loss per closed trade (EGP, after fees)", height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def gauge_fig(score: int, overall: str, n: int = 8) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={"text": f"Indicator summary: {overall}"},
        gauge={"axis": {"range": [-n, n]}, "bar": {"color": "#455a64"},
               "steps": [{"range": [-n, -2], "color": "#ef9a9a"}, {"range": [-2, 2], "color": "#e0e0e0"},
                         {"range": [2, n], "color": "#a5d6a7"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig
