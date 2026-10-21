"""Plotly figure builders. price_chart + fig_html give a pan/zoom chart whose price axes auto-fit."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GREEN, RED, GREY = "#26a69a", "#ef5350", "#9e9e9e"
CFG = {"scrollZoom": True, "displaylogo": False, "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"]}
FIXED_AXES = ("y3", "y5")        # RSI and Stochastic keep a fixed 0-100 scale


def _breaks(idx):
    full = pd.date_range(idx[0], idx[-1], freq="D")
    return [x.strftime("%Y-%m-%d") for x in full.difference(idx)]


def _L(s):
    return [None if pd.isna(v) else float(v) for v in s]


def price_chart(d: pd.DataFrame, trades: pd.DataFrame | None = None, bars: int | None = 250,
                live: dict | None = None, interactive: bool = True) -> go.Figure:
    """d from add_indicators(). trades: date, side, price. live: {'price': float, 'label': str}.
    interactive=True keeps ALL history (pan/zoom anywhere) and opens at the last `bars`;
    interactive=False slices to the last `bars`."""
    full_idx = d.index
    if bars and not interactive:
        d = d.iloc[-bars:]
    X = list(d.index.strftime("%Y-%m-%d"))
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.42, 0.11, 0.14, 0.17, 0.16])
    fig.add_trace(go.Candlestick(x=X, open=_L(d["open"]), high=_L(d["high"]), low=_L(d["low"]), close=_L(d["close"]),
                                 name="Price", increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
    for col, color in [("sma20", "#ffa726"), ("sma50", "#42a5f5"), ("sma200", "#ab47bc")]:
        if d[col].notna().any():
            fig.add_trace(go.Scatter(x=X, y=_L(d[col]), name=col.upper(), line=dict(color=color, width=1.2)), row=1, col=1)
    for col in ("bb_up", "bb_lo"):
        fig.add_trace(go.Scatter(x=X, y=_L(d[col]), name="Bollinger", legendgroup="bb", showlegend=(col == "bb_up"),
                                 line=dict(color=GREY, width=0.8, dash="dot")), row=1, col=1)
    if trades is not None and len(trades):
        t = trades.copy()
        t["date"] = pd.to_datetime(t["date"])
        for side, sym, color in (("BUY", "triangle-up", "#00c853"), ("SELL", "triangle-down", "#d50000")):
            s = t[t["side"] == side]
            if len(s):
                fig.add_trace(go.Scatter(x=list(s["date"].dt.strftime("%Y-%m-%d")), y=_L(s["price"]), mode="markers",
                                         name=f"System {side}",
                                         marker=dict(symbol=sym, size=13, color=color, line=dict(color="white", width=1))),
                              row=1, col=1)
    if live:
        fig.add_hline(y=live["price"], line_dash="dash", line_color="#fb8c00", row=1, col=1,
                      annotation_text=live["label"], annotation_position="top left")
    up = (d["close"] >= d["open"]).tolist()
    fig.add_trace(go.Bar(x=X, y=_L(d["volume"]), name="Volume", marker_color=[GREEN if u else RED for u in up],
                         showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=X, y=_L(d["rsi"]), name="RSI(14)", line=dict(color="#7e57c2")), row=3, col=1)
    for y in (30, 70):
        fig.add_hline(y=y, line_dash="dot", line_color=GREY, row=3, col=1)
    fig.add_trace(go.Bar(x=X, y=_L(d["macd_hist"]), name="MACD hist",
                         marker_color=[GREEN if v >= 0 else RED for v in d["macd_hist"]], showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=X, y=_L(d["macd"]), name="MACD", line=dict(color="#42a5f5", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=X, y=_L(d["macd_signal"]), name="Signal", line=dict(color="#ffa726", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=X, y=_L(d["stoch_k"]), name="Stoch %K", line=dict(color="#26c6da", width=1.2)), row=5, col=1)
    fig.add_trace(go.Scatter(x=X, y=_L(d["stoch_d"]), name="Stoch %D", line=dict(color="#ff7043", width=1.2)), row=5, col=1)
    for y in (20, 80):
        fig.add_hline(y=y, line_dash="dot", line_color=GREY, row=5, col=1)
    for r, label in ((2, "Volume"), (3, "RSI"), (4, "MACD"), (5, "Stochastic")):
        fig.update_yaxes(title_text=label, row=r, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=5, col=1)

    fig.update_xaxes(rangebreaks=[dict(values=_breaks(full_idx if interactive else d.index))],
                     showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
    if interactive:
        fig.update_xaxes(rangeselector=dict(buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All")]), row=1, col=1)
        if bars and len(full_idx) > bars:
            x0 = full_idx[-bars].strftime("%Y-%m-%d")
            x1 = (full_idx[-1] + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            fig.update_xaxes(range=[x0, x1])
    fig.update_layout(height=950, xaxis_rangeslider_visible=False, dragmode="pan", hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.03))
    return fig


# Re-fits every price axis to the visible x-range whenever you pan/zoom (what plain Plotly does not do).
_FIT_JS = r"""
(function () {
  var gd = document.getElementById('{plot_id}');
  var FIXED = __FIXED__;
  var timer = null;
  function toMs(v) {
    if (typeof v === 'number') return v;
    var s = String(v).replace(' ', 'T');
    if (s.length === 10) s += 'T00:00:00';
    return new Date(s).getTime();
  }
  function fit(x0, x1) {
    var mins = {}, maxs = {};
    gd.data.forEach(function (tr) {
      var xs = tr.x;
      if (!xs || tr.visible === false) return;
      var ya = tr.yaxis || 'y';
      if (FIXED.indexOf(ya) >= 0) return;
      var series = tr.type === 'candlestick' ? [tr.low, tr.high] : (tr.y ? [tr.y] : null);
      if (!series) return;
      for (var i = 0; i < xs.length; i++) {
        if (x0 !== null) { var t = toMs(xs[i]); if (t < x0 || t > x1) continue; }
        for (var s = 0; s < series.length; s++) {
          var v = series[s][i];
          if (v === null || v === undefined || isNaN(v)) continue;
          if (mins[ya] === undefined || v < mins[ya]) mins[ya] = v;
          if (maxs[ya] === undefined || v > maxs[ya]) maxs[ya] = v;
        }
      }
    });
    var upd = {};
    Object.keys(mins).forEach(function (ya) {
      var lo = mins[ya], hi = maxs[ya];
      var pad = (hi - lo) * 0.06 || 1;
      upd['yaxis' + (ya === 'y' ? '' : ya.slice(1)) + '.range'] = [ya === 'y2' ? 0 : lo - pad, hi + pad];
    });
    Plotly.relayout(gd, upd);
  }
  function xrange(ev) {
    var a = null, b = null, auto = false;
    for (var k in ev) {
      if (/^xaxis\d*\.range\[0\]$/.test(k)) a = ev[k];
      else if (/^xaxis\d*\.range\[1\]$/.test(k)) b = ev[k];
      else if (/^xaxis\d*\.range$/.test(k) && Array.isArray(ev[k])) { a = ev[k][0]; b = ev[k][1]; }
      else if (/^xaxis\d*\.autorange$/.test(k)) auto = true;
    }
    if (a !== null && b !== null) return [toMs(a), toMs(b)];
    if (auto) return [null, null];
    return undefined;
  }
  gd.on('plotly_relayout', function (ev) {
    var r = xrange(ev);
    if (r === undefined) return;
    clearTimeout(timer);
    timer = setTimeout(function () { fit(r[0], r[1]); }, 60);
  });
  setTimeout(function () {
    var xr = gd.layout.xaxis && gd.layout.xaxis.range;
    if (xr) fit(toMs(xr[0]), toMs(xr[1])); else fit(null, null);
  }, 50);
})();
"""


def fit_js() -> str:
    return _FIT_JS.replace("__FIXED__", json.dumps(list(FIXED_AXES)))


def fig_html(fig: go.Figure) -> str:
    """Standalone HTML (Plotly from CDN, needs internet) with the auto-fit script attached."""
    return fig.to_html(include_plotlyjs="cdn", full_html=True, config=CFG, post_script=fit_js(),
                       default_height="950px", default_width="100%")


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
