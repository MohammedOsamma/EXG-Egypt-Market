"""GUI: py -m streamlit run app.py  (or double-click run_gui.bat)"""
import copy
import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from egx.config import load_config
from egx.data.loader import load_csv, load_yfinance, validate
from egx.data.synthetic import make_synthetic
from egx.data.stocks import load_stock_list, ensure_prices, get_info, DATA
from egx.agents.technical import TechnicalAgent
from egx.agents.risk import RiskAgent
from egx.master.master import MasterAgent
from egx.backtest.engine import run_backtest, buy_and_hold
from egx.portfolio.portfolio import Portfolio
from egx.indicators import add_indicators, indicator_table
from egx.scanner import scan_row
from egx import charts
from egx.data.live import fetch_quotes, delay_label

st.set_page_config(page_title="EGX AI Paper Trading", layout="wide")
DATA.mkdir(exist_ok=True)
fmt = lambda v, s="": "n/a" if v is None else f"{v}{s}"


def trades_df(pf):
    t = pd.DataFrame([x.__dict__ for x in pf.trades])
    if not t.empty:
        t["date"] = pd.to_datetime(t["date"])
    return t


@st.cache_data(show_spinner=False)
def system_markers(symbol, last_date, cfg_json):
    """Default-config backtest over the whole saved history, for chart markers only."""
    cfg = json.loads(cfg_json)
    df = load_csv(str(DATA / f"{symbol}.csv"))
    cfg["technical"]["min_bars"] = max(cfg["technical"]["min_bars"], cfg["technical"]["sma_slow"] + 10)
    master = MasterAgent(cfg, TechnicalAgent(cfg), RiskAgent(cfg))
    return trades_df(run_backtest(symbol, df, cfg, master)["portfolio"])


@st.cache_data(ttl=3600, show_spinner=False)
def cached_info(ticker):
    return get_info(ticker)


@st.cache_data(ttl=60, show_spinner=False)
def get_quotes():
    return fetch_quotes()


def quote_for(sym):
    """Chart overlay dict for the live price line, or None (never raises)."""
    try:
        q = get_quotes()
        r = q[q["symbol"] == sym]
        if r.empty:
            return None
        r = r.iloc[0]
        return {"price": float(r["close"]), "label": f"Live {r['close']:.2f} ({delay_label(r['update_mode'])})"}
    except Exception:
        return None


def add_live(table):
    try:
        q = get_quotes()
    except Exception as e:
        st.warning(f"Live prices unavailable: {e}")
        return table
    m = q[["symbol", "close", "change", "volume"]].rename(
        columns={"symbol": "Symbol", "close": "Live price", "change": "Live %", "volume": "Live volume"})
    m["Live %"] = m["Live %"].round(2)
    st.caption(f"Live columns: TradingView quotes fetched {q['fetched_at'].iloc[0]:%H:%M:%S} Cairo time, "
               f"{delay_label(q['update_mode'].iloc[0])}. Display only, not used for signals.")
    return table.merge(m, on="Symbol", how="left")


def live_panel(sym, last_close, auto):
    @st.fragment(run_every=60 if auto else None)
    def _panel():
        try:
            q = get_quotes()
        except Exception as e:
            st.warning(f"Live price unavailable: {e}")
            return
        r = q[q["symbol"] == sym]
        if r.empty:
            st.warning("TradingView has no quote for this symbol code (codes can differ from Yahoo's).")
            return
        r = r.iloc[0]
        c = st.columns(4)
        c[0].metric("Live price", f"{r['close']:.2f}", f"{r['change']:.2f}% today" if pd.notna(r["change"]) else None)
        c[1].metric("vs last daily close", f"{(r['close'] / last_close - 1) * 100:.2f}%")
        c[2].metric("Day range", f"{r['low']:.2f} - {r['high']:.2f}")
        c[3].metric("Volume today", f"{int(r['volume']):,}" if pd.notna(r["volume"]) else "n/a")
        st.caption(f"TradingView quote fetched {r['fetched_at']:%H:%M:%S} Cairo time | {delay_label(r['update_mode'])}. "
                   "Display only: signals use completed daily bars.")
    _panel()


def show_price_chart(d, trades, bars, live=None):
    n = None if bars == "All" else int(bars)
    if st.session_state.get("interactive", True):
        fig = charts.price_chart(d, trades, n, live, interactive=True)
        components.html(charts.fig_html(fig), height=980, scrolling=False)
        st.caption("Drag = pan | mouse wheel = zoom | double-click = reset | buttons above the chart = 1M/3M/6M/1Y/All. "
                   "Price axes auto-fit as you move.")
    else:
        st.plotly_chart(charts.price_chart(d, trades, n, live, interactive=False), use_container_width=True, config=charts.CFG)


def color_reading(v):
    return {"Buy": "background-color:#a5d6a7;color:black", "Sell": "background-color:#ef9a9a;color:black"}.get(v, "")


# =====================================================================
# PAGE 1: STOCKS
# =====================================================================
def stocks_page():
    cfg = load_config()
    stocks = load_stock_list()
    with st.sidebar:
        st.header("Stocks")
        sectors = sorted(stocks["sector"].unique())
        sel_sectors = st.multiselect("Sector", sectors, default=sectors)
        bars = st.select_slider("Chart bars", [60, 120, 250, 500, 1000, "All"], value=250)
        markers = st.checkbox("Show system BUY/SELL markers (runs a backtest, slower)")
        use_live = st.checkbox("Live prices (TradingView, delayed)", value=True)
        auto = st.checkbox("Auto-refresh live price every 60s", value=False)
        if st.button("Refresh live prices now"):
            get_quotes.clear()
        scan = st.button("Scan all stocks", type="primary", use_container_width=True)
        st.caption("Scan downloads/updates prices for every listed stock. Edit the list in config/egx_stocks.csv.")

    st.title("EGX stocks")
    st.caption("Click a row to open the stock. Historical daily data (not live). Paper trading only, not financial advice.")
    base = stocks[stocks["sector"].isin(sel_sectors)]

    if scan:
        rows, bar = [], st.progress(0.0, "Scanning...")
        for i, r in enumerate(stocks.itertuples()):
            bar.progress((i + 1) / len(stocks), f"{r.symbol} ({i + 1}/{len(stocks)})")
            try:
                df, note = ensure_prices(r.symbol, r.ticker, max_age_hours=6)
                rows.append(scan_row(r.symbol, r.name, r.sector, df, cfg, note))
            except Exception:
                rows.append({"Symbol": r.symbol, "Name": r.name, "Sector": r.sector, "Status": "No data"})
        bar.empty()
        st.session_state["scan"] = pd.DataFrame(rows)

    table = st.session_state.get("scan")
    if table is None:
        table = base.rename(columns={"symbol": "Symbol", "name": "Name", "sector": "Sector"})[["Symbol", "Name", "Sector"]]
        st.info("Click **Scan all stocks** in the sidebar to load signals for the whole list (about 1 to 2 minutes).")
    else:
        table = table[table["Sector"].isin(sel_sectors)].reset_index(drop=True)
        ok = (table["Status"] == "OK").sum()
        st.caption(f"{ok} of {len(table)} stocks have fresh data. 'No data' means Yahoo returned nothing for that ticker.")

    if use_live:
        table = add_live(table)
    ev = st.dataframe(table, on_select="rerun", selection_mode="single-row", hide_index=True,
                      use_container_width=True, height=330)
    rows_sel = ev.selection.rows if ev is not None else []
    if rows_sel:
        sym = table.iloc[rows_sel[0]]["Symbol"]
    else:
        sym = st.selectbox("Or choose a stock", base["symbol"])
    stock_detail(sym, stocks, cfg, bars, markers, use_live, auto)


def stock_detail(sym, stocks, cfg, bars, markers, use_live, auto):
    row = stocks[stocks["symbol"] == sym].iloc[0]
    st.divider()
    st.header(f"{sym}: {row['name']}")
    try:
        with st.spinner("Loading prices..."):
            df, note = ensure_prices(sym, row["ticker"])
    except Exception as e:
        st.error(f"No price data for {row['ticker']}: {e}")
        return
    if note:
        st.warning(note)
    if len(df) < 60:
        st.warning(f"Only {len(df)} bars available; indicators and signals need more history.")
    d = add_indicators(df)
    c = df["close"]
    yr = c.tail(252)
    k = st.columns(5)
    k[0].metric("Close", f"{c.iloc[-1]:.2f}", f"{(c.iloc[-1] / c.iloc[-2] - 1) * 100:.2f}%" if len(c) > 1 else None)
    k[1].metric("52w high", f"{yr.max():.2f}")
    k[2].metric("52w low", f"{yr.min():.2f}")
    k[3].metric("Avg value 20d (EGP)", f"{int((c * df['volume']).tail(20).mean()):,}")
    k[4].metric("Last daily bar", str(df.index[-1].date()))
    if use_live:
        live_panel(sym, float(c.iloc[-1]), auto)

    t_chart, t_ind, t_sys, t_info = st.tabs(["Chart", "Buy/Sell indicators", "System decision", "Company info"])

    with t_chart:
        tr = None
        if markers:
            with st.spinner("Running backtest for markers..."):
                tr = system_markers(sym, str(df.index[-1].date()), json.dumps(cfg))
        show_price_chart(d, tr, bars, quote_for(sym) if use_live else None)
        st.caption("Candles + SMA20/50/200 + Bollinger, volume, RSI, MACD, Stochastic. Weekends/holidays are skipped. "
                   "The orange dashed line is the delayed live price.")

    with t_ind:
        table, score, overall, counts = indicator_table(d)
        a, b = st.columns([1, 2])
        a.plotly_chart(charts.gauge_fig(score, overall, len(table)), use_container_width=True)
        a.write(f"Buy: **{counts['Buy']}** | Sell: **{counts['Sell']}** | Neutral: **{counts['Neutral']}**")
        b.dataframe(table.style.map(color_reading, subset=["Reading"]), hide_index=True, use_container_width=True)
        st.caption("Conventional readings (RSI under 30 = oversold/Buy, over 70 = overbought/Sell). Descriptive only. "
                   "The system's decision uses its own rules plus risk checks, see the next tab.")

    with t_sys:
        cfg2 = copy.deepcopy(cfg)
        cfg2["technical"]["min_bars"] = max(cfg2["technical"]["min_bars"], cfg2["technical"]["sma_slow"] + 10)
        master = MasterAgent(cfg2, TechnicalAgent(cfg2), RiskAgent(cfg2))
        dec = master.decide(sym, df, Portfolio(cfg2))
        st.markdown(f"### {dec['action']}" + (f"  ({dec['shares']} shares)" if dec["shares"] else ""))
        st.caption(f"Based on the daily close of {dec['date']}, assuming no open position, using default settings.")
        for r in dec["reasons"]:
            st.write("- " + r)
        if dec["invalidation"]:
            st.write("Invalidation: " + dec["invalidation"])
        if dec["stop"]:
            st.write(f"Proposed stop {dec['stop']:.2f} | target {dec['target']:.2f}")
        with st.expander("Full agent reports"):
            st.json(dec["reports"])

    with t_info:
        info, div = cached_info(row["ticker"])
        labels = [("Sector", "sector"), ("Industry", "industry"), ("Currency", "currency"), ("Market cap", "marketCap"),
                  ("P/E (trailing)", "trailingPE"), ("EPS (trailing)", "trailingEps"), ("P/B", "priceToBook"),
                  ("Dividend yield", "dividendYield"), ("ROE", "returnOnEquity"), ("Debt/Equity", "debtToEquity"),
                  ("Revenue growth", "revenueGrowth"), ("Profit margin", "profitMargins")]
        if not info:
            st.warning("Yahoo returned no company data for this ticker. Nothing is shown rather than guessing.")
        else:
            st.write(f"**{info.get('longName', row['name'])}**" + (f" | {info['website']}" if info.get("website") else ""))
            vals = [(lab, info.get(key)) for lab, key in labels]
            cols = st.columns(4)
            for i, (lab, v) in enumerate(vals):
                cols[i % 4].metric(lab, "n/a" if v is None else (f"{v:,.4g}" if isinstance(v, (int, float)) else str(v)))
            if info.get("longBusinessSummary"):
                with st.expander("Business summary"):
                    st.write(info["longBusinessSummary"])
        if len(div):
            st.write("Dividend history (latest 10)")
            dd = div.tail(10).rename("Dividend per share").reset_index()
            dd.columns = ["Date", "Dividend per share"]
            dd["Date"] = dd["Date"].astype(str).str[:10]
            st.dataframe(dd, hide_index=True)
        st.caption("Source: Yahoo Finance via yfinance. EGX coverage can be incomplete; n/a means not provided, not zero. "
                   "Official figures come from EGX disclosures and company statements.")


# =====================================================================
# PAGE 2: BACKTEST
# =====================================================================
def backtest_page():
    from egx.data.loader import validate as _validate
    cfg = copy.deepcopy(load_config())
    st.title("Backtest")
    st.caption("Paper trading on historical data. Not live and not financial advice.")

    df, symbol = None, "DEMO"
    with st.sidebar:
        st.header("1. Data")
        src = st.radio("Source", ["Saved CSV", "Upload CSV", "Demo (synthetic)"])
        try:
            if src == "Saved CSV":
                files = sorted(DATA.glob("*.csv"))
                if files:
                    f = st.selectbox("File", files, format_func=lambda p: p.name)
                    df, symbol = load_csv(str(f)), f.stem
                else:
                    st.info("No CSV in data/. Open the Stocks page to download prices.")
            elif src == "Upload CSV":
                up = st.file_uploader("Columns: date,open,high,low,close,volume", type="csv")
                if up:
                    raw = pd.read_csv(up)
                    raw.columns = [c.lower() for c in raw.columns]
                    raw["date"] = pd.to_datetime(raw["date"])
                    df, symbol = _validate(raw.set_index("date")), Path(up.name).stem
            else:
                df = make_synthetic()
                st.warning("Synthetic data: results are meaningless.")
        except Exception as e:
            st.error(f"Data error: {e}")
            df = None
    if df is None:
        st.info("Choose data in the sidebar.")
        return

    with st.sidebar:
        st.header("2. Period")
        lo, hi = df.index[0].date(), df.index[-1].date()
        split = st.date_input("Train/test split", min(max(date(2024, 6, 30), lo), hi), min_value=lo, max_value=hi)
        start = st.date_input("Start", lo, min_value=lo, max_value=hi)
        end = st.date_input("End", min(split, hi), min_value=lo, max_value=hi)
        if end > split:
            st.warning("End is after the split: this includes the TEST period. Results stop being out-of-sample if you tune on them.")
        st.header("3. Settings")
        with st.expander("Capital & costs", expanded=True):
            cfg["capital"] = st.number_input("Capital (EGP)", 100.0, 10_000_000.0, float(cfg["capital"]), step=100.0)
            cfg["fees"]["commission_pct"] = st.number_input("Commission %", 0.0, 5.0, cfg["fees"]["commission_pct"] * 100, step=0.01, format="%.3f") / 100
            cfg["fees"]["min_commission"] = st.number_input("Min commission (EGP)", 0.0, 100.0, float(cfg["fees"]["min_commission"]), step=0.5)
            cfg["fees"]["exchange_pct"] = st.number_input("Exchange/other fees %", 0.0, 5.0, cfg["fees"]["exchange_pct"] * 100, step=0.01, format="%.3f") / 100
            cfg["slippage_pct"] = st.number_input("Slippage %", 0.0, 5.0, cfg["slippage_pct"] * 100, step=0.05, format="%.2f") / 100
            st.caption("Default fees are placeholders. Use your broker's real numbers.")
        with st.expander("Technical Agent"):
            cfg["technical"]["sma_fast"] = st.number_input("SMA fast", 5, 100, int(cfg["technical"]["sma_fast"]))
            cfg["technical"]["sma_slow"] = st.number_input("SMA slow", 10, 300, int(cfg["technical"]["sma_slow"]))
            cfg["technical"]["rsi_buy_min"] = st.number_input("RSI buy min", 0, 100, int(cfg["technical"]["rsi_buy_min"]))
            cfg["technical"]["rsi_buy_max"] = st.number_input("RSI buy max", 0, 100, int(cfg["technical"]["rsi_buy_max"]))
        with st.expander("Risk Agent"):
            cfg["risk"]["risk_per_trade_pct"] = st.number_input("Risk per trade %", 0.1, 20.0, cfg["risk"]["risk_per_trade_pct"] * 100, step=0.5) / 100
            cfg["risk"]["max_position_pct"] = st.number_input("Max position %", 5.0, 100.0, cfg["risk"]["max_position_pct"] * 100, step=5.0) / 100
            cfg["risk"]["atr_stop_mult"] = st.number_input("Stop = ATR x", 0.5, 10.0, float(cfg["risk"]["atr_stop_mult"]), step=0.5)
            cfg["risk"]["min_rr"] = st.number_input("Min reward/risk", 0.5, 10.0, float(cfg["risk"]["min_rr"]), step=0.25)
        run = st.button("Run backtest", type="primary", use_container_width=True)

    if run:
        if cfg["technical"]["sma_fast"] >= cfg["technical"]["sma_slow"]:
            st.error("SMA fast must be smaller than SMA slow.")
            return
        cfg["technical"]["min_bars"] = max(cfg["technical"]["min_bars"], cfg["technical"]["sma_slow"] + 10)
        sub = df.loc[str(start):str(end)]
        if len(sub) < cfg["technical"]["min_bars"] + 5:
            st.error(f"Not enough data in range ({len(sub)} bars).")
            return
        master = MasterAgent(cfg, TechnicalAgent(cfg), RiskAgent(cfg))
        with st.spinner("Running..."):
            res = run_backtest(symbol, sub, cfg, master)
            base = buy_and_hold(symbol, sub, cfg, cfg["technical"]["min_bars"])
            latest = master.decide(symbol, sub, Portfolio(cfg))
        st.session_state["out"] = dict(res=res, base=base, latest=latest, symbol=symbol, sub=sub)

    out = st.session_state.get("out")
    if not out:
        st.info("Set parameters in the sidebar, then click **Run backtest**.")
        return
    res, base, latest, sub = out["res"], out["base"], out["latest"], out["sub"]

    st.subheader(f"{out['symbol']}: results")
    m, b = res["metrics"], base["metrics"]
    c = st.columns(4)
    c[0].metric("Return (system)", f"{m['return_pct']}%", f"{round(m['return_pct'] - b['return_pct'], 2)}% vs buy&hold")
    c[1].metric("Return (buy & hold)", f"{b['return_pct']}%")
    c[2].metric("Max drawdown (system)", f"{m['max_drawdown_pct']}%", f"B&H {b['max_drawdown_pct']}%", delta_color="off")
    c[3].metric("Closed trades / win rate", f"{m['closed_trades']} / {fmt(m['win_rate_pct'], '%')}")
    if m["closed_trades"] < 15:
        st.warning("Fewer than 15 closed trades: result is not statistically meaningful.")

    tr = trades_df(res["portfolio"])
    tabs = st.tabs(["Summary", "Price & signals", "Latest signal", "Trades", "Decisions", "Agent reports", "Rejected orders"])

    with tabs[0]:
        st.plotly_chart(charts.equity_fig(res["equity"], base["equity"]), use_container_width=True)
        st.plotly_chart(charts.drawdown_fig(res["equity"], base["equity"]), use_container_width=True)
        if not tr.empty and (tr["side"] == "SELL").any():
            st.plotly_chart(charts.trades_fig(tr), use_container_width=True)
        else:
            st.write("No closed trades yet.")

    with tabs[1]:
        n = st.select_slider("Bars", [60, 120, 250, 500, 1000, "All"], value=250, key="bt_bars")
        show_price_chart(add_indicators(sub), tr if not tr.empty else None, n)
        st.caption("Green/red triangles are the system's simulated fills for this backtest.")

    with tabs[2]:
        st.caption(f"Based on the daily close of {latest['date']} (historical data, not live), assuming no open position.")
        st.markdown(f"### {latest['action']}" + (f"  ({latest['shares']} shares)" if latest["shares"] else ""))
        for r in latest["reasons"]:
            st.write("- " + r)
        t = latest["reports"]["technical"]
        if t.get("status") == "ok":
            st.write(f"Signal: **{t['signal']}** (confidence {t['confidence']}) | Trend: {t['trend']} | RSI {t['rsi']:.1f} | "
                     f"Support {t['support']:.2f} | Resistance {t['resistance']:.2f}")
            st.write("Invalidation: " + latest["invalidation"])
        if latest["stop"]:
            st.write(f"Proposed stop {latest['stop']:.2f} | target {latest['target']:.2f}")

    with tabs[3]:
        if tr.empty:
            st.write("No trades.")
        else:
            t2 = tr.copy()
            t2["date"] = t2["date"].astype(str).str[:10]
            st.dataframe(t2.round(2), use_container_width=True, hide_index=True)
            st.download_button("Download trades CSV", t2.to_csv(index=False), "trades.csv")

    with tabs[4]:
        dec = pd.DataFrame([{"date": d["date"], "action": d["action"], "shares": d["shares"],
                             "reasons": "; ".join(d["reasons"])} for d in res["decisions"]])
        pick = st.multiselect("Show actions", ["BUY", "SELL", "HOLD", "WAIT"], default=["BUY", "SELL"])
        st.dataframe(dec[dec["action"].isin(pick)], use_container_width=True, hide_index=True)
        st.download_button("Download decisions CSV", dec.to_csv(index=False), "decisions.csv")

    with tabs[5]:
        dates = [d["date"] for d in res["decisions"] if d["action"] in ("BUY", "SELL")] or [d["date"] for d in res["decisions"]][-1:]
        sel = st.selectbox("Decision date", dates)
        d = next(x for x in res["decisions"] if x["date"] == sel)
        st.write(f"**{d['action']}**: " + "; ".join(d["reasons"]))
        st.json(d["reports"])

    with tabs[6]:
        if res["rejected"]:
            st.dataframe(pd.DataFrame(res["rejected"]), use_container_width=True, hide_index=True)
        else:
            st.write("No orders were rejected by the execution engine.")


# =====================================================================
page = st.sidebar.radio("Page", ["Stocks", "Backtest"], horizontal=True)
st.sidebar.checkbox("Smooth interactive charts (needs internet)", value=True, key="interactive")
st.sidebar.divider()
stocks_page() if page == "Stocks" else backtest_page()
