import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from timezone_fix import setup_timezone, now_local, now_str

setup_timezone()

from streamlit_autorefresh import st_autorefresh

from src.data.market_data import fetch_ohlcv, fetch_tickers_for, SYMBOLS, TIMEFRAMES
from src.data.coingecko import (
    fetch_top20_markets, format_large_number, format_supply, FALLBACK_SYMBOLS,
)
from src.data.fear_greed import fetch_fear_greed_index, get_fg_color, get_fg_emoji
from src.data.orderbook import fetch_order_book
from src.analysis.indicators import add_all_indicators, get_current_indicator_values
from src.analysis.advanced_indicators import (
    add_all_advanced_indicators, get_advanced_indicator_values,
)
from src.analysis.smc import analyze_smc
from src.analysis.mtf import fetch_mtf_analysis, MTF_TIMEFRAMES, MTF_LABELS
from src.analysis.signals import generate_signal, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD
from src.analysis.support_resistance import find_support_resistance
from src.analysis.backtest import run_backtest
from src.ml.models import train_and_predict
from src.data.news_sentiment import get_news_sentiment
from src.risk.risk_manager import assess_risk

st.set_page_config(
    page_title="SuperSignal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
def render_sidebar(watchlist_symbols: list):

    if "symbol" in st.query_params:
        st.session_state.selected_symbol = st.query_params["symbol"]

    symbol = st.sidebar.selectbox(
        "Symbol",
        watchlist_symbols,
        index=watchlist_symbols.index(st.session_state.selected_symbol),
        key="selected_symbol"
    )

    st.query_params["symbol"] = symbol

    if "timeframe" in st.query_params:
        st.session_state.selected_timeframe = st.query_params["timeframe"]

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        list(TIMEFRAMES),
        index=list(TIMEFRAMES).index(st.session_state.selected_timeframe),
        key="selected_timeframe"
    )

    st.query_params["timeframe"] = timeframe
# ── theme CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.terminal-card {
    border: 1px solid #1e2130;
    border-radius: 8px;
    padding: 12px 16px;
    background: #0d1117;
    margin-bottom: 6px;
}
.badge-buy  { background:#1a7f37;color:#fff;padding:2px 10px;border-radius:4px;font-weight:700;font-size:.82em }
.badge-sell { background:#8b0000;color:#fff;padding:2px 10px;border-radius:4px;font-weight:700;font-size:.82em }
.badge-hold { background:#7d5a00;color:#fff;padding:2px 10px;border-radius:4px;font-weight:700;font-size:.82em }
.metric-label { color:#8b949e;font-size:.75em;text-transform:uppercase;letter-spacing:.04em }
.metric-val   { font-size:1.3em;font-weight:700 }
.up   { color:#26a69a }
.down { color:#ef5350 }
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_price(price: float, _sym: str = "") -> str:
    if price == 0:
        return "$0"
    if price >= 10_000:
        return f"${price:,.2f}"
    if price >= 1:
        return f"${price:,.3f}"
    if price >= 0.001:
        return f"${price:.5f}"
    return f"${price:.2e}"

def pct_str(v: float) -> str:
    arrow = "▲" if v >= 0 else "▼"
    color = "#26a69a" if v >= 0 else "#ef5350"
    return f"<span style='color:{color}'>{arrow} {abs(v):.2f}%</span>"

def signal_color(s: str) -> str:
    return {"BUY": "#26a69a", "SELL": "#ef5350", "HOLD": "#f39c12"}.get(s, "#888")

def sig_badge(sig: str) -> str:
    cls = {"BUY": "badge-buy", "SELL": "badge-sell"}.get(sig, "badge-hold")
    return f"<span class='{cls}'>{sig}</span>"

def sentiment_color(s: str) -> str:
    return {"positive": "#26a69a", "negative": "#ef5350", "neutral": "#f39c12"}.get(s, "#888")

def verdict_color(v: str) -> str:
    return {
        "Strong Buy":  "#1a7f37",
        "Buy":         "#26a69a",
        "Hold":        "#f39c12",
        "Sell":        "#ef5350",
        "Strong Sell": "#8b0000",
    }.get(v, "#888")

# ── cache layer ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_watchlist():
    return fetch_top20_markets()

@st.cache_data(ttl=5, show_spinner=False)
def load_tickers_for_watchlist(symbols_key: str) -> dict:
    return fetch_tickers_for(symbols_key.split("|"))

@st.cache_data(ttl=5, show_spinner=False)
def load_market_data(symbol: str, timeframe: str, limit: int = 500):
    df = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = add_all_indicators(df)
    return df

@st.cache_data(ttl=30, show_spinner=False)
def load_full_data(symbol: str, timeframe: str, limit: int = 500):
    """Loads basic + advanced indicators for the main chart."""
    df = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = add_all_indicators(df)
    df = add_all_advanced_indicators(df)
    return df

@st.cache_data(ttl=60, show_spinner=False)
def load_watchlist_data(symbol: str, timeframe: str):
    df = fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
    df = add_all_indicators(df)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_fear_greed():
    return fetch_fear_greed_index()

@st.cache_data(ttl=30, show_spinner=False)
def load_smc(symbol: str, timeframe: str, limit: int = 500):
    df = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = add_all_indicators(df)
    return analyze_smc(df)

@st.cache_data(ttl=5, show_spinner=False)
def load_orderbook(symbol: str):
    return fetch_order_book(symbol)

# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(watchlist_symbols: list):
    st.sidebar.title("⚙️ SuperSignal")

    st.sidebar.subheader("Market")

    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = watchlist_symbols[0]

    symbol = st.sidebar.selectbox(
        "Symbol",
        watchlist_symbols,
        index=watchlist_symbols.index(st.session_state.selected_symbol),
        key="selected_symbol"
    )
    if "selected_timeframe" not in st.session_state:
        st.session_state.selected_timeframe = list(TIMEFRAMES)[0]
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        list(TIMEFRAMES),
        index=list(TIMEFRAMES).index(st.session_state.selected_timeframe),
        key="selected_timeframe"
    )
    limit     = st.sidebar.slider("Candle Limit", 100, 1000, 500, 50)

    st.sidebar.subheader("Auto-Refresh")
    refresh_option = st.sidebar.select_slider(
        "Interval", options=["Off", "30s", "1m", "5m"], value="30s"
    )
    ms_map    = {"Off": None, "30s": 30_000, "1m": 60_000, "5m": 300_000}
    refresh_ms = ms_map[refresh_option]

    st.sidebar.subheader("Chart Overlays")
    show = {}
    with st.sidebar.expander("Trend", expanded=False):
        show["ema_9"]      = st.checkbox("EMA 9",    True,  key="s_e9")
        show["ema_21"]     = st.checkbox("EMA 21",   True,  key="s_e21")
        show["ema_50"]     = st.checkbox("EMA 50",   True,  key="s_e50")
        show["ema_200"]    = st.checkbox("EMA 200",  True,  key="s_e200")
        show["sma_20"]     = st.checkbox("SMA 20",   False, key="s_s20")
        show["sma_50"]     = st.checkbox("SMA 50",   False, key="s_s50")
        show["sma_200"]    = st.checkbox("SMA 200",  False, key="s_s200")
        show["vwap"]       = st.checkbox("VWAP",     False, key="s_vwap")
        show["supertrend"] = st.checkbox("Supertrend", False, key="s_st")
        show["ichimoku"]   = st.checkbox("Ichimoku", False, key="s_ich")
        show["psar"]       = st.checkbox("Parabolic SAR", False, key="s_psar")
    with st.sidebar.expander("Volatility", expanded=False):
        show["bb"]       = st.checkbox("Bollinger Bands",  True,  key="s_bb")
        show["keltner"]  = st.checkbox("Keltner Channel",  False, key="s_kc")
        show["donchian"] = st.checkbox("Donchian Channel", False, key="s_dc")
    with st.sidebar.expander("Smart Money", expanded=False):
        show["fvg"]      = st.checkbox("Fair Value Gaps",  True, key="s_fvg")
        show["ob"]       = st.checkbox("Order Blocks",     True, key="s_ob")
        show["sr_lines"] = st.checkbox("Support/Resistance", True, key="s_sr")

    st.sidebar.subheader("Risk Management")
    capital        = st.sidebar.number_input("Paper Capital ($)", 5.0, 1_000_000.0, 100.0, 1.0, format="%.2f")
    if capital < 5:
        st.sidebar.error("⚠️ Minimum capital is $5.00")
        capital = 5.0
    risk_tolerance = st.sidebar.select_slider(
        "Risk Tolerance", ["conservative", "moderate", "aggressive"], value="moderate"
    )
    stop_loss_pct   = st.sidebar.slider("Stop Loss %",   0.5, 10.0, 2.0, 0.5) / 100
    take_profit_pct = st.sidebar.slider("Take Profit %", 1.0, 20.0, 4.0, 0.5) / 100
    rr              = take_profit_pct / stop_loss_pct if stop_loss_pct else 2.0
    st.sidebar.markdown(f"**R/R:** `1:{rr:.1f}`")

    st.sidebar.subheader("Backtesting")
    bt_pos_size = st.sidebar.slider("Position Size %", 5, 50, 10, 5) / 100

    st.sidebar.markdown("---")
    st.sidebar.info("🔒 **Paper Trading Only** — no real funds.")

    return dict(
        symbol=symbol, timeframe=timeframe, limit=limit,
        refresh_ms=refresh_ms, refresh_option=refresh_option,
        auto_refresh=refresh_ms is not None,
        show=show,
        capital=capital, risk_tolerance=risk_tolerance,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        risk_reward=rr, bt_pos_size=bt_pos_size,
    )

# ── Tab 1: Overview ───────────────────────────────────────────────────────────

def render_overview(tickers, cg_data, watchlist_symbols, ind_map, signal_map, fg):
    # Top strip
    top6 = watchlist_symbols[:6]
    cols = st.columns(len(top6) + 1)
    for col, sym in zip(cols[:6], top6):
        t   = tickers.get(sym, {})
        cg  = cg_data.get(sym, {})
        price = t.get("last", 0) or cg.get("current_price", 0)
        pct   = t.get("percentage", 0) or cg.get("price_change_percentage_24h", 0)
        mc    = cg.get("market_cap", 0)
        col.markdown(
            f"<div class='terminal-card'>"
            f"<div class='metric-label'>{sym.split('/')[0]}</div>"
            f"<div class='metric-val'>{fmt_price(price, sym)}</div>"
            f"{pct_str(pct)}<br/>"
            f"<span style='font-size:.75em;color:#8b949e'>MCap {format_large_number(mc)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with cols[6]:
        fg_val = fg.get("value", 50)
        fg_c   = get_fg_color(fg_val)
        fg_cl  = fg.get("classification", "Neutral")
        st.markdown(
            f"<div class='terminal-card'>"
            f"<div class='metric-label'>Fear & Greed</div>"
            f"<div class='metric-val' style='color:{fg_c}'>{fg_val}</div>"
            f"<span style='color:{fg_c};font-size:.8em'>{get_fg_emoji(fg_cl)} {fg_cl}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### 📋 Market Scanner")
    st.caption(f"{len(watchlist_symbols)} coins · sorted by Market Cap · indicators on 200-candle 1h OHLCV")

    rows = []
    for sym in watchlist_symbols:
        t       = tickers.get(sym, {})
        cg      = cg_data.get(sym, {})
        ind     = ind_map.get(sym, {})
        sig_res = signal_map.get(sym, {})

        price = t.get("last", 0) or cg.get("current_price", 0)
        pct   = t.get("percentage", 0) or cg.get("price_change_percentage_24h", 0)
        vol   = t.get("quoteVolume", 0) or cg.get("total_volume", 0)
        mcap  = cg.get("market_cap", 0)
        sig   = sig_res.get("signal", "—")
        conf  = sig_res.get("confidence", 0.0)

        ema9  = ind.get("ema_9", 0)
        ema21 = ind.get("ema_21", 0)
        cross = (
            "🟢 Bull X" if ind.get("ema_bullish_cross")
            else "🔴 Bear X" if ind.get("ema_bearish_cross")
            else ("↑" if ind.get("ema9_above_ema21") else "↓")
        )

        rows.append({
            "Pair":       sym,
            "Name":       cg.get("name", sym.split("/")[0]),
            "Price":      fmt_price(price, sym),
            "24h %":      f"{'▲' if pct>=0 else '▼'} {abs(pct):.2f}%",
            "Market Cap": format_large_number(mcap),
            "Volume 24h": format_large_number(vol),
            "RSI":        f"{ind.get('rsi', 0):.1f}" if ind.get("rsi") else "—",
            "MACD":       f"{ind.get('macd', 0):.4f}" if ind.get("macd") is not None else "—",
            "EMA 9":      fmt_price(ema9, sym) if ema9 else "—",
            "EMA 21":     fmt_price(ema21, sym) if ema21 else "—",
            "EMA 50":     fmt_price(ind.get("ema_50", 0), sym) if ind.get("ema_50") else "—",
            "EMA 200":    fmt_price(ind.get("ema_200", 0), sym) if ind.get("ema_200") else "—",
            "Cross":      cross,
            "Signal":     sig,
            "Conf %":     f"{conf*100:.0f}%" if conf else "—",
        })

    df_table = pd.DataFrame(rows)

    def color_sig(val):
        return {
            "BUY":  "color:#26a69a;font-weight:700",
            "SELL": "color:#ef5350;font-weight:700",
            "HOLD": "color:#f39c12",
        }.get(val, "")

    def color_pct(val):
        if isinstance(val, str):
            return "color:#26a69a" if val.startswith("▲") else "color:#ef5350"
        return ""

    styled = df_table.style.map(color_sig, subset=["Signal"]).map(color_pct, subset=["24h %"])
    st.dataframe(styled, width="stretch", hide_index=True)

# ── Tab 2: Technical Analysis ─────────────────────────────────────────────────

def render_technical(df: pd.DataFrame, ind: dict, adv: dict,
                     symbol: str, sr: dict, cfg: dict, fg: dict):
    # ── Indicator summary ──────────────────────────────────────────────────
    st.markdown("### 📐 Indicator Dashboard")
    bullish_cross = ind.get("ema_bullish_cross", False)
    bearish_cross = ind.get("ema_bearish_cross", False)
    if bullish_cross:
        st.success("🟢 **EMA 9 × EMA 21 Bullish Crossover** on latest candle")
    elif bearish_cross:
        st.error("🔴 **EMA 9 × EMA 21 Bearish Crossover** on latest candle")

    close = ind["close"]

    def ind_card(label, val_str, sub="", color="#ccc"):
        return (
            f"<div class='terminal-card' style='text-align:center'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-val' style='color:{color}'>{val_str}</div>"
            f"<div style='font-size:.72em;color:#8b949e'>{sub}</div>"
            f"</div>"
        )

    rsi = ind["rsi"]
    rsi_c = "#ef5350" if rsi > 70 else ("#26a69a" if rsi < 30 else "#f1c40f")
    macd  = ind["macd"]
    macd_sig = ind["macd_signal"]
    macd_c   = "#26a69a" if macd > macd_sig else "#ef5350"

    cols = st.columns(6)
    cols[0].markdown(ind_card("RSI (14)", f"{rsi:.1f}",
        "Overbought" if rsi>70 else "Oversold" if rsi<30 else "Neutral", rsi_c),
        unsafe_allow_html=True)
    cols[1].markdown(ind_card("MACD", f"{macd:.4f}", f"Sig {macd_sig:.4f}", macd_c),
        unsafe_allow_html=True)
    stk  = adv.get("stochrsi_k", 50)
    std  = adv.get("stochrsi_d", 50)
    stk_c = "#ef5350" if stk > 80 else ("#26a69a" if stk < 20 else "#f39c12")
    cols[2].markdown(ind_card("Stoch RSI K", f"{stk:.1f}", f"D {std:.1f}", stk_c),
        unsafe_allow_html=True)
    cci = adv.get("cci", 0)
    cci_c = "#ef5350" if cci > 100 else ("#26a69a" if cci < -100 else "#f39c12")
    cols[3].markdown(ind_card("CCI (20)", f"{cci:.1f}",
        "Overbought" if cci>100 else "Oversold" if cci<-100 else "Neutral", cci_c),
        unsafe_allow_html=True)
    adx = adv.get("adx", 25)
    adx_c = "#26a69a" if adx > 30 else "#8b949e"
    cols[4].markdown(ind_card("ADX (14)", f"{adx:.1f}",
        "Strong" if adx>30 else "Weak", adx_c), unsafe_allow_html=True)
    roc = adv.get("roc", 0)
    roc_c = "#26a69a" if roc > 0 else "#ef5350"
    cols[5].markdown(ind_card("ROC (12)", f"{roc:.2f}%", "", roc_c), unsafe_allow_html=True)

    cols2 = st.columns(6)
    ema9  = ind.get("ema_9", close)
    ema21 = ind.get("ema_21", close)
    ema50 = ind.get("ema_50", close)
    ema200 = ind.get("ema_200", close)
    cols2[0].markdown(ind_card("EMA 9",   fmt_price(ema9, symbol),
        "🟢 Bull" if ema9>ema21 else "🔴 Bear", "#26a69a" if ema9>ema21 else "#ef5350"),
        unsafe_allow_html=True)
    cols2[1].markdown(ind_card("EMA 21",  fmt_price(ema21, symbol),
        f"Gap {abs(ema9-ema21)/ema21*100:.2f}%" if ema21 else "",
        "#26a69a" if ema9>ema21 else "#ef5350"), unsafe_allow_html=True)
    cols2[2].markdown(ind_card("EMA 50",  fmt_price(ema50, symbol),
        "↑ Bullish" if close>ema50 else "↓ Bearish", "#26a69a" if close>ema50 else "#ef5350"),
        unsafe_allow_html=True)
    cols2[3].markdown(ind_card("EMA 200", fmt_price(ema200, symbol),
        "Above" if close>ema200 else "Below", "#26a69a" if close>ema200 else "#ef5350"),
        unsafe_allow_html=True)
    vwap = adv.get("vwap", close)
    cols2[4].markdown(ind_card("VWAP", fmt_price(vwap, symbol),
        "Above" if close>vwap else "Below", "#26a69a" if close>vwap else "#ef5350"),
        unsafe_allow_html=True)
    sma20 = adv.get("sma_20", close)
    cols2[5].markdown(ind_card("SMA 20", fmt_price(sma20, symbol),
        "Above" if close>sma20 else "Below", "#26a69a" if close>sma20 else "#ef5350"),
        unsafe_allow_html=True)

    cols3 = st.columns(6)
    mfi = adv.get("mfi", 50)
    mfi_c = "#ef5350" if mfi > 80 else ("#26a69a" if mfi < 20 else "#f39c12")
    cols3[0].markdown(ind_card("MFI (14)", f"{mfi:.1f}",
        "Overbought" if mfi>80 else "Oversold" if mfi<20 else "Neutral", mfi_c),
        unsafe_allow_html=True)
    cmf = adv.get("cmf", 0)
    cmf_c = "#26a69a" if cmf > 0.05 else ("#ef5350" if cmf < -0.05 else "#f39c12")
    cols3[1].markdown(ind_card("CMF (20)", f"{cmf:.3f}",
        "Inflow" if cmf>0 else "Outflow", cmf_c), unsafe_allow_html=True)
    obv = adv.get("obv", 0)
    cols3[2].markdown(ind_card("OBV", format_large_number(abs(obv)).replace("$",""),
        "↑" if obv>0 else "↓", "#26a69a" if obv>0 else "#ef5350"), unsafe_allow_html=True)
    bb_pct = ind.get("bb_pct", 0.5) * 100
    bb_c   = "#ef5350" if bb_pct > 80 else ("#26a69a" if bb_pct < 20 else "#f39c12")
    cols3[3].markdown(ind_card("BB %B",   f"{bb_pct:.1f}%", "", bb_c), unsafe_allow_html=True)
    st_dir = adv.get("supertrend_dir", 0)
    st_c   = "#26a69a" if st_dir == 1 else ("#ef5350" if st_dir == -1 else "#8b949e")
    st_lbl = "Bullish" if st_dir == 1 else ("Bearish" if st_dir == -1 else "N/A")
    cols3[4].markdown(ind_card("Supertrend", st_lbl, "", st_c), unsafe_allow_html=True)
    psar_bull = adv.get("psar_bull", True)
    cols3[5].markdown(ind_card("Parabolic SAR",
        "Bullish" if psar_bull else "Bearish", "", "#26a69a" if psar_bull else "#ef5350"),
        unsafe_allow_html=True)

    st.divider()

    # ── Chart ──────────────────────────────────────────────────────────────
    chart_col, fg_col = st.columns([3, 1])
    with chart_col:
        st.markdown(f"### 📊 {symbol} Chart")
        render_advanced_chart(df, symbol, sr, cfg["show"], adv)
    with fg_col:
        st.markdown("#### 😨 Fear & Greed")
        render_fear_greed_gauge(fg)

    st.divider()

    # ── Support/Resistance ────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📏 Support / Resistance")
        nr = sr.get("nearest_resistance", 0)
        ns = sr.get("nearest_support", 0)
        st.markdown(f"🔴 **Resistance:** `{fmt_price(nr, symbol)}` (+{sr.get('resistance_pct',0):.2f}%)")
        st.markdown(f"🟢 **Support:** `{fmt_price(ns, symbol)}` (-{sr.get('support_pct',0):.2f}%)")
        for name, key in [("R2","pivot_r2"),("R1","pivot_r1"),("PP","pivot"),("S1","pivot_s1"),("S2","pivot_s2")]:
            val = sr.get(key, 0)
            c = "#ef5350" if "R" in name else ("#f1c40f" if "PP" in name else "#26a69a")
            st.markdown(f"<span style='color:{c}'><b>{name}</b>: {fmt_price(val, symbol)}</span>",
                        unsafe_allow_html=True)
    with c2:
        st.markdown("#### 📊 Ichimoku Cloud")
        ich_a = adv.get("ich_senkou_a", 0)
        ich_b = adv.get("ich_senkou_b", 0)
        ten   = adv.get("ich_tenkan", 0)
        kij   = adv.get("ich_kijun", 0)
        cloud_top    = max(ich_a, ich_b)
        cloud_bottom = min(ich_a, ich_b)
        cloud_color  = "#26a69a" if ich_a > ich_b else "#ef5350"
        cloud_label  = "Bullish Cloud" if ich_a > ich_b else "Bearish Cloud"
        st.markdown(f"**Tenkan-sen (9):** `{fmt_price(ten, symbol)}`")
        st.markdown(f"**Kijun-sen (26):** `{fmt_price(kij, symbol)}`")
        st.markdown(f"**Senkou A:** `{fmt_price(ich_a, symbol)}`")
        st.markdown(f"**Senkou B:** `{fmt_price(ich_b, symbol)}`")
        st.markdown(f"<span style='color:{cloud_color}'><b>Cloud:</b> {cloud_label} "
                    f"({fmt_price(cloud_bottom, symbol)} – {fmt_price(cloud_top, symbol)})</span>",
                    unsafe_allow_html=True)
        pos = ("Above cloud 🚀" if close > cloud_top
               else "Below cloud 📉" if close < cloud_bottom
               else "Inside cloud ⚠️")
        st.markdown(f"**Price position:** {pos}")


def render_advanced_chart(df: pd.DataFrame, symbol: str, sr: dict, show: dict, adv: dict):
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.44, 0.12, 0.15, 0.15, 0.14],
        subplot_titles=(f"{symbol} · Price", "Volume", "RSI / Stoch RSI", "MACD", "MFI / CMF"),
    )

    # ── Candles ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # ── EMA / SMA lines ───────────────────────────────────────────────────
    ema_cfg = [
        ("ema_9",   "#00e5ff", "EMA 9",   "solid", show.get("ema_9")),
        ("ema_21",  "#ff6f00", "EMA 21",  "solid", show.get("ema_21")),
        ("ema_50",  "#f39c12", "EMA 50",  "dot",   show.get("ema_50")),
        ("ema_200", "#9b59b6", "EMA 200", "dot",   show.get("ema_200")),
        ("sma_20",  "#3498db", "SMA 20",  "dash",  show.get("sma_20")),
        ("sma_50",  "#1abc9c", "SMA 50",  "dash",  show.get("sma_50")),
        ("sma_200", "#e67e22", "SMA 200", "dash",  show.get("sma_200")),
    ]
    for col_name, color, name, dash, visible in ema_cfg:
        if visible and col_name in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name], name=name,
                line=dict(color=color, width=1.4, dash=dash)), row=1, col=1)

    # ── VWAP ──────────────────────────────────────────────────────────────
    if show.get("vwap") and "vwap" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap"], name="VWAP",
            line=dict(color="#e91e63", width=1.5, dash="dot")), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────────────────
    if show.get("bb") and "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
            line=dict(color="rgba(52,152,219,0.5)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
            line=dict(color="rgba(52,152,219,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(52,152,219,0.05)"), row=1, col=1)

    # ── Keltner Channel ───────────────────────────────────────────────────
    if show.get("keltner") and "kc_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["kc_upper"], name="KC Upper",
            line=dict(color="rgba(155,89,182,0.5)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["kc_lower"], name="KC Lower",
            line=dict(color="rgba(155,89,182,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(155,89,182,0.04)"), row=1, col=1)

    # ── Donchian Channel ──────────────────────────────────────────────────
    if show.get("donchian") and "dc_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["dc_upper"], name="DC High",
            line=dict(color="rgba(230,126,34,0.5)", width=1, dash="dashdot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["dc_lower"], name="DC Low",
            line=dict(color="rgba(230,126,34,0.5)", width=1, dash="dashdot"),
            fill="tonexty", fillcolor="rgba(230,126,34,0.04)"), row=1, col=1)

    # ── Supertrend ────────────────────────────────────────────────────────
    if show.get("supertrend") and "supertrend" in df.columns:
        bull_st = df["supertrend"].where(df["supertrend_direction"] == 1)
        bear_st = df["supertrend"].where(df["supertrend_direction"] == -1)
        fig.add_trace(go.Scatter(x=df.index, y=bull_st, name="ST Bull",
            line=dict(color="#26a69a", width=2), mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bear_st, name="ST Bear",
            line=dict(color="#ef5350", width=2), mode="lines"), row=1, col=1)

    # ── Ichimoku ──────────────────────────────────────────────────────────
    if show.get("ichimoku") and "ich_senkou_a" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ich_tenkan"], name="Tenkan",
            line=dict(color="#e91e63", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ich_kijun"], name="Kijun",
            line=dict(color="#3f51b5", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ich_senkou_a"], name="Senkou A",
            line=dict(color="rgba(38,166,154,0.6)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ich_senkou_b"], name="Senkou B",
            line=dict(color="rgba(239,83,80,0.6)", width=1),
            fill="tonexty", fillcolor="rgba(100,100,100,0.07)"), row=1, col=1)

    # ── Parabolic SAR ─────────────────────────────────────────────────────
    if show.get("psar") and "psar" in df.columns:
        bull_psar = df["psar"].where(df.get("psar_bull", pd.Series(1, index=df.index)) == 1)
        bear_psar = df["psar"].where(df.get("psar_bull", pd.Series(1, index=df.index)) != 1)
        fig.add_trace(go.Scatter(x=df.index, y=bull_psar, name="SAR Bull",
            mode="markers", marker=dict(size=3, color="#26a69a", symbol="circle")),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bear_psar, name="SAR Bear",
            mode="markers", marker=dict(size=3, color="#ef5350", symbol="circle")),
            row=1, col=1)

    # ── EMA crossover markers ─────────────────────────────────────────────
    if "ema_bullish_cross" in df.columns:
        bull_x = df.index[df["ema_bullish_cross"] == True]
        if len(bull_x):
            fig.add_trace(go.Scatter(x=bull_x, y=df.loc[bull_x, "ema_9"],
                mode="markers", name="Bull X",
                marker=dict(symbol="triangle-up", size=12, color="#26a69a",
                            line=dict(color="white", width=1))), row=1, col=1)
    if "ema_bearish_cross" in df.columns:
        bear_x = df.index[df["ema_bearish_cross"] == True]
        if len(bear_x):
            fig.add_trace(go.Scatter(x=bear_x, y=df.loc[bear_x, "ema_9"],
                mode="markers", name="Bear X",
                marker=dict(symbol="triangle-down", size=12, color="#ef5350",
                            line=dict(color="white", width=1))), row=1, col=1)

    # ── Support/Resistance lines ──────────────────────────────────────────
    if show.get("sr_lines"):
        for lv in sr.get("resistance", [])[:3]:
            fig.add_hline(y=lv, line_dash="dash", line_color="rgba(239,83,80,0.5)",
                          line_width=1, row=1, col=1)
        for lv in sr.get("support", [])[:3]:
            fig.add_hline(y=lv, line_dash="dash", line_color="rgba(38,166,154,0.5)",
                          line_width=1, row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────
    bar_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
        marker_color=bar_colors, opacity=0.7), row=2, col=1)
    if "vwap" in df.columns and show.get("vwap"):
        pass  # Volume VWAP already on price

    # ── RSI + Stoch RSI ───────────────────────────────────────────────────
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
            line=dict(color="#ef5350", width=1.4)), row=3, col=1)
        for lvl, clr in [(70,"rgba(239,83,80,0.4)"),(30,"rgba(38,166,154,0.4)"),
                         (50,"rgba(128,128,128,0.2)")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=clr, line_width=1, row=3, col=1)
    if "stochrsi_k" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["stochrsi_k"], name="Stoch K",
            line=dict(color="#3498db", width=1, dash="dot")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["stochrsi_d"], name="Stoch D",
            line=dict(color="#f39c12", width=1, dash="dot")), row=3, col=1)

    # ── MACD ──────────────────────────────────────────────────────────────
    if "macd" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
            line=dict(color="#3498db", width=1.4)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
            line=dict(color="#ef5350", width=1.4)), row=4, col=1)
        hist_c = ["#26a69a" if v >= 0 else "#ef5350"
                  for v in df.get("macd_hist", pd.Series())]
        fig.add_trace(go.Bar(x=df.index, y=df.get("macd_hist", pd.Series()),
            name="Hist", marker_color=hist_c, opacity=0.65), row=4, col=1)

    # ── MFI / CMF ────────────────────────────────────────────────────────
    if "mfi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["mfi"], name="MFI",
            line=dict(color="#9b59b6", width=1.4)), row=5, col=1)
        for lvl, clr in [(80,"rgba(239,83,80,0.4)"),(20,"rgba(38,166,154,0.4)")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=clr, line_width=1, row=5, col=1)
    if "cmf" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["cmf"] * 100, name="CMF×100",
            line=dict(color="#1abc9c", width=1, dash="dot")), row=5, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="rgba(128,128,128,0.25)",
                      line_width=1, row=5, col=1)

    chart_xmin = df.index[-min(200, len(df))]
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis5=dict(range=[chart_xmin, df.index[-1]]),
    )
    for i in range(1, 6):
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", row=i, col=1)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", row=i, col=1)
    st.plotly_chart(fig, width="stretch")

# ── Tab 3: Smart Money ────────────────────────────────────────────────────────

def render_smart_money(df: pd.DataFrame, smc: dict, symbol: str):
    if not smc:
        st.warning("Insufficient data for SMC analysis (need ≥ 20 candles).")
        return

    st.markdown("### 💰 Smart Money Concepts (SMC / ICT)")

    pd_zone = smc.get("premium_discount", {})
    zone    = pd_zone.get("current_zone", "N/A")
    zone_c  = "#26a69a" if zone == "Discount" else ("#ef5350" if zone == "Premium" else "#f39c12")
    bull_fvg = smc.get("bull_fvg", [])
    bear_fvg = smc.get("bear_fvg", [])
    bull_ob  = smc.get("bull_ob", [])
    bear_ob  = smc.get("bear_ob", [])
    bos_bull = smc.get("bos_bull", [])
    bos_bear = smc.get("bos_bear", [])
    choch_b  = smc.get("choch_bull", [])
    choch_br = smc.get("choch_bear", [])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Zone",         zone,         delta=None)
    m2.metric("Bullish FVG",  len(bull_fvg), delta=None)
    m3.metric("Bearish FVG",  len(bear_fvg), delta=None)
    m4.metric("Bullish OB",   len(bull_ob),  delta=None)
    m5.metric("Bearish OB",   len(bear_ob),  delta=None)

    st.markdown(
        f"<div class='terminal-card'>"
        f"<b>Price Zone:</b> <span style='color:{zone_c};font-size:1.1em'><b>{zone}</b></span> &nbsp;|&nbsp; "
        f"BOS Bull: <b style='color:#26a69a'>{len(bos_bull)}</b> &nbsp;|&nbsp; "
        f"BOS Bear: <b style='color:#ef5350'>{len(bos_bear)}</b> &nbsp;|&nbsp; "
        f"CHoCH Bull: <b style='color:#26a69a'>{len(choch_b)}</b> &nbsp;|&nbsp; "
        f"CHoCH Bear: <b style='color:#ef5350'>{len(choch_br)}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── SMC Chart ──────────────────────────────────────────────────────────
    recent = df.tail(200)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent["open"], high=recent["high"],
        low=recent["low"], close=recent["close"],
        name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ))

    for fvg in bull_fvg[-4:]:
        fig.add_shape(type="rect",
            x0=fvg["time"], x1=recent.index[-1],
            y0=fvg["bottom"], y1=fvg["top"],
            fillcolor="rgba(38,166,154,0.12)", line_color="rgba(38,166,154,0.4)",
            line_width=1)
        fig.add_annotation(x=fvg["time"], y=fvg["mid"], text=f"FVG↑",
            font=dict(color="#26a69a", size=9), showarrow=False, xanchor="left")

    for fvg in bear_fvg[-4:]:
        fig.add_shape(type="rect",
            x0=fvg["time"], x1=recent.index[-1],
            y0=fvg["bottom"], y1=fvg["top"],
            fillcolor="rgba(239,83,80,0.12)", line_color="rgba(239,83,80,0.4)",
            line_width=1)
        fig.add_annotation(x=fvg["time"], y=fvg["mid"], text=f"FVG↓",
            font=dict(color="#ef5350", size=9), showarrow=False, xanchor="left")

    for ob in bull_ob[-3:]:
        fig.add_shape(type="rect",
            x0=ob["time"], x1=recent.index[-1],
            y0=ob["bottom"], y1=ob["top"],
            fillcolor="rgba(38,166,154,0.18)", line_color="rgba(38,166,154,0.7)",
            line_width=1, line_dash="dot")
        fig.add_annotation(x=ob["time"], y=(ob["top"]+ob["bottom"])/2, text="OB+",
            font=dict(color="#26a69a", size=9), showarrow=False, xanchor="left")

    for ob in bear_ob[-3:]:
        fig.add_shape(type="rect",
            x0=ob["time"], x1=recent.index[-1],
            y0=ob["bottom"], y1=ob["top"],
            fillcolor="rgba(239,83,80,0.18)", line_color="rgba(239,83,80,0.7)",
            line_width=1, line_dash="dot")
        fig.add_annotation(x=ob["time"], y=(ob["top"]+ob["bottom"])/2, text="OB−",
            font=dict(color="#ef5350", size=9), showarrow=False, xanchor="left")

    for b in bos_bull[-2:]:
        fig.add_hline(y=b["level"], line_dash="dash",
                      line_color="rgba(38,166,154,0.6)", line_width=1.5,
                      annotation_text="BOS ↑", annotation_position="right")
    for b in bos_bear[-2:]:
        fig.add_hline(y=b["level"], line_dash="dash",
                      line_color="rgba(239,83,80,0.6)", line_width=1.5,
                      annotation_text="BOS ↓", annotation_position="right")
    for c in choch_b[-1:]:
        fig.add_hline(y=c["level"], line_dash="dot",
                      line_color="rgba(38,166,154,0.9)", line_width=2,
                      annotation_text="CHoCH ↑", annotation_position="right")
    for c in choch_br[-1:]:
        fig.add_hline(y=c["level"], line_dash="dot",
                      line_color="rgba(239,83,80,0.9)", line_width=2,
                      annotation_text="CHoCH ↓", annotation_position="right")

    if pd_zone:
        fig.add_hrect(y0=pd_zone.get("equilibrium", 0), y1=pd_zone.get("range_high", 0),
                      fillcolor="rgba(239,83,80,0.05)", line_width=0,
                      annotation_text="Premium", annotation_position="top right")
        fig.add_hrect(y0=pd_zone.get("range_low", 0), y1=pd_zone.get("equilibrium", 0),
                      fillcolor="rgba(38,166,154,0.05)", line_width=0,
                      annotation_text="Discount", annotation_position="bottom right")
        fig.add_hline(y=pd_zone.get("equilibrium", 0), line_dash="dot",
                      line_color="rgba(241,196,15,0.6)", line_width=1,
                      annotation_text="EQ", annotation_position="right")

    fig.update_layout(
        height=580, xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
    st.plotly_chart(fig, width="stretch")

    # ── Detail tables ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Fair Value Gaps")
        fvg_rows = []
        for f in bull_fvg[-5:]:
            fvg_rows.append({"Type":"Bullish ↑","Top":fmt_price(f["top"],symbol),
                "Bottom":fmt_price(f["bottom"],symbol),"Gap%":f"{f['gap_pct']:.3f}%"})
        for f in bear_fvg[-5:]:
            fvg_rows.append({"Type":"Bearish ↓","Top":fmt_price(f["top"],symbol),
                "Bottom":fmt_price(f["bottom"],symbol),"Gap%":f"{f['gap_pct']:.3f}%"})
        if fvg_rows:
            st.dataframe(pd.DataFrame(fvg_rows), width="stretch", hide_index=True)
        else:
            st.info("No active FVGs detected.")

    with col2:
        st.markdown("#### Order Blocks")
        ob_rows = []
        for o in bull_ob[-4:]:
            ob_rows.append({"Type":"Bullish OB","Top":fmt_price(o["top"],symbol),
                "Bottom":fmt_price(o["bottom"],symbol)})
        for o in bear_ob[-4:]:
            ob_rows.append({"Type":"Bearish OB","Top":fmt_price(o["top"],symbol),
                "Bottom":fmt_price(o["bottom"],symbol)})
        if ob_rows:
            st.dataframe(pd.DataFrame(ob_rows), width="stretch", hide_index=True)
        else:
            st.info("No active Order Blocks detected.")

    # Liquidity zones
    liq = smc.get("equal_highs_above", []) + smc.get("equal_lows_below", [])
    if liq:
        st.markdown("#### Liquidity Zones (Equal Highs / Equal Lows)")
        liq_rows = [{"Type": "EQ High" if l.get("level",0) > df["close"].iloc[-1] else "EQ Low",
                     "Level": fmt_price(l["level"], symbol),
                     "Touches": l.get("touches", 0)} for l in liq[:8]]
        st.dataframe(pd.DataFrame(liq_rows), width="stretch", hide_index=True)

    # Supply/Demand
    supply = smc.get("supply_zones", [])
    demand = smc.get("demand_zones", [])
    if supply or demand:
        st.markdown("#### Supply & Demand Zones")
        sd_rows = []
        for z in demand[:4]:
            sd_rows.append({"Type":"Demand 🟢","Top":fmt_price(z["top"],symbol),
                "Bottom":fmt_price(z["bottom"],symbol)})
        for z in supply[:4]:
            sd_rows.append({"Type":"Supply 🔴","Top":fmt_price(z["top"],symbol),
                "Bottom":fmt_price(z["bottom"],symbol)})
        if sd_rows:
            st.dataframe(pd.DataFrame(sd_rows), width="stretch", hide_index=True)


# ── Tab 4: Order Book ─────────────────────────────────────────────────────────

def render_orderbook(ob: dict, symbol: str):
    if not ob:
        st.error("Order book unavailable.")
        return

    src = ob.get("source", "")
    if src == "synthetic":
        st.caption("⚠️ Showing synthetic order book (Binance rate-limited)")

    st.markdown(f"### 📖 Live Order Book — {symbol}")

    # ── Key metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Best Bid",    fmt_price(ob["best_bid"], symbol))
    m2.metric("Best Ask",    fmt_price(ob["best_ask"], symbol))
    m3.metric("Spread",      fmt_price(ob["spread"], symbol),
              f"{ob['spread_pct']:.4f}%")
    m4.metric("Buy Pressure",  f"{ob['buy_pct']:.1f}%")
    m5.metric("Sell Pressure", f"{ob['sell_pct']:.1f}%")
    imb = ob["imbalance"]
    m6.metric("Imbalance",   f"{imb:+.3f}",
              "Bid dominant" if imb > 0 else "Ask dominant")

    # ── Bid / Ask tables side by side ──────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🟢 Top Bids")
        bids_df = pd.DataFrame(ob["bids"]).rename(
            columns={"price":"Price","size":"Size","cumulative":"Cumulative","value":"Value ($)"})
        bids_df["Price"]      = bids_df["Price"].apply(lambda x: fmt_price(x, symbol))
        bids_df["Size"]       = bids_df["Size"].round(4)
        bids_df["Cumulative"] = bids_df["Cumulative"].round(4)
        bids_df["Value ($)"]  = bids_df["Value ($)"].apply(lambda x: f"${x:,.1f}")
        st.dataframe(bids_df[["Price","Size","Cumulative","Value ($)"]],
                     width="stretch", hide_index=True)

    with col2:
        st.markdown("##### 🔴 Top Asks")
        asks_df = pd.DataFrame(ob["asks"]).rename(
            columns={"price":"Price","size":"Size","cumulative":"Cumulative","value":"Value ($)"})
        asks_df["Price"]      = asks_df["Price"].apply(lambda x: fmt_price(x, symbol))
        asks_df["Size"]       = asks_df["Size"].round(4)
        asks_df["Cumulative"] = asks_df["Cumulative"].round(4)
        asks_df["Value ($)"]  = asks_df["Value ($)"].apply(lambda x: f"${x:,.1f}")
        st.dataframe(asks_df[["Price","Size","Cumulative","Value ($)"]],
                     width="stretch", hide_index=True)

    # ── Depth chart ────────────────────────────────────────────────────────
    st.markdown("##### 📊 Depth Heatmap")
    bids_list = ob["bids"]
    asks_list = ob["asks"]
    bid_prices = [b["price"] for b in bids_list]
    ask_prices = [a["price"] for a in asks_list]
    bid_cum    = [b["cumulative"] for b in bids_list]
    ask_cum    = [a["cumulative"] for a in asks_list]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bid_prices, y=bid_cum, name="Bid Depth",
        fill="tozeroy", fillcolor="rgba(38,166,154,0.3)",
        line=dict(color="#26a69a", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=ask_prices, y=ask_cum, name="Ask Depth",
        fill="tozeroy", fillcolor="rgba(239,83,80,0.3)",
        line=dict(color="#ef5350", width=2),
    ))
    fig.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Price", yaxis_title="Cumulative Volume",
    )
    st.plotly_chart(fig, width="stretch")

    # ── Buy/Sell pressure bar ──────────────────────────────────────────────
    st.markdown("##### ⚖️ Buy / Sell Pressure")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=["Buy Pressure"],  y=[ob["buy_pct"]],
        marker_color="#26a69a", name="Bids"))
    fig2.add_trace(go.Bar(x=["Sell Pressure"], y=[ob["sell_pct"]],
        marker_color="#ef5350", name="Asks"))
    fig2.update_layout(height=180, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
        showlegend=False, yaxis=dict(range=[0,100], ticksuffix="%"))
    st.plotly_chart(fig2, width="stretch")

    st.metric("Cumulative Delta",
              f"{ob['cum_delta']:+,.4f}",
              "Net buying" if ob["cum_delta"] > 0 else "Net selling")

# ── Tab 5: Multi-Timeframe ────────────────────────────────────────────────────

def render_mtf(mtf: dict, symbol: str):
    st.markdown(f"### ⏰ Multi-Timeframe Analysis — {symbol}")

    overall = mtf.get("_overall", {})
    ov = overall.get("verdict", "N/A")
    ov_c = overall.get("color", "#888")
    avg  = overall.get("avg_score", 0)

    st.markdown(
        f"<div class='terminal-card' style='text-align:center;padding:18px'>"
        f"<div class='metric-label'>Overall MTF Alignment</div>"
        f"<div style='font-size:2em;font-weight:900;color:{ov_c}'>{ov}</div>"
        f"<div style='color:#8b949e'>Avg score: {avg:+.2f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Timeframe Breakdown")
    tfs = [tf for tf in MTF_TIMEFRAMES if tf in mtf]
    cols = st.columns(len(tfs))

    for col, tf in zip(cols, tfs):
        d = mtf[tf]
        verdict = d.get("verdict", "N/A")
        color   = d.get("color", "#888")
        score   = d.get("score", 0)
        det     = d.get("details", {})
        ind     = d.get("indicators", {})

        with col:
            st.markdown(
                f"<div class='terminal-card' style='text-align:center'>"
                f"<div class='metric-label'>{MTF_LABELS.get(tf, tf)}</div>"
                f"<div style='color:{color};font-size:1.2em;font-weight:700'>{verdict}</div>"
                f"<div style='color:#8b949e;font-size:.8em'>Score: {score:+d}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Details", expanded=False):
                for k, v in det.items():
                    st.caption(f"**{k.upper()}**: {v}")
                if ind:
                    st.caption(f"RSI: {ind.get('rsi',0):.1f}")
                    st.caption(f"EMA 9/21: {'↑' if ind.get('ema_9',0) > ind.get('ema_21',0) else '↓'}")
                    macd = ind.get("macd", 0)
                    sig  = ind.get("macd_signal", 0)
                    st.caption(f"MACD: {'Bullish' if macd > sig else 'Bearish'}")

    # ── Alignment heatmap ──────────────────────────────────────────────────
    st.markdown("#### Alignment Matrix")
    metrics = ["rsi", "macd", "ema_9_21", "ema"]
    metric_labels = {"rsi": "RSI", "macd": "MACD", "ema_9_21": "EMA 9/21", "ema": "EMA 50/200"}
    heat_data = []
    heat_colors = []
    for tf in tfs:
        d = mtf[tf]
        row_d, row_c = [], []
        for m in metrics:
            v = d.get("details", {}).get(m, "N/A")
            row_d.append(v)
            if "Bull" in v or "Full Bull" in v or "Oversold" in v or "Leaning Bull" in v:
                row_c.append("rgba(38,166,154,0.3)")
            elif "Bear" in v or "Full Bear" in v or "Overbought" in v or "Leaning Bear" in v:
                row_c.append("rgba(239,83,80,0.3)")
            else:
                row_c.append("rgba(128,128,128,0.15)")
        heat_data.append(row_d)
        heat_colors.append(row_c)

    fig = go.Figure(data=go.Table(
        header=dict(
            values=["Timeframe"] + [metric_labels[m] for m in metrics],
            fill_color="#0d1117",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[[MTF_LABELS.get(tf, tf) for tf in tfs]]
                   + [[heat_data[i][j] for i in range(len(tfs))] for j in range(len(metrics))],
            fill_color=["#0d1117"] + [[heat_colors[i][j] for i in range(len(tfs))] for j in range(len(metrics))],
            font=dict(color="white", size=11),
            align="center",
            height=28,
        ),
    ))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")


# ── Tab 6: AI Signals ─────────────────────────────────────────────────────────

def render_ai_signals(ind, adv, smc, mtf, ob, sentiment, fg, signal_result, ml_result,
                      risk, symbol, cfg):
    st.markdown("### 🤖 AI Signal Engine")

    sig  = signal_result["signal"]
    conf = signal_result["confidence"]
    sc   = signal_result["score"]
    bull = signal_result.get("bull_signals", 0)
    bear = signal_result.get("bear_signals", 0)
    norm = signal_result.get("normalized_score", 0)
    color = signal_color(sig)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='terminal-card' style='text-align:center;padding:20px'>"
            f"<div class='metric-label'>Institutional Signal</div>"
            f"<div style='font-size:3em;font-weight:900;color:{color}'>{sig}</div>"
            f"<div style='color:{color}'>{conf*100:.1f}% confidence</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(conf)
    with c2:
        st.markdown("**Signal Components**")
        st.metric("Raw Score",       f"{sc:+.2f}")
        st.metric("Normalized",      f"{norm:+.3f}")
        st.metric("Bull Signals",    bull)
        st.metric("Bear Signals",    bear)
    with c3:
        st.markdown("**Trade Levels**")
        close = ind["close"]
        st.markdown(f"**Entry:** `{fmt_price(close, symbol)}`")
        st.markdown(f"**Stop Loss:** `{fmt_price(risk['stop_loss'], symbol)}`"
                    f" `(-{risk['sl_pct']:.1f}%)`")
        st.markdown(f"**Take Profit:** `{fmt_price(risk['take_profit'], symbol)}`"
                    f" `(+{risk['tp_pct']:.1f}%)`")
        st.markdown(f"**R/R:** `1:{risk['risk_reward']:.1f}`")
        pos = risk["position_size"]
        st.markdown(f"**Position:** `${pos['position_value']:,.2f}` "
                    f"({pos['position_pct']:.1f}%)")
    with c4:
        st.markdown("**Supporting Data**")
        overall = mtf.get("_overall", {})
        mtf_v = overall.get("verdict", "N/A")
        mtf_c = overall.get("color", "#888")
        st.markdown(f"MTF: <span style='color:{mtf_c}'><b>{mtf_v}</b></span>",
                    unsafe_allow_html=True)
        ob_buy = ob.get("buy_pct", 50)
        ob_c   = "#26a69a" if ob_buy > 55 else ("#ef5350" if ob_buy < 45 else "#f39c12")
        st.markdown(f"Order Book: <span style='color:{ob_c}'><b>{ob_buy:.1f}% Buy</b></span>",
                    unsafe_allow_html=True)
        sent_ov = sentiment.get("overall", "neutral")
        sent_c  = sentiment_color(sent_ov)
        st.markdown(f"Sentiment: <span style='color:{sent_c}'><b>{sent_ov.upper()}</b></span>",
                    unsafe_allow_html=True)
        fg_val = fg.get("value", 50)
        fg_c   = get_fg_color(fg_val)
        st.markdown(f"Fear & Greed: <span style='color:{fg_c}'><b>{fg_val}</b></span>",
                    unsafe_allow_html=True)
        pd_z = smc.get("premium_discount", {}).get("current_zone", "N/A") if smc else "N/A"
        st.markdown(f"SMC Zone: **{pd_z}**")

    st.divider()
    st.markdown("#### Signal Reasoning (top factors by weight)")
    reasons = signal_result.get("reasons", [])
    for i, r in enumerate(reasons[:12]):
        icon = "🟢" if any(w in r.lower() for w in
                           ["bull","oversold","above","positive","discount","dominant bid","strong buy"]) \
               else "🔴" if any(w in r.lower() for w in
                                ["bear","overbought","below","negative","premium","dominant ask","strong sell"]) \
               else "⚪"
        st.caption(f"{icon} {r}")

    st.divider()
    st.subheader("🤖 ML Predictions")
    if ml_result.get("error"):
        st.warning(f"ML: {ml_result['error']}")
    else:
        direction = ml_result.get("direction", "?")
        prob      = ml_result.get("combined_probability", 0.5)
        dir_c     = "#26a69a" if direction == "UP" else "#ef5350"
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.markdown(
            f"<div class='terminal-card' style='text-align:center'>"
            f"<div class='metric-label'>ML Consensus</div>"
            f"<div style='font-size:1.8em;color:{dir_c};font-weight:700'>"
            f"{'↑' if direction=='UP' else '↓'} {direction}</div></div>",
            unsafe_allow_html=True)
        mc2.metric("Up Probability", f"{prob*100:.1f}%")
        rf  = ml_result.get("rf", {}) or {}
        rfm = rf.get("meta", {}) or {}
        mc3.metric("RF Accuracy", f"{rfm.get('test_accuracy', rfm.get('train_accuracy',0))*100:.1f}%")
        xgb  = ml_result.get("xgb", {}) or {}
        xgbm = xgb.get("meta", {}) or {}
        mc4.metric("XGB Accuracy", f"{xgbm.get('test_accuracy', xgbm.get('train_accuracy',0))*100:.1f}%")

        fi = ml_result.get("feature_importance", {})
        if fi:
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"]).sort_values("Importance")
            fig = px.bar(fi_df.tail(10), x="Importance", y="Feature", orientation="h",
                         title="Top Feature Importances", color="Importance",
                         color_continuous_scale="teal")
            fig.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, width="stretch")


# ── Tab 7: Backtest ───────────────────────────────────────────────────────────

def render_backtest(df, cfg, symbol):
    st.subheader("🔬 Strategy Backtester")
    bc1, bc2 = st.columns(2)
    with bc1:
        bt_sl  = st.slider("Stop Loss %",   0.5, 10.0, cfg["stop_loss_pct"]*100,  0.5) / 100
        bt_tp  = st.slider("Take Profit %", 1.0, 20.0, cfg["take_profit_pct"]*100, 0.5) / 100
    with bc2:
        bt_cap = st.number_input("Starting Capital ($)", 5.0, 1_000_000.0, float(cfg["capital"]), 1.0, format="%.2f")
        if bt_cap < 5:
            st.error("⚠️ Minimum capital is $5.00")
            bt_cap = 5.0
        bt_pos = st.slider("Position Size %", 5, 50, int(cfg["bt_pos_size"]*100), 5) / 100

    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Running backtest…"):
            bt_r = run_backtest(df, initial_capital=bt_cap,
                                stop_loss_pct=bt_sl, take_profit_pct=bt_tp,
                                position_size_pct=bt_pos)
        st.session_state["bt_result"] = bt_r
        st.success("Done!")

    if "bt_result" not in st.session_state:
        st.info("Configure parameters above and click **Run Backtest**.")
        return

    m = st.session_state["bt_result"]["metrics"]
    if m["total_trades"] == 0:
        st.warning("No trades generated. Try different SL/TP or more candles.")
        return

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Return", f"${m['total_return']:,.2f}", f"{m['total_return_pct']:+.2f}%",
              delta_color="normal" if m["total_return_pct"] >= 0 else "inverse")
    r2.metric("Win Rate",     f"{m['win_rate']:.1f}%",
              f"{m['winning_trades']}W / {m['losing_trades']}L")
    r3.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.3f}")
    r4.metric("Max Drawdown", f"{m['max_drawdown']:.2f}%", delta_color="inverse")

    r5, r6, r7, r8 = st.columns(4)
    r5.metric("Total Trades",  m["total_trades"])
    r6.metric("Avg Win",       f"${m['avg_win']:,.2f}")
    r7.metric("Avg Loss",      f"${m['avg_loss']:,.2f}")
    r8.metric("Profit Factor", f"{m['profit_factor']:.3f}")

    eq = st.session_state["bt_result"]["equity_curve"].reset_index()
    if len(eq):
        fig = go.Figure(go.Scatter(x=eq["timestamp"], y=eq["equity"],
            fill="tozeroy", fillcolor="rgba(38,166,154,0.10)",
            line=dict(color="#26a69a", width=2), name="Portfolio"))
        fig.update_layout(height=260, title="Equity Curve",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickprefix="$", gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")

    trades = st.session_state["bt_result"]["trades"]
    if trades:
        td = pd.DataFrame(trades)
        td["pnl_pct"] = (td["pnl_pct"] * 100).round(2)
        td["pnl"]     = td["pnl"].round(2)
        td["entry"]   = td["entry"].round(6)
        td["exit"]    = td["exit"].round(6)
        st.caption("**Recent Trades**")
        st.dataframe(td[["timestamp","entry","exit","pnl","pnl_pct","exit_reason"]].tail(20),
                     width="stretch", hide_index=True)


# ── Tab 8: Portfolio (Paper Trading) ──────────────────────────────────────────

def render_portfolio(signal_result, ind, risk, symbol, capital):
    st.subheader("📋 Paper Trading Portfolio")
    if "paper_trades" not in st.session_state:
        st.session_state.paper_trades = []

    sig   = signal_result["signal"]
    close = ind["close"]
    pos   = risk["position_size"]
    ts    = now_str("%Y-%m-%d %H:%M:%S WIB")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Signal:** {sig_badge(sig)} at `{fmt_price(close, symbol)}`",
                    unsafe_allow_html=True)
        st.markdown(f"**Capital:** `${capital:,.2f}`")
        if sig == SIGNAL_BUY:
            if st.button("📥 Log BUY Trade", type="primary"):
                st.session_state.paper_trades.append({
                    "time": ts, "symbol": symbol, "signal": "BUY",
                    "price": close, "size": pos["position_value"],
                    "units": pos["units"], "SL": risk["stop_loss"],
                    "TP": risk["take_profit"], "conf": signal_result["confidence"],
                })
                st.success(f"Logged BUY {pos['units']:.6f} {symbol.split('/')[0]} @ {fmt_price(close, symbol)}")
        elif sig == SIGNAL_SELL:
            if st.button("📤 Log SELL Trade", type="secondary"):
                st.session_state.paper_trades.append({
                    "time": ts, "symbol": symbol, "signal": "SELL",
                    "price": close, "size": pos["position_value"],
                    "units": pos["units"], "SL": risk["stop_loss"],
                    "TP": risk["take_profit"], "conf": signal_result["confidence"],
                })
        else:
            st.info("Signal is HOLD — no action.")
    with col2:
        if st.session_state.paper_trades:
            if st.button("🗑️ Clear Log"):
                st.session_state.paper_trades = []
                st.rerun()

    if st.session_state.paper_trades:
        st.dataframe(pd.DataFrame(st.session_state.paper_trades),
                     width="stretch", hide_index=True)

    st.divider()
    st.subheader("Position Sizing Calculator")
    r1, r2 = st.columns(2)
    with r1:
        custom_cap   = st.number_input("Capital ($)", 5.0, 1_000_000.0, float(capital), 1.0, format="%.2f", key="rc")
        if custom_cap < 5:
            st.error("⚠️ Minimum capital is $5.00")
            custom_cap = 5.0
        custom_entry = st.number_input("Entry Price", 0.000001, 1_000_000.0, float(close), key="re",
                                       format="%.6f")
        custom_sl    = st.number_input("Stop Loss", 0.000001, 1_000_000.0, float(risk["stop_loss"]),
                                       key="rsl", format="%.6f")
    with r2:
        custom_tp    = st.number_input("Take Profit", 0.000001, 1_000_000.0, float(risk["take_profit"]),
                                       key="rtp", format="%.6f")
        custom_risk  = st.slider("Risk per Trade %", 0.1, 5.0, 1.0, 0.1, key="rrp") / 100
        custom_maxp  = st.slider("Max Position %",   5,   50,  25,  5,   key="rmp") / 100

    from src.risk.risk_manager import calculate_position_size
    sz     = calculate_position_size(custom_cap, custom_entry, custom_sl, custom_risk, custom_maxp)
    rr_c   = abs(custom_tp - custom_entry) / abs(custom_entry - custom_sl) if abs(custom_entry - custom_sl) > 0 else 0
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Units",          f"{sz['units']:.6f}")
    q2.metric("Position Value", f"${sz['position_value']:,.2f}")
    q3.metric("Risk Amount",    f"${sz['risk_amount']:,.2f}")
    q4.metric("Risk/Reward",    f"1:{rr_c:.2f}")


# ── Fear & Greed gauge ────────────────────────────────────────────────────────

def render_fear_greed_gauge(fg: dict):
    val   = fg.get("value", 50)
    cl    = fg.get("classification", "Neutral")
    fg_c  = get_fg_color(val)
    emoji = get_fg_emoji(cl)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        domain={"x": [0,1], "y": [0,1]},
        title={"text": f"{emoji} {cl}", "font": {"size": 15, "color": "white"}},
        number={"font": {"color": fg_c}},
        gauge={
            "axis": {"range": [0,100], "tickwidth": 1, "tickcolor": "#555"},
            "bar":  {"color": fg_c, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,25],  "color": "rgba(239,83,80,0.2)"},
                {"range": [25,45], "color": "rgba(230,126,34,0.15)"},
                {"range": [45,55], "color": "rgba(241,196,15,0.15)"},
                {"range": [55,75], "color": "rgba(38,166,154,0.15)"},
                {"range": [75,100],"color": "rgba(38,166,154,0.25)"},
            ],
            "threshold": {"line": {"color": fg_c, "width": 4},
                          "thickness": 0.75, "value": val},
        },
    ))
    fig.update_layout(
        height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=30, b=5), font={"color": "white"},
    )
    st.plotly_chart(fig, width="stretch")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    watchlist_symbols, cg_data = load_watchlist()
    if not watchlist_symbols:
        watchlist_symbols = FALLBACK_SYMBOLS[:]

    cfg = render_sidebar(watchlist_symbols)

    if cfg["auto_refresh"]:
        count = st_autorefresh(interval=cfg["refresh_ms"], key="live_refresh")
    else:
        count = 0

    st.markdown(
        f"<h2 style='margin-bottom:0'>📈 SuperSignal</h2>"
        f"<div style='color:#8b949e;font-size:.85em'>"
        f"Top {len(watchlist_symbols)} coins by MCap · Binance + CoinGecko + Fear&amp;Greed · "
        f"Paper trading only 🔒 · {now_str('%H:%M:%S')} WIB"
        + (f" · Auto-refresh {cfg['refresh_option']} (#{count})" if cfg["auto_refresh"] else "")
        + f"</div>",
        unsafe_allow_html=True,
    )

    symbol    = cfg["symbol"]
    timeframe = cfg["timeframe"]
    symbols_key = "|".join(watchlist_symbols)

    # ── Load core data ──────────────────────────────────────────────────────
    with st.spinner("Loading data…"):
        tickers = load_tickers_for_watchlist(symbols_key)
        fg      = load_fear_greed()
        df      = load_full_data(symbol, timeframe, cfg["limit"])

    ind = get_current_indicator_values(df)
    ind["bb_width"] = float(df["bb_width"].iloc[-1]) if "bb_width" in df.columns else 0.0
    adv = get_advanced_indicator_values(df)
    sr  = find_support_resistance(df)

    # ── Watchlist scanner (basic indicators only) ───────────────────────────
    ind_map    = {}
    signal_map = {}
    with st.spinner(f"Scanning {len(watchlist_symbols)} coins…"):
        for sym in watchlist_symbols:
            try:
                dft  = load_watchlist_data(sym, "1h")
                i    = get_current_indicator_values(dft)
                i["bb_width"] = float(dft["bb_width"].iloc[-1]) if "bb_width" in dft.columns else 0.0
                ind_map[sym]    = i
                signal_map[sym] = generate_signal(i, 0.0)
            except Exception:
                ind_map[sym]    = {}
                signal_map[sym] = {"signal": "HOLD", "confidence": 0.5, "reasons": []}

    # ── Load heavier per-selected-coin data in parallel ─────────────────────
    smc = load_smc(symbol, timeframe, cfg["limit"])
    ob  = load_orderbook(symbol)
    mtf = fetch_mtf_analysis(symbol)

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview",
        "📈 Technical",
        "💰 Smart Money",
        "📖 Order Book",
        "⏰ Multi-TF",
        "🤖 AI Signals",
        "🔬 Backtest",
        "📋 Portfolio",
    ])

    with tab1:
        render_overview(tickers, cg_data, watchlist_symbols, ind_map, signal_map, fg)

    with tab2:
        render_technical(df, ind, adv, symbol, sr, cfg, fg)

    with tab3:
        render_smart_money(df, smc, symbol)

    with tab4:
        render_orderbook(ob, symbol)

    with tab5:
        render_mtf(mtf, symbol)

    with tab6:
        with st.spinner("Loading sentiment…"):
            sentiment = get_news_sentiment(symbol)
        sentiment_score = sentiment.get("score", 0.0)
        fg_val = fg.get("value", 50)
        mtf_overall = mtf.get("_overall", {})

        signal_result = generate_signal(
            ind, sentiment_score,
            advanced=adv, smc=smc, mtf_overall=mtf_overall,
            orderbook=ob, fg_value=fg_val,
        )
        risk = assess_risk(
            cfg["capital"], ind["close"],
            ind.get("atr", ind["close"] * 0.02),
            signal_result["confidence"],
            cfg["risk_tolerance"],
        )
        risk["risk_reward"] = cfg["risk_reward"]

        df_hash  = str(hash(str(df.index[-1]) + symbol + timeframe))
        df_json  = df.reset_index().to_json(date_format="iso")
        with st.spinner("Training ML models…"):
            ml_result = train_and_predict(df_hash, df_json, symbol)

        render_ai_signals(ind, adv, smc, mtf, ob, sentiment, fg,
                          signal_result, ml_result, risk, symbol, cfg)

    with tab7:
        render_backtest(df, cfg, symbol)

    with tab8:
        if "signal_result" not in dir():
            sentiment = get_news_sentiment(symbol)
            sentiment_score = sentiment.get("score", 0.0)
            signal_result = generate_signal(ind, sentiment_score, advanced=adv, smc=smc)
            risk = assess_risk(cfg["capital"], ind["close"],
                               ind.get("atr", ind["close"]*0.02),
                               signal_result["confidence"], cfg["risk_tolerance"])
            risk["risk_reward"] = cfg["risk_reward"]
        render_portfolio(signal_result, ind, risk, symbol, cfg["capital"])

    st.caption(
        f"Binance (CCXT) · CoinGecko · alternative.me (Fear&Greed) · "
        f"{now_str('%Y-%m-%d %H:%M WIB')} · Paper Trading Only"
    )


if __name__ == "__main__":
    main()