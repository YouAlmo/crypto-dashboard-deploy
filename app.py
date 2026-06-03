# SuperSignal AI Engine
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
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
THEME_OPTIONS = ["Institutional Dark", "Premium Light"]
THEME_TOKENS = {
    "Institutional Dark": {
        "app_bg": "#05101d",
        "panel_bg": "rgba(9,17,31,0.94)",
        "card_bg": "rgba(10,18,30,0.96)",
        "card_border": "rgba(97,156,255,0.16)",
        "text": "#e2e8f0",
        "muted": "#94a3b8",
        "accent": "#60a5fa",
        "accent_alt": "#26c6da",
        "success": "#26a69a",
        "danger": "#ef5350",
        "warning": "#fbbf24",
        "shadow": "0 24px 60px rgba(10,18,35,0.28)",
        "sidebar": "rgba(8,12,18,0.98)",
        "tab_bg": "rgba(15,23,42,0.95)",
        "tab_active": "rgba(34,70,124,0.95)",
        "tab_border": "rgba(97,156,255,0.18)",
        "heat_bull": "rgba(38,166,154,0.28)",
        "heat_bear": "rgba(239,83,80,0.28)",
    },
    "Premium Light": {
        "app_bg": "#eff5fb",
        "panel_bg": "#f5f8ff",
        "card_bg": "rgba(255,255,255,0.98)",
        "card_border": "rgba(37,99,235,0.15)",
        "text": "#102a43",
        "muted": "#64748b",
        "accent": "#2563eb",
        "accent_alt": "#1d4ed8",
        "success": "#047857",
        "danger": "#b91c1c",
        "warning": "#c2410b",
        "shadow": "0 22px 52px rgba(15,23,42,0.08)",
        "sidebar": "#eef4fb",
        "tab_bg": "#e7f0ff",
        "tab_active": "#ffffff",
        "tab_border": "rgba(37,99,235,0.18)",
        "heat_bull": "rgba(34,197,94,0.18)",
        "heat_bear": "rgba(248,113,113,0.18)",
    },
}

def get_theme_css(theme_name: str) -> str:
    t = THEME_TOKENS.get(theme_name, THEME_TOKENS["Institutional Dark"])
    return f"""
    <style>
    :root {{
      --app-bg: {t['app_bg']};
      --panel-bg: {t['panel_bg']};
      --card-bg: {t['card_bg']};
      --card-border: {t['card_border']};
      --text: {t['text']};
      --muted: {t['muted']};
      --accent: {t['accent']};
      --accent-alt: {t['accent_alt']};
      --success: {t['success']};
      --danger: {t['danger']};
      --warning: {t['warning']};
      --shadow: {t['shadow']};
      --sidebar: {t['sidebar']};
      --tab-bg: {t['tab_bg']};
      --tab-active: {t['tab_active']};
      --tab-border: {t['tab_border']};
      --heat-bull: {t['heat_bull']};
      --heat-bear: {t['heat_bear']};
    }}
    body, .stApp, .block-container, .main {{ background: var(--app-bg) !important; color: var(--text) !important; }}
    section[data-testid="stSidebar"], .sidebar .css-1lcbmhc, .stSidebar {{ background: var(--sidebar) !important; border-right: 1px solid var(--card-border); }}
    .sidebar-block {{ background: var(--panel-bg); border-color: var(--card-border); color: var(--text); padding: 18px 20px; border-radius: 18px; box-shadow: var(--shadow); }}
    .sidebar-block h3 {{ color: var(--text); margin-bottom: 10px; }}
    .sidebar-block p {{ color: var(--muted); margin-bottom: 0; }}
    .stSidebar .element-container {{ background: transparent !important; }}
    .stSidebar .stSelectbox > div > div, .stSidebar .stSlider > div, .stSidebar .stNumberInput > div, .stSidebar .stCheckbox > div, .stSidebar .stRadio > div {{ border-radius: 16px !important; }}
    .dashboard-card, .dashboard-tile, .table-card, .terminal-card, .signal-card {{ background: var(--card-bg); border-color: var(--card-border); box-shadow: var(--shadow); color: var(--text); border-radius: 20px; padding: 18px 20px; }}
    .dashboard-card:hover, .signal-card:hover {{ transform: translateY(-3px); }}
    .dashboard-card {{ border: 1px solid var(--card-border); }}
    .dashboard-tile {{ border: 1px solid var(--card-border); }}
    .table-card {{ border: 1px solid var(--card-border); }}
    .terminal-card {{ border: 1px solid var(--card-border); }}
    .metric-pill {{ color: var(--text); padding: 7px 14px; font-size: 0.78rem; }}
    .metric-pill.buy {{ background: rgba(38,166,154,0.14); color: var(--success); }}
    .metric-pill.sell {{ background: rgba(239,83,80,0.14); color: var(--danger); }}
    .metric-pill.hold {{ background: rgba(251,191,36,0.16); color: var(--warning); }}
    .signal-card.buy {{ box-shadow: 0 20px 52px rgba(38,166,154,0.16); }}
    .signal-card.sell {{ box-shadow: 0 20px 52px rgba(239,83,80,0.16); }}
    .signal-card.hold {{ box-shadow: 0 20px 52px rgba(251,191,36,0.14); }}
    .signal-badge {{ background: var(--accent); color: #fff; }}
    .small-muted {{ color: var(--muted); }}
    .conf-wrap, .dom-wrap {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; }}
    .conf-bar {{ background: rgba(255,255,255,0.12); }}
    .dom-bar {{ background: rgba(255,255,255,0.12); }}
    .risk-badge {{ color: var(--text); }}
    .risk-low {{ background: linear-gradient(90deg, var(--success), #a5f3c9); }}
    .risk-medium {{ background: linear-gradient(90deg, var(--warning), #fde68a); }}
    .risk-high {{ background: linear-gradient(90deg, var(--danger), #fca5a5); }}
    .section-title {{ color: var(--text); font-size: 1.45rem; letter-spacing: -0.02em; margin-bottom: 0.35rem; }}
    .section-subtitle {{ color: var(--muted); margin-bottom: 1.15rem; font-size: 0.95rem; }}
    .stTabs [role="tab"] {{ display: inline-flex !important; align-items: center; justify-content: center; gap: 0.45rem; min-height: 52px; padding: 0.9rem 1rem; margin-right: 8px; border-radius: 14px 14px 0 0; border: 1px solid var(--tab-border); background: var(--tab-bg); color: var(--text); transition: transform .18s ease, box-shadow .18s ease, background .18s ease; font-size: 0.95rem; font-weight: 600; }}
    .stTabs [role="tab"]:hover {{ transform: translateY(-1px); box-shadow: 0 12px 30px rgba(0,0,0,0.08); }}
    .stTabs [role="tab"][aria-selected="true"] {{ background: var(--tab-active); color: var(--text); box-shadow: 0 10px 28px rgba(0,0,0,0.14); border-bottom: 4px solid var(--accent); }}
    .stTabs [role="tab"] span, .stTabs [role="tab"] svg {{ line-height: 1.2; }}
    .signal-item {{ background: rgba(255,255,255,0.08); color: var(--text); }}
    .table-card h5 {{ color: var(--text); }}
    button[aria-label*="Theme"], button[title*="Theme"], [data-testid="stThemeToggle"] {{ display: none !important; }}
    .stButton>button {{ border-radius: 14px; padding: 0.95rem 1rem; font-weight: 700; }}
    .stSidebar .stButton>button {{ width: 100%; }}
    .stSidebar label {{ color: var(--text) !important; font-weight: 600; }}
    .stSidebar .stSelectbox>div, .stSidebar .stSlider>div, .stSidebar .stNumberInput>div {{ background: var(--card-bg) !important; border: 1px solid var(--card-border) !important; box-shadow: inset 0 0 0 rgba(0,0,0,0.02); }}
    .stSidebar .stSlider .expy {{
      background: transparent !important;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 0.6; }}
      50% {{ opacity: 1; }}
    }}
    @media (max-width: 1024px) {{ .stTabs [role="tab"] {{ padding: 0.75rem 0.85rem; min-height: 48px; }} }}
    @media (max-width: 760px) {{ .stTabs [role="tab"] {{ flex: 1 1 auto; margin-right: 6px; min-width: 120px; }} }}
    </style>
    """

def render_theme_css(theme_name: str):
    st.markdown(get_theme_css(theme_name), unsafe_allow_html=True)

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

def render_dashboard_card(title: str, value: str, subtitle: str = "", accent: str = "var(--accent)") -> str:
    return (
        f"<div class='dashboard-card'>"
        f"<div style='font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px'>{title}</div>"
        f"<div style='font-size:2rem;font-weight:800;color:{accent};line-height:1.05;margin-bottom:6px'>{value}</div>"
        f"<div style='font-size:.92rem;color:var(--muted)'>{subtitle}</div>"
        f"</div>"
    )

def render_metric_tile(title: str, value: str, detail: str = "", badge: str = "") -> str:
    badge_html = f"<span class='metric-pill {badge}'>{badge.replace('-', ' ').title()}</span>" if badge else ""
    return (
        f"<div class='dashboard-tile'>"
        f"<h4>{title}{badge_html}</h4>"
        f"<div style='font-size:1.5rem;font-weight:800;line-height:1.1;margin-top:8px;color:var(--text)'>{value}</div>"
        f"<p style='color:var(--muted)'>{detail}</p>"
        f"</div>"
    )

def render_section_header(title: str, subtitle: str = "") -> str:
    description = f"<div class='section-subtitle'>{subtitle}</div>" if subtitle else ""
    return (
        f"<div class='section-title'>{title}</div>"
        f"{description}"
    )

def render_notice_badge(message: str, kind: str = "info") -> None:
    colors = {
        "info": ("#2563eb", "rgba(37,99,235,0.08)"),
        "warning": ("#f59e0b", "rgba(251,146,60,0.12)"),
        "error": ("#ef4444", "rgba(239,68,68,0.12)"),
    }
    fg, bg = colors.get(kind, ("#64748b", "rgba(100,116,139,0.1)"))
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:14px;border:1px solid rgba(255,255,255,0.1);"
        f"background:{bg};color:{fg};font-size:0.95rem;margin-bottom:16px;line-height:1.4;'>"
        f"<strong>{message}</strong></div>",
        unsafe_allow_html=True,
    )

def render_empty_state(message: str = "Data unavailable.", icon: str = "⚠️") -> None:
    st.markdown(
        f"<div style='padding:18px 20px;border-radius:18px;border:1px solid rgba(255,255,255,0.08);"
        f"background:rgba(255,255,255,0.05);color:var(--muted);font-size:0.95rem;'>"
        f"{icon} {message}</div>",
        unsafe_allow_html=True,
    )

def verdict_color(v: str) -> str:
    return {
        "Strong Buy":  "#1a7f37",
        "Buy":         "#26a69a",
        "Hold":        "#f39c12",
        "Sell":        "#ef5350",
        "Strong Sell": "#8b0000",
    }.get(v, "#888")

# ── PERF: Session state & lazy loading ─────────────────────────────────────
if "rendered_tabs" not in st.session_state:
    st.session_state.rendered_tabs = set()

def mark_tab_rendered(tab_name: str):
    st.session_state.rendered_tabs.add(tab_name)

def is_tab_rendered(tab_name: str) -> bool:
    return tab_name in st.session_state.rendered_tabs

def render_skeleton_loader(height: str = "200px", count: int = 1):
    """Lightweight placeholder during data load."""
    for _ in range(count):
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.08);height:{height};border-radius:12px;"
            f"margin-bottom:16px;animation:pulse 1.5s infinite' />\n",
            unsafe_allow_html=True
        )

# ── cache layer ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=45)
def load_watchlist():
    """Top 20 coins - cached 45s."""
    return fetch_top20_markets()

@st.cache_data(ttl=15, show_spinner=False)
def load_tickers_for_watchlist(symbols_key: str) -> dict:
    """Binance tickers - cached 15s."""
    return fetch_tickers_for(symbols_key.split("|"))

@st.cache_data(ttl=10, show_spinner=False)
def load_market_data(symbol: str, timeframe: str, limit: int = 200):
    """Basic + indicators - cached 10s."""
    df = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = add_all_indicators(df)
    return df

@st.cache_data(ttl=10, show_spinner=False)
def load_full_data(symbol: str, timeframe: str, limit: int = 200):
    """Full data with advanced indicators - cached 10s."""
    df = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = add_all_indicators(df)
    df = add_all_advanced_indicators(df)
    return df

@st.cache_data(ttl=10, show_spinner=False)
def load_watchlist_data(symbol: str, timeframe: str, limit: int = 80):
    """Fast watchlist scan - cached 10s."""
    df = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = add_all_indicators(df)
    return df

@st.cache_data(ttl=15)
def load_fear_greed():
    """Fear & Greed - cached 15s."""
    return fetch_fear_greed_index()

@st.cache_data(ttl=45, show_spinner=False)
def load_smc(symbol: str, timeframe: str, limit: int = 200):
    """SMC analysis - cached 45s."""
    limit = min(limit, 220)
    try:
        df = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = add_all_indicators(df)
        return analyze_smc(df)
    except Exception:
        return _default_smc()


def _default_smc() -> dict:
    return {
        "premium_discount": {
            "current_zone": "N/A",
            "equilibrium": 0.0,
            "range_high": 0.0,
            "range_low": 0.0,
        },
        "bos_bull": [], "bos_bear": [],
        "choch_bull": [], "choch_bear": [],
        "swing_highs": [], "swing_lows": [],
        "bull_fvg": [], "bear_fvg": [],
        "supply_zones": [], "demand_zones": [],
        "equal_highs_above": [], "equal_lows_below": [],
    }

@st.cache_data(ttl=10, show_spinner=False)
def load_orderbook(symbol: str):
    """Order book - cached 10s with retry, timeout, and fallback."""
    def fetch_once():
        return fetch_order_book(symbol)

    ob = None
    for attempt in range(2):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_once)
                ob = future.result(timeout=3)
            if ob and isinstance(ob, dict) and "bids" in ob and "asks" in ob:
                return ob
        except Exception:
            ob = None
    return {
        "best_bid": 0.0,
        "best_ask": 0.0,
        "spread": 0.0,
        "spread_pct": 0.0,
        "buy_pct": 50.0,
        "sell_pct": 50.0,
        "imbalance": 0.0,
        "cum_delta": 0.0,
        "bids": [{"price": 0, "size": 0, "cumulative": 0, "value": 0}],
        "asks": [{"price": 0, "size": 0, "cumulative": 0, "value": 0}],
        "source": "synthetic",
    }


def _default_mtf() -> dict:
    def _tf_template():
        return {
            "score": 0,
            "verdict": "N/A",
            "signal": "N/A",
            "color": "#999",
            "details": {"trend": "Unavailable"},
            "indicators": {},
            "momentum": 0,
            "confidence": 0.0,
        }
    results = {tf: _tf_template() for tf in ["1m", "5m", "15m", "1h", "4h"]}
    results["_overall"] = {
        "signal": "N/A",
        "score": 0,
        "alignment": "Neutral",
        "verdict": "N/A",
        "avg_score": 0.0,
        "color": "#999",
        "confidence": 0.0,
        "bullish": 0,
        "bearish": 0,
        "hold": 0,
        "agreement": 0,
        "base_tf": "1h",
    }
    return results

@st.cache_data(ttl=15, show_spinner=False)
def load_mtf_data(symbol: str, base_tf: str = "1h"):
    """Multi-Timeframe analysis - cached 15s with concurrency and safe defaults."""
    tfs = ["1m", "5m", "15m", "1h", "4h"]

    def default_tf(tf: str):
        return {
            "score": 0,
            "verdict": "N/A",
            "signal": "N/A",
            "color": "#999",
            "details": {"trend": "Unavailable"},
            "indicators": {},
            "momentum": 0,
            "confidence": 0.0,
        }

    def normalize_indicator_value(df, col, default=0.0):
        val = df[col].iloc[-1] if col in df.columns else default
        return float(val) if pd.notna(val) else default

    def classify(scores: dict) -> dict:
        score = scores.get("score", 0)
        if score >= 2.0:
            signal = "BUY"
            alignment = "Strong Bullish"
            color = "#1a7f37"
        elif score >= 1.0:
            signal = "BUY"
            alignment = "Bullish"
            color = "#2ecc71"
        elif score <= -2.0:
            signal = "SELL"
            alignment = "Strong Bearish"
            color = "#8b0000"
        elif score <= -1.0:
            signal = "SELL"
            alignment = "Bearish"
            color = "#e74c3c"
        else:
            signal = "HOLD"
            alignment = "Neutral"
            color = "#f59e0b"
        scores["signal"] = signal
        scores["alignment"] = alignment
        scores["color"] = color
        scores["confidence"] = min(max(abs(score) / 2.0, 0.0), 1.0)
        return scores

    def load_tf(tf: str):
        try:
            limit = 150 if tf == "1m" else 250
            df = fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            if df is None or df.empty or len(df) < 2:
                raise ValueError("no data")
            df = add_all_indicators(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            ind = {
                "close":       normalize_indicator_value(df, "close"),
                "rsi":         normalize_indicator_value(df, "rsi", 50),
                "macd":        normalize_indicator_value(df, "macd"),
                "macd_signal": normalize_indicator_value(df, "macd_signal"),
                "ema_9":       normalize_indicator_value(df, "ema_9"),
                "ema_21":      normalize_indicator_value(df, "ema_21"),
                "ema_50":      normalize_indicator_value(df, "ema_50"),
                "ema_200":     normalize_indicator_value(df, "ema_200"),
                "bb_pct":      normalize_indicator_value(df, "bb_pct", 0.5),
            }
            score = 0
            details = {}
            rsi = ind["rsi"]
            if rsi < 30:
                score += 2
                details["rsi"] = "Oversold"
            elif rsi < 45:
                score += 1
                details["rsi"] = "Leaning Bull"
            elif rsi > 70:
                score -= 2
                details["rsi"] = "Overbought"
            elif rsi > 55:
                score -= 1
                details["rsi"] = "Leaning Bear"
            else:
                details["rsi"] = "Neutral"
            macd = ind["macd"]
            macd_sig = ind["macd_signal"]
            macd_score = 1 if macd > macd_sig else -1
            score += macd_score
            details["macd"] = "Bullish" if macd_score > 0 else "Bearish"
            if ind["ema_9"] > ind["ema_21"]:
                score += 1
                details["ema_9_21"] = "9/21 Bullish"
            else:
                score -= 1
                details["ema_9_21"] = "9/21 Bearish"
            if ind["ema_50"] > ind["ema_200"]:
                score += 1
                details["ema"] = "Bullish"
            else:
                score -= 1
                details["ema"] = "Bearish"
            trend_score = 1 if last["close"] >= prev["close"] else -1
            score += trend_score
            details["trend"] = "Uptrend" if trend_score > 0 else "Downtrend"
            motion = float((last["close"] - prev["close"]) / prev["close"] * 100) if prev["close"] else 0
            result = {
                "score": score,
                "verdict": "BUY" if score > 0 else "SELL" if score < 0 else "HOLD",
                "details": details,
                "indicators": ind,
                "momentum": round(motion, 2),
                "confidence": min(max(abs(score) / 4.0, 0.0), 1.0),
            }
            return tf, classify(result)
        except Exception:
            return tf, default_tf(tf)

    results = {tf: default_tf(tf) for tf in tfs}
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(load_tf, tf): tf for tf in tfs}
            for future in as_completed(futures, timeout=15):
                tf = futures.get(future)
                try:
                    tf, data = future.result()
                except Exception:
                    data = default_tf(tf)
                results[tf] = data
    except Exception:
        return _default_mtf()

    if not results:
        results = {tf: default_tf(tf) for tf in tfs}

    valid_scores = [v["score"] for v in results.values() if isinstance(v, dict)]
    avg_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    overall_signal = "HOLD"
    if avg_score >= 1.0:
        overall_signal = "BUY"
    elif avg_score <= -1.0:
        overall_signal = "SELL"
    overall = {
        "signal": overall_signal,
        "score": min(max((avg_score + 2) / 4.0, 0.0), 1.0),
        "alignment": "Bullish" if avg_score > 0 else "Bearish" if avg_score < 0 else "Neutral",
        "verdict": overall_signal,
        "avg_score": avg_score,
        "color": "#2ecc71" if avg_score > 0 else ("#e74c3c" if avg_score < 0 else "#fbbf24"),
        "confidence": min(max(abs(avg_score) / 2.0, 0.0), 1.0),
        "bullish": sum(1 for v in results.values() if v.get("score", 0) > 0),
        "bearish": sum(1 for v in results.values() if v.get("score", 0) < 0),
        "hold": sum(1 for v in results.values() if v.get("score", 0) == 0),
        "agreement": sum(1 for v in results.values() if v.get("signal") == overall_signal),
        "base_tf": base_tf,
    }
    results["_overall"] = overall
    return results

@st.cache_data(ttl=120, show_spinner=False)
def load_news_sentiment(symbol: str):
    """News sentiment - cached 120s (static-like data)."""
    try:
        return get_news_sentiment(symbol)
    except Exception:
        return {"score": 0.0, "sentiment": "neutral"}

# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(watchlist_symbols: list):
    st.sidebar.markdown(
        "<div class='sidebar-block'><h3>SuperSignal</h3>Institutional crypto terminal for advanced traders.</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    theme_default = st.session_state.get("theme", THEME_OPTIONS[0])
    theme = st.sidebar.selectbox(
        "Theme",
        THEME_OPTIONS,
        index=THEME_OPTIONS.index(theme_default) if theme_default in THEME_OPTIONS else 0,
        help="Choose the interface theme for the dashboard.",
    )
    st.session_state.theme = theme

    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.sidebar.subheader("Market")

    symbol_default = st.query_params.get(
        "symbol",
        st.session_state.get("selected_symbol", watchlist_symbols[0])
    )

    if isinstance(symbol_default, list):
        symbol_default = symbol_default[0]
    if "/" not in symbol_default and symbol_default.endswith("USDT"):
        base = symbol_default.replace("USDT", "")
        symbol_default = f"{base}/USDT"

    symbol = st.sidebar.selectbox(
        "Symbol",
        watchlist_symbols,
        index=watchlist_symbols.index(symbol_default) if symbol_default in watchlist_symbols else 0,
    )

    st.session_state.selected_symbol = symbol
    st.query_params["symbol"] = symbol

    timeframe_default = st.query_params.get(
        "timeframe",
        st.session_state.get("selected_timeframe", list(TIMEFRAMES)[0])
    )

    if isinstance(timeframe_default, list):
        timeframe_default = timeframe_default[0]

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        list(TIMEFRAMES),
        index=list(TIMEFRAMES).index(timeframe_default),
    )

    st.session_state.selected_timeframe = timeframe
    st.query_params["timeframe"] = timeframe
    limit     = st.sidebar.slider("Candle Limit", 100, 1000, 500, 50)

    st.sidebar.subheader("Auto-Refresh")
    refresh_option = st.sidebar.select_slider(
        "Interval", options=["Off", "30s", "1m", "5m"], value="Off"
    )
    ms_map    = {"Off": None, "30s": 30_000, "1m": 60_000, "5m": 300_000}
    refresh_ms = ms_map[refresh_option]

    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
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

    # Paper Capital
    capital_default = st.query_params.get("capital", "100")

    if isinstance(capital_default, list):
        capital_default = capital_default[0]

    capital = st.sidebar.number_input(
        "Paper Capital ($)",
        5.0,
        1_000_000.0,
        value=float(capital_default),
        step=1.0,
        format="%.2f"
    )

    st.query_params["capital"] = str(capital)

    # Risk Tolerance
    risk_default = st.query_params.get("risk", "moderate")

    if isinstance(risk_default, list):
        risk_default = risk_default[0]

    risk_tolerance = st.sidebar.select_slider(
        "Risk Tolerance",
        ["conservative", "moderate", "aggressive"],
        value=risk_default
    )

    st.query_params["risk"] = risk_tolerance

    # Stop Loss
    sl_default = st.query_params.get("sl", "2.0")

    if isinstance(sl_default, list):
        sl_default = sl_default[0]

    sl_pct = st.sidebar.slider(
        "Stop Loss %",
        0.5,
        10.0,
        value=float(sl_default),
        step=0.5
    )

    st.query_params["sl"] = str(sl_pct)

    stop_loss_pct = sl_pct / 100

    # Take Profit
    tp_default = st.query_params.get("tp", "4.0")

    if isinstance(tp_default, list):
        tp_default = tp_default[0]

    tp_pct = st.sidebar.slider(
        "Take Profit %",
        1.0,
        20.0,
        value=float(tp_default),
        step=0.5
    )

    st.query_params["tp"] = str(tp_pct)

    take_profit_pct = tp_pct / 100

    rr = take_profit_pct / stop_loss_pct if stop_loss_pct else 2.0

    st.sidebar.markdown(f"**R/R:** 1:{rr:.1f}")

    st.sidebar.subheader("Backtesting")
    bt_pos_size = st.sidebar.slider("Position Size %", 5, 50, 10, 5) / 100

    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.sidebar.info("🔒 **Paper Trading Only** — no real funds.")

    return dict(
        symbol=symbol, timeframe=timeframe, limit=limit,
        refresh_ms=refresh_ms, refresh_option=refresh_option,
        auto_refresh=refresh_ms is not None,
        show=show,
        capital=capital, risk_tolerance=risk_tolerance,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        risk_reward=rr, bt_pos_size=bt_pos_size,
        theme=theme,
    )

# ── Tab 1: Overview ───────────────────────────────────────────────────────────

def render_overview(tickers, cg_data, watchlist_symbols, ind_map, signal_map, fg):
    st.markdown(render_section_header(
        "Market Overview",
        "Premium crypto market pulse with institutional-grade sentiment, movers, and AI signal insights."
    ), unsafe_allow_html=True)

    all_prices = []
    all_changes = []
    total_mcap = 0
    for sym in watchlist_symbols:
        t = tickers.get(sym, {})
        cg = cg_data.get(sym, {})
        price = t.get("last", 0) or cg.get("current_price", 0)
        change = t.get("percentage", 0) or cg.get("price_change_percentage_24h", 0)
        cap = cg.get("market_cap", 0)
        all_prices.append(price)
        all_changes.append(change)
        total_mcap += cap

    avg_change = np.mean(all_changes) if all_changes else 0
    buy_count = sum(1 for sig in signal_map.values() if sig.get("signal") == SIGNAL_BUY)
    sell_count = sum(1 for sig in signal_map.values() if sig.get("signal") == SIGNAL_SELL)
    hold_count = sum(1 for sig in signal_map.values() if sig.get("signal") == SIGNAL_HOLD)
    avg_conf = np.mean([sig.get("confidence", 0.0) for sig in signal_map.values()]) if signal_map else 0.0

    mover_data = []
    for sym in watchlist_symbols:
        t = tickers.get(sym, {})
        cg = cg_data.get(sym, {})
        pct = t.get("percentage", 0) or cg.get("price_change_percentage_24h", 0)
        mover_data.append((sym, pct, t.get("last", 0) or cg.get("current_price", 0)))
    movers = sorted(mover_data, key=lambda x: x[1], reverse=True)
    top_gainers = movers[:3]
    top_losers = movers[-3:][::-1]

    with st.container():
        cols = st.columns(4)
        cols[0].markdown(render_dashboard_card(
            "Total Watchlist Market Cap",
            format_large_number(total_mcap),
            "Aggregate value across tracked crypto assets.",
            accent="#60a5fa"
        ), unsafe_allow_html=True)
        cols[1].markdown(render_dashboard_card(
            "Avg 24h Change",
            f"{avg_change:+.2f}%",
            "Weighted performance across selected coins.",
            accent="#26a69a" if avg_change >= 0 else "#ef5350"
        ), unsafe_allow_html=True)
        cols[2].markdown(render_dashboard_card(
            "AI Market Pulse",
            f"{buy_count} BUY / {sell_count} SELL",
            f"{hold_count} Neutral · {avg_conf*100:.0f}% avg confidence",
            accent="#fbbf24"
        ), unsafe_allow_html=True)
        fg_val = fg.get("value", 50)
        fg_c = get_fg_color(fg_val)
        fg_cl = fg.get("classification", "Neutral")
        cols[3].markdown(render_dashboard_card(
            "Fear & Greed",
            f"{fg_val}",
            f"{get_fg_emoji(fg_cl)} {fg_cl}",
            accent=fg_c
        ), unsafe_allow_html=True)

    st.markdown("<div class='dashboard-grid'>", unsafe_allow_html=True)
    for sym, pct, price in top_gainers:
        label = "Gainer"
        color = "#26a69a"
        st.markdown(render_dashboard_card(
            f"{sym} Top Gainer",
            f"{pct:+.2f}%",
            f"{fmt_price(price, sym)}",
            accent=color
        ), unsafe_allow_html=True)
    for sym, pct, price in top_losers:
        label = "Loser"
        color = "#ef5350"
        st.markdown(render_dashboard_card(
            f"{sym} Top Loser",
            f"{pct:+.2f}%",
            f"{fmt_price(price, sym)}",
            accent=color
        ), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
            "Market Cap": format_large_number(cg.get("market_cap", 0)),
            "Volume 24h": format_large_number(
                t.get("quoteVolume", 0) or cg.get("total_volume", 0)
            ),
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
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        render_empty_state("Technical analysis data unavailable. Try a different timeframe.")
        return
    if not ind or not isinstance(ind, dict):
        render_empty_state("Technical indicators unavailable.")
        return
    # ── Indicator summary ──────────────────────────────────────────────────
    st.markdown(render_section_header(
        "Technical Dashboard",
        "Compact professional indicators, chart overlays, and support/resistance insights."
    ), unsafe_allow_html=True)
    bullish_cross = ind.get("ema_bullish_cross", False)
    bearish_cross = ind.get("ema_bearish_cross", False)
    if bullish_cross:
        st.success("🟢 **EMA 9 × EMA 21 Bullish Crossover** on latest candle")
    elif bearish_cross:
        st.error("🔴 **EMA 9 × EMA 21 Bearish Crossover** on latest candle")

    close = ind["close"]

    def ind_card(label, val_str, sub="", color="#ccc"):
        return (
            f"<div class='dashboard-card' style='text-align:center'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-val' style='color:{color};font-size:1.35em'>{val_str}</div>"
            f"<div style='font-size:.78em;color:#8b949e;margin-top:5px'>{sub}</div>"
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
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        render_empty_state("Smart Money data unavailable. Check symbol/timeframe and try again.")
        return
    if not smc or not isinstance(smc, dict):
        render_empty_state("Smart Money data unavailable. SMC analysis could not be loaded.")
        return

    st.markdown(render_section_header(
        "Smart Money Concepts",
        "Institutional order flow, FVGs, liquidity zones and SMC insights in a premium dashboard."
    ), unsafe_allow_html=True)

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
    # PERF: Validate data before rendering
    if not ob or not isinstance(ob, dict):
        render_empty_state("Order book data unavailable.")
        return
    
    if "bids" not in ob or "asks" not in ob:
        render_empty_state("Order book data unavailable.")
        return
    
    if ob.get("source") in {"unavailable", "synthetic"}:
        render_notice_badge(
            "Live order book unavailable. Showing fallback synthetic order book data.",
            kind="warning",
        )

    src = ob.get("source", "")
    if src == "synthetic":
        st.caption("⚠️ Showing synthetic order book (Binance rate-limited)")

    st.markdown(render_section_header(f"Live Order Book — {symbol}", f"Source: {src or 'Live'}"), unsafe_allow_html=True)

    imb = ob["imbalance"]
    imb_label = "Bid dominant" if imb > 0 else "Ask dominant"
    imb_badge = "buy" if imb > 0 else "sell"
    spread_note = f"{ob['spread_pct']:.4f}% of price"
    buy_pct = f"{ob['buy_pct']:.1f}%"
    sell_pct = f"{ob['sell_pct']:.1f}%"
    imbalance = f"{imb:+.3f}"
    st.markdown(
        "<div class='dashboard-grid'>"
        + render_metric_tile('Best Bid', fmt_price(ob['best_bid'], symbol), 'Near-term support', 'buy')
        + render_metric_tile('Best Ask', fmt_price(ob['best_ask'], symbol), 'Immediate resistance', 'sell')
        + render_metric_tile('Spread', fmt_price(ob['spread'], symbol), spread_note, 'hold')
        + render_metric_tile('Buy Pressure', buy_pct, 'Bid-side liquidity', 'buy')
        + render_metric_tile('Sell Pressure', sell_pct, 'Ask-side liquidity', 'sell')
        + render_metric_tile('Imbalance', imbalance, imb_label, imb_badge)
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Bid / Ask tables side by side ──────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='table-card'><h5 style='margin:0 0 12px;color:#e2e8f0'>🟢 Top Bids</h5></div>", unsafe_allow_html=True)
        bids_df = pd.DataFrame(ob["bids"]).rename(
            columns={"price":"Price","size":"Size","cumulative":"Cumulative","value":"Value ($)"})
        bids_df["Price"]      = bids_df["Price"].apply(lambda x: fmt_price(x, symbol))
        bids_df["Size"]       = bids_df["Size"].round(4)
        bids_df["Cumulative"] = bids_df["Cumulative"].round(4)
        bids_df["Value ($)"]  = bids_df["Value ($)"].apply(lambda x: f"${x:,.1f}")
        st.dataframe(bids_df[["Price","Size","Cumulative","Value ($)"]],
                     width="stretch", hide_index=True)

    with col2:
        st.markdown("<div class='table-card'><h5 style='margin:0 0 12px;color:#e2e8f0'>🔴 Top Asks</h5></div>", unsafe_allow_html=True)
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

def render_mtf(mtf: dict, symbol: str, theme_name: str = "Institutional Dark"):
    # PERF: Validate data before rendering
    if not mtf or not isinstance(mtf, dict):
        st.info("⏰ Multi-timeframe data unavailable")
        return
    
    if "_overall" not in mtf:
        st.info("⏰ Multi-timeframe data unavailable")
        return
    
    theme = THEME_TOKENS.get(theme_name, THEME_TOKENS["Institutional Dark"])
    st.markdown(render_section_header(
        f"Multi-Timeframe Analysis — {symbol}",
        "Fast institutional regime alignment view"
    ), unsafe_allow_html=True)

    overall = mtf.get("_overall", {
        "verdict": "N/A", "color": theme["muted"], "avg_score": 0,
        "bullish": 0, "bearish": 0, "hold": 0,
        "confidence": 0.0, "agreement": 0,
        "signal": "N/A", "alignment": "Unavailable",
    })
    if overall.get("signal") in {"N/A", "HOLD"} and overall.get("avg_score", 0) == 0:
        render_notice_badge(
            "Multi-timeframe analysis is partially degraded. Showing best available data.",
            kind="warning",
        )
    ov = overall.get("verdict", "N/A")
    ov_c = overall.get("color", theme["muted"])
    avg = overall.get("avg_score", 0)
    conf = int(overall.get("confidence", 0.0) * 100)

    st.markdown(
        "<div class='dashboard-grid'>"
        + render_dashboard_card("MTF Consensus", ov, f"Avg score {avg:+.2f} · {conf}% confidence", ov_c)
        + render_dashboard_card("Bullish Timeframes", str(overall.get("bullish", 0)), "Weighted alignment", theme["success"])
        + render_dashboard_card("Bearish Timeframes", str(overall.get("bearish", 0)), "Market pressure", theme["danger"])
        + render_dashboard_card("Neutral / Hold", str(overall.get("hold", 0)), "Divergence zones", theme["warning"])
        + "</div>",
        unsafe_allow_html=True,
    )

    tfs = [tf for tf in MTF_TIMEFRAMES if tf in mtf]
    if not tfs:
        st.info("No timeframe data available")
        return

    cards_html = ""
    for tf in tfs:
        d = mtf[tf]
        tf_color = d.get("color", theme["muted"])
        momentum = d.get("momentum", 0)
        confidence = int(d.get("confidence", 0.0) * 100)
        trend = d.get("details", {}).get("trend", "Neutral")
        cards_html += (
            "<div class='dashboard-card' style='padding:18px;'>"
            f"<div class='metric-label'>{MTF_LABELS.get(tf, tf)}</div>"
            f"<div style='font-size:1.4rem;font-weight:800;color:{tf_color};margin-bottom:4px'>{d.get('verdict', 'N/A')}</div>"
            f"<div style='font-size:.88rem;color:var(--muted);margin-bottom:12px'>Trend: {trend}</div>"
            f"<div style='display:flex;gap:7px;flex-wrap:wrap'>"
            f"<span class='metric-pill' style='background:rgba(37,99,235,0.12);color:{theme['accent']}'>Momentum {momentum}%</span>"
            f"<span class='metric-pill' style='background:rgba(248,113,113,0.12);color:{theme['danger']}'>Conf {confidence}%</span>"
            f"<span class='metric-pill' style='background:rgba(16,185,129,0.12);color:{theme['success']}'>Signal {d.get('signal','N/A')}</span>"
            f"</div>"
            "</div>"
        )
    st.markdown(f"<div class='dashboard-grid'>{cards_html}</div>", unsafe_allow_html=True)

    st.markdown("#### Alignment Matrix")
    metrics = ["rsi", "macd", "ema_9_21", "ema", "trend"]
    metric_labels = {
        "rsi": "RSI", "macd": "MACD", "ema_9_21": "EMA 9/21",
        "ema": "EMA 50/200", "trend": "Trend",
    }
    heat_data = []
    heat_colors = []
    for tf in tfs:
        row_d = []
        row_c = []
        for m in metrics:
            v = mtf[tf].get("details", {}).get(m, "N/A")
            row_d.append(v)
            if any(label in v for label in ["Bull", "Oversold", "Strong Bull"]):
                row_c.append(theme["heat_bull"])
            elif any(label in v for label in ["Bear", "Overbought", "Strong Bear"]):
                row_c.append(theme["heat_bear"])
            else:
                row_c.append("rgba(128,128,128,0.15)")
        heat_data.append(row_d)
        heat_colors.append(row_c)

    fig = go.Figure(data=go.Table(
        header=dict(
            values=["Timeframe"] + [metric_labels[m] for m in metrics],
            fill_color=theme["panel_bg"],
            font=dict(color=theme["text"], size=12),
            align="center",
        ),
        cells=dict(
            values=[
                [MTF_LABELS.get(tf, tf) for tf in tfs],
                *[[heat_data[i][j] for i in range(len(tfs))] for j in range(len(metrics))],
            ],
            fill_color=[theme["card_bg"]] + [[heat_colors[i][j] for i in range(len(tfs))] for j in range(len(metrics))],
            font=dict(color=theme["text"], size=11),
            align="center",
            height=32,
        ),
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


# ── Tab 6: AI Signals ─────────────────────────────────────────────────────────

def render_ai_signals(ind, adv, smc, mtf, ob, sentiment, fg, signal_result, ml_result,
                      risk, symbol, cfg):
    if not isinstance(signal_result, dict) or not signal_result:
        render_empty_state("AI signal data unavailable.")
        return
    if not isinstance(ind, dict) or not ind:
        render_empty_state("Technical indicator data unavailable.")
        return
    if not isinstance(risk, dict) or not risk:
        render_empty_state("Risk assessment unavailable.")
        return

    st.markdown("### 🤖 AI Signal Engine")

    sig  = signal_result.get("signal", "HOLD")
    conf = signal_result.get("confidence", 0.0)
    sc   = signal_result.get("score", 0.0)
    bull = signal_result.get("bull_signals", 0)
    bear = signal_result.get("bear_signals", 0)
    norm = signal_result.get("normalized_score", 0)
    color = signal_color(sig)

    inst_label = signal_result.get("institutional_bias", "Neutral")
    inst_score = signal_result.get("institutional_bias_score", 0.0)
    market_regime = signal_result.get("market_regime", "N/A")
    trend_strength = float(signal_result.get("trend_strength", 0.0))
    # Map numeric trend strength to categorical label: Weak/Moderate/Strong/Extreme
    if trend_strength >= 0.9:
        trend_category = "Extreme"
    elif trend_strength >= 0.66:
        trend_category = "Strong"
    elif trend_strength >= 0.33:
        trend_category = "Moderate"
    else:
        trend_category = "Weak"
    risk_level = signal_result.get("risk_level", "N/A")

    # dynamic classes and layout helpers
    css_state = "buy" if sig == "BUY" else ("sell" if sig == "SELL" else "hold")
    dominance_total = bull + bear
    if dominance_total > 0:
        bull_pct = bull / dominance_total * 100
    else:
        # fallback to orderbook buy_pct if available
        bull_pct = ob.get("buy_pct", 50)
    bear_pct = max(0.0, 100 - bull_pct)

    # risk badge class
    rclass = "risk-low" if risk_level == "Low" else ("risk-medium" if risk_level == "Medium" else "risk-high")

    with st.container():
        left, right = st.columns([1, 2])
        with left:
            st.markdown(
                f"<div class='signal-card {css_state}'><div style='display:flex;align-items:center;justify-content:space-between'>"
                f"<div style='display:flex;flex-direction:column;align-items:flex-start'>"
                f"<div class='signal-badge' style='background:{color}'>{sig}</div>"
                f"<div class='small-muted' style='margin-top:6px'>Confidence</div>"
                f"<div style='font-size:1.6em;font-weight:800;color:{color}'>{conf*100:.1f}%</div>"
                f"<div class='conf-wrap'><div class='conf-bar'><div class='conf-fill' style='width:{conf*100:.1f}%;background:{color}'></div></div></div>"
                f"</div>"
                f"<div style='text-align:right'>"
                f"<div class='small-muted'>Raw</div><div style='font-weight:700'>{sc:+.2f}</div>"
                f"<div class='small-muted' style='margin-top:8px'>Normalized</div><div style='font-weight:700'>{norm:+.3f}</div>"
                f"</div></div>"
                f"<div style='margin-top:8px' class='dom-wrap'><div class='dom-bar'><div class='dom-bull' style='width:{bull_pct:.1f}%;'></div><div class='dom-bear' style='width:{bear_pct:.1f}%;'></div></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with right:
            # Top meta row
            st.markdown(
                f"<div class='signal-card'><div class='signal-row'>"
                f"<div style='flex:1'><div class='signal-meta'><b>Institutional:</b> {inst_label} ({inst_score:+.3f})</div></div>"
                f"<div style='flex:1'><div class='signal-meta'><b>Regime:</b> {market_regime}</div></div>"
                f"<div style='flex:1' style='text-align:right'><div class='signal-meta'><b>Risk:</b> <span class='risk-badge {rclass}'>{risk_level}</span></div></div>"
                f"</div>"
                f"<div style='height:8px'></div>"
                f"<div class='signal-row'>"
                f"<div style='flex:1'><div class='signal-item'><b>Trend:</b> {trend_category}</div></div>"
                f"<div style='flex:1'><div class='signal-item'><b>Bull:</b> {bull}</div></div>"
                f"<div style='flex:1'><div class='signal-item'><b>Bear:</b> {bear}</div></div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

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
    if ml_result is None:
        st.warning("ML: No predictions available")
    else:
        direction = ml_result.get("direction", "?")
        prob      = ml_result.get("combined_probability", 0.5)
        dir_c     = "#26a69a" if direction == "UP" else "#ef5350"
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.markdown(
            f"<div class='terminal-card' style='text-align:center'>"
            f"<div class='metric-label'>ML Consensus</div>"
            f"<div style='font-size:1.8em;color:{dir_c};font-weight:700'>"
            f"{('↑' if direction=='UP' else '↓')} {direction}</div></div>",
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
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        render_empty_state("Backtest requires market data. Select a valid symbol and timeframe.")
        return
    st.markdown(render_section_header("Strategy Backtester", "Simulate trade performance with premium risk controls"), unsafe_allow_html=True)
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
        with st.spinner("Preparing full dataset for backtest…"):
            try:
                full_df = load_full_data(symbol, cfg.get("timeframe", "1h"), cfg["limit"])
            except Exception:
                full_df = df
        with st.spinner("Running backtest…"):
            bt_r = run_backtest(full_df, initial_capital=bt_cap,
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

    st.markdown(
        "<div class='dashboard-grid'>" +
        render_dashboard_card("Total Return", f"${m['total_return']:,.2f}", f"{m['total_return_pct']:+.2f}%") +
        render_dashboard_card("Win Rate", f"{m['win_rate']:.1f}%", f"{m['winning_trades']}W / {m['losing_trades']}L") +
        render_dashboard_card("Sharpe Ratio", f"{m['sharpe_ratio']:.3f}", "Risk-adjusted returns") +
        render_dashboard_card("Max Drawdown", f"{m['max_drawdown']:.2f}%", "Peak-to-trough", "#f97316") +
        render_dashboard_card("Total Trades", str(m['total_trades']), "Market events") +
        render_dashboard_card("Profit Factor", f"{m['profit_factor']:.3f}", "Gross profit / loss") +
        "</div>",
        unsafe_allow_html=True,
    )

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
    if not isinstance(signal_result, dict) or not signal_result:
        render_empty_state("Portfolio signal engine unavailable.")
        return
    if not isinstance(ind, dict) or not ind:
        render_empty_state("Pricing data unavailable.")
        return
    if not isinstance(risk, dict) or not risk:
        render_empty_state("Risk sizing unavailable.")
        return
    st.markdown(render_section_header("Paper Trading Portfolio", "Institutional trade log with sizing and risk controls"), unsafe_allow_html=True)
    if "paper_trades" not in st.session_state:
        st.session_state.paper_trades = []

    sig   = signal_result.get("signal", "HOLD")
    close = ind["close"]
    pos   = risk["position_size"]
    ts    = now_str("%Y-%m-%d %H:%M:%S WIB")

    st.markdown(
        "<div class='dashboard-grid'>"
        + render_dashboard_card("Active Signal", sig, f"Price @ {fmt_price(close, symbol)}")
        + render_dashboard_card("Capital", f"${capital:,.2f}", "Paper trading balance")
        + render_dashboard_card("Position Size", f"${pos['position_value']:,.2f}", f"{pos['units']:.6f} units")
        + "</div>",
        unsafe_allow_html=True,
    )

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
        watchlist_symbols = FALLBACK_SYMBOLS[:10]

    cfg = render_sidebar(watchlist_symbols)
    render_theme_css(cfg["theme"])

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
        # For UI responsiveness, load a smaller slice for initial rendering.
        ui_limit = min(cfg["limit"], 250)
        df      = load_full_data(symbol, timeframe, ui_limit)

    if df is None or df.empty:
        st.warning("No market data available for this symbol/timeframe.")
        return
    ind = get_current_indicator_values(df)
    ind["bb_width"] = float(df["bb_width"].iloc[-1]) if "bb_width" in df.columns else 0.0
    adv = get_advanced_indicator_values(df)
    sr  = find_support_resistance(df)

    # ── Watchlist scanner (basic indicators only) ───────────────────────────
    @st.cache_data(ttl=10, show_spinner=False)
    def scan_symbol_cached(sym: str):
        """Cached watchlist scan to avoid duplicate 1h fetches."""
        try:
            dft = load_watchlist_data(sym, "1h", limit=80)
            i = get_current_indicator_values(dft)
            i["bb_width"] = (
                float(dft["bb_width"].iloc[-1])
                if "bb_width" in dft.columns else 0.0
            )
            return (sym, i, generate_signal(i, 0.0))
        except Exception:
            return (sym, {}, {"signal": "HOLD", "confidence": 0.5, "reasons": []})
    
    def scan_symbol(sym):
        return scan_symbol_cached(sym)

    ind_map = {}

    signal_map = {}

    with st.spinner("Quick market scan..."):

        with ThreadPoolExecutor(max_workers=5) as executor:

            results = executor.map(scan_symbol, watchlist_symbols)

        for sym, indicators, signal in results:
            ind_map[sym] = indicators
            signal_map[sym] = signal        


    # ── Heavy sources are loaded lazily per tab to keep the dashboard responsive.
    smc = _default_smc()
    ob = {
        "best_bid": 0.0,
        "best_ask": 0.0,
        "spread": 0.0,
        "spread_pct": 0.0,
        "buy_pct": 50.0,
        "sell_pct": 50.0,
        "imbalance": 0.0,
        "cum_delta": 0.0,
        "bids": [{"price": 0, "size": 0, "cumulative": 0, "value": 0}],
        "asks": [{"price": 0, "size": 0, "cumulative": 0, "value": 0}],
        "source": "synthetic",
    }
    mtf = _default_mtf()

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
        if "smc" not in st.session_state:
            with st.spinner("Loading Smart Money data…"):
                st.session_state.smc = load_smc(symbol, timeframe, cfg["limit"])
        render_smart_money(df, st.session_state.smc, symbol)

    with tab4:
        if "ob" not in st.session_state:
            with st.spinner("Fetching order book…"):
                st.session_state.ob = load_orderbook(symbol)
        render_orderbook(st.session_state.ob, symbol)

    with tab5:
        if "mtf" not in st.session_state:
            with st.spinner("Loading multi-timeframe analysis…"):
                st.session_state.mtf = load_mtf_data(symbol, timeframe)
        render_mtf(st.session_state.mtf, symbol, cfg["theme"])

    with tab6:
        # PERF: Lazy load AI Signals - only compute when tab opened
        if not is_tab_rendered("ai_signals"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🤖 Analyzing...**")
                render_skeleton_loader("60px", 1)
            with col2:
                render_skeleton_loader("240px", 1)
            with col3:
                render_skeleton_loader("80px", 1)
            mark_tab_rendered("ai_signals")
        
        with st.spinner("Loading sentiment…"):
            sentiment = load_news_sentiment(symbol)
        sentiment_score = sentiment.get("score", 0.0)
        fg_val = fg.get("value", 50)
        # Use session-state heavy loaders if available (loaded when user opened those tabs)
        smc_used = st.session_state.get("smc", smc)
        ob_used = st.session_state.get("ob", ob)
        mtf_used = st.session_state.get("mtf", mtf)
        mtf_overall = mtf_used.get("_overall", {}) if mtf_used else {}

        signal_result = generate_signal(
            ind, sentiment_score,
            advanced=adv, smc=smc_used, mtf_overall=mtf_overall,
            orderbook=ob_used, fg_value=fg_val,
        )
        risk = assess_risk(
            cfg["capital"], ind["close"],
            ind.get("atr", ind["close"] * 0.02),
            signal_result["confidence"],
            cfg["risk_tolerance"],
        )
        risk["risk_reward"] = cfg["risk_reward"]
        ml_result = None

        # persist for Portfolio tab
        st.session_state["signal_result"] = signal_result
        render_ai_signals(ind, adv, smc_used, mtf_used, ob_used, sentiment, fg,
                  signal_result, ml_result, risk, symbol, cfg)

    with tab7:
        # PERF: Lazy load Backtest - expensive computation only when tab opened
        if not is_tab_rendered("backtest"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**🔬 Running Backtest...**")
                render_skeleton_loader("150px", 2)
            with col2:
                render_skeleton_loader("360px", 1)
            mark_tab_rendered("backtest")
        render_backtest(df, cfg, symbol)

    with tab8:
        # PERF: Lazy load Portfolio - only compute when tab opened
        if not is_tab_rendered("portfolio"):
            st.markdown("**📋 Loading Portfolio...**")
            render_skeleton_loader("200px", 2)
            mark_tab_rendered("portfolio")
        
        if "signal_result" in st.session_state:
            signal_result = st.session_state.get("signal_result")
            risk = assess_risk(cfg["capital"], ind["close"],
                               ind.get("atr", ind["close"]*0.02),
                               signal_result["confidence"], cfg["risk_tolerance"])
            risk["risk_reward"] = cfg["risk_reward"]
        else:
            sentiment = load_news_sentiment(symbol)
            sentiment_score = sentiment.get("score", 0.0)
            signal_result = generate_signal(ind, sentiment_score, advanced=adv, smc=st.session_state.get("smc", smc))
            risk = assess_risk(cfg["capital"], ind["close"],
                               ind.get("atr", ind["close"]*0.02),
                               signal_result["confidence"], cfg["risk_tolerance"])
            risk["risk_reward"] = cfg["risk_reward"]
            st.session_state["signal_result"] = signal_result
        render_portfolio(signal_result, ind, risk, symbol, cfg["capital"])

    st.caption(
        f"Binance (CCXT) · CoinGecko · alternative.me (Fear&Greed) · "
        f"{now_str('%Y-%m-%d %H:%M WIB')} · Paper Trading Only"
    )


if __name__ == "__main__":
    main()