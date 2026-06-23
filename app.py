# SuperSignal AI Engine
import sys
import os
import hashlib
import html
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone
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
from src.risk.risk_manager import assess_risk, calculate_position_size
from src.ui.layout import (
    render_header,
    get_theme_css
)
from src.ui.charts import render_price_chart

st.set_page_config(
    page_title="SuperSignal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_theme_css(), unsafe_allow_html=True)
THEME_OPTIONS = ["Default", "Dark", "Light"]
THEME_ALIASES = {
    "Institutional Dark": "Dark",
    "Premium Light": "Light",
    "System": "Default",
    "system": "Default",
    "dark": "Dark",
    "light": "Light",
    "default": "Default",
}
THEME_TOKEN_MAP = {
    "Default": "Institutional Dark",
    "Dark": "Institutional Dark",
    "Light": "Premium Light",
}

def normalize_theme_name(theme_name: str) -> str:
    raw = str(theme_name or THEME_OPTIONS[0]).strip()
    return THEME_ALIASES.get(raw, raw if raw in THEME_OPTIONS else THEME_OPTIONS[0])
THEME_TOKENS = {
    "Institutional Dark": {
        "app_bg": "#05070B",
        "app_bg_alt": "#070B12",
        "panel_bg": "#0A1018",
        "card_bg": "#070B12",
        "card_bg_hover": "#0A1018",
        "card_border": "rgba(255,255,255,0.08)",
        "text": "#F3F5F7",
        "muted": "#A8B0BD",
        "subtle": "#7B8596",
        "accent": "#00E08A",
        "accent_alt": "#00E08A",
        "success": "#00E08A",
        "danger": "#FF5C73",
        "warning": "#FFB84D",
        "shadow": "0 12px 28px rgba(2,6,23,0.26)",
        "sidebar": "#05070B",
        "sidebar_panel": "#0A1018",
        "tab_bg": "rgba(10,16,24,0.92)",
        "tab_active": "rgba(0,224,138,0.16)",
        "tab_border": "rgba(255,255,255,0.08)",
        "input_bg": "#070B12",
        "heat_bull": "rgba(0,224,138,0.20)",
        "heat_bear": "rgba(255,92,115,0.20)",
        "table_bg": "#070B12",
        "table_header_bg": "#0A1018",
    },
    "Premium Light": {
        "app_bg": "#f3f6fb",
        "app_bg_alt": "#e8eef7",
        "panel_bg": "rgba(255,255,255,0.92)",
        "card_bg": "rgba(255,255,255,0.98)",
        "card_bg_hover": "#ffffff",
        "card_border": "rgba(51,65,85,0.14)",
        "text": "#132033",
        "muted": "#526176",
        "subtle": "#7a8799",
        "accent": "#0f6fdc",
        "accent_alt": "#0891b2",
        "success": "#00E08A",
        "danger": "#FF5C73",
        "warning": "#FFB84D",
        "shadow": "0 10px 24px rgba(15,23,42,0.08)",
        "sidebar": "rgba(248,250,252,0.98)",
        "sidebar_panel": "rgba(255,255,255,0.86)",
        "tab_bg": "rgba(255,255,255,0.74)",
        "tab_active": "rgba(15,111,220,0.10)",
        "tab_border": "rgba(51,65,85,0.15)",
        "input_bg": "#ffffff",
        "heat_bull": "rgba(0,224,138,0.14)",
        "heat_bear": "rgba(255,92,115,0.14)",
        "table_bg": "#ffffff",
        "table_header_bg": "#f5f8fb",
    },
}
def get_theme_css(theme_name: str) -> str:
    theme_name = normalize_theme_name(theme_name)
    token_key = THEME_TOKEN_MAP.get(theme_name, "Institutional Dark")
    t = THEME_TOKENS.get(token_key, THEME_TOKENS["Institutional Dark"])
    return f"""
    <style>
    :root {{
      --app-bg: {t['app_bg']};
      --app-bg-alt: {t['app_bg_alt']};
      --panel-bg: {t['panel_bg']};
      --card-bg: {t['card_bg']};
      --card-bg-hover: {t['card_bg_hover']};
      --card-border: {t['card_border']};
      --text: {t['text']};
      --muted: {t['muted']};
      --subtle: {t['subtle']};
      --accent: {t['accent']};
      --accent-alt: {t['accent_alt']};
      --success: {t['success']};
      --danger: {t['danger']};
      --warning: {t['warning']};
      --shadow: {t['shadow']};
      --sidebar: {t['sidebar']};
      --sidebar-panel: {t['sidebar_panel']};
      --tab-bg: {t['tab_bg']};
      --tab-active: {t['tab_active']};
      --tab-border: {t['tab_border']};
      --input-bg: {t['input_bg']};
      --heat-bull: {t['heat_bull']};
      --heat-bear: {t['heat_bear']};
      --cg-green: #00E08A;
      --cg-red: #FF5C73;
      --cg-pos-weak: rgba(0,224,138,0.12);
      --cg-pos-medium: rgba(0,224,138,0.20);
      --cg-pos-strong: rgba(0,224,138,0.32);
      --cg-neg-weak: rgba(255,92,115,0.12);
      --cg-neg-medium: rgba(255,92,115,0.20);
      --cg-neg-strong: rgba(255,92,115,0.32);
      --table-bg: {t['table_bg']};
      --table-header-bg: {t['table_header_bg']};
      --radius-sm: 6px;
      --radius-md: 8px;
      --radius-lg: 10px;
      --dashboard-grid-min: 190px;
      --dashboard-grid-gap: 16px;
      --dashboard-grid-margin: 0.7rem 0 1rem;
      --dashboard-card-padding: 12px 14px;
      --dashboard-card-min-height: 96px;
      --dashboard-card-overflow: visible;
      --terminal-card-min-height: 76px;
      --section-subtitle-margin: 0.65rem;
      --section-subtitle-size: 0.84rem;
      --metric-value-size: clamp(1.08rem,1.22vw,1.38rem);
      --metric-tile-value-size: clamp(1rem,1.16vw,1.18rem);
    }}
    body, .stApp {{
      background: var(--app-bg) !important;
      color: var(--text) !important;
      font-feature-settings: "tnum" 1, "ss01" 1;
    }}
    .block-container {{
      background: transparent !important;
      color: var(--text) !important;
      padding-top: 1.05rem !important;
      padding-left: clamp(0.85rem, 1.6vw, 1.55rem) !important;
      padding-right: clamp(0.85rem, 1.6vw, 1.55rem) !important;
      padding-bottom: 1.1rem !important;
      width: 100%;
      max-width: 1480px;
      margin-left: auto;
      margin-right: auto;
    }}
    h1, h2, h3, h4, h5, h6, p, label, span, div {{ letter-spacing: 0 !important; }}
    h1 {{ font-size: clamp(2rem, 3vw, 2.75rem) !important; line-height: 1.08 !important; }}
    h2 {{ font-size: clamp(1.45rem, 2vw, 1.9rem) !important; }}
    h3 {{ font-size: clamp(1.08rem, 1.5vw, 1.35rem) !important; margin: 0.55rem 0 0.25rem !important; }}
    h4, h5 {{ margin: 0.45rem 0 0.25rem !important; }}
    p, .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--muted) !important; }}
    .app-header {{
      display: flex;
      align-items: center;
      gap: 14px;
      margin: 0;
      padding: 0.05rem 0 0.68rem;
    }}
    .app-header h1, .app-title, .hero-title {{ margin: 0 !important; font-size: clamp(2rem, 2.75vw, 2.35rem) !important; line-height: 1.12 !important; }}
    .app-header p {{ margin: 0.3rem 0 0; color: var(--muted) !important; font-size: 0.92rem; }}
    div[data-testid="stHorizontalBlock"]:has(.app-header) {{
      gap: 18px;
      align-items: start;
      margin: 0 0 0.8rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--card-border);
    }}
    .theme-control-label {{
      color: var(--subtle);
      font-size: 0.66rem;
      font-weight: 800;
      letter-spacing: .04em !important;
      margin: 0 0 4px;
      text-transform: uppercase;
      text-align: right;
    }}
    .header-theme-spacer {{ height: 2px; }}
    .brand-mark {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 42px;
      height: 42px;
      border-radius: var(--radius-md);
      background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 28%, transparent), color-mix(in srgb, var(--accent-alt) 20%, transparent));
      border: 1px solid color-mix(in srgb, var(--accent) 42%, var(--card-border));
      color: var(--text);
      font-weight: 900;
      font-size: 0.88rem;
      box-shadow: 0 10px 22px color-mix(in srgb, var(--accent) 12%, transparent);
    }}

    section[data-testid="stSidebar"] {{
      background: var(--sidebar) !important;
      border-right: 1px solid var(--card-border);
      box-shadow: 10px 0 30px rgba(0,0,0,0.10);
      transition: width .18s ease, background .18s ease, border-color .18s ease;
    }}
    section[data-testid="stSidebar"] .block-container {{ padding-top: 0.62rem !important; padding-inline: 0.72rem !important; }}
    .sidebar-block {{
      background: color-mix(in srgb, var(--sidebar-panel) 82%, transparent);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 9px 10px;
      border-radius: var(--radius-md);
      box-shadow: 0 8px 22px rgba(0,0,0,0.08);
      transition: background .16s ease, border-color .16s ease, box-shadow .16s ease;
    }}
    .sidebar-block:hover {{ border-color: color-mix(in srgb, var(--cg-green) 32%, var(--card-border)); }}
    .sidebar-block h3 {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      margin: 0;
      font-size: 0.98rem;
      font-weight: 850;
      line-height: 1;
      text-transform: none !important;
      white-space: nowrap;
    }}
    .sidebar-block h3::before {{
      content: "SS";
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      border-radius: var(--radius-sm);
      color: var(--text);
      background: var(--cg-pos-weak);
      border: 1px solid color-mix(in srgb, var(--cg-green) 38%, var(--card-border));
      font-size: 0.62rem;
      font-weight: 900;
      flex: 0 0 24px;
    }}
    .sidebar-block p {{ color: var(--muted); margin-bottom: 0; }}
    .sidebar-divider {{ height: 1px; background: var(--card-border); margin: 8px 0 7px; }}
    .stSidebar .element-container {{ background: transparent !important; margin-bottom: 0.26rem !important; }}
    .stSidebar [data-testid="stMarkdownContainer"] p {{ margin-bottom: 0.1rem; }}
    .stSidebar h3 {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.68rem !important;
      text-transform: uppercase;
      color: var(--subtle) !important;
      margin: 0.54rem 0 0.24rem 0 !important;
      font-weight: 850 !important;
    }}
    .stSidebar h3::before {{ color: var(--muted); font-size: 0.76rem; line-height: 1; }}
    .stSidebar h3:has(+ div) {{ letter-spacing: .02em !important; }}
    .stSidebar h3:nth-of-type(1)::before {{ content: "◐"; }}
    .stSidebar h3:nth-of-type(2)::before {{ content: "↻"; }}
    .stSidebar h3:nth-of-type(3)::before {{ content: "▦"; }}
    .stSidebar h3:nth-of-type(4)::before {{ content: "◇"; }}
    .stSidebar h3:nth-of-type(5)::before {{ content: "⌁"; }}
    .stSidebar label {{ color: var(--text) !important; font-weight: 650 !important; font-size: 0.8rem !important; }}
    .stSidebar [data-testid="stWidgetLabel"] {{ margin-bottom: 0.14rem !important; }}
    .stSidebar .stSelectbox > div > div,
    .stSidebar .stNumberInput input,
    .stSidebar [data-testid="stNumberInput"] [data-baseweb="input"],
    .stSidebar [data-testid="stTextInput"] [data-baseweb="input"],
    .stSidebar [data-baseweb="select"] > div,
    .stSidebar [data-baseweb="input"] {{
      background: color-mix(in srgb, var(--input-bg) 88%, transparent) !important;
      border: 1px solid var(--card-border) !important;
      border-radius: var(--radius-md) !important;
      min-height: 34px !important;
      color: var(--text) !important;
      box-shadow: 0 3px 10px rgba(0,0,0,0.05);
      transition: border-color .16s ease, box-shadow .16s ease, background .16s ease, transform .16s ease;
    }}
    .stSidebar .stSelectbox > div > div:hover,
    .stSidebar .stNumberInput input:hover,
    .stSidebar [data-testid="stNumberInput"] [data-baseweb="input"]:hover,
    .stSidebar [data-testid="stTextInput"] [data-baseweb="input"]:hover,
    .stSidebar [data-baseweb="select"] > div:hover {{
      border-color: var(--cg-green) !important;
      background: var(--input-bg) !important;
      box-shadow: 0 0 0 3px var(--cg-pos-weak);
    }}
    .stSidebar .stCheckbox {{ padding-block: 0; }}
    .stSidebar .stSlider {{ padding-top: 0; padding-bottom: 0.1rem; }}
    .stSidebar .stSlider [role="slider"] {{ border: 2px solid var(--accent) !important; box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 14%, transparent); }}
    .stSidebar [data-baseweb="slider"] div {{ transition: background .16s ease, box-shadow .16s ease; }}
    .stSidebar details {{
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      background: color-mix(in srgb, var(--sidebar-panel) 62%, transparent);
      padding: 1px 8px 5px;
      margin: 0.22rem 0;
      transition: border-color .16s ease, background .16s ease, box-shadow .16s ease;
    }}
    .stSidebar details:hover {{ border-color: color-mix(in srgb, var(--cg-green) 28%, var(--card-border)); }}
    .stSidebar summary {{ color: var(--text); font-weight: 720; font-size: 0.8rem; }}

    .dashboard-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(var(--dashboard-grid-min), 1fr));
      gap: var(--dashboard-grid-gap);
      row-gap: var(--dashboard-grid-gap);
      align-items: stretch;
      margin: var(--dashboard-grid-margin);
    }}
    .dashboard-card, .dashboard-tile, .table-card, .terminal-card, .signal-card {{
      --card-accent: var(--muted);
      background: linear-gradient(180deg, var(--card-bg), color-mix(in srgb, var(--card-bg) 94%, var(--table-header-bg)));
      border: 1px solid var(--card-border);
      box-shadow: 0 10px 24px rgba(0,0,0,0.22);
      color: var(--text);
      border-radius: var(--radius-lg);
      box-sizing: border-box;
      height: auto !important;
      overflow: var(--dashboard-card-overflow) !important;
      padding: var(--dashboard-card-padding);
      min-height: var(--dashboard-card-min-height);
      min-width: 0;
      margin: 0;
      transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease, background .16s ease;
    }}
    .dashboard-card {{
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      gap: 7px;
      position: relative;
    }}
    .dashboard-card::before, .dashboard-tile::before, .table-card::before, .terminal-card::before, .signal-card::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 3px;
      border-radius: var(--radius-lg) 0 0 var(--radius-lg);
      background: var(--card-accent);
      opacity: 0.9;
    }}
    .status-positive, .dashboard-card.buy, .terminal-card.buy, .signal-card.buy {{ --card-accent: var(--cg-green); }}
    .status-negative, .dashboard-card.sell, .terminal-card.sell, .signal-card.sell {{ --card-accent: var(--cg-red); }}
    .status-warning, .status-neutral, .dashboard-card.hold, .terminal-card.hold, .signal-card.hold {{ --card-accent: var(--warning); }}
    .status-muted, .status-unavailable {{ --card-accent: var(--muted); }}
    .status-positive, .signal-card.buy {{ box-shadow: 0 12px 28px rgba(0,224,138,0.10); }}
    .status-negative, .signal-card.sell {{ box-shadow: 0 12px 28px rgba(255,92,115,0.10); }}
    .status-warning, .status-neutral, .signal-card.hold {{ box-shadow: 0 12px 28px rgba(255,184,77,0.08); }}
    .dashboard-card:hover, .dashboard-tile:hover, .signal-card:hover {{
      transform: translateY(-1px);
      background: var(--card-bg-hover);
      border-color: color-mix(in srgb, var(--card-accent) 42%, var(--card-border));
      box-shadow: 0 14px 32px rgba(0,0,0,0.28);
    }}
    .dashboard-tile h4 {{ margin: 0; display: flex; align-items: center; justify-content: space-between; gap: 10px; font-size: 0.82rem; color: var(--muted); overflow-wrap: break-word; }}
    .dashboard-tile p {{ margin: 0.28rem 0 0; color: var(--muted) !important; font-size: 0.76rem; line-height: 1.28; }}
    .dashboard-tile {{ display: flex; flex-direction: column; gap: 5px; position: relative; }}
    .table-card, .terminal-card, .signal-card {{ min-height: var(--terminal-card-min-height); position: relative; }}
    .metric-label {{ color: var(--muted); font-size: 0.66rem; font-weight: 760; text-transform: uppercase; line-height: 1.2; overflow-wrap: break-word; }}
    .metric-val, .metric-value, .metric-subtext {{
      color: var(--text);
      font-weight: 800;
      overflow-wrap: anywhere;
      word-break: normal;
      max-width: 100%;
    }}
    .metric-val {{ font-variant-numeric: tabular-nums; }}
    .metric-subtitle {{ font-size: 0.74rem; color: var(--muted); margin-top: 1px; line-height: 1.28; font-weight: 550; }}
    .overview-summary-grid {{ grid-template-columns: repeat(4, minmax(210px, 1fr)); }}
    .overview-heading {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin: 0.05rem 0 0.38rem;
    }}
    .overview-heading-title {{ color: var(--text); font-size: clamp(1.02rem,1.35vw,1.28rem); font-weight: 900; line-height: 1.1; }}
    .overview-heading-meta {{ color: var(--muted); font-size: 0.75rem; font-weight: 700; white-space: nowrap; }}
    .ai-brief-card {{
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, color-mix(in srgb, var(--card-bg) 94%, var(--panel-bg)), var(--card-bg));
      box-shadow: 0 8px 20px rgba(2,6,23,0.14);
      padding: 9px 11px;
      margin: 0.02rem 0 0.42rem;
    }}
    .ai-brief-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 5px; }}
    .ai-brief-title {{ color: var(--text); font-size: 0.94rem; font-weight: 900; line-height: 1.1; }}
    .ai-brief-state {{ color: var(--muted); font-size: 0.68rem; font-weight: 800; white-space: nowrap; }}
    .ai-brief-lines {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 5px; }}
    .ai-brief-line {{
      color: var(--text);
      background: color-mix(in srgb, var(--panel-bg) 52%, transparent);
      border: 1px solid color-mix(in srgb, var(--card-border) 82%, transparent);
      border-radius: var(--radius-sm);
      padding: 6px 7px;
      font-size: 0.7rem;
      font-weight: 720;
      line-height: 1.15;
      min-height: 38px;
      overflow: hidden;
    }}
    .compact-section-head {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 0.08rem 0 0.14rem; }}
    .compact-section-title {{ color: var(--text); font-size: 0.88rem; font-weight: 900; line-height: 1.1; }}
    .compact-section-meta {{ color: var(--muted); font-size: 0.66rem; font-weight: 750; white-space: nowrap; }}
    .opportunity-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; margin: 0.08rem 0 0.28rem; }}
    .opportunity-card {{
      --opp-accent: var(--muted);
      position: relative;
      min-height: 68px;
      border: 1px solid var(--card-border);
      border-radius: var(--radius-sm);
      background: linear-gradient(180deg, var(--card-bg), color-mix(in srgb, var(--card-bg) 94%, var(--table-header-bg)));
      padding: 7px 8px;
      overflow: hidden;
      box-shadow: 0 8px 18px rgba(2,6,23,0.12);
      cursor: default;
    }}
    .opportunity-card:hover {{ border-color: color-mix(in srgb, var(--opp-accent) 44%, var(--card-border)); background: var(--card-bg-hover); }}
    .opportunity-card::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 3px; background: var(--opp-accent); }}
    .opportunity-card.status-positive {{ --opp-accent: var(--cg-green); }}
    .opportunity-card.status-negative {{ --opp-accent: var(--cg-red); }}
    .opportunity-card.status-warning, .opportunity-card.status-neutral {{ --opp-accent: var(--warning); }}
    .opportunity-top {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
    .opportunity-card .cg-symbol-cell {{ min-width: 0; gap: 8px; }}
    .opportunity-card .coin-logo {{ width: 22px; height: 22px; flex-basis: 22px; }}
    .opportunity-card .cg-name-text {{ display: none; }}
    .opportunity-rank {{ color: var(--subtle); font-size: 0.58rem; font-weight: 900; }}
    .opportunity-mid {{ display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-top: 4px; }}
    .opportunity-score {{ color: var(--text); font-size: 0.84rem; font-weight: 950; font-variant-numeric: tabular-nums; }}
    .overview-signal-pill {{ border-radius: 999px; border: 1px solid var(--card-border); padding: 2px 6px; font-size: 0.56rem; font-weight: 950; white-space: nowrap; }}
    .overview-signal-pill.buy {{ color: var(--cg-green); background: var(--cg-pos-weak); }}
    .overview-signal-pill.sell {{ color: var(--cg-red); background: var(--cg-neg-weak); }}
    .overview-signal-pill.hold {{ color: var(--warning); background: color-mix(in srgb, var(--warning) 14%, transparent); }}
    .opportunity-reason {{ color: var(--muted); font-size: 0.62rem; font-weight: 700; margin-top: 3px; line-height: 1.08; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .scanner-summary {{ display: flex; align-items: center; flex-wrap: wrap; gap: 5px; margin: 0.01rem 0 0.12rem; }}
    .scanner-summary span {{ color: var(--text); background: color-mix(in srgb, var(--panel-bg) 62%, transparent); border: 1px solid var(--card-border); border-radius: 999px; padding: 3px 7px; font-size: 0.68rem; font-weight: 850; white-space: nowrap; }}
    .market-intel-card {{
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, color-mix(in srgb, var(--card-bg) 94%, var(--panel-bg)), var(--card-bg));
      box-shadow: 0 9px 22px rgba(2,6,23,0.14);
      padding: 9px 11px;
      margin: 0.02rem 0 0.36rem;
    }}
    .market-intel-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 7px; }}
    .market-intel-title {{ color: var(--text); font-size: 0.98rem; font-weight: 950; line-height: 1.1; }}
    .market-intel-meta {{ color: var(--muted); font-size: 0.68rem; font-weight: 800; white-space: nowrap; }}
    .market-intel-body {{ display: grid; grid-template-columns: minmax(170px, 0.42fr) minmax(420px, 1.58fr); gap: 9px; align-items: stretch; }}
    .market-intel-core {{
      --intel-accent: var(--warning);
      position: relative;
      border: 1px solid color-mix(in srgb, var(--intel-accent) 28%, var(--card-border));
      border-radius: var(--radius-sm);
      background: color-mix(in srgb, var(--panel-bg) 52%, transparent);
      padding: 8px 10px 8px 12px;
      min-height: 66px;
      overflow: hidden;
    }}
    .market-intel-core::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 3px; background: var(--intel-accent); }}
    .market-intel-core.status-positive {{ --intel-accent: var(--cg-green); }}
    .market-intel-core.status-negative {{ --intel-accent: var(--cg-red); }}
    .market-intel-core.status-warning, .market-intel-core.status-neutral {{ --intel-accent: var(--warning); }}
    .market-intel-label {{ color: var(--muted); font-size: 0.56rem; font-weight: 900; text-transform: uppercase; line-height: 1.05; }}
    .market-intel-value {{ color: var(--intel-accent); font-size: clamp(1.08rem,1.45vw,1.36rem); font-weight: 950; line-height: 1.02; margin-top: 4px; }}
    .market-intel-sub {{ color: var(--text); font-size: 0.72rem; font-weight: 820; margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .market-intel-chips {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; }}
    .market-intel-chip {{
      border: 1px solid var(--card-border);
      border-radius: var(--radius-sm);
      background: color-mix(in srgb, var(--panel-bg) 54%, transparent);
      padding: 7px 8px;
      min-height: 66px;
      overflow: hidden;
    }}
    .market-intel-chip span {{ display: block; color: var(--subtle); font-size: 0.54rem; font-weight: 900; text-transform: uppercase; line-height: 1.05; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .market-intel-chip strong {{ display: block; color: var(--text); font-size: 0.82rem; font-weight: 950; line-height: 1.05; margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .market-intel-chip em {{ display: block; color: var(--muted); font-size: 0.62rem; font-style: normal; font-weight: 720; line-height: 1.08; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .direction-gauge {{
      margin: 0 0 8px;
      border: 1px solid var(--card-border);
      border-radius: var(--radius-sm);
      background: color-mix(in srgb, var(--panel-bg) 48%, transparent);
      padding: 7px 8px;
    }}
    .direction-gauge-top {{ display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 6px; }}
    .direction-gauge-label {{ color: var(--muted); font-size: 0.56rem; font-weight: 900; text-transform: uppercase; }}
    .direction-gauge-value {{ color: var(--text); font-size: 0.86rem; font-weight: 950; }}
    .direction-track {{ position: relative; height: 8px; border-radius: 999px; background: linear-gradient(90deg, var(--cg-red), var(--warning), var(--cg-green)); overflow: hidden; }}
    .direction-marker {{ position: absolute; top: -3px; width: 4px; height: 14px; border-radius: 999px; background: var(--text); box-shadow: 0 0 0 2px color-mix(in srgb, var(--app-bg) 90%, transparent); transform: translateX(-2px); }}
    .direction-scale {{ display: flex; justify-content: space-between; color: var(--subtle); font-size: 0.52rem; font-weight: 800; margin-top: 5px; }}
    .reason-list {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }}
    .reason-chip {{ color: var(--muted); background: color-mix(in srgb, var(--panel-bg) 62%, transparent); border: 1px solid var(--card-border); border-radius: 999px; padding: 2px 6px; font-size: 0.55rem; font-weight: 800; white-space: nowrap; }}
    .coin-intel-card {{
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, color-mix(in srgb, var(--card-bg) 94%, var(--panel-bg)), var(--card-bg));
      box-shadow: 0 7px 18px rgba(2,6,23,0.12);
      padding: 7px 9px;
      margin: 0.04rem 0 0.14rem;
    }}
    .coin-intel-grid {{ display: grid; grid-template-columns: minmax(180px, 0.4fr) minmax(420px, 1.6fr); gap: 7px; align-items: stretch; }}
    .coin-intel-identity {{ display: flex; align-items: center; gap: 8px; min-width: 0; }}
    .coin-intel-identity .coin-logo {{ width: 26px; height: 26px; flex-basis: 26px; }}
    .coin-intel-symbol {{ color: var(--text); font-size: 0.94rem; font-weight: 950; line-height: 1.05; }}
    .coin-intel-name {{ color: var(--muted); font-size: 0.66rem; font-weight: 750; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .coin-intel-price {{ color: var(--text); font-size: 0.8rem; font-weight: 900; margin-top: 5px; }}
    .coin-intel-summary {{ color: var(--muted); font-size: 0.68rem; font-weight: 720; line-height: 1.18; margin-top: 5px; }}
    .intel-metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 5px; }}
    .intel-metric {{ border: 1px solid var(--card-border); border-radius: var(--radius-sm); background: color-mix(in srgb, var(--panel-bg) 54%, transparent); padding: 5px 6px; min-height: 42px; overflow: hidden; }}
    .intel-metric span {{ display: block; color: var(--subtle); font-size: 0.52rem; font-weight: 900; text-transform: uppercase; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .intel-metric strong {{ display: block; color: var(--text); font-size: 0.76rem; font-weight: 920; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .smart-badges {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }}
    .smart-badge {{ border: 1px solid var(--card-border); border-radius: 999px; background: color-mix(in srgb, var(--panel-bg) 60%, transparent); padding: 2px 6px; font-size: 0.55rem; font-weight: 900; color: var(--text); white-space: nowrap; }}
    .smart-badge.positive {{ color: var(--cg-green); background: var(--cg-pos-weak); }}
    .smart-badge.negative {{ color: var(--cg-red); background: var(--cg-neg-weak); }}
    .smart-badge.warning {{ color: var(--warning); background: color-mix(in srgb, var(--warning) 14%, transparent); }}
    .market-status-strip {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 8px;
      margin: 0.12rem 0 0.48rem;
    }}
    .market-status-item {{
      --strip-accent: var(--muted);
      position: relative;
      min-height: 56px;
      padding: 7px 9px 7px 10px;
      border: 1px solid var(--card-border);
      border-radius: var(--radius-sm);
      background: color-mix(in srgb, var(--card-bg) 88%, var(--panel-bg));
      box-shadow: 0 6px 14px rgba(2,6,23,0.12);
      overflow: hidden;
    }}
    .market-status-item::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 3px;
      background: var(--strip-accent);
    }}
    .market-status-item.status-positive {{ --strip-accent: var(--cg-green); }}
    .market-status-item.status-negative {{ --strip-accent: var(--cg-red); }}
    .market-status-item.status-warning, .market-status-item.status-neutral {{ --strip-accent: var(--warning); }}
    .market-status-label {{ color: var(--muted); font-size: 0.58rem; font-weight: 850; text-transform: uppercase; line-height: 1.05; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .market-status-value {{ color: var(--text); font-size: clamp(0.9rem,1.02vw,1.05rem); font-weight: 900; line-height: 1.05; margin-top: 5px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
    .market-status-detail {{ color: var(--subtle); font-size: 0.64rem; font-weight: 650; line-height: 1.12; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .regime-card {{
      --regime-accent: var(--warning);
      position: relative;
      display: grid;
      grid-template-columns: minmax(170px, 0.58fr) minmax(360px, 1.42fr);
      gap: 10px;
      align-items: center;
      margin: 0.18rem 0 0.5rem;
      padding: 9px 11px 9px 13px;
      border: 1px solid color-mix(in srgb, var(--regime-accent) 28%, var(--card-border));
      border-radius: var(--radius-md);
      background: linear-gradient(135deg, color-mix(in srgb, var(--regime-accent) 9%, transparent), transparent 36%), linear-gradient(180deg, var(--card-bg), color-mix(in srgb, var(--card-bg) 94%, var(--table-header-bg)));
      box-shadow: 0 10px 22px rgba(2,6,23,0.14);
      overflow: hidden;
    }}
    .regime-card::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 4px;
      background: var(--regime-accent);
    }}
    .regime-card.status-positive {{ --regime-accent: var(--cg-green); }}
    .regime-card.status-negative {{ --regime-accent: var(--cg-red); }}
    .regime-card.status-warning, .regime-card.status-neutral {{ --regime-accent: var(--warning); }}
    .regime-label {{ color: var(--muted); font-size: 0.58rem; font-weight: 850; text-transform: uppercase; }}
    .regime-value {{ color: var(--regime-accent); font-size: clamp(1.05rem,1.42vw,1.34rem); font-weight: 950; line-height: 1.02; margin-top: 3px; }}
    .regime-confidence {{ color: var(--text); font-size: 0.72rem; font-weight: 800; margin-top: 4px; }}
    .regime-drivers {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; }}
    .regime-driver {{
      min-height: 44px;
      border: 1px solid var(--card-border);
      border-radius: var(--radius-sm);
      background: color-mix(in srgb, var(--panel-bg) 56%, transparent);
      padding: 7px 8px;
      color: var(--text);
      font-size: 0.72rem;
      font-weight: 760;
      line-height: 1.16;
      overflow: hidden;
    }}
    .regime-driver span {{ display: block; color: var(--subtle); font-size: 0.55rem; font-weight: 850; text-transform: uppercase; margin-bottom: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .smc-command-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:0.05rem 0 0.28rem; }}
    .smc-command-title {{ color:var(--text); font-size:clamp(1.02rem,1.35vw,1.28rem); font-weight:950; line-height:1.1; }}
    .smc-command-meta {{ color:var(--muted); font-size:0.72rem; font-weight:780; white-space:nowrap; }}
    .smc-command-card {{ border:1px solid var(--card-border); border-radius:var(--radius-md); background:linear-gradient(180deg,color-mix(in srgb,var(--card-bg) 94%,var(--panel-bg)),var(--card-bg)); box-shadow:0 9px 22px rgba(2,6,23,0.14); padding:9px 11px; margin:0.02rem 0 0.42rem; }}
    .smc-command-grid {{ display:grid; grid-template-columns:minmax(210px,0.46fr) minmax(480px,1.54fr); gap:9px; align-items:stretch; }}
    .smc-core {{ --smc-accent:var(--warning); position:relative; border:1px solid color-mix(in srgb,var(--smc-accent) 30%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 52%,transparent); padding:8px 10px 8px 12px; overflow:hidden; }}
    .smc-core::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--smc-accent); }}
    .smc-core.status-positive {{ --smc-accent:var(--cg-green); }}
    .smc-core.status-negative {{ --smc-accent:var(--cg-red); }}
    .smc-core.status-warning, .smc-core.status-neutral {{ --smc-accent:var(--warning); }}
    .smc-label {{ color:var(--muted); font-size:0.56rem; font-weight:900; text-transform:uppercase; line-height:1.05; }}
    .smc-value {{ color:var(--smc-accent); font-size:clamp(1.05rem,1.38vw,1.32rem); font-weight:950; line-height:1.02; margin-top:4px; }}
    .smc-sub {{ color:var(--text); font-size:0.72rem; font-weight:820; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .smc-gauge {{ margin-top:8px; }}
    .smc-gauge-track {{ position:relative; height:8px; border-radius:999px; background:linear-gradient(90deg,var(--cg-red),var(--warning),var(--cg-green)); overflow:hidden; }}
    .smc-gauge-marker {{ position:absolute; top:-3px; width:4px; height:14px; border-radius:999px; background:var(--text); box-shadow:0 0 0 2px color-mix(in srgb,var(--app-bg) 90%,transparent); transform:translateX(-2px); }}
    .smc-gauge-scale {{ display:flex; justify-content:space-between; color:var(--subtle); font-size:0.50rem; font-weight:800; margin-top:5px; }}
    .smc-health-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:6px; }}
    .smc-health-chip, .smc-setup-item {{ --smc-chip-accent:var(--warning); border:1px solid color-mix(in srgb,var(--smc-chip-accent) 25%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 54%,transparent); padding:7px 8px; min-height:56px; overflow:hidden; }}
    .smc-health-chip.status-positive, .smc-setup-item.status-positive {{ --smc-chip-accent:var(--cg-green); }}
    .smc-health-chip.status-negative, .smc-setup-item.status-negative {{ --smc-chip-accent:var(--cg-red); }}
    .smc-health-chip.status-warning, .smc-health-chip.status-neutral, .smc-setup-item.status-warning, .smc-setup-item.status-neutral {{ --smc-chip-accent:var(--warning); }}
    .smc-health-chip span, .smc-setup-item span {{ display:block; color:var(--subtle); font-size:0.52rem; font-weight:900; text-transform:uppercase; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .smc-health-chip strong, .smc-setup-item strong {{ display:block; color:var(--text); font-size:0.78rem; font-weight:950; line-height:1.08; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .smc-health-chip em, .smc-setup-item em {{ display:block; color:var(--muted); font-size:0.58rem; font-style:normal; font-weight:720; line-height:1.08; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .smc-reasons {{ display:flex; flex-wrap:wrap; gap:5px; margin:0.1rem 0 0.45rem; }}
    .smc-reason-chip {{ color:var(--text); background:color-mix(in srgb,var(--panel-bg) 62%,transparent); border:1px solid var(--card-border); border-radius:999px; padding:3px 8px; font-size:0.66rem; font-weight:850; white-space:nowrap; }}
    .smc-chart-control-row {{ display:flex; align-items:end; justify-content:space-between; gap:12px; margin:0.1rem 0 0.32rem; flex-wrap:wrap; }}
    .smc-chart-context {{ display:flex; flex-wrap:wrap; gap:5px; align-items:center; }}
    .smc-chart-chip {{ color:var(--text); background:color-mix(in srgb,var(--panel-bg) 62%,transparent); border:1px solid var(--card-border); border-radius:999px; padding:4px 8px; font-size:0.64rem; font-weight:850; white-space:nowrap; line-height:1; }}
    .smc-chart-chip strong {{ color:var(--muted); font-size:0.58rem; font-weight:900; text-transform:uppercase; margin-right:4px; }}
    .smc-setup-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.44rem; }}
    .smc-summary-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.42rem; }}
    .smc-summary-card {{ --smc-summary-accent:var(--warning); border:1px solid color-mix(in srgb,var(--smc-summary-accent) 26%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 56%,transparent); padding:7px 8px; min-height:54px; overflow:hidden; }}
    .smc-summary-card.status-positive {{ --smc-summary-accent:var(--cg-green); }}
    .smc-summary-card.status-negative {{ --smc-summary-accent:var(--cg-red); }}
    .smc-summary-card.status-warning, .smc-summary-card.status-neutral {{ --smc-summary-accent:var(--warning); }}
    .smc-summary-card span {{ display:block; color:var(--subtle); font-size:0.52rem; font-weight:900; text-transform:uppercase; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .smc-summary-card strong {{ display:block; color:var(--smc-summary-accent); font-size:0.92rem; font-weight:950; line-height:1.04; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .smc-summary-card em {{ display:block; color:var(--muted); font-size:0.58rem; font-style:normal; font-weight:720; line-height:1.08; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .smc-empty-note {{ border:1px solid var(--card-border); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 56%,transparent); padding:10px 11px; color:var(--text); font-size:0.82rem; font-weight:850; line-height:1.25; }}
    .smc-empty-note span {{ display:block; color:var(--muted); font-size:0.68rem; font-weight:720; margin-top:4px; }}
    .indicator-grid {{ grid-template-columns: repeat(auto-fit, minmax(188px, 1fr)); gap: 16px; margin-top: 0.7rem; }}
    .indicator-grid .dashboard-card {{ min-height: 132px; align-items: center; justify-content: center; text-align: center; }}
    .tech-command-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:0.06rem 0 0.22rem; }}
    .tech-command-title {{ color:var(--text); font-size:clamp(1.02rem,1.35vw,1.28rem); font-weight:950; line-height:1.1; }}
    .tech-command-meta {{ color:var(--muted); font-size:0.72rem; font-weight:780; white-space:nowrap; }}
    .tech-control-row {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin:0 0 0.36rem; flex-wrap:wrap; }}
    .tech-anchor-pills {{ display:flex; align-items:center; gap:6px; flex-wrap:wrap; }}
    .tech-anchor-pills a {{ color:var(--muted); text-decoration:none; border:1px solid var(--card-border); border-radius:999px; background:color-mix(in srgb,var(--panel-bg) 58%,transparent); padding:4px 9px; font-size:0.62rem; font-weight:900; line-height:1; }}
    .tech-anchor-pills a:hover {{ color:var(--text); border-color:color-mix(in srgb,var(--accent) 48%,var(--card-border)); }}
    .tech-summary-card {{ border:1px solid var(--card-border); border-radius:var(--radius-md); background:linear-gradient(180deg,color-mix(in srgb,var(--card-bg) 94%,var(--panel-bg)),var(--card-bg)); box-shadow:0 9px 22px rgba(2,6,23,0.14); padding:9px 11px; margin:0.02rem 0 0.42rem; }}
    .tech-summary-grid {{ display:grid; grid-template-columns:minmax(185px,0.42fr) minmax(420px,1.58fr); gap:9px; align-items:stretch; }}
    .tech-bias-core {{ --tech-accent:var(--warning); position:relative; border:1px solid color-mix(in srgb,var(--tech-accent) 30%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 52%,transparent); padding:8px 10px 8px 12px; overflow:hidden; }}
    .tech-bias-core::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--tech-accent); }}
    .tech-bias-core.status-positive {{ --tech-accent:var(--cg-green); }}
    .tech-bias-core.status-negative {{ --tech-accent:var(--cg-red); }}
    .tech-bias-core.status-warning, .tech-bias-core.status-neutral {{ --tech-accent:var(--warning); }}
    .tech-label {{ color:var(--muted); font-size:0.56rem; font-weight:900; text-transform:uppercase; line-height:1.05; }}
    .tech-bias-value {{ color:var(--tech-accent); font-size:clamp(1.08rem,1.45vw,1.36rem); font-weight:950; line-height:1.02; margin-top:4px; }}
    .tech-bias-sub {{ color:var(--text); font-size:0.72rem; font-weight:820; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tech-gauge {{ margin-top:8px; }}
    .tech-gauge-track {{ position:relative; height:8px; border-radius:999px; background:linear-gradient(90deg,var(--cg-red),var(--warning),var(--cg-green)); overflow:hidden; }}
    .tech-gauge-marker {{ position:absolute; top:-3px; width:4px; height:14px; border-radius:999px; background:var(--text); box-shadow:0 0 0 2px color-mix(in srgb,var(--app-bg) 90%,transparent); transform:translateX(-2px); }}
    .tech-gauge-scale {{ display:flex; justify-content:space-between; color:var(--subtle); font-size:0.52rem; font-weight:800; margin-top:5px; }}
    .tech-health-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6px; }}
    .tech-health-chip {{ border:1px solid var(--card-border); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 54%,transparent); padding:7px 8px; min-height:58px; overflow:hidden; }}
    .tech-health-chip span {{ display:block; color:var(--subtle); font-size:0.54rem; font-weight:900; text-transform:uppercase; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tech-health-chip strong {{ display:block; color:var(--text); font-size:0.82rem; font-weight:950; line-height:1.05; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tech-health-chip em {{ display:block; color:var(--muted); font-size:0.62rem; font-style:normal; font-weight:720; line-height:1.08; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tech-reasons {{ display:flex; flex-wrap:wrap; gap:5px; margin:0.1rem 0 0.45rem; }}
    .tech-reason-chip {{ color:var(--text); background:color-mix(in srgb,var(--panel-bg) 62%,transparent); border:1px solid var(--card-border); border-radius:999px; padding:3px 8px; font-size:0.66rem; font-weight:850; white-space:nowrap; }}
    .tech-action-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:7px; margin:0.16rem 0 0.38rem; }}
    .tech-action-item {{ --action-accent:var(--warning); position:relative; border:1px solid color-mix(in srgb,var(--action-accent) 28%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 56%,transparent); padding:7px 8px 7px 10px; min-height:54px; overflow:hidden; }}
    .tech-action-item::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--action-accent); }}
    .tech-action-item.status-positive {{ --action-accent:var(--cg-green); }}
    .tech-action-item.status-negative {{ --action-accent:var(--cg-red); }}
    .tech-action-item.status-warning, .tech-action-item.status-neutral {{ --action-accent:var(--warning); }}
    .tech-action-item.status-muted, .tech-action-item.status-unavailable {{ --action-accent:var(--muted); }}
    .tech-action-item span {{ display:block; color:var(--subtle); font-size:0.52rem; font-weight:900; text-transform:uppercase; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tech-action-item strong {{ display:block; color:var(--text); font-size:0.78rem; font-weight:950; line-height:1.08; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tech-action-item em {{ display:block; color:var(--muted); font-size:0.58rem; font-style:normal; font-weight:720; line-height:1.08; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tech-checklist {{ display:grid; grid-template-columns:repeat(7,minmax(0,1fr)); gap:6px; margin:0.12rem 0 0.44rem; }}
    .tech-check {{ --check-accent:var(--warning); border:1px solid color-mix(in srgb,var(--check-accent) 26%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 58%,transparent); padding:6px 7px; min-height:44px; overflow:hidden; }}
    .tech-check.pass {{ --check-accent:var(--cg-green); }}
    .tech-check.fail {{ --check-accent:var(--cg-red); }}
    .tech-check.warn {{ --check-accent:var(--warning); }}
    .tech-check b {{ display:block; color:var(--check-accent); font-size:0.58rem; font-weight:950; text-transform:uppercase; line-height:1; }}
    .tech-check span {{ display:block; color:var(--text); font-size:0.68rem; font-weight:850; line-height:1.08; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tech-score-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.4rem; }}
    .tech-score-card {{ --score-accent:var(--warning); border:1px solid color-mix(in srgb,var(--score-accent) 26%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 56%,transparent); padding:7px 8px; min-height:54px; overflow:hidden; }}
    .tech-score-card.status-positive {{ --score-accent:var(--cg-green); }}
    .tech-score-card.status-negative {{ --score-accent:var(--cg-red); }}
    .tech-score-card.status-warning, .tech-score-card.status-neutral {{ --score-accent:var(--warning); }}
    .tech-score-card span {{ display:block; color:var(--subtle); font-size:0.52rem; font-weight:900; text-transform:uppercase; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tech-score-card strong {{ display:block; color:var(--score-accent); font-size:1rem; font-weight:950; line-height:1.04; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tech-score-card em {{ display:block; color:var(--muted); font-size:0.58rem; font-style:normal; font-weight:720; line-height:1.08; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .indicator-group {{ margin:0.24rem 0 0.58rem; }}
    .indicator-group-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin:0 0 0.2rem; }}
    .indicator-group-title {{ color:var(--text); font-size:0.86rem; font-weight:950; line-height:1.1; }}
    .indicator-group-meta {{ color:var(--muted); font-size:0.64rem; font-weight:750; white-space:nowrap; }}
    .indicator-group .indicator-grid {{ grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:8px; margin:0; }}
    .indicator-group .dashboard-card {{ min-height:92px; padding:9px 10px !important; }}
    .tech-chart-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:10px; margin:0.08rem 0 0.02rem; }}
    .tech-chart-title {{ color:var(--text); font-size:1rem; font-weight:950; line-height:1.1; }}
    .tech-chart-meta {{ color:var(--muted); font-size:0.66rem; font-weight:760; }}
    .tech-fg-card {{ border:1px solid var(--card-border); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 56%,transparent); padding:7px 9px 2px; margin:0.12rem 0 0.3rem; }}
    .tech-fg-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:-4px; }}
    .tech-fg-title {{ color:var(--text); font-size:0.78rem; font-weight:950; line-height:1.05; }}
    .tech-fg-meta {{ color:var(--muted); font-size:0.62rem; font-weight:760; white-space:nowrap; }}
    .tech-fg-compact {{ --fg-accent:var(--warning); border:1px solid color-mix(in srgb,var(--fg-accent) 28%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 58%,transparent); padding:8px 10px; margin:0.12rem 0 0.28rem; }}
    .tech-fg-compact-top {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
    .tech-fg-compact-title {{ color:var(--subtle); font-size:0.54rem; font-weight:900; text-transform:uppercase; line-height:1.05; }}
    .tech-fg-compact-value {{ color:var(--fg-accent); font-size:1rem; font-weight:950; line-height:1; white-space:nowrap; }}
    .tech-fg-bar {{ height:6px; border-radius:999px; background:linear-gradient(90deg,var(--cg-red),var(--warning),var(--cg-green)); margin-top:7px; position:relative; overflow:hidden; }}
    .tech-fg-pin {{ position:absolute; top:-3px; width:4px; height:12px; border-radius:999px; background:var(--text); transform:translateX(-2px); }}
    .tech-detail-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin-top:0.45rem; }}
    .tech-detail-card {{ border:1px solid var(--card-border); border-radius:var(--radius-md); background:linear-gradient(180deg,color-mix(in srgb,var(--card-bg) 94%,var(--panel-bg)),var(--card-bg)); box-shadow:0 8px 20px rgba(2,6,23,0.12); padding:10px 11px; }}
    .tech-detail-title {{ color:var(--text); font-size:0.98rem; font-weight:950; margin-bottom:8px; }}
    .tech-metric-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; }}
    .tech-mini-metric {{ border:1px solid var(--card-border); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 54%,transparent); padding:6px 7px; min-height:48px; overflow:hidden; }}
    .tech-mini-label, .tech-mini-metric span {{ display:block; color:var(--subtle); font-size:0.52rem; font-weight:900; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; line-height:1.05; }}
    .tech-mini-value, .tech-mini-metric strong {{ display:block; color:var(--text); font-size:0.76rem; font-weight:920; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; line-height:1.12; writing-mode:horizontal-tb; text-orientation:mixed; }}
    .tech-level-list {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:5px; margin-top:7px; }}
    .tech-level {{ color:var(--muted); border:1px solid var(--card-border); border-radius:999px; padding:3px 6px; font-size:0.58rem; font-weight:850; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; line-height:1.1; writing-mode:horizontal-tb; text-orientation:mixed; }}
    .order-command-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:0.05rem 0 0.28rem; }}
    .order-command-title {{ color:var(--text); font-size:clamp(1.02rem,1.35vw,1.28rem); font-weight:950; line-height:1.1; }}
    .order-command-meta {{ color:var(--muted); font-size:0.72rem; font-weight:780; white-space:nowrap; }}
    .order-command-card {{ border:1px solid var(--card-border); border-radius:var(--radius-md); background:linear-gradient(180deg,color-mix(in srgb,var(--card-bg) 94%,var(--panel-bg)),var(--card-bg)); box-shadow:0 9px 22px rgba(2,6,23,0.14); padding:9px 11px; margin:0.02rem 0 0.42rem; }}
    .order-command-grid {{ display:grid; grid-template-columns:minmax(210px,0.44fr) minmax(520px,1.56fr); gap:9px; align-items:stretch; }}
    .order-core {{ --order-accent:var(--warning); position:relative; border:1px solid color-mix(in srgb,var(--order-accent) 30%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 52%,transparent); padding:8px 10px 8px 12px; overflow:hidden; }}
    .order-core::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--order-accent); }}
    .order-core.status-positive {{ --order-accent:var(--cg-green); }}
    .order-core.status-negative {{ --order-accent:var(--cg-red); }}
    .order-core.status-warning, .order-core.status-neutral {{ --order-accent:var(--warning); }}
    .order-label {{ color:var(--muted); font-size:0.56rem; font-weight:900; text-transform:uppercase; line-height:1.05; }}
    .order-value {{ color:var(--order-accent); font-size:clamp(1.05rem,1.38vw,1.32rem); font-weight:950; line-height:1.02; margin-top:4px; }}
    .order-sub {{ color:var(--text); font-size:0.72rem; font-weight:820; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .order-gauge {{ margin-top:8px; }}
    .order-gauge-track {{ position:relative; height:8px; border-radius:999px; background:linear-gradient(90deg,var(--cg-red),var(--warning),var(--cg-green)); overflow:hidden; }}
    .order-gauge-marker {{ position:absolute; top:-3px; width:4px; height:14px; border-radius:999px; background:var(--text); box-shadow:0 0 0 2px color-mix(in srgb,var(--app-bg) 90%,transparent); transform:translateX(-2px); }}
    .order-gauge-scale {{ display:flex; justify-content:space-between; color:var(--subtle); font-size:0.52rem; font-weight:800; margin-top:5px; }}
    .order-health-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:6px; }}
    .order-chip, .order-action-item {{ --order-chip-accent:var(--warning); border:1px solid color-mix(in srgb,var(--order-chip-accent) 25%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 54%,transparent); padding:7px 8px; min-height:56px; overflow:hidden; }}
    .order-chip.status-positive, .order-action-item.status-positive {{ --order-chip-accent:var(--cg-green); }}
    .order-chip.status-negative, .order-action-item.status-negative {{ --order-chip-accent:var(--cg-red); }}
    .order-chip.status-warning, .order-chip.status-neutral, .order-action-item.status-warning, .order-action-item.status-neutral {{ --order-chip-accent:var(--warning); }}
    .order-chip span, .order-action-item span {{ display:block; color:var(--subtle); font-size:0.52rem; font-weight:900; text-transform:uppercase; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .order-chip strong, .order-action-item strong {{ display:block; color:var(--text); font-size:0.78rem; font-weight:950; line-height:1.08; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .order-chip em, .order-action-item em {{ display:block; color:var(--muted); font-size:0.58rem; font-style:normal; font-weight:720; line-height:1.08; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .order-reasons, .order-context-chips {{ display:flex; flex-wrap:wrap; gap:5px; margin:0.1rem 0 0.45rem; }}
    .order-reason-chip, .order-context-chip {{ color:var(--text); background:color-mix(in srgb,var(--panel-bg) 62%,transparent); border:1px solid var(--card-border); border-radius:999px; padding:3px 8px; font-size:0.66rem; font-weight:850; white-space:nowrap; }}
    .order-context-chip strong {{ color:var(--muted); font-size:0.58rem; font-weight:900; text-transform:uppercase; margin-right:4px; }}
    .order-action-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.44rem; }}
    .order-pressure-panel {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:7px; margin:0.1rem 0 0.35rem; }}
    .mtf-command-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:0.05rem 0 0.28rem; }}
    .mtf-command-title {{ color:var(--text); font-size:clamp(1.02rem,1.35vw,1.28rem); font-weight:950; line-height:1.1; }}
    .mtf-command-meta {{ color:var(--muted); font-size:0.72rem; font-weight:780; white-space:nowrap; }}
    .mtf-command-card {{ border:1px solid var(--card-border); border-radius:var(--radius-md); background:linear-gradient(180deg,color-mix(in srgb,var(--card-bg) 94%,var(--panel-bg)),var(--card-bg)); box-shadow:0 9px 22px rgba(2,6,23,0.14); padding:9px 11px; margin:0.02rem 0 0.42rem; }}
    .mtf-command-grid {{ display:grid; grid-template-columns:minmax(210px,0.42fr) minmax(560px,1.58fr); gap:9px; align-items:stretch; }}
    .mtf-core {{ --mtf-accent:var(--warning); position:relative; border:1px solid color-mix(in srgb,var(--mtf-accent) 30%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 52%,transparent); padding:8px 10px 8px 12px; overflow:hidden; }}
    .mtf-core::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--mtf-accent); }}
    .mtf-core.status-positive {{ --mtf-accent:var(--cg-green); }}
    .mtf-core.status-negative {{ --mtf-accent:var(--cg-red); }}
    .mtf-core.status-warning, .mtf-core.status-neutral {{ --mtf-accent:var(--warning); }}
    .mtf-label {{ color:var(--muted); font-size:0.56rem; font-weight:900; text-transform:uppercase; line-height:1.05; }}
    .mtf-value {{ color:var(--mtf-accent); font-size:clamp(1.05rem,1.38vw,1.32rem); font-weight:950; line-height:1.02; margin-top:4px; }}
    .mtf-sub {{ color:var(--text); font-size:0.72rem; font-weight:820; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .mtf-gauge {{ margin-top:8px; }}
    .mtf-gauge-track {{ position:relative; height:8px; border-radius:999px; background:linear-gradient(90deg,var(--cg-red),var(--warning),var(--cg-green)); overflow:hidden; }}
    .mtf-gauge-marker {{ position:absolute; top:-3px; width:4px; height:14px; border-radius:999px; background:var(--text); box-shadow:0 0 0 2px color-mix(in srgb,var(--app-bg) 90%,transparent); transform:translateX(-2px); }}
    .mtf-gauge-scale {{ display:flex; justify-content:space-between; color:var(--subtle); font-size:0.52rem; font-weight:800; margin-top:5px; }}
    .mtf-health-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:6px; }}
    .mtf-chip, .mtf-action-item, .mtf-summary-card {{ --mtf-chip-accent:var(--warning); border:1px solid color-mix(in srgb,var(--mtf-chip-accent) 25%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 54%,transparent); padding:7px 8px; min-height:56px; overflow:hidden; }}
    .mtf-chip.status-positive, .mtf-action-item.status-positive, .mtf-summary-card.status-positive {{ --mtf-chip-accent:var(--cg-green); }}
    .mtf-chip.status-negative, .mtf-action-item.status-negative, .mtf-summary-card.status-negative {{ --mtf-chip-accent:var(--cg-red); }}
    .mtf-chip.status-warning, .mtf-chip.status-neutral, .mtf-action-item.status-warning, .mtf-action-item.status-neutral, .mtf-summary-card.status-warning, .mtf-summary-card.status-neutral {{ --mtf-chip-accent:var(--warning); }}
    .mtf-chip span, .mtf-action-item span, .mtf-summary-card span {{ display:block; color:var(--subtle); font-size:0.52rem; font-weight:900; text-transform:uppercase; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .mtf-chip strong, .mtf-action-item strong, .mtf-summary-card strong {{ display:block; color:var(--text); font-size:0.78rem; font-weight:950; line-height:1.08; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .mtf-summary-card strong {{ color:var(--mtf-chip-accent); font-size:0.9rem; }}
    .mtf-chip em, .mtf-action-item em, .mtf-summary-card em {{ display:block; color:var(--muted); font-size:0.58rem; font-style:normal; font-weight:720; line-height:1.08; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .mtf-reasons {{ display:flex; flex-wrap:wrap; gap:5px; margin:0.1rem 0 0.45rem; }}
    .mtf-reason-chip {{ color:var(--text); background:color-mix(in srgb,var(--panel-bg) 62%,transparent); border:1px solid var(--card-border); border-radius:999px; padding:3px 8px; font-size:0.66rem; font-weight:850; white-space:nowrap; }}
    .mtf-action-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.44rem; }}
    .mtf-summary-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.38rem; }}
    .mtf-card-high {{ transform:translateY(-2px); box-shadow:0 16px 32px rgba(2,6,23,0.18) !important; min-height:150px !important; }}
    .mtf-card-mid {{ min-height:136px !important; }}
    .mtf-weight-badge {{ display:inline-flex; align-items:center; border:1px solid color-mix(in srgb,var(--accent) 28%,var(--card-border)); border-radius:999px; background:color-mix(in srgb,var(--accent) 12%,transparent); color:var(--text); padding:3px 7px; font-size:0.58rem; font-weight:900; line-height:1; white-space:nowrap; }}
    .ai-command-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:0.05rem 0 0.28rem; }}
    .ai-command-title {{ color:var(--text); font-size:clamp(1.02rem,1.35vw,1.28rem); font-weight:950; line-height:1.1; }}
    .ai-command-meta {{ color:var(--muted); font-size:0.72rem; font-weight:780; white-space:nowrap; }}
    .ai-command-card {{ border:1px solid var(--card-border); border-radius:var(--radius-md); background:linear-gradient(180deg,color-mix(in srgb,var(--card-bg) 94%,var(--panel-bg)),var(--card-bg)); box-shadow:0 9px 22px rgba(2,6,23,0.14); padding:9px 11px; margin:0.02rem 0 0.42rem; }}
    .ai-command-grid {{ display:grid; grid-template-columns:minmax(210px,0.42fr) minmax(560px,1.58fr); gap:9px; align-items:stretch; }}
    .ai-core {{ --ai-accent:var(--warning); position:relative; border:1px solid color-mix(in srgb,var(--ai-accent) 30%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 52%,transparent); padding:8px 10px 8px 12px; overflow:hidden; }}
    .ai-core::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--ai-accent); }}
    .ai-core.status-positive {{ --ai-accent:var(--cg-green); }}
    .ai-core.status-negative {{ --ai-accent:var(--cg-red); }}
    .ai-core.status-warning, .ai-core.status-neutral {{ --ai-accent:var(--warning); }}
    .ai-label {{ color:var(--muted); font-size:0.56rem; font-weight:900; text-transform:uppercase; line-height:1.05; }}
    .ai-value {{ color:var(--ai-accent); font-size:clamp(1.05rem,1.38vw,1.32rem); font-weight:950; line-height:1.02; margin-top:4px; }}
    .ai-sub {{ color:var(--text); font-size:0.72rem; font-weight:820; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .ai-gauge {{ margin-top:8px; }}
    .ai-gauge-track {{ position:relative; height:8px; border-radius:999px; background:linear-gradient(90deg,var(--cg-red),var(--warning),var(--cg-green)); overflow:hidden; }}
    .ai-gauge-marker {{ position:absolute; top:-3px; width:4px; height:14px; border-radius:999px; background:var(--text); box-shadow:0 0 0 2px color-mix(in srgb,var(--app-bg) 90%,transparent); transform:translateX(-2px); }}
    .ai-gauge-scale {{ display:flex; justify-content:space-between; color:var(--subtle); font-size:0.52rem; font-weight:800; margin-top:5px; }}
    .ai-health-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:6px; }}
    .ai-chip, .ai-action-item, .ai-consensus-card, .ai-breakdown-card {{ --ai-chip-accent:var(--warning); border:1px solid color-mix(in srgb,var(--ai-chip-accent) 25%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 54%,transparent); padding:7px 8px; min-height:56px; overflow:hidden; }}
    .ai-chip.status-positive, .ai-action-item.status-positive, .ai-consensus-card.status-positive, .ai-breakdown-card.status-positive {{ --ai-chip-accent:var(--cg-green); }}
    .ai-chip.status-negative, .ai-action-item.status-negative, .ai-consensus-card.status-negative, .ai-breakdown-card.status-negative {{ --ai-chip-accent:var(--cg-red); }}
    .ai-chip.status-warning, .ai-chip.status-neutral, .ai-action-item.status-warning, .ai-action-item.status-neutral, .ai-consensus-card.status-warning, .ai-consensus-card.status-neutral, .ai-breakdown-card.status-warning, .ai-breakdown-card.status-neutral {{ --ai-chip-accent:var(--warning); }}
    .ai-chip span, .ai-action-item span, .ai-consensus-card span, .ai-breakdown-card span {{ display:block; color:var(--subtle); font-size:0.52rem; font-weight:900; text-transform:uppercase; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .ai-chip strong, .ai-action-item strong, .ai-consensus-card strong, .ai-breakdown-card strong {{ display:block; color:var(--text); font-size:0.78rem; font-weight:950; line-height:1.08; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .ai-consensus-card strong, .ai-breakdown-card strong {{ color:var(--ai-chip-accent); font-size:0.92rem; }}
    .ai-chip em, .ai-action-item em, .ai-consensus-card em, .ai-breakdown-card em {{ display:block; color:var(--muted); font-size:0.58rem; font-style:normal; font-weight:720; line-height:1.08; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .ai-reasons {{ display:flex; flex-wrap:wrap; gap:5px; margin:0.1rem 0 0.45rem; }}
    .ai-reason-chip {{ color:var(--text); background:color-mix(in srgb,var(--panel-bg) 62%,transparent); border:1px solid var(--card-border); border-radius:999px; padding:3px 8px; font-size:0.66rem; font-weight:850; white-space:nowrap; }}
    .ai-action-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.44rem; }}
    .ai-consensus-grid, .ai-breakdown-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.44rem; }}
    .bt-command-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:0.05rem 0 0.28rem; }}
    .bt-command-title {{ color:var(--text); font-size:clamp(1.02rem,1.35vw,1.28rem); font-weight:950; line-height:1.1; text-transform:uppercase; }}
    .bt-command-meta {{ color:var(--muted); font-size:0.72rem; font-weight:780; white-space:nowrap; }}
    .bt-command-card {{ border:1px solid var(--card-border); border-radius:var(--radius-md); background:linear-gradient(180deg,color-mix(in srgb,var(--card-bg) 94%,var(--panel-bg)),var(--card-bg)); box-shadow:0 9px 22px rgba(2,6,23,0.14); padding:9px 11px; margin:0.02rem 0 0.42rem; }}
    .bt-command-grid {{ display:grid; grid-template-columns:minmax(210px,0.42fr) minmax(560px,1.58fr); gap:9px; align-items:stretch; }}
    .bt-core {{ --bt-accent:var(--warning); position:relative; border:1px solid color-mix(in srgb,var(--bt-accent) 30%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 52%,transparent); padding:8px 10px 8px 12px; overflow:hidden; }}
    .bt-core::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--bt-accent); }}
    .bt-core.status-positive {{ --bt-accent:var(--cg-green); }}
    .bt-core.status-negative {{ --bt-accent:var(--cg-red); }}
    .bt-core.status-warning, .bt-core.status-neutral {{ --bt-accent:var(--warning); }}
    .bt-label {{ color:var(--muted); font-size:0.56rem; font-weight:900; text-transform:uppercase; line-height:1.05; }}
    .bt-value {{ color:var(--bt-accent); font-size:clamp(1.05rem,1.38vw,1.32rem); font-weight:950; line-height:1.02; margin-top:4px; }}
    .bt-sub {{ color:var(--text); font-size:0.72rem; font-weight:820; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .bt-gauge {{ margin-top:8px; }}
    .bt-gauge-track {{ position:relative; height:8px; border-radius:999px; background:linear-gradient(90deg,var(--cg-red),var(--warning),var(--cg-green)); overflow:hidden; }}
    .bt-gauge-marker {{ position:absolute; top:-3px; width:4px; height:14px; border-radius:999px; background:var(--text); box-shadow:0 0 0 2px color-mix(in srgb,var(--app-bg) 90%,transparent); transform:translateX(-2px); }}
    .bt-gauge-scale {{ display:flex; justify-content:space-between; color:var(--subtle); font-size:0.52rem; font-weight:800; margin-top:5px; }}
    .bt-health-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:6px; }}
    .bt-chip, .bt-kpi, .bt-analytics-card {{ --bt-chip-accent:var(--warning); border:1px solid color-mix(in srgb,var(--bt-chip-accent) 25%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 54%,transparent); padding:7px 8px; min-height:56px; overflow:hidden; }}
    .bt-chip.status-positive, .bt-kpi.status-positive, .bt-analytics-card.status-positive {{ --bt-chip-accent:var(--cg-green); }}
    .bt-chip.status-negative, .bt-kpi.status-negative, .bt-analytics-card.status-negative {{ --bt-chip-accent:var(--cg-red); }}
    .bt-chip.status-warning, .bt-chip.status-neutral, .bt-kpi.status-warning, .bt-kpi.status-neutral, .bt-analytics-card.status-warning, .bt-analytics-card.status-neutral {{ --bt-chip-accent:var(--warning); }}
    .bt-chip span, .bt-kpi span, .bt-analytics-card span {{ display:block; color:var(--subtle); font-size:0.52rem; font-weight:900; text-transform:uppercase; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .bt-chip strong, .bt-kpi strong, .bt-analytics-card strong {{ display:block; color:var(--text); font-size:0.78rem; font-weight:950; line-height:1.08; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .bt-kpi strong, .bt-analytics-card strong {{ color:var(--bt-chip-accent); font-size:0.92rem; }}
    .bt-chip em, .bt-kpi em, .bt-analytics-card em {{ display:block; color:var(--muted); font-size:0.58rem; font-style:normal; font-weight:720; line-height:1.08; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .bt-kpi-grid {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.44rem; }}
    .bt-analytics-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.44rem; }}
    .bt-reasons {{ display:flex; flex-wrap:wrap; gap:5px; margin:0.1rem 0 0.45rem; }}
    .bt-reason-chip {{ color:var(--text); background:color-mix(in srgb,var(--panel-bg) 62%,transparent); border:1px solid var(--card-border); border-radius:999px; padding:3px 8px; font-size:0.66rem; font-weight:850; white-space:nowrap; }}
    .pf-command-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:0.05rem 0 0.28rem; }}
    .pf-command-title {{ color:var(--text); font-size:clamp(1.02rem,1.35vw,1.28rem); font-weight:950; line-height:1.1; text-transform:uppercase; }}
    .pf-command-meta {{ color:var(--muted); font-size:0.72rem; font-weight:780; white-space:nowrap; }}
    .pf-command-card {{ border:1px solid var(--card-border); border-radius:var(--radius-md); background:linear-gradient(180deg,color-mix(in srgb,var(--card-bg) 94%,var(--panel-bg)),var(--card-bg)); box-shadow:0 9px 22px rgba(2,6,23,0.14); padding:9px 11px; margin:0.02rem 0 0.42rem; }}
    .pf-command-grid {{ display:grid; grid-template-columns:minmax(210px,0.42fr) minmax(560px,1.58fr); gap:9px; align-items:stretch; }}
    .pf-core {{ --pf-accent:var(--warning); position:relative; border:1px solid color-mix(in srgb,var(--pf-accent) 30%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 52%,transparent); padding:8px 10px 8px 12px; overflow:hidden; }}
    .pf-core::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--pf-accent); }}
    .pf-core.status-positive {{ --pf-accent:var(--cg-green); }}
    .pf-core.status-negative {{ --pf-accent:var(--cg-red); }}
    .pf-core.status-warning, .pf-core.status-neutral {{ --pf-accent:var(--warning); }}
    .pf-label {{ color:var(--muted); font-size:0.56rem; font-weight:900; text-transform:uppercase; line-height:1.05; }}
    .pf-value {{ color:var(--pf-accent); font-size:clamp(1.05rem,1.38vw,1.32rem); font-weight:950; line-height:1.02; margin-top:4px; }}
    .pf-sub {{ color:var(--text); font-size:0.72rem; font-weight:820; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .pf-gauge {{ margin-top:8px; }}
    .pf-gauge-track {{ position:relative; height:8px; border-radius:999px; background:linear-gradient(90deg,var(--cg-red),var(--warning),var(--cg-green)); overflow:hidden; }}
    .pf-gauge-marker {{ position:absolute; top:-3px; width:4px; height:14px; border-radius:999px; background:var(--text); box-shadow:0 0 0 2px color-mix(in srgb,var(--app-bg) 90%,transparent); transform:translateX(-2px); }}
    .pf-gauge-scale {{ display:flex; justify-content:space-between; color:var(--subtle); font-size:0.52rem; font-weight:800; margin-top:5px; }}
    .pf-health-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6px; }}
    .pf-chip, .pf-kpi, .pf-analytics-card {{ --pf-chip-accent:var(--warning); border:1px solid color-mix(in srgb,var(--pf-chip-accent) 25%,var(--card-border)); border-radius:var(--radius-sm); background:color-mix(in srgb,var(--panel-bg) 54%,transparent); padding:7px 8px; min-height:56px; overflow:hidden; }}
    .pf-chip.status-positive, .pf-kpi.status-positive, .pf-analytics-card.status-positive {{ --pf-chip-accent:var(--cg-green); }}
    .pf-chip.status-negative, .pf-kpi.status-negative, .pf-analytics-card.status-negative {{ --pf-chip-accent:var(--cg-red); }}
    .pf-chip.status-warning, .pf-chip.status-neutral, .pf-kpi.status-warning, .pf-kpi.status-neutral, .pf-analytics-card.status-warning, .pf-analytics-card.status-neutral {{ --pf-chip-accent:var(--warning); }}
    .pf-chip span, .pf-kpi span, .pf-analytics-card span {{ display:block; color:var(--subtle); font-size:0.52rem; font-weight:900; text-transform:uppercase; line-height:1.05; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .pf-chip strong, .pf-kpi strong, .pf-analytics-card strong {{ display:block; color:var(--text); font-size:0.78rem; font-weight:950; line-height:1.08; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .pf-kpi strong, .pf-analytics-card strong {{ color:var(--pf-chip-accent); font-size:0.92rem; }}
    .pf-chip em, .pf-kpi em, .pf-analytics-card em {{ display:block; color:var(--muted); font-size:0.58rem; font-style:normal; font-weight:720; line-height:1.08; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .pf-kpi-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.44rem; }}
    .pf-analytics-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; margin:0.14rem 0 0.44rem; }}
    .pf-reasons {{ display:flex; flex-wrap:wrap; gap:5px; margin:0.1rem 0 0.45rem; }}
    .pf-reason-chip {{ color:var(--text); background:color-mix(in srgb,var(--panel-bg) 62%,transparent); border:1px solid var(--card-border); border-radius:999px; padding:3px 8px; font-size:0.66rem; font-weight:850; white-space:nowrap; }}
    .metric-pill {{ color: var(--text); padding: 3px 7px; font-size: 0.62rem; border-radius: 999px; border: 1px solid var(--card-border); white-space: nowrap; }}
    .metric-pill.buy {{ background: var(--cg-pos-weak); color: var(--cg-green); }}
    .metric-pill.sell {{ background: var(--cg-neg-weak); color: var(--cg-red); }}
    .metric-pill.hold {{ background: color-mix(in srgb, var(--warning) 16%, transparent); color: var(--warning); }}
    .signal-badge {{ background: var(--accent); color: #fff; border-radius: 999px; padding: 4px 9px; font-weight: 800; letter-spacing: .04em !important; }}
    .small-muted {{ color: var(--muted); }}
    .conf-wrap, .dom-wrap {{ background: color-mix(in srgb, var(--panel-bg) 70%, transparent); border: 1px solid var(--card-border); border-radius: var(--radius-md); padding: 3px; }}
    .conf-bar, .dom-bar {{ background: color-mix(in srgb, var(--muted) 16%, transparent); border-radius: 999px; overflow: hidden; min-height: 8px; }}
    .conf-fill, .dom-bull, .dom-bear {{ min-height: 8px; }}
    .signal-row {{ display: flex; gap: 10px; align-items: stretch; flex-wrap: wrap; }}
    .signal-meta, .signal-item {{ color: var(--text); background: color-mix(in srgb, var(--panel-bg) 55%, transparent); border: 1px solid var(--card-border); border-radius: var(--radius-md); padding: 7px 9px; }}
    .risk-badge {{ color: var(--text); border-radius: 999px; padding: 3px 8px; font-size: 0.78rem; font-weight: 800; }}
    .risk-low {{ background: var(--cg-pos-medium); color: var(--cg-green); }}
    .risk-medium {{ background: color-mix(in srgb, var(--warning) 24%, transparent); color: var(--warning); }}
    .risk-high {{ background: var(--cg-neg-medium); color: var(--cg-red); }}

    .section-title {{ color: var(--text); font-size: clamp(1.08rem, 1.45vw, 1.34rem); font-weight: 800; margin: 0.35rem 0 0.12rem; }}
    .section-subtitle {{ color: var(--muted); margin-bottom: var(--section-subtitle-margin); font-size: var(--section-subtitle-size); max-width: 980px; line-height: 1.35; }}
    .table-card {{ min-height: auto !important; padding: 10px 12px !important; }}
    .table-card h5 {{ color: var(--text); }}
    div[data-testid="stMetric"] {{
      background: var(--card-bg) !important;
      border: 1px solid var(--card-border) !important;
      border-radius: var(--radius-lg) !important;
      padding: 10px 12px !important;
      box-shadow: var(--shadow);
      min-height: 78px;
      overflow: hidden;
    }}
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{ color: var(--muted) !important; font-size: 0.7rem !important; font-weight: 750 !important; text-transform: uppercase; }}
    div[data-testid="stMetricValue"] {{ color: var(--text) !important; font-size: clamp(1.02rem, 1.35vw, 1.36rem) !important; font-weight: 800 !important; line-height: 1.08 !important; overflow-wrap: anywhere; }}
    div[data-testid="stMetricDelta"] {{ font-size: 0.74rem !important; }}
    .stDataFrame, [data-testid="stDataFrame"] {{
      border-radius: var(--radius-md);
      overflow: hidden;
      border: 1px solid var(--card-border);
      box-shadow: 0 10px 26px rgba(2,6,23,0.10);
      background: var(--table-bg);
    }}
    [data-testid="stDataFrame"] [role="grid"] {{
      background: var(--card-bg) !important;
    }}
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] [data-testid="stTableStyledTable"] thead tr th {{
      background: var(--table-header-bg) !important;
      color: var(--muted) !important;
      font-size: 0.72rem !important;
      font-weight: 800 !important;
      text-transform: uppercase;
      border-color: var(--card-border) !important;
    }}
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] [data-testid="stTableStyledTable"] tbody tr td {{
      color: var(--text) !important;
      border-color: color-mix(in srgb, var(--card-border) 72%, transparent) !important;
      font-variant-numeric: tabular-nums;
    }}
    [data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"],
    [data-testid="stDataFrame"] [data-testid="stTableStyledTable"] tbody tr:hover td {{
      background: color-mix(in srgb, var(--accent) 7%, transparent) !important;
    }}
    [data-testid="stDataFrame"] input,
    [data-testid="stDataFrame"] textarea,
    [data-testid="stDataFrame"] [contenteditable="true"] {{
      background: var(--input-bg) !important;
      color: var(--text) !important;
      border-color: var(--card-border) !important;
      caret-color: var(--accent) !important;
    }}
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {{
      background: var(--input-bg) !important;
      color: var(--text) !important;
      border-color: var(--card-border) !important;
      caret-color: var(--accent) !important;
    }}
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stNumberInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"] > div {{
      background: var(--input-bg) !important;
      color: var(--text) !important;
      border-color: var(--card-border) !important;
      border-radius: var(--radius-md) !important;
      box-shadow: none !important;
    }}
    [data-testid="stTextInput"] [data-baseweb="input"]:hover,
    [data-testid="stNumberInput"] [data-baseweb="input"]:hover,
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
    [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {{
      border-color: var(--cg-green) !important;
      box-shadow: 0 0 0 3px var(--cg-pos-weak) !important;
    }}
    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] [role="button"],
    [data-testid="stNumberInput"] [data-baseweb="button"],
    [data-testid="stNumberInput"] [data-baseweb="base-input"] > div:last-child,
    [data-testid="stNumberInput"] [data-baseweb="input"] button,
    [data-testid="stNumberInput"] [data-baseweb="input"] [aria-label],
    [data-testid="stNumberInput"] [data-testid*="Step"] {{
      background: var(--input-bg) !important;
      color: var(--text) !important;
      border-color: var(--card-border) !important;
    }}
    [data-testid="stNumberInput"] svg,
    [data-testid="stTextInput"] svg {{
      color: var(--muted) !important;
      fill: currentColor !important;
    }}
    [data-testid="stNumberInput"] button:hover,
    [data-testid="stNumberInput"] [role="button"]:hover,
    [data-testid="stNumberInput"] [data-baseweb="button"]:hover,
    [data-testid="stNumberInput"] [data-baseweb="input"] button:hover,
    [data-testid="stNumberInput"] [data-baseweb="input"] [aria-label]:hover,
    [data-testid="stNumberInput"] [data-testid*="Step"]:hover {{
      background: color-mix(in srgb, var(--accent) 10%, var(--input-bg)) !important;
      color: var(--text) !important;
    }}
    .ag-root-wrapper, .ag-root {{
      background: var(--table-bg) !important;
      color: var(--text) !important;
      border-color: var(--card-border) !important;
    }}
    .ag-header {{ background: var(--table-header-bg) !important; }}
    .cg-table-wrap {{
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      background: var(--table-bg);
      box-shadow: 0 16px 34px rgba(2,6,23,0.18);
      scrollbar-width: thin;
    }}
    .cg-table {{
      width: 100%;
      min-width: 1080px;
      border-collapse: separate;
      border-spacing: 0;
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }}
    .cg-table thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--table-header-bg);
      color: color-mix(in srgb, var(--text) 86%, var(--accent) 14%);
      border-bottom: 1px solid var(--card-border);
      font-size: 0.72rem;
      font-weight: 850;
      line-height: 1.1;
      padding: 14px 14px;
      text-align: right;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .cg-table thead th:first-child {{ text-align: left; padding-left: 18px; }}
    .cg-table tbody tr {{ transition: background .16s ease; }}
    .cg-table tbody tr:hover {{ background: var(--cg-pos-weak); }}
    .cg-table tbody td {{
      border-bottom: 1px solid color-mix(in srgb, var(--card-border) 74%, transparent);
      padding: 13px 14px;
      text-align: right;
      vertical-align: middle;
      white-space: nowrap;
      font-size: 0.9rem;
      font-weight: 650;
    }}
    .cg-table tbody tr:last-child td {{ border-bottom: 0; }}
    .cg-table tbody td:first-child {{ text-align: left; padding-left: 18px; }}
    .cg-symbol-cell {{ display: inline-flex; align-items: center; gap: 10px; min-width: 150px; }}
    .coin-logo {{
      width: 26px;
      height: 26px;
      border-radius: 50%;
      flex: 0 0 26px;
      background: color-mix(in srgb, var(--panel-bg) 78%, var(--accent) 8%);
      border: 1px solid color-mix(in srgb, var(--card-border) 70%, transparent);
      object-fit: cover;
      box-shadow: 0 4px 12px rgba(2,6,23,0.16);
    }}
    .coin-logo-fallback {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: var(--text);
      font-size: .66rem;
      font-weight: 900;
    }}
    .cg-symbol-text {{ color: var(--text); font-weight: 900; letter-spacing: .01em !important; }}
    .cg-name-text {{ display: block; color: var(--muted); font-size: 0.68rem; font-weight: 650; margin-top: 2px; }}
    .cg-heat-pos {{ background: var(--cg-pos-weak); color: var(--cg-green); font-weight: 850; }}
    .cg-heat-neg {{ background: var(--cg-neg-weak); color: var(--cg-red); font-weight: 850; }}
    .cg-signal-buy {{ color: var(--success); font-weight: 900; }}
    .cg-signal-sell {{ color: var(--danger); font-weight: 900; }}
    .cg-signal-hold {{ color: var(--warning); font-weight: 850; }}
    @media (max-width: 760px) {{
      .cg-table thead th, .cg-table tbody td {{ padding: 11px 12px; font-size: 0.84rem; }}
      .cg-symbol-cell {{ min-width: 132px; }}
    }}
    .mover-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
      gap: 7px;
      margin: 0.14rem 0 0.3rem;
    }}
    .mover-card {{
      min-height: 72px !important;
      padding: 8px 9px !important;
      border-radius: var(--radius-sm) !important;
      display: flex;
      flex-direction: column;
      align-items: stretch;
      justify-content: space-between;
      gap: 5px;
      overflow: hidden !important;
    }}
    .mover-topline {{ display: flex; align-items: center; justify-content: space-between; gap: 7px; min-width: 0; }}
    .mover-label {{ font-size: 0.52rem; font-weight: 900; text-transform: uppercase; border-radius: 999px; padding: 2px 6px; border: 1px solid var(--card-border); white-space: nowrap; }}
    .status-positive .mover-label {{ color: var(--cg-green); background: var(--cg-pos-weak); }}
    .status-negative .mover-label {{ color: var(--cg-red); background: var(--cg-neg-weak); }}
    .mover-metric-row {{ display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }}
    .mover-card .metric-val {{ font-size: clamp(0.95rem,1.22vw,1.12rem) !important; margin-top: 0 !important; }}
    .mover-card .metric-subtext {{ color: var(--muted); font-size: 0.66rem; font-weight: 650; white-space: nowrap; }}
    .mover-card .cg-symbol-cell {{ min-width: 0; gap: 7px; }}
    .mover-card .coin-logo {{ width: 19px; height: 19px; flex-basis: 19px; }}
    .mover-card .cg-name-text {{ display: none; }}
    .mover-card:hover {{ border-color: color-mix(in srgb, var(--card-accent) 44%, var(--card-border)); }}
    .scanner-head {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 0.02rem 0 0.08rem; }}
    .scanner-title {{ color: var(--text); font-size: 0.95rem; font-weight: 900; line-height: 1.1; }}
    .scanner-status-line {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
      gap: 5px;
      color: var(--muted);
      font-size: 0.7rem;
      font-weight: 650;
      margin: 0;
    }}
    .scanner-status-line span {{
      color: var(--text);
      background: color-mix(in srgb, var(--panel-bg) 62%, transparent);
      border: 1px solid var(--card-border);
      border-radius: 999px;
      padding: 3px 7px;
      font-weight: 800;
      white-space: nowrap;
    }}
    #MainMenu,
    footer,
    div[data-testid="stStatusWidget"],
    div[data-testid="stDeployButton"],
    div[data-testid="stMainMenu"],
    div[data-testid="stAppDeployButton"],
    div[data-testid="viewerBadge_container__1QSob"],
    button[title="View fullscreen"] {{
      display: none !important;
      visibility: hidden !important;
      pointer-events: none !important;
    }}
    header[data-testid="stHeader"] {{
      display: block !important;
      visibility: visible !important;
      pointer-events: none !important;
      background: transparent !important;
      box-shadow: none !important;
      height: 3rem !important;
      min-height: 3rem !important;
      z-index: 999998 !important;
      overflow: visible !important;
    }}
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"] {{
      display: flex !important;
      visibility: visible !important;
      pointer-events: auto !important;
      background: transparent !important;
      box-shadow: none !important;
    }}
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[kind="header"],
    button[aria-label*="sidebar" i],
    button[title*="sidebar" i] {{
      display: inline-flex !important;
      visibility: visible !important;
      pointer-events: auto !important;
      align-items: center !important;
      justify-content: center !important;
      color: var(--text) !important;
      background: color-mix(in srgb, var(--panel-bg) 82%, transparent) !important;
      border: 1px solid var(--card-border) !important;
      border-radius: var(--radius-md) !important;
      box-shadow: 0 10px 24px rgba(2,6,23,0.24) !important;
    }}
    [data-testid="stSidebarCollapsedControl"] {{
      position: fixed !important;
      top: 0.72rem !important;
      left: 0.72rem !important;
      z-index: 999999 !important;
    }}
    div[data-testid="stVerticalBlock"] {{ gap: var(--vertical-block-gap, 0.65rem) !important; }}
    div[data-testid="column"] {{ min-width: 0 !important; }}
    hr {{ margin: 0.75rem 0 !important; border-color: var(--card-border) !important; }}
    [data-testid="stExpander"] {{ border-color: var(--card-border) !important; border-radius: var(--radius-lg) !important; }}
    [data-testid="stExpander"],
    [data-testid="stForm"],
    [data-testid="stForm"] > div {{
      background: color-mix(in srgb, var(--panel-bg) 72%, transparent) !important;
      color: var(--text) !important;
      border-color: var(--card-border) !important;
    }}

    div[role="radiogroup"] {{
      display: flex;
      gap: 6px;
      flex-wrap: nowrap;
      justify-content: flex-start;
      align-items: center;
      overflow-x: auto;
      overflow-y: hidden;
      scrollbar-width: thin;
      padding: 4px;
      margin: 0.1rem 0 1.05rem;
      background: color-mix(in srgb, var(--panel-bg) 78%, transparent);
      border: 1px solid var(--card-border);
      border-radius: 999px;
      box-shadow: 0 10px 24px rgba(2,6,23,0.10);
      width: 100%;
      max-width: 1180px;
    }}
    div[role="radiogroup"] label {{
      border-radius: 999px;
      border: 1px solid transparent;
      padding: 7px 13px;
      min-height: 34px;
      flex: 0 0 auto;
      justify-content: center;
      color: var(--muted) !important;
      background: transparent;
      transition: background .16s ease, border-color .16s ease, color .16s ease, transform .16s ease;
    }}
    div[role="radiogroup"] label:hover {{
      color: var(--text) !important;
      background: var(--tab-bg);
      border-color: var(--tab-border);
      transform: translateY(-1px);
    }}
    div[role="radiogroup"] label:has(input:checked) {{
      color: var(--text) !important;
      background: var(--tab-active);
      border-color: var(--cg-green);
      box-shadow: 0 7px 18px color-mix(in srgb, var(--accent) 10%, transparent);
      font-weight: 800;
    }}
    div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {{ margin: 0 !important; white-space: nowrap; font-size: 0.84rem; line-height: 1.1; }}
    div[role="radiogroup"] label > div:first-child {{ display: none !important; }}

    button[aria-label*="Theme"], button[title*="Theme"], [data-testid="stThemeToggle"] {{ display: none !important; }}
    .stButton>button {{
      border-radius: var(--radius-md);
      padding: 0.72rem 0.95rem;
      font-weight: 750;
      border: 1px solid var(--card-border);
      background: var(--input-bg) !important;
      color: var(--text) !important;
      transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease, background .16s ease, color .16s ease;
    }}
    .stButton>button[kind="primary"],
    .stButton>button[data-testid="baseButton-primary"] {{
      background: var(--accent) !important;
      border-color: var(--accent) !important;
      color: var(--app-bg) !important;
    }}
    .stButton>button:hover {{ transform: translateY(-1px); background: color-mix(in srgb, var(--accent) 10%, var(--input-bg)) !important; color: var(--text) !important; border-color: var(--accent); box-shadow: 0 10px 24px color-mix(in srgb, var(--accent) 16%, transparent); }}
    .stButton>button[kind="primary"]:hover,
    .stButton>button[data-testid="baseButton-primary"]:hover {{
      background: color-mix(in srgb, var(--accent) 86%, #ffffff) !important;
      color: var(--app-bg) !important;
    }}
    .stSidebar .stButton>button {{ width: 100%; }}
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
      background: var(--input-bg) !important;
      border: 1px solid var(--card-border) !important;
      border-radius: var(--radius-md) !important;
      color: var(--text) !important;
      min-height: 36px !important;
      box-shadow: 0 8px 20px rgba(2,6,23,0.08);
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {{
      border-color: var(--cg-green) !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] span {{
      background-color: transparent !important;
      color: var(--text) !important;
      border-color: var(--card-border) !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] input {{
      caret-color: transparent !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] svg {{
      color: var(--muted) !important;
      fill: currentColor !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {{
      border-color: var(--cg-green) !important;
      box-shadow: 0 0 0 3px var(--cg-pos-weak) !important;
    }}
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] > div > div,
    [data-baseweb="menu"],
    [data-baseweb="menu"] ul,
    [role="listbox"],
    [role="menu"],
    ul[role="listbox"],
    div[data-baseweb="select-dropdown"],
    div[data-baseweb="select-dropdown"] > div,
    div[data-baseweb="select-dropdown"] ul {{
      background: var(--panel-bg) !important;
      color: var(--text) !important;
      border-color: var(--card-border) !important;
      box-shadow: var(--shadow) !important;
    }}
    [role="option"],
    [role="menuitem"],
    [data-baseweb="menu"] li,
    [data-baseweb="select-dropdown"] li,
    [data-baseweb="select-dropdown"] li > div {{
      background: var(--panel-bg) !important;
      color: var(--text) !important;
    }}
    [role="option"]:hover,
    [role="option"][aria-selected="true"],
    [role="menuitem"]:hover,
    [data-baseweb="menu"] li:hover,
    [data-baseweb="select-dropdown"] li:hover,
    [data-baseweb="select-dropdown"] li:hover > div {{
      background: color-mix(in srgb, var(--accent) 13%, var(--panel-bg)) !important;
      color: var(--text) !important;
    }}
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {{
      color: var(--text) !important;
    }}
    input, textarea, [data-baseweb="input"], [data-baseweb="input"] > div, [data-baseweb="input"] input {{
      background: var(--input-bg) !important;
      color: var(--text) !important;
      border-color: var(--card-border) !important;
      caret-color: var(--accent) !important;
    }}
    input::placeholder, textarea::placeholder {{ color: var(--subtle) !important; }}
    [data-testid="stMarkdownContainer"] code,
    [data-testid="stMarkdownContainer"] p code {{
      background: color-mix(in srgb, var(--panel-bg) 86%, var(--accent) 10%) !important;
      color: var(--success) !important;
      border: 1px solid var(--card-border) !important;
      border-radius: var(--radius-sm) !important;
      padding: 0.12rem 0.34rem !important;
      font-weight: 750 !important;
    }}
    button:focus-visible, input:focus-visible, [role="button"]:focus-visible, [role="slider"]:focus-visible {{ outline: 2px solid var(--cg-green) !important; outline-offset: 2px !important; }}
    .stAlert {{ border-radius: var(--radius-md) !important; border: 1px solid var(--card-border) !important; }}
    .news-category-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin: 0.08rem 0 0.72rem;
    }}
    .news-category-pill {{
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 6px 11px;
      border: 1px solid var(--card-border);
      border-radius: 999px;
      background: color-mix(in srgb, var(--panel-bg) 76%, transparent);
      color: var(--muted);
      font-size: 0.76rem;
      font-weight: 780;
      line-height: 1;
    }}
    .news-hero {{
      position: relative;
      display: grid;
      grid-template-columns: minmax(0, 1.65fr) minmax(220px, 0.7fr);
      gap: 14px;
      padding: 18px;
      margin: 0 0 0.85rem;
      border: 1px solid color-mix(in srgb, var(--accent) 30%, var(--card-border));
      border-radius: var(--radius-lg);
      background:
        radial-gradient(circle at 5% 10%, color-mix(in srgb, var(--accent) 13%, transparent), transparent 28%),
        linear-gradient(180deg, var(--card-bg), color-mix(in srgb, var(--panel-bg) 86%, transparent));
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .news-eyebrow, .news-meta, .news-card-meta {{
      color: var(--subtle);
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
      line-height: 1.18;
    }}
    .news-hero h2 {{
      margin: 0.32rem 0 0.4rem !important;
      color: var(--text) !important;
      font-size: clamp(1.45rem, 2.3vw, 2.2rem) !important;
      line-height: 1.06 !important;
      font-weight: 950 !important;
    }}
    .news-summary, .news-intel, .news-card-summary {{
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.42;
    }}
    .news-badge-row, .news-tag-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }}
    .news-badge, .news-tag {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      border: 1px solid var(--card-border);
      background: color-mix(in srgb, var(--panel-bg) 72%, transparent);
      padding: 4px 8px;
      color: var(--muted);
      font-size: 0.68rem;
      font-weight: 820;
      line-height: 1;
    }}
    .news-badge.bullish {{ color: var(--cg-green); border-color: color-mix(in srgb, var(--cg-green) 38%, var(--card-border)); background: var(--cg-pos-weak); }}
    .news-badge.bearish {{ color: var(--cg-red); border-color: color-mix(in srgb, var(--cg-red) 38%, var(--card-border)); background: var(--cg-neg-weak); }}
    .news-badge.neutral {{ color: var(--warning); border-color: color-mix(in srgb, var(--warning) 36%, var(--card-border)); background: var(--cg-warn-weak); }}
    .news-badge.high {{ color: var(--cg-red); }}
    .news-badge.medium {{ color: var(--warning); }}
    .news-badge.low {{ color: var(--muted); }}
    .news-link {{
      color: var(--accent) !important;
      font-weight: 850;
      text-decoration: none !important;
    }}
    .news-link:hover {{ text-decoration: underline !important; }}
    .news-hero-side {{
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      background: color-mix(in srgb, var(--panel-bg) 74%, transparent);
      padding: 13px;
      align-self: stretch;
    }}
    .news-page-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.75fr) minmax(280px, 0.75fr);
      gap: 14px;
      align-items: start;
    }}
    .news-feed, .news-sidebar {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-width: 0;
    }}
    .news-section-title {{
      color: var(--text);
      font-size: 1rem;
      font-weight: 950;
      line-height: 1.08;
      margin: 0.12rem 0 0.18rem;
    }}
    .news-card, .news-widget {{
      border: 1px solid var(--card-border);
      border-radius: var(--radius-lg);
      background: linear-gradient(180deg, var(--card-bg), color-mix(in srgb, var(--panel-bg) 86%, transparent));
      box-shadow: var(--shadow);
      padding: 13px 14px;
      overflow: hidden;
    }}
    .news-card h3 {{
      color: var(--text) !important;
      font-size: 1rem !important;
      line-height: 1.18 !important;
      margin: 0.22rem 0 0.28rem !important;
      font-weight: 930 !important;
    }}
    .news-widget-title {{
      color: var(--text);
      font-size: 0.86rem;
      font-weight: 920;
      margin-bottom: 8px;
    }}
    .news-stat-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0,1fr));
      gap: 7px;
    }}
    .news-stat {{
      border: 1px solid var(--card-border);
      border-radius: var(--radius-md);
      padding: 9px;
      background: color-mix(in srgb, var(--input-bg) 76%, transparent);
    }}
    .news-stat span {{
      display: block;
      color: var(--subtle);
      font-size: 0.66rem;
      font-weight: 850;
      text-transform: uppercase;
    }}
    .news-stat strong {{
      display: block;
      color: var(--text);
      font-size: 1rem;
      font-weight: 950;
      margin-top: 4px;
    }}
    .news-list {{
      display: flex;
      flex-direction: column;
      gap: 7px;
      margin: 0;
      padding: 0;
    }}
    .news-list-item {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid color-mix(in srgb, var(--card-border) 70%, transparent);
      padding: 7px 0;
      color: var(--muted);
      font-size: 0.78rem;
      line-height: 1.25;
    }}
    .news-list-item:last-child {{ border-bottom: 0; }}
    .news-list-item strong {{ color: var(--text); }}
    @media (max-width: 980px) {{
      .news-hero, .news-page-grid {{ grid-template-columns: 1fr; }}
      .news-stat-grid {{ grid-template-columns: 1fr; }}
    }}
    .trade-decision-card {{
      --decision-accent: var(--warning);
      position: relative;
      display: grid;
      grid-template-columns: minmax(190px, 0.72fr) minmax(520px, 1.9fr) minmax(180px, 0.58fr);
      gap: 12px;
      align-items: stretch;
      margin: 0.02rem 0 0.78rem;
      padding: 12px 14px;
      border: 1px solid color-mix(in srgb, var(--decision-accent) 58%, var(--card-border));
      border-radius: var(--radius-lg);
      background:
        radial-gradient(circle at 6% 20%, color-mix(in srgb, var(--decision-accent) 18%, transparent), transparent 30%),
        linear-gradient(135deg, color-mix(in srgb, var(--decision-accent) 10%, transparent), transparent 36%),
        linear-gradient(180deg, color-mix(in srgb, var(--card-bg) 96%, var(--panel-bg)), var(--card-bg));
      box-shadow: 0 14px 32px rgba(2,6,23,0.22), 0 0 0 1px rgba(255,255,255,0.02) inset;
      overflow: visible;
    }}
    .trade-decision-card.status-positive {{ --decision-accent: var(--cg-green); }}
    .trade-decision-card.status-negative {{ --decision-accent: var(--cg-red); }}
    .trade-decision-card.status-warning {{ --decision-accent: var(--warning); }}
    .trade-decision-card.status-neutral {{ --decision-accent: var(--subtle); }}
    .trade-decision-card::before {{
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 4px;
      background: var(--decision-accent);
      box-shadow: 0 0 18px color-mix(in srgb, var(--decision-accent) 72%, transparent);
    }}
    .trade-decision-core {{
      display: flex;
      flex-direction: column;
      justify-content: center;
      min-width: 0;
      padding-left: 4px;
    }}
    .trade-decision-symbol {{
      color: var(--muted);
      font-size: 0.72rem;
      font-weight: 850;
      line-height: 1.05;
      text-transform: uppercase;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .trade-decision-verdict {{
      color: var(--decision-accent);
      font-size: clamp(1.28rem, 2.1vw, 2rem);
      font-weight: 950;
      line-height: 1.02;
      margin-top: 5px;
      white-space: nowrap;
      text-shadow: 0 0 18px color-mix(in srgb, var(--decision-accent) 28%, transparent);
    }}
    .trade-decision-action {{
      color: var(--text);
      font-size: 0.78rem;
      font-weight: 840;
      line-height: 1.1;
      margin-top: 6px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .trade-decision-metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 7px;
      align-items: stretch;
      overflow: visible;
    }}
    .trade-decision-metric {{
      border: 1px solid color-mix(in srgb, var(--decision-accent) 20%, var(--card-border));
      border-radius: var(--radius-sm);
      background: color-mix(in srgb, var(--panel-bg) 58%, transparent);
      padding: 7px 8px;
      min-height: 58px;
      overflow: visible;
    }}
    .trade-decision-metric span {{
      display: block;
      color: var(--subtle);
      font-size: 0.52rem;
      font-weight: 900;
      text-transform: uppercase;
      line-height: 1.05;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .trade-decision-metric strong {{
      display: block;
      color: var(--text);
      font-size: 0.86rem;
      font-weight: 930;
      line-height: 1.08;
      margin-top: 6px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-variant-numeric: tabular-nums;
    }}
    .trade-decision-metric.is-primary strong {{ color: var(--decision-accent); }}
    .trade-decision-reason {{
      display: flex;
      flex-direction: column;
      justify-content: center;
      min-width: 0;
      border-left: 1px solid var(--card-border);
      padding-left: 12px;
    }}
    .trade-decision-reason span {{
      color: var(--subtle);
      font-size: 0.52rem;
      font-weight: 900;
      text-transform: uppercase;
      line-height: 1.05;
    }}
    .trade-decision-reason strong {{
      color: var(--text);
      font-size: 0.8rem;
      font-weight: 820;
      line-height: 1.18;
      margin-top: 6px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .metric-label-with-info {{
      display: inline-flex !important;
      align-items: center;
      min-width: 0;
      max-width: 100%;
      vertical-align: baseline;
      overflow: visible !important;
      gap: 0;
    }}
    .metric-label-text {{
      display: inline-flex !important;
      align-items: center;
      min-width: 0;
      overflow: visible !important;
    }}
    .metric-info-wrap {{
      position: relative;
      display: inline-flex !important;
      align-items: center;
      justify-content: center;
      width: 11px !important;
      height: 11px !important;
      min-width: 11px !important;
      max-width: 11px !important;
      min-height: 11px !important;
      max-height: 11px !important;
      flex: 0 0 11px !important;
      margin: 0 0 0 4px !important;
      padding: 0 !important;
      border: 0 !important;
      background: transparent !important;
      color: inherit !important;
      font-size: 0 !important;
      font-weight: 400 !important;
      line-height: 1 !important;
      text-transform: none !important;
      white-space: normal !important;
      overflow: visible !important;
      text-overflow: clip !important;
      vertical-align: text-top;
      outline: none;
    }}
    .metric-info-icon {{
      box-sizing: border-box !important;
      width: 11px !important;
      height: 11px !important;
      min-width: 11px !important;
      max-width: 11px !important;
      min-height: 11px !important;
      max-height: 11px !important;
      flex: 0 0 11px !important;
      border-radius: 50% !important;
      border: 1px solid var(--muted) !important;
      color: var(--muted) !important;
      background: transparent !important;
      padding: 0 !important;
      margin: 0 !important;
      font-size: 8px !important;
      font-weight: 700 !important;
      line-height: 11px !important;
      text-align: center !important;
      display: inline-flex !important;
      align-items: center !important;
      justify-content: center !important;
      opacity: 0.58 !important;
      cursor: help !important;
      text-transform: none !important;
      letter-spacing: 0 !important;
      box-shadow: none !important;
      text-decoration: none !important;
      overflow: hidden !important;
      text-overflow: clip !important;
      transform: none !important;
    }}
    .metric-info-wrap:hover .metric-info-icon,
    .metric-info-wrap:focus .metric-info-icon,
    .metric-info-wrap:focus-within .metric-info-icon {{
      opacity: 1 !important;
      color: var(--text) !important;
      border-color: var(--text) !important;
      background: transparent !important;
    }}
    .metric-label-with-info .metric-tooltip-content {{
      position: absolute !important;
      z-index: 9999999 !important;
      left: 50% !important;
      bottom: calc(100% + 10px) !important;
      top: auto !important;
      display: block !important;
      width: 250px !important;
      max-width: min(260px, 80vw) !important;
      min-width: 0 !important;
      height: auto !important;
      max-height: none !important;
      transform: translateX(-50%) translateY(3px) !important;
      margin: 0 !important;
      padding: 11px 13px !important;
      border: 1px solid var(--card-border) !important;
      border-radius: 11px !important;
      background: var(--panel-bg) !important;
      box-shadow: 0 16px 40px rgba(0,0,0,.35) !important;
      color: var(--muted) !important;
      font-size: 12px !important;
      font-weight: 500 !important;
      line-height: 1.45 !important;
      text-transform: none !important;
      white-space: normal !important;
      text-align: left !important;
      pointer-events: none !important;
      opacity: 0 !important;
      visibility: hidden !important;
      text-overflow: clip !important;
      transition: opacity 90ms ease, transform 90ms ease, visibility 90ms ease !important;
    }}
    .trade-decision-metric span:has(.metric-info-wrap),
    .market-status-label:has(.metric-info-wrap),
    .market-intel-label:has(.metric-info-wrap),
    .market-intel-chip span:has(.metric-info-wrap),
    .direction-gauge-label:has(.metric-info-wrap),
    .intel-metric span:has(.metric-info-wrap),
    .metric-label:has(.metric-info-wrap),
    .tech-label:has(.metric-info-wrap),
    .tech-mini-label:has(.metric-info-wrap),
    .tech-health-chip span:has(.metric-info-wrap),
    .tech-action-item span:has(.metric-info-wrap),
    .tech-score-card span:has(.metric-info-wrap),
    .smc-label:has(.metric-info-wrap),
    .smc-health-chip span:has(.metric-info-wrap),
    .smc-setup-item span:has(.metric-info-wrap),
    .smc-summary-card span:has(.metric-info-wrap),
    .order-label:has(.metric-info-wrap),
    .order-chip span:has(.metric-info-wrap),
    .order-action-item span:has(.metric-info-wrap),
    .mtf-label:has(.metric-info-wrap),
    .mtf-chip span:has(.metric-info-wrap),
    .mtf-action-item span:has(.metric-info-wrap),
    .mtf-summary-card span:has(.metric-info-wrap),
    .ai-label:has(.metric-info-wrap),
    .ai-chip span:has(.metric-info-wrap),
    .ai-action-item span:has(.metric-info-wrap),
    .ai-consensus-card span:has(.metric-info-wrap),
    .ai-breakdown-card span:has(.metric-info-wrap),
    .bt-label:has(.metric-info-wrap),
    .bt-chip span:has(.metric-info-wrap),
    .bt-kpi span:has(.metric-info-wrap),
    .bt-analytics-card span:has(.metric-info-wrap),
    .pf-label:has(.metric-info-wrap),
    .pf-chip span:has(.metric-info-wrap),
    .pf-kpi span:has(.metric-info-wrap),
    .pf-analytics-card span:has(.metric-info-wrap),
    .dashboard-card h4:has(.metric-info-wrap),
    .dashboard-tile h4:has(.metric-info-wrap) {{
      overflow: visible !important;
      text-overflow: clip !important;
    }}
    .metric-info-wrap::after {{
      display: none !important;
    }}
    .metric-info-wrap::before {{
      content: "";
      position: absolute;
      z-index: 10000000;
      left: 50%;
      bottom: calc(100% + 5px);
      top: auto;
      width: 9px;
      height: 9px;
      border-right: 1px solid var(--card-border);
      border-bottom: 1px solid var(--card-border);
      background: var(--panel-bg);
      transform: translateX(-50%) rotate(45deg);
      pointer-events: none;
      opacity: 0;
      visibility: hidden;
      transition: opacity 120ms ease, visibility 120ms ease;
    }}
    .metric-info-wrap.tooltip-below .metric-tooltip-content {{
      top: calc(100% + 10px) !important;
      bottom: auto !important;
      transform: translateX(-50%) translateY(-4px) !important;
    }}
    .metric-info-wrap.tooltip-below::before {{
      top: calc(100% + 5px) !important;
      bottom: auto !important;
      border-right: 0 !important;
      border-bottom: 0 !important;
      border-left: 1px solid var(--card-border) !important;
      border-top: 1px solid var(--card-border) !important;
      transform: translateX(-50%) rotate(45deg) !important;
    }}
    .metric-info-wrap.tooltip-left .metric-tooltip-content {{
      left: auto !important;
      right: calc(100% + 10px) !important;
      top: 50% !important;
      bottom: auto !important;
      transform: translateY(-50%) translateX(4px) !important;
    }}
    .metric-info-wrap.tooltip-right .metric-tooltip-content {{
      left: calc(100% + 10px) !important;
      top: 50% !important;
      bottom: auto !important;
      transform: translateY(-50%) translateX(-4px) !important;
    }}
    .trade-decision-card {{
      z-index: 20;
    }}
    .trade-decision-card,
    .trade-decision-card .trade-decision-core,
    .trade-decision-card .trade-decision-metrics,
    .trade-decision-card .trade-decision-metric,
    .trade-decision-card .metric-label-with-info,
    .trade-decision-card .metric-info-wrap {{
      overflow: visible !important;
    }}
    .metric-info-wrap:hover .metric-tooltip-content,
    .metric-info-wrap:focus .metric-tooltip-content,
    .metric-info-wrap:focus-within .metric-tooltip-content {{
      opacity: 1 !important;
      visibility: visible !important;
      transform: translateX(-50%) translateY(0) !important;
    }}
    .trade-decision-card .metric-info-wrap:hover .metric-tooltip-content,
    .trade-decision-card .metric-info-wrap:focus .metric-tooltip-content,
    .trade-decision-card .metric-info-wrap:focus-within .metric-tooltip-content,
    .metric-info-wrap.tooltip-below:hover .metric-tooltip-content,
    .metric-info-wrap.tooltip-below:focus .metric-tooltip-content,
    .metric-info-wrap.tooltip-below:focus-within .metric-tooltip-content {{
      transform: translateX(-50%) translateY(0) !important;
    }}
    .metric-info-wrap.tooltip-left:hover .metric-tooltip-content,
    .metric-info-wrap.tooltip-left:focus .metric-tooltip-content,
    .metric-info-wrap.tooltip-left:focus-within .metric-tooltip-content {{
      transform: translateY(-50%) translateX(0) !important;
    }}
    .metric-info-wrap.tooltip-right:hover .metric-tooltip-content,
    .metric-info-wrap.tooltip-right:focus .metric-tooltip-content,
    .metric-info-wrap.tooltip-right:focus-within .metric-tooltip-content {{
      transform: translateY(-50%) translateX(0) !important;
    }}
    .metric-info-wrap:hover::before,
    .metric-info-wrap:focus::before,
    .metric-info-wrap:focus-within::before {{
      opacity: 1 !important;
      visibility: visible !important;
    }}
    .dashboard-grid,
    .paper-trade-signal-grid,
    .indicator-grid,
    .indicator-group,
    .terminal-card,
    .market-intel-card,
    .market-intel-body,
    .market-intel-chips,
    .tech-summary-card,
    .tech-summary-grid,
    .tech-detail-grid,
    .tech-health-grid,
    .tech-action-grid,
    .tech-score-grid,
    .tech-metric-grid,
    .smc-command-card,
    .smc-command-grid,
    .smc-health-grid,
    .smc-setup-grid,
    .smc-summary-grid,
    .order-command-card,
    .order-command-grid,
    .order-health-grid,
    .order-action-grid,
    .order-pressure-panel,
    .mtf-command-card,
    .mtf-command-grid,
    .mtf-health-grid,
    .mtf-action-grid,
    .mtf-summary-grid,
    .ai-command-card,
    .ai-command-grid,
    .ai-health-grid,
    .ai-action-grid,
    .ai-consensus-grid,
    .ai-breakdown-grid,
    .bt-command-card,
    .bt-command-grid,
    .bt-health-grid,
    .bt-kpi-grid,
    .bt-analytics-grid,
    .pf-command-card,
    .pf-command-grid,
    .pf-health-grid,
    .pf-kpi-grid,
    .pf-analytics-grid {{
      overflow: visible !important;
    }}
    .trade-decision-metric:has(.metric-info-wrap),
    .market-status-item:has(.metric-info-wrap),
    .market-intel-chip:has(.metric-info-wrap),
    .market-intel-core:has(.metric-info-wrap),
    .intel-metric:has(.metric-info-wrap),
    .direction-gauge:has(.metric-info-wrap),
    .smc-core:has(.metric-info-wrap),
    .tech-bias-core:has(.metric-info-wrap),
    .order-core:has(.metric-info-wrap),
    .mtf-core:has(.metric-info-wrap),
    .mtf-chip:has(.metric-info-wrap),
    .mtf-action-item:has(.metric-info-wrap),
    .mtf-summary-card:has(.metric-info-wrap),
    .ai-core:has(.metric-info-wrap),
    .tech-health-chip:has(.metric-info-wrap),
    .tech-action-item:has(.metric-info-wrap),
    .tech-score-card:has(.metric-info-wrap),
    .tech-mini-metric:has(.metric-info-wrap),
    .smc-health-chip:has(.metric-info-wrap),
    .smc-setup-item:has(.metric-info-wrap),
    .smc-summary-card:has(.metric-info-wrap),
    .order-chip:has(.metric-info-wrap),
    .order-action-item:has(.metric-info-wrap),
    .ai-chip:has(.metric-info-wrap),
    .ai-action-item:has(.metric-info-wrap),
    .ai-consensus-card:has(.metric-info-wrap),
    .ai-breakdown-card:has(.metric-info-wrap),
    .bt-core:has(.metric-info-wrap),
    .bt-chip:has(.metric-info-wrap),
    .bt-kpi:has(.metric-info-wrap),
    .bt-analytics-card:has(.metric-info-wrap),
    .pf-core:has(.metric-info-wrap),
    .pf-chip:has(.metric-info-wrap),
    .pf-kpi:has(.metric-info-wrap),
    .pf-analytics-card:has(.metric-info-wrap),
    .dashboard-card:has(.metric-info-wrap),
    .dashboard-tile:has(.metric-info-wrap),
    .terminal-card:has(.metric-info-wrap) {{
      overflow: visible;
    }}
    .paper-trade-signal-grid,
    .paper-trade-signal-grid .dashboard-card,
    .paper-trade-signal-grid .metric-label,
    .paper-trade-signal-grid .metric-label-with-info,
    .paper-trade-signal-grid .metric-info-wrap {{
      overflow: visible !important;
    }}
    .paper-trade-signal-grid .metric-tooltip-content {{
      top: auto !important;
      bottom: calc(100% + 10px) !important;
      transform: translateX(-50%) translateY(3px) !important;
    }}
    .paper-trade-signal-grid .metric-info-wrap::before {{
      top: auto !important;
      bottom: calc(100% + 5px) !important;
      border-left: 0 !important;
      border-top: 0 !important;
      border-right: 1px solid var(--card-border) !important;
      border-bottom: 1px solid var(--card-border) !important;
      transform: translateX(-50%) rotate(45deg) !important;
    }}
    .paper-trade-signal-grid .dashboard-card:has(.metric-info-wrap:hover),
    .paper-trade-signal-grid .dashboard-card:has(.metric-info-wrap:focus-within) {{
      position: relative;
      z-index: 1000000;
    }}
    .mtf-timeframe-grid,
    .mtf-timeframe-grid .dashboard-card,
    .mtf-timeframe-grid .metric-label,
    .mtf-timeframe-grid .metric-label-with-info,
    .mtf-timeframe-grid .metric-info-wrap {{
      overflow: visible !important;
    }}
    .mtf-timeframe-grid .metric-tooltip-content {{
      top: auto !important;
      bottom: calc(100% + 16px) !important;
      transform: translateX(-50%) translateY(4px) !important;
    }}
    .mtf-timeframe-grid .metric-info-wrap::before {{
      top: auto !important;
      bottom: calc(100% + 11px) !important;
      border-left: 0 !important;
      border-top: 0 !important;
      border-right: 1px solid var(--card-border) !important;
      border-bottom: 1px solid var(--card-border) !important;
      transform: translateX(-50%) rotate(45deg) !important;
    }}
    .mtf-timeframe-grid .dashboard-card:has(.metric-info-wrap:hover),
    .mtf-timeframe-grid .dashboard-card:has(.metric-info-wrap:focus-within) {{
      position: relative;
      z-index: 1000000;
    }}
    .mtf-summary-dashboard-grid,
    .mtf-summary-dashboard-grid .dashboard-card,
    .mtf-summary-dashboard-grid .metric-label,
    .mtf-summary-dashboard-grid .metric-label-with-info,
    .mtf-summary-dashboard-grid .metric-info-wrap {{
      overflow: visible !important;
    }}
    .mtf-summary-dashboard-grid .metric-tooltip-content {{
      top: auto !important;
      bottom: calc(100% + 16px) !important;
      transform: translateX(-50%) translateY(4px) !important;
    }}
    .mtf-summary-dashboard-grid .metric-info-wrap::before {{
      top: auto !important;
      bottom: calc(100% + 11px) !important;
      border-left: 0 !important;
      border-top: 0 !important;
      border-right: 1px solid var(--card-border) !important;
      border-bottom: 1px solid var(--card-border) !important;
      transform: translateX(-50%) rotate(45deg) !important;
    }}
    .mtf-summary-dashboard-grid .dashboard-card:has(.metric-info-wrap:hover),
    .mtf-summary-dashboard-grid .dashboard-card:has(.metric-info-wrap:focus-within) {{
      position: relative;
      z-index: 1000000;
    }}
    .order-book-tile-grid,
    .order-book-tile-grid .dashboard-tile,
    .order-book-tile-grid .dashboard-tile h4,
    .order-book-tile-grid .metric-label-with-info,
    .order-book-tile-grid .metric-info-wrap {{
      overflow: visible !important;
    }}
    .order-book-tile-grid .metric-tooltip-content {{
      top: auto !important;
      bottom: calc(100% + 16px) !important;
      transform: translateX(-50%) translateY(4px) !important;
    }}
    .order-book-tile-grid .metric-info-wrap::before {{
      top: auto !important;
      bottom: calc(100% + 11px) !important;
      border-left: 0 !important;
      border-top: 0 !important;
      border-right: 1px solid var(--card-border) !important;
      border-bottom: 1px solid var(--card-border) !important;
      transform: translateX(-50%) rotate(45deg) !important;
    }}
    .order-book-tile-grid .dashboard-tile:has(.metric-info-wrap:hover),
    .order-book-tile-grid .dashboard-tile:has(.metric-info-wrap:focus-within) {{
      position: relative;
      z-index: 1000000;
    }}
    .order-pressure-panel,
    .order-pressure-panel .order-chip,
    .order-pressure-panel .dashboard-card,
    .order-pressure-panel span,
    .order-pressure-panel .metric-label,
    .order-pressure-panel .metric-label-with-info,
    .order-pressure-panel .metric-info-wrap {{
      overflow: visible !important;
    }}
    .order-pressure-panel .metric-tooltip-content {{
      top: auto !important;
      bottom: calc(100% + 16px) !important;
      transform: translateX(-50%) translateY(4px) !important;
    }}
    .order-pressure-panel .metric-info-wrap::before {{
      top: auto !important;
      bottom: calc(100% + 11px) !important;
      border-left: 0 !important;
      border-top: 0 !important;
      border-right: 1px solid var(--card-border) !important;
      border-bottom: 1px solid var(--card-border) !important;
      transform: translateX(-50%) rotate(45deg) !important;
    }}
    .order-pressure-panel .order-chip:has(.metric-info-wrap:hover),
    .order-pressure-panel .order-chip:has(.metric-info-wrap:focus-within),
    .order-pressure-panel .dashboard-card:has(.metric-info-wrap:hover),
    .order-pressure-panel .dashboard-card:has(.metric-info-wrap:focus-within) {{
      position: relative;
      z-index: 1000000;
    }}
    .trade-decision-metric:has(.metric-info-wrap:hover),
    .trade-decision-metric:has(.metric-info-wrap:focus-within),
    .market-status-item:has(.metric-info-wrap:hover),
    .market-status-item:has(.metric-info-wrap:focus-within),
    .market-intel-chip:has(.metric-info-wrap:hover),
    .market-intel-chip:has(.metric-info-wrap:focus-within),
    .market-intel-core:has(.metric-info-wrap:hover),
    .market-intel-core:has(.metric-info-wrap:focus-within),
    .intel-metric:has(.metric-info-wrap:hover),
    .intel-metric:has(.metric-info-wrap:focus-within),
    .direction-gauge:has(.metric-info-wrap:hover),
    .direction-gauge:has(.metric-info-wrap:focus-within),
    .smc-core:has(.metric-info-wrap:hover),
    .smc-core:has(.metric-info-wrap:focus-within),
    .tech-bias-core:has(.metric-info-wrap:hover),
    .tech-bias-core:has(.metric-info-wrap:focus-within),
    .order-core:has(.metric-info-wrap:hover),
    .order-core:has(.metric-info-wrap:focus-within),
    .mtf-core:has(.metric-info-wrap:hover),
    .mtf-core:has(.metric-info-wrap:focus-within),
    .mtf-chip:has(.metric-info-wrap:hover),
    .mtf-chip:has(.metric-info-wrap:focus-within),
    .mtf-action-item:has(.metric-info-wrap:hover),
    .mtf-action-item:has(.metric-info-wrap:focus-within),
    .mtf-summary-card:has(.metric-info-wrap:hover),
    .mtf-summary-card:has(.metric-info-wrap:focus-within),
    .ai-core:has(.metric-info-wrap:hover),
    .ai-core:has(.metric-info-wrap:focus-within),
    .tech-health-chip:has(.metric-info-wrap:hover),
    .tech-health-chip:has(.metric-info-wrap:focus-within),
    .tech-action-item:has(.metric-info-wrap:hover),
    .tech-action-item:has(.metric-info-wrap:focus-within),
    .tech-score-card:has(.metric-info-wrap:hover),
    .tech-score-card:has(.metric-info-wrap:focus-within),
    .tech-mini-metric:has(.metric-info-wrap:hover),
    .tech-mini-metric:has(.metric-info-wrap:focus-within),
    .smc-health-chip:has(.metric-info-wrap:hover),
    .smc-health-chip:has(.metric-info-wrap:focus-within),
    .smc-setup-item:has(.metric-info-wrap:hover),
    .smc-setup-item:has(.metric-info-wrap:focus-within),
    .smc-summary-card:has(.metric-info-wrap:hover),
    .smc-summary-card:has(.metric-info-wrap:focus-within),
    .order-chip:has(.metric-info-wrap:hover),
    .order-chip:has(.metric-info-wrap:focus-within),
    .order-action-item:has(.metric-info-wrap:hover),
    .order-action-item:has(.metric-info-wrap:focus-within),
    .ai-chip:has(.metric-info-wrap:hover),
    .ai-chip:has(.metric-info-wrap:focus-within),
    .ai-action-item:has(.metric-info-wrap:hover),
    .ai-action-item:has(.metric-info-wrap:focus-within),
    .ai-consensus-card:has(.metric-info-wrap:hover),
    .ai-consensus-card:has(.metric-info-wrap:focus-within),
    .ai-breakdown-card:has(.metric-info-wrap:hover),
    .ai-breakdown-card:has(.metric-info-wrap:focus-within),
    .bt-core:has(.metric-info-wrap:hover),
    .bt-core:has(.metric-info-wrap:focus-within),
    .bt-chip:has(.metric-info-wrap:hover),
    .bt-chip:has(.metric-info-wrap:focus-within),
    .bt-kpi:has(.metric-info-wrap:hover),
    .bt-kpi:has(.metric-info-wrap:focus-within),
    .bt-analytics-card:has(.metric-info-wrap:hover),
    .bt-analytics-card:has(.metric-info-wrap:focus-within),
    .pf-core:has(.metric-info-wrap:hover),
    .pf-core:has(.metric-info-wrap:focus-within),
    .pf-chip:has(.metric-info-wrap:hover),
    .pf-chip:has(.metric-info-wrap:focus-within),
    .pf-kpi:has(.metric-info-wrap:hover),
    .pf-kpi:has(.metric-info-wrap:focus-within),
    .pf-analytics-card:has(.metric-info-wrap:hover),
    .pf-analytics-card:has(.metric-info-wrap:focus-within),
    .dashboard-card:has(.metric-info-wrap:hover),
    .dashboard-card:has(.metric-info-wrap:focus-within),
    .dashboard-tile:has(.metric-info-wrap:hover),
    .dashboard-tile:has(.metric-info-wrap:focus-within),
    .terminal-card:has(.metric-info-wrap:hover),
    .terminal-card:has(.metric-info-wrap:focus-within) {{
      position: relative;
      z-index: 999998;
    }}

    @keyframes pulse {{
      0%, 100% {{ opacity: 0.72; }}
      50% {{ opacity: 1; }}
    }}
    @media (max-width: 1400px) {{
      .dashboard-grid {{ grid-template-columns: repeat(auto-fit, minmax(min(var(--dashboard-grid-min), 100%), 1fr)); }}
      div[role="radiogroup"] {{ max-width: 100%; }}
      div[role="radiogroup"] label {{ padding: 6px 10px; font-size: 0.8rem; }}
    }}
    @media (max-width: 760px) {{
      .block-container {{ padding-inline: 0.75rem !important; }}
      .dashboard-grid {{ grid-template-columns: 1fr; }}
      .overview-summary-grid, .indicator-grid {{ grid-template-columns: 1fr; }}
      .trade-decision-card {{ grid-template-columns: 1fr; gap: 8px; padding: 11px 12px; }}
      .trade-decision-metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .trade-decision-reason {{ border-left: 0; border-top: 1px solid var(--card-border); padding-left: 0; padding-top: 9px; }}
      .trade-decision-verdict {{ white-space: normal; }}
      .tech-command-head, .indicator-group-head, .tech-chart-head, .tech-control-row, .smc-command-head, .mtf-command-head, .ai-command-head, .bt-command-head, .pf-command-head {{ align-items:flex-start; flex-direction:column; gap:5px; }}
      .tech-command-meta, .indicator-group-meta, .tech-chart-meta, .mtf-command-meta, .ai-command-meta, .bt-command-meta, .pf-command-meta {{ white-space:normal; }}
      .tech-summary-grid, .tech-detail-grid, .tech-action-grid, .tech-score-grid, .smc-command-grid, .smc-setup-grid, .smc-summary-grid, .order-command-grid, .order-action-grid, .order-pressure-panel, .mtf-command-grid, .mtf-action-grid, .mtf-summary-grid, .ai-command-grid, .ai-action-grid, .ai-consensus-grid, .ai-breakdown-grid, .bt-command-grid, .bt-kpi-grid, .bt-analytics-grid, .pf-command-grid, .pf-kpi-grid, .pf-analytics-grid {{ grid-template-columns:1fr; }}
      .smc-health-grid, .order-health-grid, .mtf-health-grid, .ai-health-grid, .bt-health-grid, .pf-health-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .tech-checklist {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .tech-health-grid, .tech-metric-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .tech-level-list {{ grid-template-columns:1fr; }}
      .overview-heading {{ align-items: flex-start; flex-direction: column; gap: 3px; }}
      .overview-heading-meta {{ white-space: normal; }}
      .ai-brief-head, .compact-section-head {{ align-items: flex-start; flex-direction: column; gap: 3px; }}
      .ai-brief-state, .compact-section-meta {{ white-space: normal; }}
      .ai-brief-lines {{ grid-template-columns: 1fr; }}
      .market-intel-head {{ align-items: flex-start; flex-direction: column; gap: 3px; }}
      .market-intel-meta {{ white-space: normal; }}
      .market-intel-body {{ grid-template-columns: 1fr; }}
      .market-intel-chips {{ grid-template-columns: 1fr; }}
      .coin-intel-grid {{ grid-template-columns: 1fr; }}
      .intel-metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .market-status-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .opportunity-grid {{ grid-template-columns: 1fr; }}
      .regime-card {{ grid-template-columns: 1fr; }}
      .regime-drivers {{ grid-template-columns: 1fr; }}
      .scanner-head {{ align-items: flex-start; flex-direction: column; gap: 5px; }}
      .scanner-status-line {{ justify-content: flex-start; }}
      div[role="radiogroup"] {{ gap: 6px; max-width: 100%; }}
      div[role="radiogroup"] label {{ flex: 0 0 auto; justify-content: center; }}
      .section-subtitle {{ font-size: 0.86rem; }}
      .app-header {{ align-items: flex-start; }}
      .theme-control-label {{ text-align: left; }}
    }}
    </style>
    """

def render_theme_css(theme_name: str):
    st.markdown(get_theme_css(theme_name), unsafe_allow_html=True)


def render_app_header(theme_name: str) -> None:
    theme_name = normalize_theme_name(theme_name)
    left, right = st.columns([0.78, 0.22])
    with left:
        st.markdown("""
        <div class='app-header'>
            <div class='brand-mark'>SS</div>
            <div>
                <h1>SuperSignal</h1>
                <p>Markets · Signals · Risk</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("<div class='theme-control-label'>Theme</div>", unsafe_allow_html=True)
        selected_theme = st.selectbox(
            "Theme",
            THEME_OPTIONS,
            index=THEME_OPTIONS.index(theme_name) if theme_name in THEME_OPTIONS else 0,
            key="header_theme",
            label_visibility="collapsed",
        )
        st.markdown("<div class='header-theme-spacer'></div>", unsafe_allow_html=True)
    selected_theme = normalize_theme_name(selected_theme)
    if selected_theme != normalize_theme_name(st.session_state.get("theme")):
        st.session_state.theme = selected_theme
    qp_set("theme", selected_theme)

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
    color = "#00E08A" if v >= 0 else "#FF5C73"
    return f"<span style='color:{color}'>{arrow} {abs(v):.2f}%</span>"

METRIC_TOOLTIPS = {
    "Market Direction": "Shows whether the current market bias is bullish, bearish, or neutral based on the active signal model.",
    "Market Regime": "Summarizes the current market environment such as bullish, bearish, neutral, or choppy.",
    "Market Breadth": "Shows how many tracked assets are trading above the EMA200. Higher breadth means stronger market participation.",
    "Momentum Score": "Measures the strength and direction of recent price movement.",
    "AI Conviction": "Shows how strongly the model agrees with the current trading bias.",
    "Fear & Greed": "Reflects overall crypto market sentiment. Low values indicate fear; high values indicate greed.",
    "Top Strength": "Shows the strongest tracked assets ranked by AI score and signal quality.",
    "Risk / Reward": "Compares potential reward against risk using the selected stop-loss and take-profit settings.",
    "Confidence": "Shows how much conviction the current model output has in the displayed bias.",
    "Entry Zone": "Suggested price area for considering entry based on the current setup.",
    "Stop Loss": "Suggested invalidation level based on the selected stop-loss setting.",
    "Take Profit": "Suggested target based on the selected take-profit setting.",
    "Allocation": "Suggested position allocation based on user risk settings and signal confidence.",
    "Suggested Allocation": "Suggested position allocation based on user risk settings and signal confidence.",
    "Market regime": "Summarizes the current market environment such as bullish, bearish, neutral, or choppy.",
    "MTF Bias": "Summarizes directional bias across multiple timeframes.",
    "Weighted Score": "Combines timeframe signals with higher timeframes carrying more weight.",
    "Trend Alignment": "Shows whether trend direction agrees across the selected timeframes.",
    "Momentum Alignment": "Shows whether momentum conditions agree across the selected timeframes.",
    "Signal Agreement": "Measures how much the weighted timeframe signals agree with each other.",
    "Risk Level": "Summarizes setup risk from confidence, disagreement, and conflicting signals.",
    "MTF Consensus": "Summarizes the combined signal across all tracked timeframes.",
    "Bullish Timeframes": "Counts how many tracked timeframes currently lean bullish.",
    "Bearish Timeframes": "Counts how many tracked timeframes currently lean bearish.",
    "Neutral / Hold": "Counts timeframes without a clear bullish or bearish signal.",
    "Technical Bias": "Summarizes the technical read from trend, momentum, volatility, and volume indicators.",
    "Institutional Bias": "Summarizes smart-money structure from liquidity, sweeps, gaps, and premium or discount zones.",
    "Order Flow Bias": "Summarizes order-book pressure from depth, spread, imbalance, and liquidity walls.",
    "AI Bias": "Summarizes the combined model view from AI signal components.",
    "Strategy Verdict": "Summarizes how the backtest view rates the current strategy conditions.",
    "Portfolio Verdict": "Summarizes current paper portfolio exposure, risk, and performance.",
    "Portfolio Health": "Shows the overall account condition from portfolio score, exposure, and risk.",
    "Portfolio Score": "Composite account quality score based on return, exposure, risk, and concentration.",
    "Portfolio Risk": "Summarizes the current account risk from exposure, concentration, and drawdown.",
    "Exposure Level": "Shows how much capital is currently deployed in open paper positions.",
    "Capital Utilization": "Shows the percentage of configured capital currently allocated to positions.",
    "Active Positions": "Counts the open paper positions currently tracked in the portfolio.",
    "Unrealized PnL": "Shows open profit or loss from active positions marked to current price.",
    "Realized PnL": "Shows profit or loss already recorded from closed paper trades.",
    "Total Capital": "Shows the configured paper account capital used for sizing and reporting.",
    "Available Cash": "Shows capital that is not currently allocated to open positions.",
    "Used Margin": "Shows capital currently tied up in open paper positions.",
    "Portfolio Value": "Combines available cash, deployed capital, and current PnL.",
    "Total Return %": "Shows combined open and realized return as a percentage of capital.",
    "Daily Return %": "Shows return from positions logged over the last day.",
    "Weekly Return %": "Shows return from positions logged over the last seven days.",
    "Monthly Return %": "Shows return from positions logged over the last thirty days.",
    "Portfolio Risk Score": "Scores account risk after exposure, concentration, and drawdown are considered.",
    "Current Exposure": "Shows the current deployed capital as a percentage of account capital.",
    "Maximum Exposure": "Shows the reference exposure ceiling used in the risk view.",
    "Risk Utilization": "Compares current exposure with the displayed maximum exposure level.",
    "Position Concentration": "Shows how much open exposure is concentrated in the largest position.",
    "Largest Position": "Shows the biggest open paper allocation by value.",
    "Portfolio Drawdown": "Shows the current negative account return from the configured capital.",
    "Open Positions": "Counts open paper trades currently being tracked.",
    "Winning Positions": "Counts open positions with positive unrealized PnL.",
    "Losing Positions": "Counts open positions with negative unrealized PnL.",
    "ML Consensus": "Shows the combined machine-learning signal across the active model inputs.",
    "Trend Status": "Shows whether the short, medium, and long trend indicators are aligned.",
    "Momentum Status": "Shows whether recent momentum supports or conflicts with the current bias.",
    "Volatility Status": "Shows whether price action is quiet, stretched, or unstable.",
    "Volume Flow": "Shows whether volume pressure is leaning toward inflow or outflow.",
    "Liquidity Pressure": "Shows whether nearby liquidity is more likely to pressure price up or down.",
    "FVG Bias": "Summarizes whether fair-value gaps lean bullish, bearish, or mixed.",
    "Structure Bias": "Shows whether recent market structure favors continuation or reversal.",
    "Sweep Risk": "Highlights the chance of a liquidity sweep around nearby levels.",
    "Price Zone": "Shows whether price is trading in premium, discount, or equilibrium context.",
    "Bias": "Shows the preferred directional lean for this section.",
    "Risk": "Summarizes whether this setup looks low, moderate, or high risk.",
    "Preferred Direction": "Shows the favored directional play based on the current setup.",
    "Preferred Setup": "Shows the setup type that best fits current conditions.",
    "Entry Context": "Explains where to look for a trigger relative to the current setup.",
    "Confirmation Needed": "Shows the next condition that would strengthen the setup.",
    "Invalidation": "Shows what would weaken or cancel the current setup idea.",
    "AI Score": "Scores the current setup using the active signal model inputs.",
    "Signal": "Shows the current model signal for the selected market.",
    "24h Change": "Shows recent percentage price movement over the last day.",
    "Volume": "Shows recent traded volume used to judge market participation.",
    "Price": "Shows the latest available market price for this view.",
    "Spread": "Shows the gap between best bid and best ask. Lower is usually cleaner.",
    "Depth": "Shows available order book liquidity around the current price.",
    "Bid Wall": "Shows stronger visible buy-side liquidity in the order book.",
    "Ask Wall": "Shows stronger visible sell-side liquidity in the order book.",
    "Imbalance": "Shows whether order book liquidity leans toward bids or asks.",
    "Buy Pressure": "Shows the share of visible order-book liquidity on the bid side.",
    "Sell Pressure": "Shows the share of visible order-book liquidity on the ask side.",
    "Interpretation": "Summarizes whether buyers, sellers, or neither side controls short-term flow.",
    "Cumulative Delta": "Tracks net buying versus selling pressure from recent order-flow changes.",
    "Flow Pressure": "Summarizes whether order flow currently favors buyers or sellers.",
    "Liquidity Bias": "Shows whether nearby liquidity conditions favor upside or downside pressure.",
    "Execution Quality": "Summarizes spread and depth conditions for cleaner entries.",
    "Slippage Risk": "Estimates how likely execution may move away from the expected price.",
    "Timeframe": "Shows the chart interval used for this signal or metric.",
    "Direction": "Shows the directional lean for this specific model or timeframe.",
    "Strength": "Shows how strong the current signal or condition appears.",
    "Agreement": "Shows whether multiple inputs are pointing in the same direction.",
    "Model Vote": "Shows the direction favored by this model component.",
    "Model Confidence": "Shows how strongly this model component supports its vote.",
    "Consensus": "Shows whether the active model components agree with each other.",
    "Feature Impact": "Shows which inputs are contributing most to the model output.",
    "Strategy Score": "Scores the backtest quality using performance and risk metrics.",
    "Profitability": "Summarizes whether the backtest has positive expectancy.",
    "Robustness": "Shows whether the backtest has enough trades and stable behavior.",
    "Trade Quality": "Summarizes average trade quality and win/loss balance.",
    "Net Profit": "Shows total backtest profit or loss in account currency.",
    "Final Capital": "Shows ending capital after applying the backtest results.",
    "Win Rate": "Shows the percentage of closed trades that were profitable.",
    "Profit Factor": "Compares gross profit with gross loss. Above 1 is better.",
    "Sharpe Ratio": "Shows risk-adjusted return. Higher values imply smoother performance.",
    "Max Drawdown": "Shows the largest peak-to-trough decline in the backtest.",
    "Recovery Factor": "Compares net profit with drawdown to judge recovery strength.",
    "Average Trade": "Shows the average profit or loss per closed trade.",
    "Best Trade": "Shows the largest winning trade in the sample.",
    "Worst Trade": "Shows the largest losing trade in the sample.",
    "Total Trades": "Shows how many trades are included in the result.",
    "Current Drawdown": "Shows the current decline from the latest equity high.",
    "Avg Winner": "Shows the average profit from winning trades.",
    "Avg Loser": "Shows the average loss from losing trades.",
    "Win/Loss Ratio": "Compares average winning trade size with average losing trade size.",
    "Profit Consistency": "Shows whether returns are steady or concentrated in few trades.",
    "Capital Preservation": "Shows how well the strategy avoids damaging drawdowns.",
    "Position Risk": "Shows risk created by the selected backtest position size.",
    "Risk Verdict": "Summarizes the backtest risk profile in one label.",
    "Open Risk": "Shows estimated risk currently carried by open paper positions.",
    "Risk Grade": "Classifies position risk using current allocation size.",
    "Position Value": "Shows the notional value of the paper position.",
    "Active Signal": "Shows the current trade signal used by the paper trading panel.",
    "Capital": "Shows the paper trading balance used for sizing.",
    "PnL": "Shows profit or loss for the selected trade or period.",
    "PnL %": "Shows profit or loss as a percentage.",
    "Sizing Capital": "Shows capital used for position sizing calculations.",
    "Position Size": "Shows the suggested or configured position size.",
    "Stop Distance": "Shows distance from entry to the stop level.",
    "Target Distance": "Shows distance from entry to the take-profit target.",
    "Units": "Shows the estimated number of units for the position size.",
}

def render_info_label(label: str, tooltip: str | None = None, placement: str = "above") -> str:
    label_text = str(label)
    normalized_label = " ".join(label_text.split())
    tooltip_lookup = (
        METRIC_TOOLTIPS.get(normalized_label)
        or METRIC_TOOLTIPS.get(normalized_label.title())
        or METRIC_TOOLTIPS.get(normalized_label.upper())
        or METRIC_TOOLTIPS.get(normalized_label.lower())
    )
    text = str(tooltip).strip() if tooltip is not None else ""
    if not text:
        text = str(tooltip_lookup or "").strip()
    if not text:
        text = "Explains this metric and how to interpret it."
    safe_label = html.escape(label_text)
    safe_label_attr = html.escape(label_text, quote=True)
    safe_tooltip = html.escape(str(text), quote=True)
    safe_placement = str(placement).strip().lower()
    if safe_placement not in {"above", "below", "left", "right"}:
        safe_placement = "above"
    return (
        "<span class='metric-label-with-info'>"
        f"<span class='metric-label-text'>{safe_label}</span>"
        f"<span class='metric-info-wrap tooltip-{safe_placement}' tabindex='0' aria-label='More information: {safe_label_attr}' data-tooltip='{safe_tooltip}'>"
        "<span class='metric-info-icon'>i</span>"
        f"<span class='metric-tooltip-content'>{safe_tooltip}</span>"
        "</span>"
        "</span>"
    )

def base_symbol(symbol: str) -> str:
    return str(symbol).split("/")[0].replace("USDT", "").upper()

@st.cache_data(show_spinner=False)
def fallback_logo_map() -> dict:
    return {
        "BTC": "https://assets.coingecko.com/coins/images/1/small/bitcoin.png",
        "ETH": "https://assets.coingecko.com/coins/images/279/small/ethereum.png",
        "SOL": "https://assets.coingecko.com/coins/images/4128/small/solana.png",
        "XRP": "https://assets.coingecko.com/coins/images/44/small/xrp-symbol-white-128.png",
        "BNB": "https://assets.coingecko.com/coins/images/825/small/bnb-icon2_2x.png",
        "DOGE": "https://assets.coingecko.com/coins/images/5/small/dogecoin.png",
        "ADA": "https://assets.coingecko.com/coins/images/975/small/cardano.png",
        "LINK": "https://assets.coingecko.com/coins/images/877/small/chainlink-new-logo.png",
        "AVAX": "https://assets.coingecko.com/coins/images/12559/small/Avalanche_Circle_RedWhite_Trans.png",
        "SUI": "https://assets.coingecko.com/coins/images/26375/small/sui_asset.jpeg",
        "AR": "https://assets.coingecko.com/coins/images/4343/small/oRt6SiEN_400x400.jpg",
        "ZEC": "https://assets.coingecko.com/coins/images/486/small/circle-zcash-color.png",
        "FIL": "https://assets.coingecko.com/coins/images/12817/small/filecoin.png",
        "ALGO": "https://assets.coingecko.com/coins/images/4380/small/download.png",
        "PYTH": "https://assets.coingecko.com/coins/images/31924/small/pyth.png",
    }

def coin_logo_url(symbol: str, coin_data: dict | None = None) -> str:
    coin_data = coin_data or {}
    return coin_data.get("image") or fallback_logo_map().get(base_symbol(symbol), "")

def render_coin_identity(symbol: str, coin_data: dict | None = None) -> str:
    ticker = base_symbol(symbol)
    name = html.escape(str((coin_data or {}).get("name") or ticker))
    logo = coin_logo_url(symbol, coin_data)
    if logo:
        logo_html = f"<img class='coin-logo' src='{html.escape(logo, quote=True)}' alt='{ticker} logo'>"
    else:
        logo_html = f"<span class='coin-logo coin-logo-fallback'>{html.escape(ticker[:2])}</span>"
    return (
        "<span class='cg-symbol-cell'>"
        f"{logo_html}"
        "<span>"
        f"<span class='cg-symbol-text'>{html.escape(ticker)}</span>"
        f"<span class='cg-name-text'>{name}</span>"
        "</span>"
        "</span>"
    )

def heat_class(value: float) -> str:
    return "cg-heat-pos" if value >= 0 else "cg-heat-neg"

def signal_class(value: str) -> str:
    return {
        SIGNAL_BUY: "cg-signal-buy",
        SIGNAL_SELL: "cg-signal-sell",
        SIGNAL_HOLD: "cg-signal-hold",
    }.get(value, "")

def signal_color(s: str) -> str:
    return {"BUY": "#00E08A", "SELL": "#FF5C73", "HOLD": "#FFB84D"}.get(s, "#A8B0BD")

def _normalize_status_text(text) -> str | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    if not value:
        return None
    unavailable_terms = ("unknown", "n/a", "na", "no data", "no signal", "unchanged", "no movement", "unavailable", "not available")
    positive_terms = ("buy", "strong buy", "bull", "bullish", "uptrend", "up", "↑", "gain", "gainer", "positive", "above", "inflow", "discount")
    negative_terms = ("sell", "strong sell", "bear", "bearish", "downtrend", "down", "↓", "loss", "loser", "negative", "below", "outflow", "extreme fear", "fear", "premium")
    warning_terms = ("hold", "neutral", "mixed", "sideways", "warning", "divergence", "balanced")
    if any(term in value for term in unavailable_terms):
        return "unavailable"
    if any(term in value for term in positive_terms):
        return "positive"
    if any(term in value for term in negative_terms):
        return "negative"
    if any(term in value for term in warning_terms):
        return "warning"
    return None

def card_status_class(
    status: str | None = None,
    value=None,
    label: str | None = None,
    signal: str | None = None,
    trend: str | None = None,
) -> str:
    resolved = _normalize_status_text(signal)
    if resolved is None:
        resolved = _normalize_status_text(status)
    if resolved is None:
        resolved = _normalize_status_text(label)
    if resolved is None:
        resolved = _normalize_status_text(trend)
    if resolved is None and value is not None:
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            resolved = "positive" if value > 0 else "negative" if value < 0 else "unavailable"
        else:
            resolved = _normalize_status_text(value)
    return {
        "positive": "status-positive",
        "negative": "status-negative",
        "warning": "status-warning",
        "neutral": "status-neutral",
        "hold": "status-neutral",
        "unavailable": "status-unavailable",
        "muted": "status-muted",
    }.get(str(resolved or "unavailable").lower(), "status-unavailable")

def get_status_class(value=None, label: str | None = None, status: str | None = None, signal: str | None = None, trend: str | None = None) -> str:
    return card_status_class(status=status, value=value, label=label, signal=signal, trend=trend)

def status_color(status: str | None = None, signal: str | None = None, trend: str | None = None) -> str:
    cls = card_status_class(status=status, signal=signal, trend=trend)
    if cls == "status-positive":
        return "var(--cg-green)"
    if cls == "status-negative":
        return "var(--cg-red)"
    if cls in {"status-warning", "status-neutral"}:
        return "var(--warning)"
    return "var(--muted)"

def status_from_color(color: str) -> str:
    color = str(color).lower()
    if "00e08a" in color or "--success" in color or "--cg-green" in color:
        return "positive"
    if "ff5c73" in color or "--danger" in color or "--cg-red" in color:
        return "negative"
    if "ffb84d" in color or "warning" in color:
        return "warning"
    return "muted"

def sig_badge(sig: str) -> str:
    cls = {"BUY": "badge-buy", "SELL": "badge-sell"}.get(sig, "badge-hold")
    return f"<span class='{cls}'>{sig}</span>"

def sentiment_color(s: str) -> str:
    return {"positive": "#00E08A", "negative": "#FF5C73", "neutral": "#FFB84D"}.get(s, "#A8B0BD")

def render_dashboard_card(
    title: str,
    value: str,
    subtitle: str = "",
    accent: str | None = None,
    status: str | None = None,
    signal: str | None = None,
    trend: str | None = None,
) -> str:
    if status is None and signal is None and trend is None:
        status = status_from_color(accent or "") if accent else None
    card_class = card_status_class(status=status, value=value, label=title, signal=signal, trend=trend)
    accent_color = accent or status_color(status=status, signal=signal, trend=trend)
    title_html = render_info_label(title)
    return (
        f"<div class='dashboard-card {card_class}'>"
        f"<div class='metric-label'>{title_html}</div>"
        f"<div class='metric-val' style='font-size:var(--metric-value-size);color:{accent_color};line-height:1.08;margin-top:4px'>{value}</div>"
        f"<div class='metric-subtitle'>{subtitle}</div>"
        f"</div>"
    )

def render_metric_tile(title: str, value: str, detail: str = "", badge: str = "") -> str:
    badge_html = f"<span class='metric-pill {badge}'>{badge.replace('-', ' ').title()}</span>" if badge else ""
    status = badge or None
    title_html = render_info_label(title)
    return (
        f"<div class='dashboard-tile {card_status_class(status=status, value=value, label=title)}'>"
        f"<h4>{title_html}{badge_html}</h4>"
        f"<div class='metric-val' style='font-size:var(--metric-tile-value-size);line-height:1.08;margin-top:4px;color:{status_color(status=status)}'>{value}</div>"
        f"<p>{detail}</p>"
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
        "warning": ("#FFB84D", "rgba(251,146,60,0.12)"),
        "error": ("#ef4444", "rgba(239,68,68,0.12)"),
    }
    fg, bg = colors.get(kind, ("#7B8596", "rgba(123,133,150,0.10)"))
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


NEWS_CATEGORIES = ["Crypto", "Bitcoin", "Ethereum", "Altcoins", "Macro", "Regulation", "ETFs", "Exchanges", "DeFi", "AI & Tech"]
NEWS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt", "https://decrypt.co/feed"),
    ("CryptoSlate", "https://cryptoslate.com/feed/"),
]
NEWS_ASSETS = {
    "BTC": ("BTC", "Bitcoin"),
    "ETH": ("ETH", "Ethereum"),
    "SOL": ("SOL", "Solana"),
    "BNB": ("BNB",),
    "XRP": ("XRP",),
    "AR": ("AR", "Arweave"),
    "FIL": ("FIL", "Filecoin"),
    "DOGE": ("DOGE", "Dogecoin"),
    "ADA": ("ADA", "Cardano"),
    "AVAX": ("AVAX", "Avalanche"),
    "LINK": ("LINK", "Chainlink"),
    "TON": ("TON",),
    "USDT": ("USDT", "Tether"),
    "USDC": ("USDC",),
}
NEWS_BULLISH_KEYWORDS = {"approval", "inflow", "adoption", "partnership", "upgrade", "rally", "breakout", "accumulation", "institutional", "demand", "launch", "recovery", "record", "surge"}
NEWS_BEARISH_KEYWORDS = {"hack", "exploit", "lawsuit", "ban", "outflow", "liquidation", "selloff", "decline", "crash", "recession", "rate hike", "investigation", "probe", "fraud", "breach"}
NEWS_HIGH_IMPACT_KEYWORDS = {"etf", "sec", "fed", "cpi", "rate decision", "hack", "binance", "coinbase", "blackrock", "liquidation", "regulation", "lawsuit"}
NEWS_MEDIUM_IMPACT_KEYWORDS = {"btc", "bitcoin", "eth", "ethereum", "sol", "bnb", "xrp", "macro", "etf"}


def strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", str(text or ""))
    clean = re.sub(r"\s+", " ", clean).strip()
    return html.unescape(clean)


def news_time_ago(published: datetime | None) -> str:
    if not published:
        return "Recently"
    now = datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    seconds = max(0, int((now - published.astimezone(timezone.utc)).total_seconds()))
    if seconds < 3600:
        return f"{max(1, seconds // 60)}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def parse_news_date(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(str(value or ""))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):
        return None


def detect_affected_assets(text: str) -> list[str]:
    found = []
    haystack = f" {str(text or '').lower()} "
    for symbol, terms in NEWS_ASSETS.items():
        for term in terms:
            if re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", haystack):
                found.append(symbol)
                break
    return found[:5] or ["Crypto"]


def classify_news_sentiment(text: str) -> str:
    lowered = str(text or "").lower()
    bullish = sum(1 for word in NEWS_BULLISH_KEYWORDS if word in lowered)
    bearish = sum(1 for word in NEWS_BEARISH_KEYWORDS if word in lowered)
    if bullish > bearish:
        return "Bullish"
    if bearish > bullish:
        return "Bearish"
    return "Neutral"


def estimate_news_impact(text: str) -> str:
    lowered = str(text or "").lower()
    if any(word in lowered for word in NEWS_HIGH_IMPACT_KEYWORDS):
        return "High"
    if any(word in lowered for word in NEWS_MEDIUM_IMPACT_KEYWORDS):
        return "Medium"
    return "Low"


def normalize_news_item(item: dict) -> dict:
    title = strip_html(item.get("title", "Market update"))
    summary = strip_html(item.get("summary", ""))[:260]
    published = item.get("published")
    if isinstance(published, str):
        published = parse_news_date(published)
    text = f"{title} {summary}"
    normalized = {
        "title": title,
        "source": strip_html(item.get("source", "Public RSS")),
        "summary": summary or "No summary available.",
        "link": str(item.get("link", "#") or "#"),
        "published": published,
        "assets": detect_affected_assets(text),
        "sentiment": classify_news_sentiment(text),
        "impact": estimate_news_impact(text),
    }
    normalized["time"] = news_time_ago(published)
    return normalized


def fallback_news_items() -> list[dict]:
    now = datetime.now(timezone.utc)
    samples = [
        ("Bitcoin traders watch ETF flows as market searches for direction", "ETF demand and liquidity conditions remain the key short-term focus for BTC positioning.", "CoinDesk", "https://www.coindesk.com/"),
        ("Ethereum upgrade discussion keeps staking and L2 activity in focus", "Developers and investors continue to track network upgrades, scaling activity, and fee trends.", "Cointelegraph", "https://cointelegraph.com/"),
        ("Regulation headlines keep exchange tokens and stablecoins on watch", "Policy risk can quickly affect liquidity, exchange activity, and market confidence.", "Decrypt", "https://decrypt.co/"),
        ("Solana ecosystem activity supports broader altcoin watchlist", "Altcoin momentum remains selective as traders compare liquidity and user activity across chains.", "CryptoSlate", "https://cryptoslate.com/"),
    ]
    return [normalize_news_item({"title": title, "summary": summary, "source": source, "link": link, "published": now}) for title, summary, source, link in samples]


def parse_news_items(source: str, payload: bytes) -> list[dict]:
    root = ET.fromstring(payload)
    parsed = []
    for node in root.findall(".//item")[:8]:
        def child_text(name: str) -> str:
            child = node.find(name)
            return child.text if child is not None and child.text else ""

        parsed.append(normalize_news_item({
            "title": child_text("title"),
            "source": source,
            "summary": child_text("description"),
            "link": child_text("link"),
            "published": child_text("pubDate"),
        }))
    if parsed:
        return parsed
    ns = "{http://www.w3.org/2005/Atom}"
    for node in root.findall(f".//{ns}entry")[:8]:
        title = node.find(f"{ns}title")
        summary = node.find(f"{ns}summary")
        updated = node.find(f"{ns}updated")
        link = node.find(f"{ns}link")
        parsed.append(normalize_news_item({
            "title": title.text if title is not None and title.text else "",
            "source": source,
            "summary": summary.text if summary is not None and summary.text else "",
            "link": link.attrib.get("href", "#") if link is not None else "#",
            "published": updated.text if updated is not None and updated.text else "",
        }))
    return parsed


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_feed(refresh_key: int = 0) -> tuple[list[dict], bool]:
    del refresh_key
    items = []
    for source, url in NEWS_FEEDS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SuperSignal/1.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                items.extend(parse_news_items(source, response.read()))
        except Exception:
            continue
    deduped = []
    seen = set()
    for item in sorted(items, key=lambda x: x.get("published") or datetime.min.replace(tzinfo=timezone.utc), reverse=True):
        key = item.get("title", "").lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    if deduped:
        return deduped[:12], True
    return fallback_news_items(), False


def news_reason(item: dict) -> tuple[str, str]:
    assets = ", ".join(item.get("assets", ["Crypto"]))
    impact = str(item.get("impact", "Low")).lower()
    sentiment = str(item.get("sentiment", "Neutral")).lower()
    return (
        f"{assets} is in focus and the story has {impact} market relevance.",
        f"Likely {sentiment} read-through unless price action rejects the headline.",
    )


def render_news_badges(item: dict) -> str:
    sentiment = str(item.get("sentiment", "Neutral"))
    impact = str(item.get("impact", "Low"))
    tags = "".join(f"<span class='news-tag'>{html.escape(asset)}</span>" for asset in item.get("assets", ["Crypto"]))
    return (
        "<div class='news-badge-row'>"
        f"<span class='news-badge {sentiment.lower()}'>{html.escape(sentiment)}</span>"
        f"<span class='news-badge {impact.lower()}'>{html.escape(impact)} impact</span>{tags}</div>"
    )


def render_news_card(item: dict) -> str:
    why, likely = news_reason(item)
    link = html.escape(item.get("link", "#"), quote=True)
    return (
        "<article class='news-card'>"
        f"<div class='news-card-meta'>{html.escape(item.get('source', 'RSS'))} · {html.escape(item.get('time', 'Recently'))}</div>"
        f"<h3>{html.escape(item.get('title', 'Market update'))}</h3>"
        f"<div class='news-card-summary'>{html.escape(item.get('summary', 'No summary available.'))}</div>"
        f"{render_news_badges(item)}"
        f"<div class='news-intel'><strong>Why it matters:</strong> {html.escape(why)}<br><strong>Likely market impact:</strong> {html.escape(likely)}</div>"
        f"<div style='margin-top:9px'><a class='news-link' href='{link}' target='_blank' rel='noopener noreferrer'>Read original</a></div>"
        "</article>"
    )


def render_news_hero(item: dict) -> str:
    why, likely = news_reason(item)
    assets = ", ".join(item.get("assets", ["Crypto"]))
    link = html.escape(item.get("link", "#"), quote=True)
    return (
        "<section class='news-hero'><div>"
        f"<div class='news-eyebrow'>Top market story · {html.escape(item.get('source', 'RSS'))} · {html.escape(item.get('time', 'Recently'))}</div>"
        f"<h2>{html.escape(item.get('title', 'Market update'))}</h2>"
        f"<div class='news-summary'>{html.escape(item.get('summary', 'No summary available.'))}</div>"
        f"{render_news_badges(item)}"
        f"<div style='margin-top:12px'><a class='news-link' href='{link}' target='_blank' rel='noopener noreferrer'>Read original</a></div>"
        "</div><aside class='news-hero-side'>"
        "<div class='news-widget-title'>News Intelligence</div>"
        f"<div class='news-intel'><strong>Why it matters:</strong> {html.escape(why)}</div>"
        f"<div class='news-intel' style='margin-top:8px'><strong>Likely market impact:</strong> {html.escape(likely)}</div>"
        f"<div class='news-tag-row'><span class='news-badge'>Affected: {html.escape(assets)}</span></div>"
        "</aside></section>"
    )


def render_news_sidebar(items: list[dict]) -> str:
    asset_counts = {}
    sentiment_counts = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
    high_impact = []
    for item in items:
        sentiment = item.get("sentiment", "Neutral")
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        if item.get("impact") == "High":
            high_impact.append(item)
        for asset in item.get("assets", []):
            asset_counts[asset] = asset_counts.get(asset, 0) + 1
    mentioned = sorted(asset_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    mentioned_html = "".join(f"<div class='news-list-item'><strong>{html.escape(asset)}</strong><span>{count} mentions</span></div>" for asset, count in mentioned) or "<div class='news-list-item'>No asset mentions yet</div>"
    high_html = "".join(f"<div class='news-list-item'><strong>{html.escape(item['title'][:54])}</strong><span>{html.escape(item['source'])}</span></div>" for item in high_impact[:4]) or "<div class='news-list-item'>No high-impact headlines in the current feed</div>"
    return (
        "<aside class='news-sidebar'>"
        "<div class='news-widget'><div class='news-widget-title'>Crypto Market Pulse</div><div class='news-stat-grid'>"
        f"<div class='news-stat'><span>Stories</span><strong>{len(items)}</strong></div>"
        f"<div class='news-stat'><span>High Impact</span><strong>{len(high_impact)}</strong></div>"
        f"<div class='news-stat'><span>Bullish</span><strong>{sentiment_counts.get('Bullish', 0)}</strong></div>"
        f"<div class='news-stat'><span>Bearish</span><strong>{sentiment_counts.get('Bearish', 0)}</strong></div>"
        "</div></div>"
        f"<div class='news-widget'><div class='news-widget-title'>Most Mentioned Assets</div><div class='news-list'>{mentioned_html}</div></div>"
        f"<div class='news-widget'><div class='news-widget-title'>High Impact Watchlist</div><div class='news-list'>{high_html}</div></div>"
        "<div class='news-widget'><div class='news-widget-title'>Upcoming Macro / Calendar</div><div class='news-list'>"
        "<div class='news-list-item'><strong>CPI / inflation prints</strong><span>Monitor USD liquidity</span></div>"
        "<div class='news-list-item'><strong>Fed rate guidance</strong><span>Macro risk driver</span></div>"
        "<div class='news-list-item'><strong>ETF flow updates</strong><span>BTC demand gauge</span></div>"
        "</div></div></aside>"
    )


def render_news_tab(symbol: str) -> None:
    st.markdown(render_section_header("News", "Crypto market headlines, sentiment, and event risk"), unsafe_allow_html=True)
    if st.button("Refresh News", key="refresh_news_btn"):
        st.session_state.news_refresh_key = int(st.session_state.get("news_refresh_key", 0)) + 1
        fetch_news_feed.clear()
        st.rerun()

    categories_html = "".join(f"<span class='news-category-pill'>{html.escape(category)}</span>" for category in NEWS_CATEGORIES)
    st.markdown(f"<div class='news-category-bar'>{categories_html}</div>", unsafe_allow_html=True)

    items, live_ok = fetch_news_feed(int(st.session_state.get("news_refresh_key", 0)))
    if not live_ok:
        render_notice_badge("Live news feed is temporarily unavailable. Showing fallback market headlines.", kind="warning")

    selected_symbol = symbol.split("/")[0] if "/" in symbol else symbol
    ranked = sorted(
        items,
        key=lambda item: (
            {"High": 3, "Medium": 2, "Low": 1}.get(item.get("impact"), 0),
            1 if selected_symbol in item.get("assets", []) else 0,
            item.get("published") or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    hero = ranked[0] if ranked else fallback_news_items()[0]
    feed_items = [item for item in items if item.get("title") != hero.get("title")][:9]
    feed_html = "<section class='news-feed'><div class='news-section-title'>Latest News Feed</div>" + "".join(render_news_card(item) for item in feed_items) + "</section>"

    st.markdown(render_news_hero(hero), unsafe_allow_html=True)
    st.markdown(f"<div class='news-page-grid'>{feed_html}{render_news_sidebar(items)}</div>", unsafe_allow_html=True)


def dataframe_theme_styles(df: pd.DataFrame):
    return (
        df.style
        .set_properties(
            **{
                "background-color": "#070B12",
                "color": "#F3F5F7",
                "border-color": "rgba(255,255,255,0.08)",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#0A1018"),
                        ("color", "#A8B0BD"),
                        ("border-color", "rgba(255,255,255,0.08)"),
                    ],
                }
            ]
        )
    )


def render_interactive_dataframe(
    df: pd.DataFrame,
    *,
    column_config: dict | None = None,
    column_order: list[str] | None = None,
    height: int | None = None,
    signed_columns: set[str] | None = None,
) -> None:
    signed_columns = signed_columns or set()
    styler = dataframe_theme_styles(df)

    def signed_style(value):
        if pd.isna(value):
            return ""
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"buy", "bullish", "bullish ob", "demand"} or value.startswith("+"):
                return "color:#00E08A;font-weight:850"
            if lowered in {"sell", "bearish", "bearish ob", "supply"} or value.startswith("-"):
                return "color:#FF5C73;font-weight:850"
            if lowered in {"hold", "neutral"}:
                return "color:#FFB84D;font-weight:850"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if number > 0:
            return "color:#00E08A;font-weight:850"
        if number < 0:
            return "color:#FF5C73;font-weight:850"
        return ""

    for col in signed_columns:
        if col in df.columns:
            styler = styler.map(signed_style, subset=[col])

    dataframe_kwargs = {
        "width": "stretch",
        "hide_index": True,
        "column_order": column_order,
        "column_config": column_config,
    }
    if height is not None:
        dataframe_kwargs["height"] = height
    st.dataframe(styler, **dataframe_kwargs)


def render_compact_state(message: str, detail: str = "") -> None:
    detail_html = f"<span style='color:var(--muted);font-weight:500;margin-left:8px'>{detail}</span>" if detail else ""
    st.markdown(
        f"<div style='display:inline-flex;align-items:center;padding:9px 12px;border-radius:12px;"
        f"border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.06);"
        f"color:var(--text);font-size:0.9rem;margin:4px 0 12px 0;'>"
        f"<strong>{message}</strong>{detail_html}</div>",
        unsafe_allow_html=True,
    )

def render_ml_prediction_state(ml_result: dict) -> bool:
    if not isinstance(ml_result, dict) or not ml_result:
        render_empty_state("ML predictions are unavailable for the current market dataset.", icon="ℹ️")
        return False
    if ml_result.get("error"):
        render_notice_badge(f"ML predictions unavailable: {ml_result['error']}", kind="warning")
        return False
    if ml_result.get("combined_probability") is None:
        render_notice_badge("ML models trained, but no current prediction probability is available.", kind="warning")
        return False
    return True

TAB_OPTIONS = [
    ("overview", "Overview"),
    ("technical", "Technical"),
    ("smart_money", "Smart Money"),
    ("order_book", "Order Book"),
    ("multi_tf", "Multi-TF"),
    ("ai_signals", "AI Signals"),
    ("news", "News"),
    ("backtest", "Backtest"),
    ("portfolio", "Portfolio"),
]
TAB_LABEL_BY_ID = dict(TAB_OPTIONS)
TAB_ID_BY_LABEL = {label: tab_id for tab_id, label in TAB_OPTIONS}


def qp_get(name: str, default=None):
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def qp_set(name: str, value) -> None:
    text = str(value)
    if qp_get(name) != text:
        st.query_params[name] = text


def qp_choice(name: str, choices: list, default):
    value = qp_get(name, default)
    return value if value in choices else default


def qp_float(name: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        value = float(qp_get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value


def qp_int(name: str, default: int, min_value: int | None = None, max_value: int | None = None, step: int | None = None) -> int:
    try:
        value = int(float(qp_get(name, default)))
    except (TypeError, ValueError):
        value = int(default)
    if step:
        value = int(round(value / step) * step)
    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value


def qp_bool(name: str, default: bool) -> bool:
    value = str(qp_get(name, "1" if default else "0")).lower()
    return value in {"1", "true", "yes", "on"}


def init_widget_from_query(widget_key: str, query_key: str, default, cast=str):
    if widget_key in st.session_state:
        return
    value = qp_get(query_key, default)
    try:
        st.session_state[widget_key] = cast(value)
    except (TypeError, ValueError):
        st.session_state[widget_key] = default


def render_persistent_tabs() -> str:
    initial_tab = qp_choice("tab", [tab_id for tab_id, _ in TAB_OPTIONS], "overview")
    tab_labels = [label for _, label in TAB_OPTIONS]
    if "active_tab_label" not in st.session_state or st.session_state.active_tab_label not in tab_labels:
        st.session_state.active_tab_label = TAB_LABEL_BY_ID[initial_tab]

    selected_label = st.radio(
        "Navigation",
        tab_labels,
        key="active_tab_label",
        horizontal=True,
        label_visibility="collapsed",
    )
    active_tab = TAB_ID_BY_LABEL.get(selected_label, "overview")
    st.session_state.active_tab = active_tab
    qp_set("tab", active_tab)
    return active_tab


def sync_overlay_query(show: dict) -> None:
    for key, value in show.items():
        qp_set(f"show_{key}", int(bool(value)))


def render_tab_density_css(active_tab: str) -> None:
    heavy_tabs = {"overview", "technical", "smart_money"}
    if active_tab in heavy_tabs:
        density = {
            "grid_min": "220px",
            "grid_gap": "16px",
            "grid_margin": "0.7rem 0 1rem",
            "card_padding": "14px 16px",
            "card_min": "110px",
            "overflow": "visible",
            "terminal_min": "96px",
            "subtitle_margin": "0.78rem",
            "subtitle_size": "0.88rem",
            "metric_value": "clamp(1.08rem,1.28vw,1.42rem)",
            "metric_tile_value": "clamp(1rem,1.16vw,1.18rem)",
            "vertical_gap": "0.7rem",
        }
    else:
        density = {
            "grid_min": "170px",
            "grid_gap": "16px",
            "grid_margin": "0.65rem 0 0.95rem",
            "card_padding": "13px 15px",
            "card_min": "96px",
            "overflow": "hidden",
            "terminal_min": "78px",
            "subtitle_margin": "0.58rem",
            "subtitle_size": "0.84rem",
            "metric_value": "clamp(1.04rem,1.2vw,1.3rem)",
            "metric_tile_value": "clamp(0.98rem,1.12vw,1.1rem)",
            "vertical_gap": "0.65rem",
        }
    st.markdown(
        f"""
        <style>
        :root {{
          --dashboard-grid-min: {density['grid_min']};
          --dashboard-grid-gap: {density['grid_gap']};
          --dashboard-grid-margin: {density['grid_margin']};
          --dashboard-card-padding: {density['card_padding']};
          --dashboard-card-min-height: {density['card_min']};
          --dashboard-card-overflow: {density['overflow']};
          --terminal-card-min-height: {density['terminal_min']};
          --section-subtitle-margin: {density['subtitle_margin']};
          --section-subtitle-size: {density['subtitle_size']};
          --metric-value-size: {density['metric_value']};
          --metric-tile-value-size: {density['metric_tile_value']};
          --vertical-block-gap: {density['vertical_gap']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def verdict_color(v: str) -> str:
    return {
        "Strong Buy":  "#00A868",
        "Buy":         "#00E08A",
        "Hold":        "#FFB84D",
        "Sell":        "#FF5C73",
        "Strong Sell": "#8b0000",
    }.get(v, "#7B8596")

def compute_trade_verdict(symbol: str, ind: dict, adv: dict, signal_result: dict, cfg: dict) -> dict:
    ind = ind or {}
    adv = adv or {}
    signal_result = signal_result or {}
    cfg = cfg or {}

    def safe_num(value, default=0.0) -> float:
        try:
            if value is None or pd.isna(value):
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    signal = str(signal_result.get("signal", SIGNAL_HOLD) or SIGNAL_HOLD).upper()
    confidence = int(round(max(0.0, min(100.0, safe_num(signal_result.get("confidence"), 0.0) * 100))))
    norm = safe_num(signal_result.get("normalized_score"), 0.0)
    directional_score = int(round(max(0.0, min(100.0, 50.0 + norm * 50.0))))
    regime = str(signal_result.get("market_regime", "") or "").lower()
    risk_level = str(signal_result.get("risk_level", "") or "").lower()

    close = safe_num(ind.get("close"), 0.0)
    ema9 = safe_num(ind.get("ema_9"), close)
    ema21 = safe_num(ind.get("ema_21"), close)
    ema50 = safe_num(ind.get("ema_50"), close)
    ema200 = safe_num(ind.get("ema_200"), close)
    macd = safe_num(ind.get("macd"), 0.0)
    macd_sig = safe_num(ind.get("macd_signal"), 0.0)
    rsi = safe_num(ind.get("rsi"), 50.0)
    roc = safe_num(adv.get("roc"), 0.0)

    bullish_regime = regime in {"bull", "bullish"} or (close > ema50 >= ema200 and ema9 >= ema21)
    bearish_regime = regime in {"bear", "bearish"} or (close < ema50 <= ema200 and ema9 <= ema21)
    positive_momentum = macd >= macd_sig and roc >= 0 and rsi >= 45
    negative_momentum = macd <= macd_sig and roc <= 0 and rsi <= 55

    if risk_level == "high" and confidence < 58 and signal == SIGNAL_HOLD:
        verdict = "AVOID"
        action = "Stay Out"
        status = "neutral"
        reason = "Weak signal quality with elevated risk"
    elif signal == SIGNAL_BUY and confidence >= 75 and bullish_regime and positive_momentum:
        verdict = "STRONG BUY"
        action = "Enter Long"
        status = "positive"
        reason = "Strong trend plus positive momentum"
    elif signal == SIGNAL_BUY or (directional_score >= 58 and bullish_regime):
        verdict = "BUY"
        action = "Enter Long"
        status = "positive"
        reason = "Bullish structure with supportive confirmation"
    elif signal == SIGNAL_SELL and confidence >= 75 and bearish_regime and negative_momentum:
        verdict = "STRONG SELL"
        action = "Enter Short"
        status = "negative"
        reason = "Strong downtrend plus negative momentum"
    elif signal == SIGNAL_SELL or (directional_score <= 42 and bearish_regime):
        verdict = "SELL"
        action = "Enter Short"
        status = "negative"
        reason = "Bearish structure with downside confirmation"
    else:
        verdict = "WAIT"
        action = "Wait for better setup"
        status = "warning"
        reason = "No clean setup"

    position_size_pct = safe_num(cfg.get("bt_pos_size"), 0.0) * 100.0
    if verdict in {"WAIT", "AVOID"}:
        allocation = 0.0
    elif confidence < 60:
        allocation = min(position_size_pct, 3.0)
    elif confidence < 75:
        allocation = min(position_size_pct, 5.0)
    else:
        allocation = position_size_pct

    stop_loss_pct = safe_num(cfg.get("stop_loss_pct"), 0.0) * 100.0
    take_profit_pct = safe_num(cfg.get("take_profit_pct"), 0.0) * 100.0
    is_long = verdict in {"BUY", "STRONG BUY"}
    is_short = verdict in {"SELL", "STRONG SELL"}
    has_price = close > 0

    if has_price and (is_long or is_short):
        entry = close
        if is_long:
            stop = entry * (1 - stop_loss_pct / 100.0)
            target = entry * (1 + take_profit_pct / 100.0)
        else:
            stop = entry * (1 + stop_loss_pct / 100.0)
            target = entry * (1 - take_profit_pct / 100.0)
        entry_text = fmt_price(entry, symbol)
        stop_text = fmt_price(stop, symbol)
        target_text = fmt_price(target, symbol)
        rr_text = f"1:{safe_num(cfg.get('risk_reward'), 0.0):.1f}"
    else:
        entry_text = "-"
        stop_text = "-"
        target_text = "-"
        rr_text = "-"

    allocation_text = f"{allocation:.0f}%" if allocation == round(allocation) else f"{allocation:.1f}%"
    return {
        "symbol": symbol,
        "verdict": verdict,
        "confidence": f"{confidence}%",
        "action": action,
        "entry": entry_text,
        "stop": stop_text,
        "target": target_text,
        "risk_reward": rr_text,
        "allocation": allocation_text,
        "reason": reason,
        "status": status,
    }


def render_trade_decision_card(decision: dict) -> None:
    if not isinstance(decision, dict):
        return

    def metric(label: str, value: str, primary: bool = False) -> str:
        class_name = "trade-decision-metric is-primary" if primary else "trade-decision-metric"
        return (
            f"<div class='{class_name}'>"
            f"<span>{render_info_label(str(label))}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            "</div>"
        )

    status = html.escape(str(decision.get("status", "warning")))
    symbol = html.escape(str(decision.get("symbol", "")))
    verdict = html.escape(str(decision.get("verdict", "WAIT")))
    action = html.escape(str(decision.get("action", "Wait for better setup")))
    reason = html.escape(str(decision.get("reason", "No clean setup")))

    st.markdown(
        f"""
        <div class="trade-decision-card status-{status}">
          <div class="trade-decision-core">
            <div class="trade-decision-symbol">{symbol}</div>
            <div class="trade-decision-verdict">{verdict}</div>
            <div class="trade-decision-action">{action}</div>
          </div>
          <div class="trade-decision-metrics">
            {metric("Confidence", decision.get("confidence", "0%"), True)}
            {metric("Entry Zone", decision.get("entry", "-"))}
            {metric("Stop Loss", decision.get("stop", "-"))}
            {metric("Take Profit", decision.get("target", "-"))}
            {metric("Risk / Reward", decision.get("risk_reward", "-"))}
            {metric("Allocation", decision.get("allocation", "0%"), True)}
          </div>
          <div class="trade-decision-reason">
            <span>Reason</span>
            <strong>{reason}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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


def load_ml_prediction(df: pd.DataFrame, symbol: str) -> dict:
    """Run the cached ML prediction path for the active market dataset."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "market dataset is empty"}
    try:
        df_serialized = df.to_json(date_format="iso")
        df_hash = hashlib.sha256(df_serialized.encode("utf-8")).hexdigest()
        return train_and_predict(df_hash, df_serialized, symbol)
    except Exception as exc:
        return {"error": str(exc)}


def market_data_cache_key(df: pd.DataFrame, symbol: str, timeframe: str, limit: int) -> str:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return f"{symbol}|{timeframe}|{limit}|empty"
    close_tail = df["close"].tail(5).round(8).astype(str).tolist() if "close" in df.columns else []
    last_index = str(df.index[-1])
    return "|".join([symbol, timeframe, str(limit), str(len(df)), last_index, *close_tail])


def trim_session_cache(cache_name: str, max_items: int = 6) -> None:
    cache = st.session_state.get(cache_name, {})
    if isinstance(cache, dict) and len(cache) > max_items:
        for key in list(cache.keys())[:-max_items]:
            cache.pop(key, None)


def get_cached_ml_prediction(df: pd.DataFrame, symbol: str, timeframe: str, limit: int) -> dict:
    key = market_data_cache_key(df, symbol, timeframe, limit)
    cache = st.session_state.setdefault("ml_prediction_cache", {})
    if key in cache:
        result = dict(cache[key])
        result["_cache_status"] = "cached"
        return result

    start = time.perf_counter()
    result = dict(load_ml_prediction(df, symbol) or {})
    result["_elapsed_ms"] = int((time.perf_counter() - start) * 1000)
    result["_cache_status"] = "calculated"
    cache[key] = result
    trim_session_cache("ml_prediction_cache")
    return dict(result)


def has_cached_ml_prediction(df: pd.DataFrame, symbol: str, timeframe: str, limit: int) -> bool:
    key = market_data_cache_key(df, symbol, timeframe, limit)
    return key in st.session_state.get("ml_prediction_cache", {})


@st.cache_data(ttl=600, show_spinner=False)
def load_backtest_result(symbol: str, timeframe: str, limit: int, initial_capital: float,
                         stop_loss_pct: float, take_profit_pct: float, position_size_pct: float) -> dict:
    full_df = load_full_data(symbol, timeframe, limit)
    return run_backtest(
        full_df,
        initial_capital=initial_capital,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        position_size_pct=position_size_pct,
    )


def backtest_cache_key(symbol: str, timeframe: str, limit: int, initial_capital: float,
                       stop_loss_pct: float, take_profit_pct: float, position_size_pct: float) -> tuple:
    return (
        symbol, timeframe, int(limit), round(float(initial_capital), 2),
        round(float(stop_loss_pct), 4), round(float(take_profit_pct), 4),
        round(float(position_size_pct), 4),
    )


@st.cache_data(ttl=120, show_spinner=False)
def load_portfolio_risk(capital: float, close: float, atr: float, confidence: float,
                        risk_tolerance: str, risk_reward: float) -> dict:
    risk = assess_risk(capital, close, atr, confidence, risk_tolerance)
    risk["risk_reward"] = risk_reward
    return risk


@st.cache_data(ttl=120, show_spinner=False)
def load_position_size_cached(capital: float, entry: float, stop_loss: float,
                              risk_per_trade: float, max_position_pct: float) -> dict:
    return calculate_position_size(capital, entry, stop_loss, risk_per_trade, max_position_pct)


def orderbook_source_label(ob: dict) -> str:
    source = str((ob or {}).get("source", "live")).lower()
    return {"live": "Live", "cached": "Cached", "synthetic": "Synthetic"}.get(source, source.title() or "Live")


def orderbook_source_message(ob: dict) -> tuple[str, str] | None:
    label = orderbook_source_label(ob)
    if label == "Live":
        return None
    if label == "Cached":
        return "Displaying the most recent cached order book snapshot.", "info"
    if label == "Synthetic":
        return "Displaying synthetic order book data because a live snapshot is not currently available.", "warning"
    return f"Displaying {label.lower()} order book data.", "info"


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
            color = "#00A868"
        elif score >= 1.0:
            signal = "BUY"
            alignment = "Bullish"
            color = "#00E08A"
        elif score <= -2.0:
            signal = "SELL"
            alignment = "Strong Bearish"
            color = "#8b0000"
        elif score <= -1.0:
            signal = "SELL"
            alignment = "Bearish"
            color = "#FF5C73"
        else:
            signal = "HOLD"
            alignment = "Neutral"
            color = "#FFB84D"
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
        "color": "#00E08A" if avg_score > 0 else ("#FF5C73" if avg_score < 0 else "#FFB84D"),
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
    if st.session_state.get("header_theme"):
        st.session_state.theme = normalize_theme_name(st.session_state.header_theme)

    st.sidebar.markdown(
        "<div class='sidebar-block'><h3>SuperSignal</h3></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    init_widget_from_query("theme", "theme", THEME_OPTIONS[0], str)
    st.session_state.theme = normalize_theme_name(st.session_state.theme)
    theme = st.session_state.theme
    qp_set("theme", theme)
    st.session_state.header_theme = theme

    st.sidebar.subheader("Market")

    raw_symbol_default = qp_get("symbol", st.session_state.get("selected_symbol", watchlist_symbols[0]))
    if "/" not in str(raw_symbol_default) and str(raw_symbol_default).endswith("USDT"):
        base = str(raw_symbol_default).replace("USDT", "")
        raw_symbol_default = f"{base}/USDT"
    symbol_default = raw_symbol_default if raw_symbol_default in watchlist_symbols else watchlist_symbols[0]
    st.session_state.setdefault("selected_symbol", symbol_default)
    if st.session_state.selected_symbol not in watchlist_symbols:
        st.session_state.selected_symbol = symbol_default

    symbol = st.sidebar.selectbox(
        "Symbol",
        watchlist_symbols,
        index=watchlist_symbols.index(st.session_state.selected_symbol)
        if st.session_state.selected_symbol in watchlist_symbols else 0,
        key="selected_symbol",
    )
    qp_set("symbol", symbol)

    timeframe_options = list(TIMEFRAMES)
    timeframe_default = qp_choice(
        "timeframe", timeframe_options,
        st.session_state.get("selected_timeframe", timeframe_options[0]),
    )
    st.session_state.setdefault("selected_timeframe", timeframe_default)
    if st.session_state.selected_timeframe not in timeframe_options:
        st.session_state.selected_timeframe = timeframe_options[0]

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        timeframe_options,
        index=timeframe_options.index(st.session_state.selected_timeframe)
        if st.session_state.selected_timeframe in timeframe_options else 0,
        key="selected_timeframe",
    )
    qp_set("timeframe", timeframe)

    init_widget_from_query("candle_limit", "limit", 500, lambda v: qp_int("limit", 500, 100, 1000, 50))
    limit = st.sidebar.slider("Candle Limit", 100, 1000, 500, 50, key="candle_limit")
    qp_set("limit", limit)

    st.sidebar.subheader("Auto-Refresh")
    refresh_options = ["Off", "30s", "1m", "5m"]
    refresh_default = qp_choice("refresh", refresh_options, "Off")
    st.session_state.setdefault("refresh_option", refresh_default)
    if st.session_state.refresh_option not in refresh_options:
        st.session_state.refresh_option = "Off"
    refresh_option = st.sidebar.select_slider(
        "Interval", options=refresh_options, value=st.session_state.refresh_option, key="refresh_option"
    )
    qp_set("refresh", refresh_option)
    ms_map    = {"Off": None, "30s": 30_000, "1m": 60_000, "5m": 300_000}
    refresh_ms = ms_map[refresh_option]

    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.sidebar.subheader("Chart Overlays")
    show_defaults = {
        "ema_9": True, "ema_21": True, "ema_50": True, "ema_200": True,
        "sma_20": False, "sma_50": False, "sma_200": False, "vwap": False,
        "supertrend": False, "ichimoku": False, "psar": False, "bb": True,
        "keltner": False, "donchian": False, "fvg": True, "ob": True, "sr_lines": True,
    }
    overlay_widget_keys = {
        "ema_9": "s_e9", "ema_21": "s_e21", "ema_50": "s_e50", "ema_200": "s_e200",
        "sma_20": "s_s20", "sma_50": "s_s50", "sma_200": "s_s200", "vwap": "s_vwap",
        "supertrend": "s_st", "ichimoku": "s_ich", "psar": "s_psar", "bb": "s_bb",
        "keltner": "s_kc", "donchian": "s_dc", "fvg": "s_fvg", "ob": "s_ob", "sr_lines": "s_sr",
    }
    for overlay_key, widget_key in overlay_widget_keys.items():
        if widget_key not in st.session_state:
            st.session_state[widget_key] = qp_bool(f"show_{overlay_key}", show_defaults[overlay_key])

    show = {}
    with st.sidebar.expander("Trend", expanded=False):
        show["ema_9"]      = st.checkbox("EMA 9",    key="s_e9")
        show["ema_21"]     = st.checkbox("EMA 21",   key="s_e21")
        show["ema_50"]     = st.checkbox("EMA 50",   key="s_e50")
        show["ema_200"]    = st.checkbox("EMA 200",  key="s_e200")
        show["sma_20"]     = st.checkbox("SMA 20",   key="s_s20")
        show["sma_50"]     = st.checkbox("SMA 50",   key="s_s50")
        show["sma_200"]    = st.checkbox("SMA 200",  key="s_s200")
        show["vwap"]       = st.checkbox("VWAP",     key="s_vwap")
        show["supertrend"] = st.checkbox("Supertrend", key="s_st")
        show["ichimoku"]   = st.checkbox("Ichimoku", key="s_ich")
        show["psar"]       = st.checkbox("Parabolic SAR", key="s_psar")
    with st.sidebar.expander("Volatility", expanded=False):
        show["bb"]       = st.checkbox("Bollinger Bands",  key="s_bb")
        show["keltner"]  = st.checkbox("Keltner Channel",  key="s_kc")
        show["donchian"] = st.checkbox("Donchian Channel", key="s_dc")
    with st.sidebar.expander("Smart Money", expanded=False):
        show["fvg"]      = st.checkbox("Fair Value Gaps",  key="s_fvg")
        show["ob"]       = st.checkbox("Order Blocks",     key="s_ob")
        show["sr_lines"] = st.checkbox("Support/Resistance", key="s_sr")
    sync_overlay_query(show)

    st.sidebar.subheader("Risk Management")

    init_widget_from_query("paper_capital", "capital", 100.0, lambda v: qp_float("capital", 100.0, 5.0, 1_000_000.0))
    capital = st.sidebar.number_input(
        "Paper Capital ($)",
        5.0,
        1_000_000.0,
        step=1.0,
        format="%.2f",
        key="paper_capital",
    )
    qp_set("capital", capital)

    risk_options = ["conservative", "moderate", "aggressive"]
    risk_default = qp_choice("risk", risk_options, "moderate")
    st.session_state.setdefault("risk_tolerance", risk_default)
    if st.session_state.risk_tolerance not in risk_options:
        st.session_state.risk_tolerance = "moderate"
    risk_tolerance = st.sidebar.select_slider(
        "Risk Tolerance",
        risk_options,
        value=st.session_state.risk_tolerance,
        key="risk_tolerance",
    )
    qp_set("risk", risk_tolerance)

    init_widget_from_query("stop_loss_pct_input", "sl", 2.0, lambda v: qp_float("sl", 2.0, 0.5, 10.0))
    sl_pct = st.sidebar.slider(
        "Stop Loss %",
        0.5,
        10.0,
        step=0.5,
        key="stop_loss_pct_input",
    )
    qp_set("sl", sl_pct)
    stop_loss_pct = sl_pct / 100

    init_widget_from_query("take_profit_pct_input", "tp", 4.0, lambda v: qp_float("tp", 4.0, 1.0, 20.0))
    tp_pct = st.sidebar.slider(
        "Take Profit %",
        1.0,
        20.0,
        step=0.5,
        key="take_profit_pct_input",
    )
    qp_set("tp", tp_pct)
    take_profit_pct = tp_pct / 100

    rr = take_profit_pct / stop_loss_pct if stop_loss_pct else 2.0
    st.sidebar.markdown(f"**R/R:** 1:{rr:.1f}")

    st.sidebar.subheader("Backtesting")
    init_widget_from_query("bt_pos_size_pct", "bt_pos", 10, lambda v: qp_int("bt_pos", 10, 5, 50, 5))
    bt_pos_size_pct = st.sidebar.slider("Position Size %", 5, 50, 10, 5, key="bt_pos_size_pct")
    qp_set("bt_pos", bt_pos_size_pct)
    bt_pos_size = bt_pos_size_pct / 100

    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.sidebar.info("**Paper Trading Only** · no real funds.")

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

def render_market_status_item(title: str, value: str, detail: str = "", status: str = "muted") -> str:
    card_class = card_status_class(status=status, value=value, label=title)
    return (
        f"<div class='market-status-item {card_class}'>"
        f"<div class='market-status-label'>{render_info_label(str(title))}</div>"
        f"<div class='market-status-value'>{html.escape(str(value))}</div>"
        f"<div class='market-status-detail'>{html.escape(str(detail))}</div>"
        f"</div>"
    )


def overview_ai_score(pct, rsi, macd, ema9, ema21, ema50, ema200, signal) -> int:
    score = 50.0
    try:
        pct_value = float(pct or 0)
        score += max(min(pct_value, 10), -10) * 1.4
    except (TypeError, ValueError):
        pass

    if rsi is not None and not pd.isna(rsi):
        rsi_value = float(rsi)
        if 45 <= rsi_value <= 60:
            score += 8
        elif 60 < rsi_value <= 70:
            score += 5
        elif 30 <= rsi_value < 45:
            score += 3
        elif rsi_value > 70:
            score -= 6
        else:
            score -= 4

    if macd is not None and not pd.isna(macd):
        macd_value = float(macd)
        score += 8 if macd_value > 0 else -8 if macd_value < 0 else 0

    if ema9 and ema21:
        score += 7 if float(ema9) > float(ema21) else -7
    if ema50 and ema200:
        score += 7 if float(ema50) > float(ema200) else -7

    if signal == SIGNAL_BUY:
        score += 18
    elif signal == SIGNAL_SELL:
        score -= 18

    return int(round(max(0, min(100, score))))


def render_overview(tickers, cg_data, watchlist_symbols, ind_map, signal_map, fg):
    rows = []
    total_mcap = 0.0
    total_volume = 0.0
    all_changes = []
    for sym in watchlist_symbols:
        t       = tickers.get(sym, {})
        cg      = cg_data.get(sym, {})
        ind     = ind_map.get(sym, {})
        sig_res = signal_map.get(sym, {})

        price = t.get("last", 0) or cg.get("current_price", 0)
        pct   = t.get("percentage", 0) or cg.get("price_change_percentage_24h", 0)
        vol   = t.get("quoteVolume", 0) or cg.get("total_volume", 0)
        mcap  = cg.get("market_cap", 0)
        sig   = sig_res.get("signal", "N/A")
        conf  = sig_res.get("confidence", 0.0)

        ema9  = ind.get("ema_9", 0)
        ema21 = ind.get("ema_21", 0)
        ema50 = ind.get("ema_50")
        ema200 = ind.get("ema_200")
        rsi = ind.get("rsi")
        macd = ind.get("macd")
        cross = (
            "Bull X" if ind.get("ema_bullish_cross")
            else "Bear X" if ind.get("ema_bearish_cross")
            else ("Above" if ind.get("ema9_above_ema21") else "Below")
        )
        ai_score = overview_ai_score(pct, rsi, macd, ema9, ema21, ema50, ema200, sig)
        total_mcap += float(mcap or 0)
        total_volume += float(vol or 0)
        all_changes.append(float(pct or 0))
        rows.append({
            "Logo":       coin_logo_url(sym, cg),
            "Symbol":     base_symbol(sym),
            "Pair":       sym,
            "Name":       cg.get("name", base_symbol(sym)),
            "Price":      float(price or 0),
            "24h %":      float(pct or 0),
            "Market Cap": float(mcap or 0),
            "Volume 24h": float(vol or 0),
            "RSI":        float(rsi) if rsi is not None else None,
            "MACD":       float(macd) if macd is not None else None,
            "EMA 9":      float(ema9) if ema9 else None,
            "EMA 21":     float(ema21) if ema21 else None,
            "EMA 50":     float(ema50) if ema50 else None,
            "EMA 200":    float(ema200) if ema200 else None,
            "Cross":      cross,
            "Signal":     sig,
            "AI Score":   ai_score,
            "Conf %":     float(conf * 100) if conf else None,
        })

    df_master = pd.DataFrame(rows).sort_values("Market Cap", ascending=False, kind="mergesort")
    tracked_count = len(watchlist_symbols)
    avg_change = float(np.mean(all_changes)) if all_changes else 0.0
    buy_count = sum(1 for sig in signal_map.values() if sig.get("signal") == SIGNAL_BUY)
    sell_count = sum(1 for sig in signal_map.values() if sig.get("signal") == SIGNAL_SELL)
    hold_count = sum(1 for sig in signal_map.values() if sig.get("signal") == SIGNAL_HOLD)
    signal_total = max(buy_count + sell_count + hold_count, 1)
    avg_conf = float(np.mean([sig.get("confidence", 0.0) for sig in signal_map.values()])) if signal_map else 0.0
    ai_conviction = int(round(avg_conf * 100))

    rsi_values = [float(ind.get("rsi")) for ind in ind_map.values() if ind.get("rsi") is not None]
    avg_rsi = float(np.mean(rsi_values)) if rsi_values else None
    macd_values = [float(ind.get("macd")) for ind in ind_map.values() if ind.get("macd") is not None]
    avg_macd = float(np.mean(macd_values)) if macd_values else None
    ai_scores = df_master["AI Score"].dropna().astype(float).tolist() if "AI Score" in df_master else []
    momentum_score = int(round(float(np.mean(ai_scores)))) if ai_scores else 50

    ema200_ready = df_master.dropna(subset=["EMA 200"]) if "EMA 200" in df_master else pd.DataFrame()
    above_ema200 = int((ema200_ready["Price"] > ema200_ready["EMA 200"]).sum()) if not ema200_ready.empty else 0
    breadth_total = int(len(ema200_ready))
    breadth_pct = int(round(above_ema200 / breadth_total * 100)) if breadth_total else 0
    breadth_value = f"{breadth_pct}%" if breadth_total else "N/A"
    breadth_detail = f"{above_ema200}/{breadth_total} above EMA200" if breadth_total else "EMA200 unavailable"

    fg_val = fg.get("value", 50)
    try:
        fg_num = int(fg_val)
    except (TypeError, ValueError):
        fg_num = 50
    fg_cl = fg.get("classification", "Neutral")

    regime_points = 0
    regime_points += 1 if avg_change > 0.25 else -1 if avg_change < -0.25 else 0
    regime_points += 1 if buy_count > sell_count else -1 if sell_count > buy_count else 0
    regime_points += 1 if fg_num >= 55 else -1 if fg_num < 35 else 0
    if avg_rsi is not None:
        regime_points += 1 if 45 <= avg_rsi <= 65 else -1 if avg_rsi > 72 or avg_rsi < 28 else 0
    if avg_macd is not None:
        regime_points += 1 if avg_macd > 0 else -1 if avg_macd < 0 else 0

    regime = "Bullish" if regime_points >= 2 else "Bearish" if regime_points <= -2 else "Neutral"
    regime_status = "positive" if regime == "Bullish" else "negative" if regime == "Bearish" else "warning"
    dominance_edge = abs(buy_count - sell_count) / signal_total
    confidence = int(round(min(95, max(45, 50 + abs(regime_points) * 8 + dominance_edge * 22 + avg_conf * 18))))
    risk_score = 0
    risk_score += 2 if momentum_score < 40 else 1 if momentum_score < 55 else 0
    risk_score += 2 if breadth_total and breadth_pct < 35 else 1 if breadth_total and breadth_pct < 50 else 0
    risk_score += 2 if fg_num < 25 else 1 if fg_num < 40 else 0
    risk_score += 1 if sell_count > buy_count else 0
    risk_level = "High" if risk_score >= 4 else "Low" if risk_score == 0 and momentum_score >= 62 and (not breadth_total or breadth_pct >= 55) else "Moderate"
    risk_status = "negative" if risk_level == "High" else "positive" if risk_level == "Low" else "warning"

    def overview_signal_class(signal: str) -> str:
        return "buy" if signal == SIGNAL_BUY else "sell" if signal == SIGNAL_SELL else "hold"

    def display_signal(row: pd.Series) -> str:
        signal = row.get("Signal", "N/A")
        score = float(row.get("AI Score", 0) or 0)
        if signal == SIGNAL_BUY and score >= 85:
            return "Strong Buy"
        if signal == SIGNAL_SELL and score <= 30:
            return "Strong Sell"
        return str(signal).title() if signal in {SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD} else "N/A"

    def opportunity_reason(row: pd.Series) -> str:
        if row.get("Signal") == SIGNAL_BUY and row.get("EMA 50") and row.get("EMA 200") and row.get("EMA 50") > row.get("EMA 200"):
            return "Momentum + EMA alignment"
        if float(row.get("24h %", 0) or 0) > max(avg_change, 0):
            return "High relative strength"
        if row.get("MACD") is not None and not pd.isna(row.get("MACD")) and row.get("MACD") > 0:
            return "Positive MACD momentum"
        if row.get("Signal") == SIGNAL_BUY:
            return "Signal engine accumulation"
        return "Balanced risk-adjusted setup"

    def row_reasons(row: pd.Series) -> list[str]:
        reasons = []
        if row.get("EMA 200") is not None and not pd.isna(row.get("EMA 200")):
            reasons.append("Above EMA200" if row.get("Price", 0) > row.get("EMA 200") else "Below EMA200")
        if row.get("MACD") is not None and not pd.isna(row.get("MACD")):
            reasons.append("Bullish MACD" if row.get("MACD") > 0 else "Bearish MACD")
        if row.get("RSI") is not None and not pd.isna(row.get("RSI")):
            rsi_value = float(row.get("RSI"))
            reasons.append("RSI healthy" if 35 <= rsi_value <= 70 else "RSI stretched")
        if float(row.get("24h %", 0) or 0) >= 0:
            reasons.append("Positive momentum")
        else:
            reasons.append("Negative momentum")
        return reasons[:4]

    def row_trend(row: pd.Series) -> str:
        if row.get("EMA 200") is not None and not pd.isna(row.get("EMA 200")):
            return "Bullish" if row.get("Price", 0) > row.get("EMA 200") else "Bearish"
        return "Neutral"

    def ema_status(row: pd.Series) -> str:
        if row.get("EMA 200") is None or pd.isna(row.get("EMA 200")):
            return "N/A"
        return "Bullish" if row.get("Price", 0) > row.get("EMA 200") else "Bearish"

    def rsi_status(row: pd.Series) -> str:
        if row.get("RSI") is None or pd.isna(row.get("RSI")):
            return "N/A"
        rsi_value = float(row.get("RSI"))
        if rsi_value > 70:
            return "Overbought"
        if rsi_value < 30:
            return "Oversold"
        return "Healthy"

    def macd_status(row: pd.Series) -> str:
        if row.get("MACD") is None or pd.isna(row.get("MACD")):
            return "N/A"
        return "Bullish" if row.get("MACD") > 0 else "Bearish" if row.get("MACD") < 0 else "Neutral"

    def volume_status(row: pd.Series) -> str:
        if not total_volume:
            return "N/A"
        avg_volume = total_volume / max(tracked_count, 1)
        return "Rising" if row.get("Volume 24h", 0) >= avg_volume else "Quiet"

    def badge_status(value: str) -> str:
        value = str(value).lower()
        if any(term in value for term in ("bull", "healthy", "rising", "buy", "strong")):
            return "positive"
        if any(term in value for term in ("bear", "overbought", "oversold", "sell", "negative")):
            return "negative"
        return "warning"

    def ai_summary(row: pd.Series) -> str:
        trend_text = "above EMA200" if ema_status(row) == "Bullish" else "below EMA200" if ema_status(row) == "Bearish" else "without EMA200 confirmation"
        momentum_text = "positive momentum" if float(row.get("24h %", 0) or 0) >= 0 else "negative momentum"
        return (
            f"{row.get('Symbol', 'Asset')} is trading {trend_text} with {momentum_text}. "
            f"RSI remains {rsi_status(row).lower()} and MACD trend remains {macd_status(row).lower()}. "
            f"Current signal bias remains {row.get('Signal', 'N/A')}."
        )

    sorted_strength = df_master.sort_values(
        by=["AI Score", "Conf %", "24h %", "Market Cap"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    top_strength = sorted_strength.head(3)
    top_symbols = top_strength["Symbol"].tolist()
    top_text = ", ".join(top_symbols[:-1]) + (f" and {top_symbols[-1]}" if len(top_symbols) > 1 else (top_symbols[0] if top_symbols else "N/A"))
    momentum_word = "positive" if momentum_score >= 60 else "negative" if momentum_score < 45 else "mixed"
    direction_raw = (
        momentum_score * 0.35
        + (breadth_pct if breadth_total else 50) * 0.25
        + ai_conviction * 0.20
        + fg_num * 0.10
        + ((buy_count - sell_count + signal_total) / (signal_total * 2)) * 100 * 0.10
    )
    direction_score = int(round(max(0, min(100, direction_raw))))
    if direction_score >= 82:
        direction_state = "Strong Bullish"
    elif direction_score >= 62:
        direction_state = "Bullish"
    elif direction_score <= 18:
        direction_state = "Strong Bearish"
    elif direction_score <= 38:
        direction_state = "Bearish"
    else:
        direction_state = "Neutral"
    direction_status = "positive" if "Bullish" in direction_state else "negative" if "Bearish" in direction_state else "warning"
    market_summary = (
        f"Market remains {regime.lower()} despite {str(fg_cl).lower()} sentiment. "
        f"Momentum leadership is concentrated in {top_text}."
        if regime == "Bullish"
        else f"Trend conditions remain {regime.lower()}. {breadth_detail}."
    )
    market_chips = [
        ("Market Breadth", breadth_value, breadth_detail),
        ("Momentum Score", f"{momentum_score}/100", momentum_word.title()),
        ("AI Conviction", f"{ai_conviction}%", f"{buy_count} BUY / {sell_count} SELL"),
        ("Fear & Greed", str(fg_num), str(fg_cl)),
        ("Top Strength", top_text, "Ranked by AI Score"),
    ]
    chips_html = "".join(
        f"<div class='market-intel-chip'><span>{render_info_label(label)}</span>"
        f"<strong>{html.escape(str(value))}</strong><em>{html.escape(str(detail))}</em></div>"
        for label, value, detail in market_chips
    )
    st.markdown(
        "<div class='market-intel-card'>"
        "<div class='market-intel-head'><div class='market-intel-title'>Market Intelligence</div>"
        f"<div class='market-intel-meta'>{html.escape(market_summary)}</div></div>"
        f"<div class='direction-gauge {card_status_class(status=direction_status)}'>"
        f"<div class='direction-gauge-top'><div><div class='direction-gauge-label'>{render_info_label('Market Direction')}</div>"
        f"<div class='direction-gauge-value'>{html.escape(direction_state)}</div></div>"
        f"<div class='market-intel-meta'>Confidence {confidence}%</div></div>"
        "<div class='direction-track'>"
        f"<div class='direction-marker' style='left:{direction_score}%'></div></div>"
        "<div class='direction-scale'><span>Strong Bearish</span><span>Neutral</span><span>Strong Bullish</span></div>"
        "</div>"
        "<div class='market-intel-body'>"
        f"<div class='market-intel-core {card_status_class(status=regime_status)}'>"
        f"<div class='market-intel-label'>{render_info_label('Market Regime')}</div>"
        f"<div class='market-intel-value'>{html.escape(regime)}</div>"
        f"<div class='market-intel-sub'>Confidence {confidence}% · Risk {risk_level}</div>"
        "</div>"
        f"<div class='market-intel-chips'>{chips_html}</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    opportunity_cards = []
    for rank, (_, row) in enumerate(top_strength.head(3).iterrows(), start=1):
        signal = row.get("Signal", "N/A")
        signal_cls = overview_signal_class(signal)
        status = "positive" if signal == SIGNAL_BUY else "negative" if signal == SIGNAL_SELL else "warning"
        reasons_html = "".join(f"<span class='reason-chip'>{html.escape(reason)}</span>" for reason in row_reasons(row))
        opportunity_cards.append(
            f"<div class='opportunity-card {card_status_class(status=status)}'>"
            f"<div class='opportunity-top'>{render_coin_identity(row['Pair'], cg_data.get(row['Pair'], {}))}<span class='opportunity-rank'>#{rank}</span></div>"
            f"<div class='opportunity-mid'><div class='opportunity-score'>AI Score {int(row['AI Score'])}</div>"
            f"<span class='overview-signal-pill {signal_cls}'>{html.escape(display_signal(row))}</span></div>"
            f"<div class='opportunity-reason'>{html.escape(opportunity_reason(row))}</div>"
            f"<div class='reason-list'>{reasons_html}</div>"
            f"</div>"
        )
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Top Opportunities</div>"
        "<div class='compact-section-meta'>Ranked by AI Score, confidence, strength</div></div>"
        f"<div class='opportunity-grid'>{''.join(opportunity_cards)}</div>",
        unsafe_allow_html=True,
    )
    movers = df_master.sort_values("24h %", ascending=False, kind="mergesort")
    top_gainers = movers.head(3)
    top_losers = movers.tail(3).sort_values("24h %", ascending=True, kind="mergesort")

    mover_cards = []
    for label, card_status, items in (("GAINER", "status-positive", top_gainers), ("LOSER", "status-negative", top_losers)):
        for _, row in items.iterrows():
            color_var = "var(--success)" if card_status == "status-positive" else "var(--danger)"
            signal_cls = overview_signal_class(row.get("Signal", "N/A"))
            mover_cards.append(
                f"<div class='dashboard-card mover-card {card_status}'>"
                f"<div class='mover-topline'>{render_coin_identity(row['Pair'], cg_data.get(row['Pair'], {}))}<span class='mover-label'>{label}</span></div>"
                f"<div class='mover-metric-row'><div class='metric-val' style='color:{color_var}'>{row['24h %']:+.2f}%</div>"
                f"<div class='metric-subtext'>{fmt_price(row['Price'], row['Pair'])}</div></div>"
                f"<div class='mover-metric-row'><div class='metric-subtext'>AI Score {int(row['AI Score'])}</div>"
                f"<span class='overview-signal-pill {signal_cls}'>{html.escape(str(row.get('Signal', 'N/A')))}</span></div>"
                f"</div>"
            )
    st.markdown(f"<div class='mover-grid'>{''.join(mover_cards)}</div>", unsafe_allow_html=True)
    coin_options = df_master["Pair"].tolist()
    if coin_options:
        selected_default = st.session_state.get("overview_selected_coin", coin_options[0])
        if selected_default not in coin_options:
            selected_default = coin_options[0]
        selected_pair = st.selectbox(
            "Inspect coin",
            coin_options,
            index=coin_options.index(selected_default),
            key="overview_coin_inspector",
            format_func=lambda pair: base_symbol(pair),
        )
        st.session_state.overview_selected_coin = selected_pair
        selected_row = df_master.loc[df_master["Pair"] == selected_pair].iloc[0]
        selected_cg = cg_data.get(selected_pair, {})
        signal_cls = overview_signal_class(selected_row.get("Signal", "N/A"))
        metric_items = [
            ("24h Change", f"{selected_row['24h %']:+.2f}%"),
            ("AI Score", f"{int(selected_row['AI Score'])}"),
            ("Signal", str(selected_row.get("Signal", "N/A"))),
            ("Confidence", "N/A" if pd.isna(selected_row.get("Conf %")) else f"{selected_row['Conf %']:.0f}%"),
            ("Market Trend", row_trend(selected_row)),
            ("EMA Status", ema_status(selected_row)),
            ("RSI Status", rsi_status(selected_row)),
            ("MACD Status", macd_status(selected_row)),
        ]
        metrics_html = "".join(
            f"<div class='intel-metric'><span>{render_info_label(label)}</span><strong>{html.escape(value)}</strong></div>"
            for label, value in metric_items
        )
        badge_items = [
            ("EMA200", ema_status(selected_row)),
            ("Momentum", "Strong" if selected_row.get("AI Score", 0) >= 80 else "Mixed" if selected_row.get("AI Score", 0) >= 50 else "Weak"),
            ("RSI", rsi_status(selected_row)),
            ("Volume", volume_status(selected_row)),
            ("Signal", str(selected_row.get("Signal", "N/A"))),
        ]
        badges_html = "".join(
            f"<span class='smart-badge {badge_status(value)}'>{html.escape(label)} · {html.escape(str(value))}</span>"
            for label, value in badge_items
        )
        logo = coin_logo_url(selected_pair, selected_cg)
        logo_html = f"<img class='coin-logo' src='{html.escape(logo, quote=True)}' alt='{base_symbol(selected_pair)} logo'>" if logo else f"<span class='coin-logo coin-logo-fallback'>{html.escape(base_symbol(selected_pair)[:2])}</span>"
        st.markdown(
            "<div class='coin-intel-card'>"
            "<div class='market-intel-head'><div class='market-intel-title'>Coin Intelligence</div>"
            f"<div class='market-intel-meta'>Selected from Overview scanner</div></div>"
            "<div class='coin-intel-grid'>"
            "<div>"
            f"<div class='coin-intel-identity'>{logo_html}<div><div class='coin-intel-symbol'>{html.escape(base_symbol(selected_pair))}</div>"
            f"<div class='coin-intel-name'>{html.escape(str(selected_cg.get('name', selected_row.get('Name', base_symbol(selected_pair)))))} </div></div></div>"
            f"<div class='coin-intel-price'>{fmt_price(selected_row['Price'], selected_pair)} <span class='overview-signal-pill {signal_cls}'>{html.escape(str(selected_row.get('Signal', 'N/A')))}</span></div>"
            f"<div class='coin-intel-summary'>{html.escape(ai_summary(selected_row))}</div>"
            "</div>"
            f"<div><div class='intel-metrics'>{metrics_html}</div><div class='smart-badges'>{badges_html}</div></div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='scanner-head'><div class='scanner-title'>Market Scanner</div>"
        f"<div class='scanner-status-line'><span>{tracked_count} coins</span><span>Market Cap sort</span><span>1h candles</span><span>80 bars</span></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='scanner-summary'>"
        f"<span>{tracked_count} assets scanned</span>"
        f"<span>{buy_count} bullish</span>"
        f"<span>{hold_count} neutral</span>"
        f"<span>{sell_count} bearish</span>"
        f"<span>Average AI Score: {momentum_score}</span>"
        + "</div>",
        unsafe_allow_html=True,
    )

    df_table = df_master.copy()
    filter_col, search_col = st.columns([0.28, 0.72])
    with filter_col:
        quick_filter = st.selectbox(
            "Quick filter",
            ["All", "BUY", "HOLD", "SELL", "Gainers", "Losers"],
            key="market_scanner_quick_filter",
        )
    with search_col:
        search_query = st.text_input(
            "Search/filter by symbol",
            placeholder="Search BTC, ETH, Solana...",
            key="market_scanner_search",
        ).strip().lower()

    if quick_filter in {"BUY", "HOLD", "SELL"}:
        df_table = df_table.loc[df_table["Signal"] == quick_filter].copy()
    elif quick_filter == "Gainers":
        df_table = df_table.loc[df_table["24h %"] > 0].copy()
    elif quick_filter == "Losers":
        df_table = df_table.loc[df_table["24h %"] < 0].copy()

    if search_query:
        mask = (
            df_table["Symbol"].str.lower().str.contains(search_query, na=False)
            | df_table["Pair"].str.lower().str.contains(search_query, na=False)
            | df_table["Name"].str.lower().str.contains(search_query, na=False)
        )
        df_table = df_table.loc[mask].copy()

    def style_change(value):
        if pd.isna(value):
            return ""
        magnitude = abs(float(value))
        if value >= 0:
            bg = "rgba(0,224,138,0.32)" if magnitude >= 5 else "rgba(0,224,138,0.20)" if magnitude >= 2 else "rgba(0,224,138,0.12)"
            return f"background:{bg};color:#00E08A;font-weight:850"
        bg = "rgba(255,92,115,0.32)" if magnitude >= 5 else "rgba(255,92,115,0.20)" if magnitude >= 2 else "rgba(255,92,115,0.12)"
        return f"background:{bg};color:#FF5C73;font-weight:850"

    def style_signal(value):
        return {
            SIGNAL_BUY: "background:rgba(0,224,138,0.14);color:#00E08A;font-weight:950",
            SIGNAL_SELL: "background:rgba(255,92,115,0.14);color:#FF5C73;font-weight:950",
            SIGNAL_HOLD: "background:rgba(255,184,77,0.14);color:#FFB84D;font-weight:900",
        }.get(value, "")

    def style_ai_score(value):
        if pd.isna(value):
            return ""
        value = float(value)
        if value >= 66:
            return "color:#00E08A;font-weight:900"
        if value <= 34:
            return "color:#FF5C73;font-weight:900"
        return "color:#FFB84D;font-weight:850"

    styled_table = (
        dataframe_theme_styles(df_table)
        .map(style_change, subset=["24h %"])
        .map(style_signal, subset=["Signal"])
        .map(style_ai_score, subset=["AI Score"])
    )
    st.dataframe(
        styled_table,
        width="stretch",
        hide_index=True,
        height=min(640, 72 + max(len(df_table), 1) * 42),
        column_order=[
            "Logo", "Symbol", "Name", "Price", "24h %", "Market Cap", "Volume 24h",
            "AI Score", "RSI", "MACD", "EMA 9", "EMA 21", "EMA 50", "EMA 200", "Cross", "Signal", "Conf %",
        ],
        column_config={
            "Logo": st.column_config.ImageColumn("", width="small"),
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Price": st.column_config.NumberColumn("Price", format="$%.4f"),
            "24h %": st.column_config.NumberColumn("24h %", format="%+.2f%%"),
            "Market Cap": st.column_config.NumberColumn("Market Cap", format="$%.0f"),
            "Volume 24h": st.column_config.NumberColumn("Volume 24h", format="$%.0f"),
            "AI Score": st.column_config.NumberColumn("AI Score", format="%d", width="small"),
            "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
            "MACD": st.column_config.NumberColumn("MACD", format="%.4f"),
            "EMA 9": st.column_config.NumberColumn("EMA 9", format="$%.4f"),
            "EMA 21": st.column_config.NumberColumn("EMA 21", format="$%.4f"),
            "EMA 50": st.column_config.NumberColumn("EMA 50", format="$%.4f"),
            "EMA 200": st.column_config.NumberColumn("EMA 200", format="$%.4f"),
            "Cross": st.column_config.TextColumn("Cross", width="small"),
            "Signal": st.column_config.TextColumn("Signal", width="small"),
            "Conf %": st.column_config.NumberColumn("Conf %", format="%.0f%%"),
        },
    )

# ── Tab 2: Technical Analysis ─────────────────────────────────────────────────

def render_technical(df: pd.DataFrame, ind: dict, adv: dict,
                     symbol: str, sr: dict, cfg: dict, fg: dict):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        render_empty_state("Technical analysis data unavailable. Try a different timeframe.")
        return
    if not ind or not isinstance(ind, dict):
        render_empty_state("Technical indicators unavailable.")
        return

    close = ind["close"]
    rsi = ind["rsi"]
    macd = ind["macd"]
    macd_sig = ind["macd_signal"]
    stk = adv.get("stochrsi_k", 50)
    std = adv.get("stochrsi_d", 50)
    cci = adv.get("cci", 0)
    adx = adv.get("adx", 25)
    roc = adv.get("roc", 0)
    ema9 = ind.get("ema_9", close)
    ema21 = ind.get("ema_21", close)
    ema50 = ind.get("ema_50", close)
    ema200 = ind.get("ema_200", close)
    vwap = adv.get("vwap", close)
    sma20 = adv.get("sma_20", close)
    mfi = adv.get("mfi", 50)
    cmf = adv.get("cmf", 0)
    obv = adv.get("obv", 0)
    atr = ind.get("atr")
    bb_pct = ind.get("bb_pct", 0.5) * 100
    st_dir = adv.get("supertrend_dir", 0)
    st_lbl = "Bullish" if st_dir == 1 else ("Bearish" if st_dir == -1 else "N/A")
    psar_bull = adv.get("psar_bull", True)
    bullish_cross = ind.get("ema_bullish_cross", False)
    bearish_cross = ind.get("ema_bearish_cross", False)

    def ind_card(label, val_str, sub="", color="#ccc"):
        return (
            f"<div class='dashboard-card {card_status_class(status=status_from_color(color), label=sub, value=val_str)}' style='text-align:center'>"
            f"<div class='metric-label'>{render_info_label(str(label))}</div>"
            f"<div class='metric-val' style='color:{color};font-size:var(--metric-tile-value-size);line-height:1.08'>{val_str}</div>"
            f"<div class='metric-subtitle'>{sub}</div>"
            f"</div>"
        )

    def mini_metric(label: str, value: str) -> str:
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))
        return (
            "<div class='tech-mini-metric'>"
            f"<div class='tech-mini-label'>{render_info_label(str(label))}</div>"
            f"<div class='tech-mini-value'>{safe_value}</div>"
            "</div>"
        )

    rsi_label = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Healthy" if 40 <= rsi <= 65 else "Neutral"
    rsi_c = "#FF5C73" if rsi > 70 else ("#00E08A" if rsi < 30 else "#FFB84D")
    macd_c = "#00E08A" if macd > macd_sig else "#FF5C73"
    stk_c = "#FF5C73" if stk > 80 else ("#00E08A" if stk < 20 else "#FFB84D")
    cci_c = "#FF5C73" if cci > 100 else ("#00E08A" if cci < -100 else "#FFB84D")
    adx_c = "#00E08A" if adx > 30 else "#7B8596"
    roc_c = "#00E08A" if roc > 0 else "#FF5C73"
    mfi_c = "#FF5C73" if mfi > 80 else ("#00E08A" if mfi < 20 else "#FFB84D")
    cmf_c = "#00E08A" if cmf > 0.05 else ("#FF5C73" if cmf < -0.05 else "#FFB84D")
    bb_c = "#FF5C73" if bb_pct > 80 else ("#00E08A" if bb_pct < 20 else "#FFB84D")
    st_c = "#00E08A" if st_dir == 1 else ("#FF5C73" if st_dir == -1 else "#7B8596")

    score = 50
    score += 10 if ema9 > ema21 else -10
    score += 10 if close > ema50 else -10
    score += 14 if close > ema200 else -14
    score += 12 if macd > macd_sig else -12
    score += 8 if 40 <= rsi <= 65 else -8 if rsi > 75 or rsi < 25 else 0
    score += 8 if st_dir == 1 else -8 if st_dir == -1 else 0
    score += 6 if psar_bull else -6
    score += 5 if cmf > 0.05 else -5 if cmf < -0.05 else 0
    score += 4 if obv > 0 else -4 if obv < 0 else 0
    if adx > 30:
        score += 5 if close > ema200 else -5
    direction_score = int(round(max(0, min(100, score))))
    technical_bias = "Bullish" if direction_score >= 60 else "Bearish" if direction_score <= 40 else "Neutral"
    direction_state = "Strong Bullish" if direction_score >= 82 else "Bullish" if direction_score >= 62 else "Strong Bearish" if direction_score <= 18 else "Bearish" if direction_score <= 38 else "Neutral"
    confidence = int(round(min(95, max(45, 50 + abs(direction_score - 50) * 0.9))))
    bias_status = "positive" if technical_bias == "Bullish" else "negative" if technical_bias == "Bearish" else "warning"

    trend_status = "Bullish trend" if close > ema200 and ema9 > ema21 else "Bearish trend" if close < ema200 and ema9 < ema21 else "Mixed trend"
    momentum_status = "Bullish momentum" if macd > macd_sig and roc > 0 else "Bearish momentum" if macd < macd_sig and roc < 0 else "Mixed momentum"
    volatility_status = "Upper band stretch" if bb_pct > 80 else "Lower band stretch" if bb_pct < 20 else "Normal range"
    volume_flow = "Inflow" if cmf > 0.05 else "Outflow" if cmf < -0.05 else "Balanced"

    nr = sr.get("nearest_resistance", 0)
    ns = sr.get("nearest_support", 0)
    resistance_pct = sr.get("resistance_pct", 0)
    support_pct = sr.get("support_pct", 0)
    nearest_label = "Resistance" if resistance_pct <= support_pct else "Support"
    nearest_distance = min(resistance_pct, support_pct)

    near_support = ns > 0 and support_pct <= 1.5
    near_resistance = nr > 0 and resistance_pct <= 1.5
    trend_bullish = close > ema200 and ema9 > ema21 and st_dir == 1
    trend_bearish = close < ema200 and ema9 < ema21 and st_dir == -1
    momentum_bullish = macd > macd_sig and 35 <= rsi <= 70
    momentum_bearish = macd < macd_sig and (rsi < 55 or close < ema200)

    if technical_bias == "Bullish" and trend_bullish and momentum_bullish and adx >= 25:
        preferred_setup = "Long continuation"
    elif technical_bias == "Bearish" and trend_bearish and momentum_bearish and adx >= 25:
        preferred_setup = "Short continuation"
    elif near_support and technical_bias != "Bearish":
        preferred_setup = "Pullback entry"
    elif near_resistance and technical_bias != "Bearish":
        preferred_setup = "Breakout watch"
    else:
        preferred_setup = "Wait"

    if near_support:
        entry_context = "Above support"
    elif near_resistance:
        entry_context = "Near resistance" if preferred_setup != "Breakout watch" else "Breakout zone"
    elif resistance_pct <= 3 and technical_bias == "Bullish":
        entry_context = "Breakout zone"
    else:
        entry_context = "Range middle"

    invalidation = f"Below {fmt_price(ns, symbol)}" if preferred_setup in {"Long continuation", "Pullback entry", "Breakout watch"} and ns else f"Above {fmt_price(nr, symbol)}" if nr else "Nearest S/R"
    risk_points = 0
    risk_points += 1 if adx < 20 else -1 if adx >= 30 else 0
    risk_points += 1 if rsi > 72 or rsi < 28 else 0
    risk_points += 1 if bb_pct > 88 or bb_pct < 12 else 0
    risk_points += 1 if nearest_distance <= 0.8 else 0
    risk_points += -1 if st_dir == 1 and technical_bias == "Bullish" else -1 if st_dir == -1 and technical_bias == "Bearish" else 0
    risk_level = "High" if risk_points >= 2 else "Low" if risk_points <= -1 else "Moderate"

    sr_action = "Breakout watch" if near_resistance and technical_bias != "Bearish" else "Support bounce zone" if near_support and technical_bias != "Bearish" else "Resistance rejection risk" if near_resistance else "Range trading zone"

    reasons = []
    reasons.append("Price above EMA200" if close > ema200 else "Price below EMA200")
    reasons.append("MACD bullish" if macd > macd_sig else "MACD bearish")
    reasons.append("RSI healthy" if 35 <= rsi <= 70 else "RSI stretched")
    reasons.append("Supertrend bullish" if st_dir == 1 else "Supertrend bearish" if st_dir == -1 else "Supertrend neutral")
    reasons.append("CMF inflow" if cmf > 0 else "CMF outflow")
    reason_html = "".join(f"<span class='tech-reason-chip'>{html.escape(reason)}</span>" for reason in reasons)

    st.markdown(
        "<span id='technical-top'></span><div class='tech-command-head'>"
        f"<div class='tech-command-title'>Technical Command Center</div>"
        f"<div class='tech-command-meta'>{html.escape(symbol)} · {cfg.get('timeframe', '1h')} · {len(df)} candles</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if "technical_view_mode" not in st.session_state:
        st.session_state.technical_view_mode = "Summary"
    if "technical_chart_mode" not in st.session_state:
        st.session_state.technical_chart_mode = "Standard"

    nav_col, view_col, chart_mode_col = st.columns([1.0, 0.34, 0.48])
    with nav_col:
        st.markdown(
            "<div class='tech-control-row'><div class='tech-anchor-pills'>"
            "<a href='#technical-top'>Top</a><a href='#technical-chart'>Chart</a><a href='#technical-support-resistance'>Support/Resistance</a>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with view_col:
        view_mode = st.radio(
            "View Mode",
            ["Summary", "Full"],
            key="technical_view_mode",
            horizontal=True,
        )
    with chart_mode_col:
        chart_mode = st.radio(
            "Chart Overlays",
            ["Clean", "Standard", "Full"],
            key="technical_chart_mode",
            horizontal=True,
        )
    full_mode = view_mode == "Full"

    st.markdown(
        "<div class='tech-summary-card'>"
        "<div class='tech-summary-grid'>"
        f"<div class='tech-bias-core {card_status_class(status=bias_status)}'>"
        f"<div class='tech-label'>{render_info_label('Technical Bias')}</div>"
        f"<div class='tech-bias-value'>{technical_bias}</div>"
        f"<div class='tech-bias-sub'>Confidence {confidence}% · {direction_state}</div>"
        "<div class='tech-gauge'><div class='tech-gauge-track'>"
        f"<div class='tech-gauge-marker' style='left:{direction_score}%'></div></div>"
        "<div class='tech-gauge-scale'><span>Strong Bearish</span><span>Neutral</span><span>Strong Bullish</span></div></div>"
        "</div>"
        "<div class='tech-health-grid'>"
        f"<div class='tech-health-chip'><span>{render_info_label('Trend Status')}</span><strong>{html.escape(trend_status)}</strong><em>EMA 9/21/200</em></div>"
        f"<div class='tech-health-chip'><span>{render_info_label('Momentum Status')}</span><strong>{html.escape(momentum_status)}</strong><em>RSI {rsi:.1f} · ROC {roc:.2f}%</em></div>"
        f"<div class='tech-health-chip'><span>{render_info_label('Volatility Status')}</span><strong>{html.escape(volatility_status)}</strong><em>BB %B {bb_pct:.1f}%</em></div>"
        f"<div class='tech-health-chip'><span>{render_info_label('Volume Flow')}</span><strong>{html.escape(volume_flow)}</strong><em>CMF {cmf:.3f}</em></div>"
        "</div></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Technical Action Plan</div>"
        "<div class='compact-section-meta'>Decision context from price, S/R, EMA, MACD, RSI, ADX, Supertrend</div></div>"
        "<div class='tech-action-grid'>"
        f"<div class='tech-action-item {card_status_class(status=bias_status)}'><span>{render_info_label('Bias')}</span><strong>{technical_bias}</strong><em>{direction_state}</em></div>"
        f"<div class='tech-action-item {card_status_class(status='positive' if preferred_setup.startswith('Long') or preferred_setup == 'Pullback entry' else 'negative' if preferred_setup.startswith('Short') else 'warning')}'>"
        f"<span>{render_info_label('Preferred Setup')}</span><strong>{html.escape(preferred_setup)}</strong><em>ADX {adx:.1f}</em></div>"
        f"<div class='tech-action-item {card_status_class(status='positive' if entry_context in {'Above support', 'Breakout zone'} else 'negative' if entry_context == 'Near resistance' else 'warning')}'>"
        f"<span>{render_info_label('Entry Context')}</span><strong>{html.escape(entry_context)}</strong><em>{nearest_distance:.2f}% to {nearest_label}</em></div>"
        f"<div class='tech-action-item {card_status_class(status='warning')}'>"
        f"<span>{render_info_label('Invalidation')}</span><strong>{html.escape(invalidation)}</strong><em>Nearest level</em></div>"
        f"<div class='tech-action-item {card_status_class(status='negative' if risk_level == 'High' else 'positive' if risk_level == 'Low' else 'warning')}'>"
        f"<span>{render_info_label('Risk')}</span><strong>{risk_level}</strong><em>RSI {rsi:.1f} / BB {bb_pct:.1f}%</em></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    def checklist_item(label: str, state: str) -> str:
        state_label = {"pass": "Pass", "warn": "Warning", "fail": "Fail"}.get(state, "Warning")
        return f"<div class='tech-check {state}'><b>{state_label}</b><span>{html.escape(label)}</span></div>"

    checklist_html = "".join([
        checklist_item("Price above EMA200", "pass" if close > ema200 else "fail"),
        checklist_item("EMA9 above EMA21", "pass" if ema9 > ema21 else "fail"),
        checklist_item("MACD bullish", "pass" if macd > macd_sig else "fail"),
        checklist_item("RSI healthy", "pass" if 40 <= rsi <= 65 else "warn" if 35 <= rsi <= 70 else "fail"),
        checklist_item("CMF inflow", "pass" if cmf > 0.05 else "warn" if cmf >= 0 else "fail"),
        checklist_item("Supertrend bullish", "pass" if st_dir == 1 else "warn" if st_dir == 0 else "fail"),
        checklist_item("ADX strong" if adx > 30 else "ADX weak", "pass" if adx > 30 else "warn" if adx >= 20 else "fail"),
    ])
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Technical Checklist</div>"
        "<div class='compact-section-meta'>Pass / warning / fail</div></div>"
        f"<div class='tech-checklist'>{checklist_html}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Why this technical bias?</div>"
        f"<div class='compact-section-meta'>Score {direction_score}/100</div></div>"
        f"<div class='tech-reasons'>{reason_html}</div>",
        unsafe_allow_html=True,
    )

    if bullish_cross:
        st.success("EMA 9 x EMA 21 bullish crossover on latest candle")
    elif bearish_cross:
        st.error("EMA 9 x EMA 21 bearish crossover on latest candle")

    momentum_cards = [
        ind_card("RSI (14)", f"{rsi:.1f}", rsi_label, rsi_c),
        ind_card("Stoch RSI K", f"{stk:.1f}", f"D {std:.1f}", stk_c),
        ind_card("ROC (12)", f"{roc:.2f}%", "Positive" if roc > 0 else "Negative", roc_c),
        ind_card("MFI (14)", f"{mfi:.1f}", "Overbought" if mfi > 80 else "Oversold" if mfi < 20 else "Neutral", mfi_c),
        ind_card("CCI (20)", f"{cci:.1f}", "Overbought" if cci > 100 else "Oversold" if cci < -100 else "Neutral", cci_c),
    ]
    trend_cards = [
        ind_card("EMA 9", fmt_price(ema9, symbol), "Bull" if ema9 > ema21 else "Bear", "#00E08A" if ema9 > ema21 else "#FF5C73"),
        ind_card("EMA 21", fmt_price(ema21, symbol), f"Gap {abs(ema9-ema21)/ema21*100:.2f}%" if ema21 else "", "#00E08A" if ema9 > ema21 else "#FF5C73"),
        ind_card("EMA 50", fmt_price(ema50, symbol), "Bullish" if close > ema50 else "Bearish", "#00E08A" if close > ema50 else "#FF5C73"),
        ind_card("EMA 200", fmt_price(ema200, symbol), "Above" if close > ema200 else "Below", "#00E08A" if close > ema200 else "#FF5C73"),
        ind_card("SMA 20", fmt_price(sma20, symbol), "Above" if close > sma20 else "Below", "#00E08A" if close > sma20 else "#FF5C73"),
        ind_card("Supertrend", st_lbl, "Trend filter", st_c),
        ind_card("Parabolic SAR", "Bullish" if psar_bull else "Bearish", "Trailing regime", "#00E08A" if psar_bull else "#FF5C73"),
        ind_card("ADX (14)", f"{adx:.1f}", "Strong" if adx > 30 else "Weak", adx_c),
    ]
    flow_cards = [
        ind_card("OBV", format_large_number(abs(obv)).replace("$", ""), "Up" if obv > 0 else "Down", "#00E08A" if obv > 0 else "#FF5C73"),
        ind_card("CMF (20)", f"{cmf:.3f}", "Inflow" if cmf > 0 else "Outflow", cmf_c),
        ind_card("VWAP", fmt_price(vwap, symbol), "Above" if close > vwap else "Below", "#00E08A" if close > vwap else "#FF5C73"),
    ]
    volatility_cards = [
        ind_card("BB %B", f"{bb_pct:.1f}%", volatility_status, bb_c),
    ]
    if atr is not None:
        volatility_cards.append(ind_card("ATR (14)", fmt_price(float(atr), symbol), "Range proxy", "#FFB84D"))

    momentum_score = int(round(100 * sum([macd > macd_sig, 40 <= rsi <= 65, roc > 0, 20 <= stk <= 80]) / 4))
    trend_score = int(round(100 * sum([close > ema200, ema9 > ema21, close > ema50, st_dir == 1, adx > 25]) / 5))
    flow_score = int(round(100 * sum([cmf > 0.05, obv > 0, close > vwap]) / 3))
    volatility_risk_score = int(round(100 * sum([bb_pct > 80 or bb_pct < 20, adx < 20, rsi > 70 or rsi < 30]) / 3))

    def score_status(value: int, inverse: bool = False) -> str:
        if inverse:
            return "negative" if value >= 67 else "positive" if value <= 33 else "warning"
        return "positive" if value >= 67 else "negative" if value <= 33 else "warning"

    score_cards = [
        ("Momentum Score", momentum_score, momentum_status, score_status(momentum_score)),
        ("Trend Score", trend_score, trend_status, score_status(trend_score)),
        ("Flow Score", flow_score, volume_flow, score_status(flow_score)),
        ("Volatility Risk", volatility_risk_score, volatility_status, score_status(volatility_risk_score, inverse=True)),
    ]
    score_html = "".join(
        f"<div class='tech-score-card {card_status_class(status=status)}'><span>{render_info_label(label)}</span><strong>{value}/100</strong><em>{html.escape(detail)}</em></div>"
        for label, value, detail, status in score_cards
    )
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Indicator Group Summary</div>"
        "<div class='compact-section-meta'>Compact read before raw cards</div></div>"
        f"<div class='tech-score-grid'>{score_html}</div>",
        unsafe_allow_html=True,
    )

    groups = [
        ("Momentum", "RSI, oscillators, rate of change", momentum_cards),
        ("Trend", "EMA structure, trend filters, ADX", trend_cards),
        ("Volume / Flow", "Participation and money flow", flow_cards),
        ("Volatility", "Band position and range", volatility_cards),
    ]
    if full_mode:
        for title, meta, cards in groups:
            st.markdown(
                f"<div class='indicator-group'><div class='indicator-group-head'><div class='indicator-group-title'>{title}</div>"
                f"<div class='indicator-group-meta'>{meta}</div></div>"
                f"<div class='dashboard-grid indicator-grid'>{''.join(cards)}</div></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    active_overlays = [name.replace('_', ' ').upper() for name, enabled in cfg.get("show", {}).items() if enabled]
    overlay_text = "EMA 9/21/50/200 + Volume" if chart_mode == "Clean" else "All available overlays" if chart_mode == "Full" else ", ".join(active_overlays[:5]) if active_overlays else "Core OHLC, volume, RSI, MACD, MFI/CMF"
    st.markdown(
        f"<span id='technical-chart'></span><div class='tech-chart-head'><div class='tech-chart-title'>{html.escape(symbol)} Technical Chart</div>"
        f"<div class='tech-chart-meta'>{html.escape(chart_mode)} · {html.escape(overlay_text)}</div></div>",
        unsafe_allow_html=True,
    )
    render_advanced_chart(df, symbol, sr, cfg["show"], adv, chart_mode=chart_mode)
    fg_value = fg.get("value", 50)
    fg_class = fg.get("classification", "Neutral")
    fg_num = int(fg_value or 0) if str(fg_value).isdigit() else 50
    fg_c = get_fg_color(fg_num)
    if full_mode:
        st.markdown(
            "<div class='tech-fg-card'><div class='tech-fg-head'>"
            f"<div class='tech-fg-title'>Fear & Greed</div><div class='tech-fg-meta'>{html.escape(str(fg_class))} · {html.escape(str(fg_value))}/100</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        render_fear_greed_gauge(fg)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='tech-fg-compact' style='--fg-accent:{fg_c}'>"
            "<div class='tech-fg-compact-top'>"
            f"<div><div class='tech-fg-compact-title'>Fear & Greed</div><strong>{html.escape(str(fg_class))}</strong></div>"
            f"<div class='tech-fg-compact-value'>{html.escape(str(fg_value))}/100</div></div>"
            f"<div class='tech-fg-bar'><div class='tech-fg-pin' style='left:{max(0, min(100, fg_num))}%'></div></div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    sr_bias = sr_action

    ich_a = adv.get("ich_senkou_a", 0)
    ich_b = adv.get("ich_senkou_b", 0)
    ten = adv.get("ich_tenkan", 0)
    kij = adv.get("ich_kijun", 0)
    cloud_top = max(ich_a, ich_b)
    cloud_bottom = min(ich_a, ich_b)
    cloud_label = "Bullish Cloud" if ich_a > ich_b else "Bearish Cloud"
    cloud_status = "Above cloud" if close > cloud_top else "Below cloud" if close < cloud_bottom else "Inside cloud"
    tk_status = "Tenkan above Kijun" if ten > kij else "Tenkan below Kijun" if ten < kij else "Tenkan equals Kijun"
    ich_bias = "Bullish" if ich_a > ich_b and close > cloud_top and ten >= kij else "Bearish" if ich_a < ich_b and close < cloud_bottom and ten <= kij else "Mixed"
    ich_summary = "Bullish confirmation" if ich_bias == "Bullish" else "Bearish cloud" if ich_bias == "Bearish" else "Mixed cloud"
    ich_trend = "Trend confirmation" if (technical_bias == "Bullish" and ich_bias == "Bullish") or (technical_bias == "Bearish" and ich_bias == "Bearish") else "Trend conflict" if ich_bias != "Mixed" else "Trend mixed"

    level_html = "".join(
        f"<div class='tech-level'>{html.escape(name)}: {html.escape(fmt_price(sr.get(key, 0), symbol))}</div>"
        for name, key in [("R2", "pivot_r2"), ("R1", "pivot_r1"), ("PP", "pivot"), ("S1", "pivot_s1"), ("S2", "pivot_s2")]
    )
    sr_metrics = "".join([
        mini_metric("Current Price", fmt_price(close, symbol)),
        mini_metric("Nearest Support", fmt_price(ns, symbol)),
        mini_metric("Nearest Resistance", fmt_price(nr, symbol)),
        mini_metric("Nearest Distance", f"{nearest_distance:.2f}% to {nearest_label}"),
        mini_metric("Action", sr_bias),
        mini_metric("Range Width", f"{(resistance_pct + support_pct):.2f}%"),
    ])
    ich_metrics = "".join([
        mini_metric("Summary", ich_summary),
        mini_metric("Trend Read", ich_trend),
        mini_metric("Cloud Status", cloud_label),
        mini_metric("Price Position", cloud_status),
        mini_metric("Tenkan / Kijun", tk_status),
        mini_metric("Ichimoku Bias", ich_bias),
    ])
    st.markdown("<div id='technical-support-resistance'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tech-detail-grid'>"
        "<div class='tech-detail-card'><div class='tech-detail-title'>Support / Resistance</div>"
        f"<div class='tech-metric-grid'>{sr_metrics}</div><div class='tech-level-list'>{level_html}</div></div>"
        "<div class='tech-detail-card'><div class='tech-detail-title'>Ichimoku Cloud</div>"
        f"<div class='tech-metric-grid'>{ich_metrics}</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_advanced_chart(df: pd.DataFrame, symbol: str, sr: dict, show: dict, adv: dict, chart_mode: str = "Standard"):
    chart_mode = chart_mode if chart_mode in {"Clean", "Standard", "Full"} else "Standard"
    clean_mode = chart_mode == "Clean"
    if chart_mode == "Full":
        effective_show = dict(show or {})
        for key in ("ema_9", "ema_21", "ema_50", "ema_200", "sma_20", "sma_50", "sma_200", "vwap", "bb", "keltner", "donchian", "supertrend", "ichimoku", "psar", "sr_lines"):
            effective_show[key] = True
    elif clean_mode:
        effective_show = {"ema_9": True, "ema_21": True, "ema_50": True, "ema_200": True}
    else:
        effective_show = dict(show or {})

    rows = 2 if clean_mode else 5
    volume_row = 2
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.78, 0.22] if clean_mode else [0.44, 0.12, 0.15, 0.15, 0.14],
        subplot_titles=(f"{symbol} · Price", "Volume") if clean_mode else (f"{symbol} · Price", "Volume", "RSI / Stoch RSI", "MACD", "MFI / CMF"),
    )
    show = effective_show

    # ── Candles ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC",
        increasing_line_color="#00E08A", decreasing_line_color="#FF5C73",
        increasing_fillcolor="#00E08A", decreasing_fillcolor="#FF5C73",
    ), row=1, col=1)

    # ── EMA / SMA lines ───────────────────────────────────────────────────
    ema_cfg = [
        ("ema_9",   "#00e5ff", "EMA 9",   "solid", show.get("ema_9")),
        ("ema_21",  "#ff6f00", "EMA 21",  "solid", show.get("ema_21")),
        ("ema_50",  "#FFB84D", "EMA 50",  "dot",   show.get("ema_50")),
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
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", showlegend=False,
            line=dict(color="rgba(52,152,219,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(52,152,219,0.05)"), row=1, col=1)

    # ── Keltner Channel ───────────────────────────────────────────────────
    if show.get("keltner") and "kc_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["kc_upper"], name="KC Upper",
            line=dict(color="rgba(155,89,182,0.5)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["kc_lower"], name="KC Lower", showlegend=False,
            line=dict(color="rgba(155,89,182,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(155,89,182,0.04)"), row=1, col=1)

    # ── Donchian Channel ──────────────────────────────────────────────────
    if show.get("donchian") and "dc_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["dc_upper"], name="DC High",
            line=dict(color="rgba(230,126,34,0.5)", width=1, dash="dashdot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["dc_lower"], name="DC Low", showlegend=False,
            line=dict(color="rgba(230,126,34,0.5)", width=1, dash="dashdot"),
            fill="tonexty", fillcolor="rgba(230,126,34,0.04)"), row=1, col=1)

    # ── Supertrend ────────────────────────────────────────────────────────
    if show.get("supertrend") and "supertrend" in df.columns:
        bull_st = df["supertrend"].where(df["supertrend_direction"] == 1)
        bear_st = df["supertrend"].where(df["supertrend_direction"] == -1)
        fig.add_trace(go.Scatter(x=df.index, y=bull_st, name="ST Bull",
            line=dict(color="#00E08A", width=2), mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bear_st, name="ST Bear",
            line=dict(color="#FF5C73", width=2), mode="lines"), row=1, col=1)

    # ── Ichimoku ──────────────────────────────────────────────────────────
    if show.get("ichimoku") and "ich_senkou_a" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ich_tenkan"], name="Tenkan",
            line=dict(color="#e91e63", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ich_kijun"], name="Kijun",
            line=dict(color="#3f51b5", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ich_senkou_a"], name="Senkou A",
            line=dict(color="rgba(38,166,154,0.6)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ich_senkou_b"], name="Senkou B", showlegend=False,
            line=dict(color="rgba(239,83,80,0.6)", width=1),
            fill="tonexty", fillcolor="rgba(100,100,100,0.07)"), row=1, col=1)

    # ── Parabolic SAR ─────────────────────────────────────────────────────
    if show.get("psar") and "psar" in df.columns:
        bull_psar = df["psar"].where(df.get("psar_bull", pd.Series(1, index=df.index)) == 1)
        bear_psar = df["psar"].where(df.get("psar_bull", pd.Series(1, index=df.index)) != 1)
        fig.add_trace(go.Scatter(x=df.index, y=bull_psar, name="SAR Bull",
            mode="markers", marker=dict(size=3, color="#00E08A", symbol="circle")),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bear_psar, name="SAR Bear",
            mode="markers", marker=dict(size=3, color="#FF5C73", symbol="circle")),
            row=1, col=1)

    # ── EMA crossover markers ─────────────────────────────────────────────
    if "ema_bullish_cross" in df.columns:
        bull_x = df.index[df["ema_bullish_cross"] == True]
        if len(bull_x):
            fig.add_trace(go.Scatter(x=bull_x, y=df.loc[bull_x, "ema_9"],
                mode="markers", name="Bull X",
                marker=dict(symbol="triangle-up", size=12, color="#00E08A",
                            line=dict(color="white", width=1))), row=1, col=1)
    if "ema_bearish_cross" in df.columns:
        bear_x = df.index[df["ema_bearish_cross"] == True]
        if len(bear_x):
            fig.add_trace(go.Scatter(x=bear_x, y=df.loc[bear_x, "ema_9"],
                mode="markers", name="Bear X",
                marker=dict(symbol="triangle-down", size=12, color="#FF5C73",
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
    bar_colors = ["#00E08A" if c >= o else "#FF5C73"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
        marker_color=bar_colors, opacity=0.7), row=2, col=1)
    if "vwap" in df.columns and show.get("vwap"):
        pass  # Volume VWAP already on price

    if not clean_mode:
        # ── RSI + Stoch RSI ───────────────────────────────────────────────
        if "rsi" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                line=dict(color="#FF5C73", width=1.4)), row=3, col=1)
            for lvl, clr in [(70,"rgba(239,83,80,0.4)"),(30,"rgba(38,166,154,0.4)"),
                             (50,"rgba(128,128,128,0.2)")]:
                fig.add_hline(y=lvl, line_dash="dash", line_color=clr, line_width=1, row=3, col=1)
        if "stochrsi_k" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["stochrsi_k"], name="Stoch K",
                line=dict(color="#3498db", width=1, dash="dot")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["stochrsi_d"], name="Stoch D",
                line=dict(color="#FFB84D", width=1, dash="dot")), row=3, col=1)

        # ── MACD ──────────────────────────────────────────────────────────
        if "macd" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
                line=dict(color="#3498db", width=1.4)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                line=dict(color="#FF5C73", width=1.4)), row=4, col=1)
            hist_c = ["#00E08A" if v >= 0 else "#FF5C73"
                      for v in df.get("macd_hist", pd.Series())]
            fig.add_trace(go.Bar(x=df.index, y=df.get("macd_hist", pd.Series()),
                name="Hist", marker_color=hist_c, opacity=0.65), row=4, col=1)

        # ── MFI / CMF ────────────────────────────────────────────────────
        if "mfi" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["mfi"], name="MFI",
                line=dict(color="#9b59b6", width=1.4)), row=5, col=1)
            for lvl, clr in [(80,"rgba(239,83,80,0.4)"),(20,"rgba(38,166,154,0.4)")]:
                fig.add_hline(y=lvl, line_dash="dash", line_color=clr, line_width=1, row=5, col=1)
        if "cmf" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["cmf"] * 100, name="CMF x100",
                line=dict(color="#1abc9c", width=1, dash="dot")), row=5, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="rgba(128,128,128,0.25)",
                          line_width=1, row=5, col=1)

    chart_xmin = df.index[-min(200, len(df))]
    fig.update_layout(
        height=460 if clean_mode else 690,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0,
                    font=dict(size=9), itemwidth=30, tracegroupgap=4),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=32 if clean_mode else 36, b=8),
    )
    fig.update_xaxes(range=[chart_xmin, df.index[-1]], row=rows, col=1)
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", row=i, col=1)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", row=i, col=1)
    render_price_chart(fig)

# ── Tab 3: Smart Money ────────────────────────────────────────────────────────

def render_smart_money(df: pd.DataFrame, smc: dict, symbol: str):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        render_empty_state("Smart Money data unavailable. Check symbol/timeframe and try again.")
        return
    if not smc or not isinstance(smc, dict):
        render_empty_state("Smart Money data unavailable. SMC analysis could not be loaded.")
        return

    close = float(df["close"].iloc[-1])
    recent = df.tail(200)
    pd_zone = smc.get("premium_discount", {})
    zone = pd_zone.get("current_zone", "N/A")
    zone_c = "#00E08A" if zone == "Discount" else ("#FF5C73" if zone == "Premium" else "#FFB84D")
    bull_fvg = smc.get("bull_fvg", [])
    bear_fvg = smc.get("bear_fvg", [])
    bull_ob = smc.get("bull_ob", [])
    bear_ob = smc.get("bear_ob", [])
    bos_bull = smc.get("bos_bull", [])
    bos_bear = smc.get("bos_bear", [])
    choch_b = smc.get("choch_bull", [])
    choch_br = smc.get("choch_bear", [])
    eq_highs = smc.get("equal_highs_above", [])
    eq_lows = smc.get("equal_lows_below", [])

    def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def signed_ratio(pos: int, neg: int, weight: float) -> float:
        total = pos + neg
        if total <= 0:
            return 0
        return ((pos - neg) / total) * weight

    def smc_state_status(value: str) -> str:
        v = str(value).lower()
        if any(term in v for term in ("accumulation", "bull", "buy-side", "long", "discount", "low", "active")):
            return "positive"
        if any(term in v for term in ("distribution", "bear", "sell-side", "short", "premium", "high")):
            return "negative"
        return "warning"

    def summary_card(label: str, value: str, detail: str = "", status: str = "warning") -> str:
        return (
            f"<div class='smc-summary-card {card_status_class(status=status)}'>"
            f"<span>{render_info_label(str(label))}</span><strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(detail))}</em></div>"
        )

    def setup_item(label: str, value: str, detail: str = "", status: str = "warning") -> str:
        return (
            f"<div class='smc-setup-item {card_status_class(status=status)}'>"
            f"<span>{render_info_label(str(label))}</span><strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(detail))}</em></div>"
        )

    fvg_bias_score = signed_ratio(len(bull_fvg), len(bear_fvg), 20)
    structure_bull = len(bos_bull) + len(choch_b)
    structure_bear = len(bos_bear) + len(choch_br)
    structure_score = signed_ratio(structure_bull, structure_bear, 25)
    liquidity_score = signed_ratio(len(eq_highs), len(eq_lows), 20)
    zone_score = 15 if zone == "Discount" else -15 if zone == "Premium" else 0
    price_change = 0.0
    volume_confirm = 0.0
    if len(df) >= 12:
        price_change = ((close - float(df["close"].iloc[-12])) / float(df["close"].iloc[-12])) * 100 if float(df["close"].iloc[-12]) else 0
        recent_vol = float(df["volume"].tail(5).mean()) if "volume" in df.columns else 0
        base_vol = float(df["volume"].tail(30).mean()) if "volume" in df.columns else 0
        volume_confirm = 20 if price_change > 0 and recent_vol >= base_vol else -20 if price_change < 0 and recent_vol >= base_vol else 10 if price_change > 0 else -10 if price_change < 0 else 0
    smc_score = int(round(clamp(50 + fvg_bias_score + structure_score + liquidity_score + zone_score + volume_confirm, 0, 100)))
    institutional_bias = "Accumulation" if smc_score >= 65 else "Distribution" if smc_score <= 35 else "Neutral"
    bias_status = "positive" if institutional_bias == "Accumulation" else "negative" if institutional_bias == "Distribution" else "warning"

    liquidity_pressure = "Buy-side" if len(eq_highs) > len(eq_lows) else "Sell-side" if len(eq_lows) > len(eq_highs) else "Balanced"
    fvg_bias = "Bullish" if len(bull_fvg) > len(bear_fvg) else "Bearish" if len(bear_fvg) > len(bull_fvg) else "Mixed"
    structure_bias = "Bullish" if structure_bull > structure_bear else "Bearish" if structure_bear > structure_bull else "Mixed"

    def detect_last_sweep() -> dict:
        if len(df) < 12:
            return {"type": "None", "level": None, "age": None}
        lookback = min(45, len(df) - 1)
        start_i = max(5, len(df) - lookback)
        last = {"type": "None", "level": None, "age": None}
        for i in range(start_i, len(df)):
            left = df.iloc[max(0, i - 20):i]
            if left.empty:
                continue
            prev_high = float(left["high"].max())
            prev_low = float(left["low"].min())
            high = float(df["high"].iloc[i])
            low = float(df["low"].iloc[i])
            candle_close = float(df["close"].iloc[i])
            sweep_high_level = prev_high
            sweep_low_level = prev_low
            nearby_highs = [float(z.get("level", 0)) for z in eq_highs if z.get("level")]
            nearby_lows = [float(z.get("level", 0)) for z in eq_lows if z.get("level")]
            if nearby_highs:
                sweep_high_level = min(nearby_highs, key=lambda x: abs(x - high))
            if nearby_lows:
                sweep_low_level = min(nearby_lows, key=lambda x: abs(x - low))
            if high > sweep_high_level and candle_close < sweep_high_level:
                last = {"type": "Sweep High", "level": sweep_high_level, "age": len(df) - 1 - i}
            if low < sweep_low_level and candle_close > sweep_low_level:
                last = {"type": "Sweep Low", "level": sweep_low_level, "age": len(df) - 1 - i}
        return last

    sweep = detect_last_sweep()
    nearest_high = min(eq_highs, key=lambda z: abs(float(z.get("level", close)) - close), default=None)
    nearest_low = min(eq_lows, key=lambda z: abs(float(z.get("level", close)) - close), default=None)
    liq_candidates = []
    if nearest_high:
        liq_candidates.append(("Buy-side", float(nearest_high.get("level", close)), nearest_high.get("touches", 0)))
    if nearest_low:
        liq_candidates.append(("Sell-side", float(nearest_low.get("level", close)), nearest_low.get("touches", 0)))
    nearest_liq = min(liq_candidates, key=lambda x: abs(x[1] - close), default=None)
    nearest_liq_label = nearest_liq[0] if nearest_liq else "None"
    nearest_liq_level = nearest_liq[1] if nearest_liq else None
    nearest_liq_dist = abs(nearest_liq_level - close) / close * 100 if nearest_liq_level and close else 0
    sweep_age = sweep.get("age")
    sweep_risk = "High" if sweep_age is not None and sweep_age <= 3 else "Moderate" if (sweep_age is not None and sweep_age <= 12) or nearest_liq_dist <= 0.75 else "Low"

    reasons = []
    reasons.append("Bullish FVGs dominate" if len(bull_fvg) > len(bear_fvg) else "Bearish FVGs dominate" if len(bear_fvg) > len(bull_fvg) else "FVG balance mixed")
    reasons.append("BOS bullish lead" if len(bos_bull) > len(bos_bear) else "BOS bearish lead" if len(bos_bear) > len(bos_bull) else "BOS balanced")
    if len(choch_br) > len(choch_b):
        reasons.append("CHoCH bearish warning")
    elif len(choch_b) > len(choch_br):
        reasons.append("CHoCH bullish warning")
    reasons.append("Buy-side liquidity nearby" if nearest_liq_label == "Buy-side" else "Sell-side liquidity nearby" if nearest_liq_label == "Sell-side" else "Liquidity balanced")
    reasons.append("Price in discount" if zone == "Discount" else "Price in premium" if zone == "Premium" else "Price near equilibrium")
    if sweep_risk != "Low":
        reasons.append("Sweep risk elevated")
    reason_html = "".join(f"<span class='smc-reason-chip'>{html.escape(r)}</span>" for r in reasons)

    setup_type = "Accumulation" if institutional_bias == "Accumulation" and zone != "Premium" else "Distribution" if institutional_bias == "Distribution" and zone != "Discount" else "Reversal Watch" if sweep.get("type") != "None" else "Breakout Watch" if nearest_liq_dist <= 1.0 and nearest_liq else "Neutral"
    preferred_direction = "Long" if setup_type in {"Accumulation", "Reversal Watch"} and structure_bias != "Bearish" else "Short" if setup_type == "Distribution" or (setup_type == "Breakout Watch" and nearest_liq_label == "Sell-side") else "Wait"
    confirmation_needed = "Bullish BOS or CHoCH hold" if preferred_direction == "Long" else "Bearish BOS or CHoCH hold" if preferred_direction == "Short" else "Directional BOS/CHoCH"
    invalidation_level = nearest_low["level"] if preferred_direction == "Long" and nearest_low else nearest_high["level"] if preferred_direction == "Short" and nearest_high else pd_zone.get("equilibrium")
    risk_level = "High" if sweep_risk == "High" or (zone == "Premium" and preferred_direction == "Long") or (zone == "Discount" and preferred_direction == "Short") else "Low" if sweep_risk == "Low" and institutional_bias != "Neutral" else "Moderate"

    st.markdown(
        "<div class='smc-command-head'>"
        "<div class='smc-command-title'>Smart Money Command Center</div>"
        f"<div class='smc-command-meta'>{html.escape(symbol)} / {len(df)} candles / {html.escape(str(zone))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='smc-command-card'><div class='smc-command-grid'>"
        f"<div class='smc-core {card_status_class(status=bias_status)}'>"
        f"<div class='smc-label'>{render_info_label('Institutional Bias')}</div>"
        f"<div class='smc-value'>{html.escape(institutional_bias)}</div>"
        f"<div class='smc-sub'>SMC Score {smc_score}/100</div>"
        "<div class='smc-gauge'><div class='smc-gauge-track'>"
        f"<div class='smc-gauge-marker' style='left:{smc_score}%'></div></div>"
        "<div class='smc-gauge-scale'><span>Distribution</span><span>Sell-side</span><span>Neutral</span><span>Buy-side</span><span>Accumulation</span></div></div>"
        "</div>"
        "<div class='smc-health-grid'>"
        f"<div class='smc-health-chip {card_status_class(status=smc_state_status(liquidity_pressure))}'><span>{render_info_label('Liquidity Pressure')}</span><strong>{html.escape(liquidity_pressure)}</strong><em>EQH {len(eq_highs)} / EQL {len(eq_lows)}</em></div>"
        f"<div class='smc-health-chip {card_status_class(status=smc_state_status(fvg_bias))}'><span>{render_info_label('FVG Bias')}</span><strong>{html.escape(fvg_bias)}</strong><em>{len(bull_fvg)} bull / {len(bear_fvg)} bear</em></div>"
        f"<div class='smc-health-chip {card_status_class(status=smc_state_status(structure_bias))}'><span>{render_info_label('Structure Bias')}</span><strong>{html.escape(structure_bias)}</strong><em>BOS/CHoCH {structure_bull}:{structure_bear}</em></div>"
        f"<div class='smc-health-chip {card_status_class(status='negative' if sweep_risk == 'High' else 'warning' if sweep_risk == 'Moderate' else 'positive')}'><span>{render_info_label('Sweep Risk')}</span><strong>{html.escape(sweep_risk)}</strong><em>{html.escape(sweep.get('type', 'None'))}</em></div>"
        f"<div class='smc-health-chip {card_status_class(status=zone_status if (zone_status := ('positive' if zone == 'Discount' else 'negative' if zone == 'Premium' else 'warning')) else 'warning')}'><span>{render_info_label('Price Zone')}</span><strong>{html.escape(str(zone))}</strong><em>Premium / discount</em></div>"
        "</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Why this SMC bias?</div>"
        f"<div class='compact-section-meta'>FVG {fvg_bias_score:+.0f} / Structure {structure_score:+.0f} / Liquidity {liquidity_score:+.0f}</div></div>"
        f"<div class='smc-reasons'>{reason_html}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Smart Money Setup</div>"
        "<div class='compact-section-meta'>Display-only context from current SMC map</div></div>"
        "<div class='smc-setup-grid'>"
        + setup_item("Setup Type", setup_type, f"Zone {zone}", smc_state_status(setup_type))
        + setup_item("Preferred Direction", preferred_direction, confirmation_needed, smc_state_status(preferred_direction))
        + setup_item("Confirmation Needed", confirmation_needed, "BOS / CHoCH", "warning")
        + setup_item("Invalidation", fmt_price(float(invalidation_level), symbol) if invalidation_level else "N/A", "Nearest opposing liquidity", "warning")
        + setup_item("Risk Level", risk_level, f"Sweep {sweep_risk}", "negative" if risk_level == "High" else "positive" if risk_level == "Low" else "warning")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Liquidity Sweep Monitor</div>"
        f"<div class='compact-section-meta'>Recent wick reclaim scan, OHLC only</div></div>"
        "<div class='smc-summary-grid'>"
        + summary_card("Last Sweep", sweep.get("type", "None"), fmt_price(float(sweep["level"]), symbol) if sweep.get("level") else "No recent sweep", smc_state_status(sweep.get("type", "None")))
        + summary_card("Sweep Age", f"{sweep_age} candles" if sweep_age is not None else "N/A", "Most recent sweep", "warning")
        + summary_card("Nearest Liquidity", nearest_liq_label, fmt_price(nearest_liq_level, symbol) if nearest_liq_level else "No active EQ level", smc_state_status(nearest_liq_label))
        + summary_card("Distance", f"{nearest_liq_dist:.2f}%" if nearest_liq else "N/A", "To nearest liquidity", "negative" if nearest_liq_dist <= 0.5 and nearest_liq else "warning")
        + "</div>",
        unsafe_allow_html=True,
    )

    if "smc_chart_view" not in st.session_state:
        st.session_state.smc_chart_view = "Standard"

    chart_control_col, chart_context_col = st.columns([0.34, 1.2])
    with chart_control_col:
        chart_view = st.radio(
            "Chart View",
            ["Clean", "Standard", "Full"],
            key="smc_chart_view",
            horizontal=True,
        )

    all_fvgs_for_chart = [("Bullish", f) for f in bull_fvg] + [("Bearish", f) for f in bear_fvg]
    all_fvgs_for_chart = sorted(all_fvgs_for_chart, key=lambda item: abs(float(item[1].get("mid", close)) - close))
    nearest_fvg_for_chart = all_fvgs_for_chart[0] if all_fvgs_for_chart else None
    nearest_liq_text = f"{nearest_liq_label[:3].upper()} {fmt_price(nearest_liq_level, symbol)}" if nearest_liq_level else "None"
    with chart_context_col:
        st.markdown(
            "<div class='smc-chart-control-row'><div class='smc-chart-context'>"
            f"<span class='smc-chart-chip'><strong>View</strong>{html.escape(chart_view)}</span>"
            f"<span class='smc-chart-chip'><strong>Nearest Liquidity</strong>{html.escape(nearest_liq_text)}</span>"
            f"<span class='smc-chart-chip'><strong>Last Sweep</strong>{html.escape(str(sweep.get('type', 'None')))}</span>"
            f"<span class='smc-chart-chip'><strong>Active FVGs</strong>{len(bull_fvg) + len(bear_fvg)}</span>"
            f"<span class='smc-chart-chip'><strong>Zone</strong>{html.escape(str(zone))}</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

    def selected_fvgs_for_view() -> list:
        if chart_view == "Clean":
            return all_fvgs_for_chart[:2]
        if chart_view == "Standard":
            return all_fvgs_for_chart[:5]
        return [("Bullish", f) for f in bull_fvg] + [("Bearish", f) for f in bear_fvg]

    def recent_structure_for_view() -> list:
        items = []
        for label, rows in (("BOS+", bos_bull), ("BOS-", bos_bear), ("CHoCH+", choch_b), ("CHoCH-", choch_br)):
            for row in rows:
                t = row.get("break_time") or row.get("time") or df.index[-1]
                items.append((t, label, row))
        items = sorted(items, key=lambda x: x[0])
        if chart_view == "Clean":
            return []
        if chart_view == "Standard":
            return items[-5:]
        return items

    def selected_liquidity_for_view(levels: list, side: str) -> list:
        sorted_levels = sorted(
            [z for z in levels if z.get("level")],
            key=lambda z: abs(float(z.get("level", close)) - close),
        )
        if chart_view == "Clean":
            return sorted_levels[:1]
        if chart_view == "Standard":
            return sorted_levels[:3]
        return sorted_levels

    def label_liquidity_levels(levels: list, base_label: str) -> list:
        labeled = []
        previous_level = None
        cluster_open = False
        for idx, z in enumerate(levels):
            lv = float(z.get("level", 0))
            if not lv:
                continue
            close_to_previous = previous_level is not None and abs(lv - previous_level) / close * 100 <= 0.18
            if close_to_previous:
                label = f"{base_label} Cluster" if not cluster_open else ""
                cluster_open = True
            else:
                label = base_label if idx < 2 or chart_view != "Full" else ""
                cluster_open = False
            labeled.append((z, label))
            previous_level = lv
        return labeled

    chart_fvgs = selected_fvgs_for_view()
    chart_structure = recent_structure_for_view()
    chart_bsl = selected_liquidity_for_view(eq_highs, "BSL")
    chart_ssl = selected_liquidity_for_view(eq_lows, "SSL")

    # SMC Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=recent.index, open=recent["open"], high=recent["high"],
        low=recent["low"], close=recent["close"],
        name="OHLC",
        increasing_line_color="#00E08A", decreasing_line_color="#FF5C73",
        increasing_fillcolor="#00E08A", decreasing_fillcolor="#FF5C73",
    ))

    for idx, (fvg_type, fvg) in enumerate(chart_fvgs):
        is_nearest = nearest_fvg_for_chart is not None and fvg is nearest_fvg_for_chart[1]
        is_bull = fvg_type == "Bullish"
        fill = "rgba(38,166,154,0.18)" if is_nearest and is_bull else "rgba(239,83,80,0.18)" if is_nearest else "rgba(38,166,154,0.08)" if is_bull else "rgba(239,83,80,0.08)"
        line = "rgba(38,166,154,0.78)" if is_nearest and is_bull else "rgba(239,83,80,0.78)" if is_nearest else "rgba(38,166,154,0.35)" if is_bull else "rgba(239,83,80,0.35)"
        text = "FVG+" if is_bull else "FVG-"
        fig.add_shape(type="rect", x0=fvg["time"], x1=recent.index[-1], y0=fvg["bottom"], y1=fvg["top"], fillcolor=fill, line_color=line, line_width=2 if is_nearest else 1)
        if chart_view != "Clean" or is_nearest:
            fig.add_annotation(x=fvg["time"], y=fvg["mid"], text=text, font=dict(color="#00E08A" if is_bull else "#FF5C73", size=9), showarrow=False, xanchor="left")

    if chart_view == "Full":
        for ob in bull_ob:
            fig.add_shape(type="rect", x0=ob["time"], x1=recent.index[-1], y0=ob["bottom"], y1=ob["top"], fillcolor="rgba(38,166,154,0.14)", line_color="rgba(38,166,154,0.58)", line_width=1, line_dash="dot")
            fig.add_annotation(x=ob["time"], y=(ob["top"] + ob["bottom"]) / 2, text="OB+", font=dict(color="#00E08A", size=9), showarrow=False, xanchor="left")
        for ob in bear_ob:
            fig.add_shape(type="rect", x0=ob["time"], x1=recent.index[-1], y0=ob["bottom"], y1=ob["top"], fillcolor="rgba(239,83,80,0.14)", line_color="rgba(239,83,80,0.58)", line_width=1, line_dash="dot")
            fig.add_annotation(x=ob["time"], y=(ob["top"] + ob["bottom"]) / 2, text="OB-", font=dict(color="#FF5C73", size=9), showarrow=False, xanchor="left")

    for _time, label, row in chart_structure:
        bullish = "+" in label
        fig.add_hline(
            y=row["level"],
            line_dash="dash" if label.startswith("BOS") else "dot",
            line_color="rgba(38,166,154,0.54)" if bullish else "rgba(239,83,80,0.54)",
            line_width=1.2 if label.startswith("BOS") else 1.6,
            annotation_text=label,
            annotation_position="right",
        )

    for z, label in label_liquidity_levels(chart_bsl, "BSL"):
        lv = float(z.get("level", 0))
        is_nearest = nearest_liq_label == "Buy-side" and nearest_liq_level and abs(lv - nearest_liq_level) < 1e-9
        fig.add_hline(y=lv, line_dash="dot", line_color="rgba(255,92,115,0.82)" if is_nearest else "rgba(255,92,115,0.42)", line_width=2 if is_nearest else 1, annotation_text=label, annotation_position="right")
    for z, label in label_liquidity_levels(chart_ssl, "SSL"):
        lv = float(z.get("level", 0))
        is_nearest = nearest_liq_label == "Sell-side" and nearest_liq_level and abs(lv - nearest_liq_level) < 1e-9
        fig.add_hline(y=lv, line_dash="dot", line_color="rgba(0,224,138,0.82)" if is_nearest else "rgba(0,224,138,0.42)", line_width=2 if is_nearest else 1, annotation_text=label, annotation_position="right")

    if sweep.get("level"):
        sweep_color = "rgba(255,92,115,0.92)" if sweep.get("type") == "Sweep High" else "rgba(0,224,138,0.92)"
        sweep_idx = len(df) - 1 - int(sweep.get("age") or 0)
        sweep_x = df.index[max(0, min(len(df) - 1, sweep_idx))]
        fig.add_hline(y=float(sweep["level"]), line_dash="dashdot", line_color=sweep_color, line_width=2, annotation_text=sweep.get("type", "Sweep"), annotation_position="right")
        fig.add_trace(go.Scatter(x=[sweep_x], y=[float(sweep["level"])], mode="markers", name="Last Sweep", marker=dict(size=11, color=sweep_color, symbol="diamond", line=dict(color="white", width=1)), showlegend=False))

    if pd_zone:
        fig.add_hrect(y0=pd_zone.get("equilibrium", 0), y1=pd_zone.get("range_high", 0), fillcolor="rgba(239,83,80,0.035)", line_width=0, annotation_text="Premium", annotation_position="top right")
        fig.add_hrect(y0=pd_zone.get("range_low", 0), y1=pd_zone.get("equilibrium", 0), fillcolor="rgba(38,166,154,0.035)", line_width=0, annotation_text="Discount", annotation_position="bottom right")
        fig.add_hline(y=pd_zone.get("equilibrium", 0), line_dash="dot", line_color="rgba(241,196,15,0.56)", line_width=1, annotation_text="EQ", annotation_position="right")

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=24, b=10), showlegend=False)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
    st.plotly_chart(fig, width="stretch")

    all_fvgs = [("Bullish", f) for f in bull_fvg] + [("Bearish", f) for f in bear_fvg]
    nearest_fvg = min(all_fvgs, key=lambda item: abs(float(item[1].get("mid", close)) - close), default=None)
    nearest_fvg_text = f"{nearest_fvg[0]} {fmt_price(float(nearest_fvg[1].get('mid', close)), symbol)}" if nearest_fvg else "None"
    fvg_status = "Imbalance active" if len(bull_fvg) + len(bear_fvg) > 0 and fvg_bias != "Mixed" else "Mixed imbalance" if bull_fvg and bear_fvg else "Neutral"
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Fair Value Gaps</div>"
        "<div class='compact-section-meta'>Active imbalance map</div></div>"
        "<div class='smc-summary-grid'>"
        + summary_card("Bullish FVG", str(len(bull_fvg)), "Demand imbalance", "positive" if bull_fvg else "muted")
        + summary_card("Bearish FVG", str(len(bear_fvg)), "Supply imbalance", "negative" if bear_fvg else "muted")
        + summary_card("Net FVG Bias", fvg_bias, nearest_fvg_text, smc_state_status(fvg_bias))
        + summary_card("Status", fvg_status, "Gap magnet" if nearest_fvg else "No active gap", "positive" if fvg_status == "Imbalance active" else "warning")
        + "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fvg_rows = []
        for f in bull_fvg[-5:]:
            fvg_rows.append({"Type": "Bullish", "Top": fmt_price(f["top"], symbol), "Bottom": fmt_price(f["bottom"], symbol), "Gap%": f"{f['gap_pct']:.3f}%"})
        for f in bear_fvg[-5:]:
            fvg_rows.append({"Type": "Bearish", "Top": fmt_price(f["top"], symbol), "Bottom": fmt_price(f["bottom"], symbol), "Gap%": f"{f['gap_pct']:.3f}%"})
        if fvg_rows:
            render_interactive_dataframe(pd.DataFrame(fvg_rows), signed_columns={"Type"})
        else:
            st.info("No active FVGs detected.")

    with col2:
        nearest_ob_candidates = [("Bullish", o) for o in bull_ob] + [("Bearish", o) for o in bear_ob]
        nearest_ob = min(nearest_ob_candidates, key=lambda item: abs(((float(item[1].get("top", close)) + float(item[1].get("bottom", close))) / 2) - close), default=None)
        nearest_ob_text = f"{nearest_ob[0]} {fmt_price((float(nearest_ob[1].get('top', close)) + float(nearest_ob[1].get('bottom', close))) / 2, symbol)}" if nearest_ob else "None"
        ob_bias = "Bullish" if len(bull_ob) > len(bear_ob) else "Bearish" if len(bear_ob) > len(bull_ob) else "Mixed"
        st.markdown(
            "<div class='compact-section-head'><div class='compact-section-title'>Order Blocks</div>"
            "<div class='compact-section-meta'>Validated zones only</div></div>"
            "<div class='smc-summary-grid'>"
            + summary_card("Bullish OB", str(len(bull_ob)), "Demand blocks", "positive" if bull_ob else "muted")
            + summary_card("Bearish OB", str(len(bear_ob)), "Supply blocks", "negative" if bear_ob else "muted")
            + summary_card("Nearest OB", nearest_ob_text, "Tested" if nearest_ob and float(nearest_ob[1].get("bottom", 0)) <= close <= float(nearest_ob[1].get("top", 0)) else "Untested / none", smc_state_status(ob_bias))
            + summary_card("OB Bias", ob_bias, "Active block balance", smc_state_status(ob_bias))
            + "</div>",
            unsafe_allow_html=True,
        )
        ob_rows = []
        for o in bull_ob[-4:]:
            ob_rows.append({"Type": "Bullish OB", "Top": fmt_price(o["top"], symbol), "Bottom": fmt_price(o["bottom"], symbol)})
        for o in bear_ob[-4:]:
            ob_rows.append({"Type": "Bearish OB", "Top": fmt_price(o["top"], symbol), "Bottom": fmt_price(o["bottom"], symbol)})
        if ob_rows:
            render_interactive_dataframe(pd.DataFrame(ob_rows), signed_columns={"Type"})
        else:
            st.markdown(
                "<div class='smc-empty-note'>No active Order Blocks detected"
                "<span>No validated unmitigated order block found in current lookback. Increase candle limit or switch timeframe.</span></div>",
                unsafe_allow_html=True,
            )

    liq = eq_highs + eq_lows
    if liq:
        liq_direction = "Buy-side" if len(eq_highs) > len(eq_lows) else "Sell-side" if len(eq_lows) > len(eq_highs) else "Balanced"
        st.markdown(
            "<div class='compact-section-head'><div class='compact-section-title'>Liquidity Zones</div>"
            "<div class='compact-section-meta'>Equal highs / equal lows</div></div>"
            "<div class='smc-summary-grid'>"
            + summary_card("Equal Highs", str(len(eq_highs)), "Buy-side liquidity", "negative" if eq_highs else "muted")
            + summary_card("Equal Lows", str(len(eq_lows)), "Sell-side liquidity", "positive" if eq_lows else "muted")
            + summary_card("Nearest Target", nearest_liq_label, fmt_price(nearest_liq_level, symbol) if nearest_liq_level else "None", smc_state_status(nearest_liq_label))
            + summary_card("Distance", f"{nearest_liq_dist:.2f}%" if nearest_liq else "N/A", liq_direction, "warning")
            + "</div>",
            unsafe_allow_html=True,
        )
        liq_rows = [{"Type": "EQ High" if float(l.get("level", 0)) > close else "EQ Low", "Level": fmt_price(l["level"], symbol), "Touches": l.get("touches", 0)} for l in liq[:8]]
        render_interactive_dataframe(pd.DataFrame(liq_rows), column_config={"Touches": st.column_config.NumberColumn("Touches", format="%d")})

    supply = smc.get("supply_zones", [])
    demand = smc.get("demand_zones", [])
    if supply or demand:
        st.markdown("#### Supply & Demand Zones")
        sd_rows = []
        for z in demand[:4]:
            sd_rows.append({"Type": "Demand", "Top": fmt_price(z["top"], symbol), "Bottom": fmt_price(z["bottom"], symbol)})
        for z in supply[:4]:
            sd_rows.append({"Type": "Supply", "Top": fmt_price(z["top"], symbol), "Bottom": fmt_price(z["bottom"], symbol)})
        if sd_rows:
            render_interactive_dataframe(pd.DataFrame(sd_rows), signed_columns={"Type"})


# ── Tab 4: Order Book ─────────────────────────────────────────────────────────

def render_orderbook(ob: dict, symbol: str):
    # PERF: Validate data before rendering
    if not ob or not isinstance(ob, dict):
        render_empty_state("Order book data unavailable.")
        return

    if "bids" not in ob or "asks" not in ob:
        render_empty_state("Order book data unavailable.")
        return

    source_notice = orderbook_source_message(ob)
    if source_notice:
        message, kind = source_notice
        render_notice_badge(message, kind=kind)

    src_label = orderbook_source_label(ob)
    st.markdown(render_section_header(f"Order Book — {symbol}", f"Source: {src_label}"), unsafe_allow_html=True)

    imb = float(ob.get("imbalance", 0) or 0)
    buy_pressure = float(ob.get("buy_pct", 50) or 50)
    sell_pressure = float(ob.get("sell_pct", 50) or 50)
    spread_pct = float(ob.get("spread_pct", 0) or 0)
    cum_delta = float(ob.get("cum_delta", 0) or 0)
    spread = float(ob.get("spread", 0) or 0)

    def order_status(value: str) -> str:
        v = str(value).lower()
        if any(term in v for term in ("bull", "buyer", "bid", "buying", "tight", "long", "low")):
            return "positive"
        if any(term in v for term in ("bear", "seller", "ask", "selling", "wide", "short", "high")):
            return "negative"
        return "warning"

    def flow_card(label: str, value: str, detail: str, status: str = "warning") -> str:
        return (
            f"<div class='order-chip {card_status_class(status=status)}'>"
            f"<span>{render_info_label(str(label))}</span><strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(detail))}</em></div>"
        )

    def action_card(label: str, value: str, detail: str, status: str = "warning") -> str:
        return (
            f"<div class='order-action-item {card_status_class(status=status)}'>"
            f"<span>{render_info_label(str(label))}</span><strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(detail))}</em></div>"
        )

    pressure_component = (buy_pressure - sell_pressure) * 0.45
    imbalance_component = imb * 100 * 0.30
    delta_component = (15 if cum_delta > 0 else -15 if cum_delta < 0 else 0)
    spread_penalty = -8 if spread_pct > 0.15 else -4 if spread_pct > 0.08 else 4 if spread_pct <= 0.04 else 0
    flow_score = int(round(max(0, min(100, 50 + pressure_component + imbalance_component + delta_component + spread_penalty))))

    order_flow_bias = "Bullish" if flow_score >= 65 else "Bearish" if flow_score <= 35 else "Neutral"
    bias_status = "positive" if order_flow_bias == "Bullish" else "negative" if order_flow_bias == "Bearish" else "warning"
    confidence = int(round(min(100, max(0, 45 + abs(flow_score - 50) * 1.1 + (8 if spread_pct <= 0.05 else -5 if spread_pct > 0.15 else 0)))))

    liquidity_dominance = "Buyers" if buy_pressure - sell_pressure >= 8 else "Sellers" if sell_pressure - buy_pressure >= 8 else "Balanced"
    spread_status = "Tight" if spread_pct <= 0.04 else "Wide" if spread_pct > 0.15 else "Normal"
    depth_imbalance = "Bid Dominant" if imb >= 0.08 else "Ask Dominant" if imb <= -0.08 else "Balanced"
    delta_status = "Net Buying" if cum_delta > 0 else "Net Selling" if cum_delta < 0 else "Neutral"

    reasons = []
    if liquidity_dominance == "Buyers":
        reasons.append("Bid pressure dominant")
    elif liquidity_dominance == "Sellers":
        reasons.append("Ask pressure rising")
    else:
        reasons.append("Balanced book")
    reasons.append("Positive cumulative delta" if cum_delta > 0 else "Negative cumulative delta" if cum_delta < 0 else "Neutral cumulative delta")
    reasons.append("Tight spread" if spread_status == "Tight" else "Wide spread caution" if spread_status == "Wide" else "Normal spread")
    if depth_imbalance == "Bid Dominant":
        reasons.append("Bid depth dominant")
    elif depth_imbalance == "Ask Dominant":
        reasons.append("Ask depth dominant")
    reasons_html = "".join(f"<span class='order-reason-chip'>{html.escape(r)}</span>" for r in reasons)

    healthy_spread = spread_status in {"Tight", "Normal"}
    if order_flow_bias == "Bullish" and healthy_spread:
        preferred_setup = "Long pullback"
        entry_context = "Bid support"
    elif order_flow_bias == "Bearish" and healthy_spread:
        preferred_setup = "Short bounce"
        entry_context = "Ask resistance"
    elif order_flow_bias == "Bullish":
        preferred_setup = "Long only after spread tightens"
        entry_context = "Bid support, spread caution"
    elif order_flow_bias == "Bearish":
        preferred_setup = "Short only after spread tightens"
        entry_context = "Ask resistance, spread caution"
    else:
        preferred_setup = "Wait"
        entry_context = "Balanced book"
    invalidation = "Below best bid" if order_flow_bias == "Bullish" else "Above best ask" if order_flow_bias == "Bearish" else "Break of book balance"
    risk = "High" if spread_status == "Wide" else "Low" if confidence >= 70 and spread_status == "Tight" else "Moderate"

    st.markdown(
        "<div class='order-command-head'>"
        "<div class='order-command-title'>Order Flow Command Center</div>"
        f"<div class='order-command-meta'>{html.escape(symbol)} / {html.escape(src_label)} / Confidence {confidence}%</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='order-command-card'><div class='order-command-grid'>"
        f"<div class='order-core {card_status_class(status=bias_status)}'>"
        f"<div class='order-label'>{render_info_label('Order Flow Bias')}</div>"
        f"<div class='order-value'>{html.escape(order_flow_bias)}</div>"
        f"<div class='order-sub'>Score {flow_score}/100 / Confidence {confidence}%</div>"
        "<div class='order-gauge'><div class='order-gauge-track'>"
        f"<div class='order-gauge-marker' style='left:{flow_score}%'></div></div>"
        "<div class='order-gauge-scale'><span>Bearish</span><span>Neutral</span><span>Bullish</span></div></div>"
        "</div>"
        "<div class='order-health-grid'>"
        + flow_card("Liquidity Dominance", liquidity_dominance, f"Buy {buy_pressure:.1f}% / Sell {sell_pressure:.1f}%", order_status(liquidity_dominance))
        + flow_card("Spread Status", spread_status, f"{spread_pct:.4f}% of price", order_status(spread_status))
        + flow_card("Depth Imbalance", depth_imbalance, f"Imbalance {imb:+.3f}", order_status(depth_imbalance))
        + flow_card("Delta Status", delta_status, f"{cum_delta:+,.4f}", order_status(delta_status))
        + flow_card("Confidence", f"{confidence}/100", f"Risk {risk}", "positive" if confidence >= 70 else "warning" if confidence >= 45 else "negative")
        + "</div></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Why this order flow bias?</div>"
        f"<div class='compact-section-meta'>Pressure {pressure_component:+.1f} / Depth {imbalance_component:+.1f} / Delta {delta_component:+.0f}</div></div>"
        f"<div class='order-reasons'>{reasons_html}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Order Flow Action Plan</div>"
        "<div class='compact-section-meta'>Display-only context from live book values</div></div>"
        "<div class='order-action-grid'>"
        + action_card("Bias", order_flow_bias, f"Score {flow_score}/100", bias_status)
        + action_card("Preferred Setup", preferred_setup, f"Spread {spread_status.lower()}", order_status(order_flow_bias))
        + action_card("Entry Context", entry_context, depth_imbalance, order_status(entry_context))
        + action_card("Invalidation", invalidation, f"Bid {fmt_price(ob['best_bid'], symbol)} / Ask {fmt_price(ob['best_ask'], symbol)}", "warning")
        + action_card("Risk", risk, f"Spread {spread_pct:.4f}%", "negative" if risk == "High" else "positive" if risk == "Low" else "warning")
        + "</div>",
        unsafe_allow_html=True,
    )

    imb_label = "Bid dominant" if imb > 0 else "Ask dominant" if imb < 0 else "Balanced"
    imb_badge = "buy" if imb > 0 else "sell" if imb < 0 else "hold"
    spread_note = f"{spread_pct:.4f}% of price"
    buy_pct = f"{buy_pressure:.1f}%"
    sell_pct = f"{sell_pressure:.1f}%"
    imbalance = f"{imb:+.3f}"
    st.markdown(
        "<div class='dashboard-grid order-book-tile-grid'>"
        + render_metric_tile('Best Bid', fmt_price(ob['best_bid'], symbol), 'Near-term support', 'buy')
        + render_metric_tile('Best Ask', fmt_price(ob['best_ask'], symbol), 'Immediate resistance', 'sell')
        + render_metric_tile('Spread', fmt_price(spread, symbol), spread_note, 'hold')
        + render_metric_tile('Buy Pressure', buy_pct, 'Bid-side liquidity', 'buy')
        + render_metric_tile('Sell Pressure', sell_pct, 'Ask-side liquidity', 'sell')
        + render_metric_tile('Imbalance', imbalance, imb_label, imb_badge)
        + "</div>",
        unsafe_allow_html=True,
    )

    # Bid / Ask tables side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='table-card status-positive'><h5 style='margin:0 0 10px;color:var(--text)'>Top Bids</h5></div>", unsafe_allow_html=True)
        bids_df = pd.DataFrame(ob["bids"]).rename(
            columns={"price":"Price","size":"Size","cumulative":"Cumulative","value":"Value ($)"})
        bids_df["Price"] = bids_df["Price"].astype(float)
        bids_df["Size"] = bids_df["Size"].astype(float).round(4)
        bids_df["Cumulative"] = bids_df["Cumulative"].astype(float).round(4)
        bids_df["Value ($)"] = bids_df["Value ($)"].astype(float)
        render_interactive_dataframe(
            bids_df[["Price","Size","Cumulative","Value ($)"]],
            column_config={
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Size": st.column_config.NumberColumn("Size", format="%.4f"),
                "Cumulative": st.column_config.NumberColumn("Cumulative", format="%.4f"),
                "Value ($)": st.column_config.NumberColumn("Value ($)", format="$%.1f"),
            },
            height=600,
        )

    with col2:
        st.markdown("<div class='table-card status-negative'><h5 style='margin:0 0 10px;color:var(--text)'>Top Asks</h5></div>", unsafe_allow_html=True)
        asks_df = pd.DataFrame(ob["asks"]).rename(
            columns={"price":"Price","size":"Size","cumulative":"Cumulative","value":"Value ($)"})
        asks_df["Price"] = asks_df["Price"].astype(float)
        asks_df["Size"] = asks_df["Size"].astype(float).round(4)
        asks_df["Cumulative"] = asks_df["Cumulative"].astype(float).round(4)
        asks_df["Value ($)"] = asks_df["Value ($)"].astype(float)
        render_interactive_dataframe(
            asks_df[["Price","Size","Cumulative","Value ($)"]],
            column_config={
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Size": st.column_config.NumberColumn("Size", format="%.4f"),
                "Cumulative": st.column_config.NumberColumn("Cumulative", format="%.4f"),
                "Value ($)": st.column_config.NumberColumn("Value ($)", format="$%.1f"),
            },
            height=600,
        )

    # Depth chart
    context_html = "".join([
        f"<span class='order-context-chip'><strong>Spread</strong>{html.escape(fmt_price(spread, symbol))}</span>",
        f"<span class='order-context-chip'><strong>Bid pressure</strong>{buy_pressure:.1f}%</span>",
        f"<span class='order-context-chip'><strong>Ask pressure</strong>{sell_pressure:.1f}%</span>",
        f"<span class='order-context-chip'><strong>Imbalance</strong>{imb:+.3f}</span>",
        f"<span class='order-context-chip'><strong>Delta</strong>{cum_delta:+,.2f}</span>",
    ])
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Depth Heatmap</div>"
        f"<div class='compact-section-meta'>{html.escape(depth_imbalance)} / {html.escape(spread_status)} spread</div></div>"
        f"<div class='order-context-chips'>{context_html}</div>",
        unsafe_allow_html=True,
    )
    bids_list = ob["bids"]
    asks_list = ob["asks"]
    bid_prices = [b["price"] for b in bids_list]
    ask_prices = [a["price"] for a in asks_list]
    bid_cum = [b["cumulative"] for b in bids_list]
    ask_cum = [a["cumulative"] for a in asks_list]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bid_prices, y=bid_cum, name="Bid Depth",
        fill="tozeroy", fillcolor="rgba(38,166,154,0.3)",
        line=dict(color="#00E08A", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=ask_prices, y=ask_cum, name="Ask Depth",
        fill="tozeroy", fillcolor="rgba(239,83,80,0.3)",
        line=dict(color="#FF5C73", width=2),
    ))
    fig.update_layout(
        height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Price", yaxis_title="Cumulative Volume",
    )
    st.plotly_chart(fig, width="stretch")

    # Buy/Sell pressure bar
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Buy / Sell Pressure</div>"
        f"<div class='compact-section-meta'>{html.escape(liquidity_dominance)} controlling short-term flow</div></div>"
        "<div class='order-pressure-panel'>"
        + flow_card("Buy Pressure", f"{buy_pressure:.1f}%", "Bid-side liquidity", "positive")
        + flow_card("Sell Pressure", f"{sell_pressure:.1f}%", "Ask-side liquidity", "negative")
        + flow_card("Interpretation", liquidity_dominance, delta_status, order_status(liquidity_dominance))
        + "</div>",
        unsafe_allow_html=True,
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=["Buy Pressure"], y=[buy_pressure], marker_color="#00E08A", name="Bids"))
    fig2.add_trace(go.Bar(x=["Sell Pressure"], y=[sell_pressure], marker_color="#FF5C73", name="Asks"))
    fig2.update_layout(height=145, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0),
        showlegend=False, yaxis=dict(range=[0,100], ticksuffix="%"))
    st.plotly_chart(fig2, width="stretch")

    st.markdown(
        "<div class='dashboard-grid order-pressure-panel'>"
        + render_dashboard_card(
            "Cumulative Delta",
            f"{cum_delta:+,.4f}",
            "Net buying" if cum_delta > 0 else "Net selling" if cum_delta < 0 else "Balanced flow",
            status="positive" if cum_delta > 0 else "negative" if cum_delta < 0 else "neutral",
        )
        + "</div>",
        unsafe_allow_html=True,
    )


# ── Tab 5: Multi-Timeframe ────────────────────────────────────────────────────

def render_mtf(mtf: dict, symbol: str, theme_name: str = "Default"):
    theme_name = normalize_theme_name(theme_name)
    # PERF: Validate data before rendering
    if not mtf or not isinstance(mtf, dict):
        st.info("Multi-timeframe data unavailable")
        return

    if "_overall" not in mtf:
        st.info("Multi-timeframe data unavailable")
        return

    theme = THEME_TOKENS.get(THEME_TOKEN_MAP.get(theme_name, "Institutional Dark"), THEME_TOKENS["Institutional Dark"])
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

    tfs = [tf for tf in MTF_TIMEFRAMES if tf in mtf]
    if not tfs:
        st.info("No timeframe data available")
        return

    tf_weights = {"1m": 1, "5m": 2, "15m": 3, "1h": 5, "4h": 8}

    def safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def clamp(value, low, high):
        return max(low, min(high, value))

    def tf_weight(tf: str) -> int:
        return tf_weights.get(tf, 0)

    def tf_direction(d: dict) -> int:
        score = safe_float(d.get("score"), 0.0)
        verdict = str(d.get("verdict", d.get("signal", ""))).lower()
        if score >= 2 or "buy" in verdict or "bull" in verdict:
            return 1
        if score <= -2 or "sell" in verdict or "bear" in verdict:
            return -1
        return 0

    def status_from_bias(label: str) -> str:
        text = str(label).lower()
        if any(word in text for word in ["bull", "buy", "long", "aligned", "low"]):
            return "positive"
        if any(word in text for word in ["bear", "sell", "short", "diverg", "high"]):
            return "negative"
        return "warning"

    weighted_sum = 0.0
    total_weight = 0
    bullish_weight = 0
    bearish_weight = 0
    neutral_weight = 0
    htf_weighted_sum = 0.0
    htf_total_weight = 0
    ltf_conflict_weight = 0
    momentum_bull_weight = 0
    momentum_bear_weight = 0
    trend_bull_weight = 0
    trend_bear_weight = 0
    contributions = {}

    for tf in tfs:
        d = mtf.get(tf, {})
        if d.get("verdict") == "N/A":
            continue
        weight = tf_weight(tf)
        if weight <= 0:
            contributions[tf] = 0.0
            continue
        score = safe_float(d.get("score"), 0.0)
        normalized = clamp(score / 6.0, -1.0, 1.0)
        direction = tf_direction(d)
        weighted_sum += normalized * weight
        total_weight += weight
        contributions[tf] = normalized * weight
        if direction > 0:
            bullish_weight += weight
        elif direction < 0:
            bearish_weight += weight
        else:
            neutral_weight += weight
        if weight >= 5:
            htf_weighted_sum += normalized * weight
            htf_total_weight += weight
        else:
            ltf_conflict_weight += 0
        momentum = safe_float(d.get("momentum"), 50.0)
        if momentum >= 55:
            momentum_bull_weight += weight
        elif momentum <= 45:
            momentum_bear_weight += weight
        trend_text = str(d.get("details", {}).get("trend", "Neutral"))
        if "Bull" in trend_text or trend_text == "Uptrend":
            trend_bull_weight += weight
        elif "Bear" in trend_text or trend_text == "Downtrend":
            trend_bear_weight += weight

    if total_weight:
        weighted_raw = weighted_sum / total_weight
        weighted_score = int(round(clamp((weighted_raw + 1.0) * 50, 0, 100)))
    else:
        weighted_raw = 0.0
        weighted_score = 50

    mtf_bias = "Bullish" if weighted_score >= 65 else "Bearish" if weighted_score <= 35 else "Neutral"
    bias_status = status_from_bias(mtf_bias)
    agreement_weight = max(bullish_weight, bearish_weight, neutral_weight)
    agreement_pct = int(round((agreement_weight / total_weight) * 100)) if total_weight else 0
    direction_spread = abs(bullish_weight - bearish_weight)
    confidence = int(round(clamp(40 + abs(weighted_score - 50) * 1.1 + agreement_pct * 0.35, 0, 100)))

    if trend_bull_weight > trend_bear_weight * 1.25:
        trend_alignment = "Bullish aligned"
    elif trend_bear_weight > trend_bull_weight * 1.25:
        trend_alignment = "Bearish aligned"
    elif trend_bull_weight or trend_bear_weight:
        trend_alignment = "Trend divergence"
    else:
        trend_alignment = "Unavailable"

    if momentum_bull_weight > momentum_bear_weight * 1.2:
        momentum_alignment = "Bullish momentum"
    elif momentum_bear_weight > momentum_bull_weight * 1.2:
        momentum_alignment = "Bearish momentum"
    elif momentum_bull_weight or momentum_bear_weight:
        momentum_alignment = "Mixed momentum"
    else:
        momentum_alignment = "Unavailable"

    signal_agreement = "Strong agreement" if agreement_pct >= 70 else "Mixed agreement" if agreement_pct >= 48 else "Low agreement"
    htf_raw = htf_weighted_sum / htf_total_weight if htf_total_weight else weighted_raw
    htf_direction = 1 if htf_raw > 0.22 else -1 if htf_raw < -0.22 else 0
    bias_direction = 1 if mtf_bias == "Bullish" else -1 if mtf_bias == "Bearish" else 0
    for tf in ["1m", "5m"]:
        if tf in mtf and bias_direction and tf_direction(mtf[tf]) and tf_direction(mtf[tf]) != bias_direction:
            ltf_conflict_weight += tf_weight(tf)

    divergence = "divergence" in trend_alignment.lower() or "mixed" in momentum_alignment.lower()
    if confidence >= 72 and not divergence and agreement_pct >= 60:
        risk_level = "Low"
    elif confidence < 55 or divergence or ltf_conflict_weight >= 2:
        risk_level = "High"
    else:
        risk_level = "Moderate"

    def command_chip(label, value, sub, status=None):
        cls = status or status_from_bias(value)
        return (
            f"<div class='mtf-chip status-{cls}'>"
            f"<span>{render_info_label(str(label))}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(sub))}</em>"
            "</div>"
        )

    st.markdown(
        "<div class='mtf-command-head'>"
        "<div class='mtf-command-title'>MTF Command Center</div>"
        f"<div class='mtf-command-meta'>{html.escape(symbol)} / weighted 1m-4H consensus</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='mtf-command-card'><div class='mtf-command-grid'>"
        f"<div class='mtf-core status-{bias_status}'>"
        f"<div class='mtf-label'>{render_info_label('MTF Bias')}</div>"
        f"<div class='mtf-value'>{html.escape(mtf_bias)}</div>"
        f"<div class='mtf-sub'>Weighted score {weighted_score}/100 · Confidence {confidence}%</div>"
        "<div class='mtf-gauge'><div class='mtf-gauge-track'>"
        f"<span class='mtf-gauge-marker' style='left:{weighted_score}%'></span>"
        "</div><div class='mtf-gauge-scale'><span>Bearish</span><span>Neutral</span><span>Bullish</span></div></div>"
        "</div>"
        "<div class='mtf-health-grid'>"
        + command_chip("Weighted Score", f"{weighted_score}/100", f"Raw {weighted_raw:+.2f}", bias_status)
        + command_chip("Confidence", f"{confidence}%", f"Agreement {agreement_pct}%", "positive" if confidence >= 70 else "warning" if confidence >= 55 else "negative")
        + command_chip("Trend Alignment", trend_alignment, f"Bull {trend_bull_weight} / Bear {trend_bear_weight}")
        + command_chip("Momentum Alignment", momentum_alignment, f"Bull {momentum_bull_weight} / Bear {momentum_bear_weight}")
        + command_chip("Signal Agreement", signal_agreement, f"{agreement_pct}% weighted")
        + command_chip("Risk Level", risk_level, f"LTF conflict {ltf_conflict_weight}", "negative" if risk_level == "High" else "positive" if risk_level == "Low" else "warning")
        + "</div></div></div>",
        unsafe_allow_html=True,
    )

    reason_labels = []
    for tf in ["4h", "1h"]:
        if tf in mtf:
            direction = tf_direction(mtf[tf])
            if direction > 0:
                reason_labels.append(f"{MTF_LABELS.get(tf, tf)} bullish")
            elif direction < 0:
                reason_labels.append(f"{MTF_LABELS.get(tf, tf)} bearish")
            else:
                reason_labels.append(f"{MTF_LABELS.get(tf, tf)} neutral")
    if "15m" in mtf and bias_direction and tf_direction(mtf["15m"]) and tf_direction(mtf["15m"]) != bias_direction:
        reason_labels.append("15m conflicting")
    if ltf_conflict_weight:
        reason_labels.append("Lower timeframe noise")
    if momentum_alignment.lower().startswith(mtf_bias.lower()[:4]):
        reason_labels.append("Momentum aligned")
    if "divergence" in trend_alignment.lower():
        reason_labels.append("Trend divergence")
    if not reason_labels:
        reason_labels.append("Mixed timeframe structure")
    reason_labels = reason_labels[:7]
    reasons_html = "".join(f"<span class='mtf-reason-chip'>{html.escape(reason)}</span>" for reason in reason_labels)
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Why this MTF bias?</div>"
        f"<div class='compact-section-meta'>Higher timeframes carry heavier weight</div></div>"
        f"<div class='mtf-reasons'>{reasons_html}</div>",
        unsafe_allow_html=True,
    )

    if mtf_bias == "Bullish":
        preferred_direction = "Long"
        preferred_setup = "HTF continuation" if htf_direction >= 0 else "Pullback alignment"
        entry_context = "Use 15m / 5m trigger with 1H and 4H bias"
        confirmation_needed = "15m bullish follow-through" if "15m" in mtf and tf_direction(mtf["15m"]) <= 0 else "Lower timeframe holds bid"
        invalidation = "4H bias flips" if "4h" in mtf else "1H bias flips"
    elif mtf_bias == "Bearish":
        preferred_direction = "Short"
        preferred_setup = "HTF continuation" if htf_direction <= 0 else "Bounce alignment"
        entry_context = "Use 15m / 5m trigger with 1H and 4H pressure"
        confirmation_needed = "15m bearish follow-through" if "15m" in mtf and tf_direction(mtf["15m"]) >= 0 else "Lower timeframe loses bid"
        invalidation = "4H bias flips" if "4h" in mtf else "1H bias flips"
    else:
        preferred_direction = "Wait"
        preferred_setup = "Wait for alignment"
        entry_context = "Range / mixed timeframe structure"
        confirmation_needed = "1H and 4H resolve the same way"
        invalidation = "New HTF conflict"

    def action_card(label, value, sub, status=None):
        cls = status or status_from_bias(value)
        return (
            f"<div class='mtf-action-item status-{cls}'>"
            f"<span>{render_info_label(str(label))}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(sub))}</em>"
            "</div>"
        )

    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>MTF Trade Setup Generator</div>"
        "<div class='compact-section-meta'>Display-only setup from existing timeframe states</div></div>"
        "<div class='mtf-action-grid'>"
        + action_card("Preferred Direction", preferred_direction, mtf_bias, bias_status)
        + action_card("Preferred Setup", preferred_setup, trend_alignment)
        + action_card("Entry Context", entry_context, signal_agreement)
        + action_card("Confirmation Needed", confirmation_needed, momentum_alignment)
        + action_card("Invalidation", invalidation, "Higher timeframe state")
        + action_card("Risk Level", risk_level, f"Confidence {confidence}%", "negative" if risk_level == "High" else "positive" if risk_level == "Low" else "warning")
        + "</div>",
        unsafe_allow_html=True,
    )

    ov = overall.get("verdict", "N/A")
    ov_c = overall.get("color", theme["muted"])
    avg = safe_float(overall.get("avg_score"), 0.0)
    conf = int(safe_float(overall.get("confidence"), 0.0) * 100)
    st.markdown(
        "<div class='dashboard-grid mtf-summary-dashboard-grid'>"
        + render_dashboard_card("MTF Consensus", ov, f"Avg score {avg:+.2f} · {conf}% confidence", ov_c, signal=overall.get("signal"), status=ov, trend=overall.get("alignment"),)
        + render_dashboard_card("Bullish Timeframes", str(overall.get("bullish", 0)), "Raw timeframe count", theme["success"], status="positive")
        + render_dashboard_card("Bearish Timeframes", str(overall.get("bearish", 0)), "Raw timeframe count", theme["danger"], status="negative")
        + render_dashboard_card("Neutral / Hold", str(overall.get("hold", 0)), "Divergence zones", theme["warning"], status="neutral")
        + "</div>",
        unsafe_allow_html=True,
    )

    cards_html = ""
    for tf in tfs:
        d = mtf[tf]
        tf_color = d.get("color", theme["muted"])
        momentum = int(safe_float(d.get("momentum"), 0.0))
        tf_confidence = int(safe_float(d.get("confidence"), 0.0) * 100)
        trend = str(d.get("details", {}).get("trend", "Neutral"))
        tf_signal = d.get("signal", d.get("verdict", "N/A"))
        weight = tf_weight(tf)
        contribution = contributions.get(tf, 0.0)
        importance = "HTF driver" if weight >= 5 else "Midframe" if weight >= 3 else "LTF trigger" if weight else "Reference"
        weight_label = f"W{weight}" if weight else "Ref"
        emphasis_class = "mtf-card-high" if weight >= 5 else "mtf-card-mid" if weight >= 3 else ""
        tf_status_class = card_status_class(signal=tf_signal, status=d.get("verdict"), trend=trend, value=momentum)
        cards_html += (
            f"<div class='dashboard-card {tf_status_class} {emphasis_class}' style='padding:18px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px'>"
            f"<div class='metric-label'>{render_info_label(MTF_LABELS.get(tf, tf))}</div>"
            f"<span class='mtf-weight-badge'>{weight_label} · {html.escape(importance)}</span>"
            "</div>"
            f"<div style='font-size:1.4rem;font-weight:800;color:{tf_color};margin-bottom:4px'>{html.escape(str(d.get('verdict', 'N/A')))}</div>"
            f"<div style='font-size:.88rem;color:var(--muted);margin-bottom:12px'>Trend: {html.escape(trend)}</div>"
            f"<div style='display:flex;gap:7px;flex-wrap:wrap'>"
            f"<span class='metric-pill' style='background:rgba(37,99,235,0.12);color:{theme['accent']}'>Momentum {momentum}%</span>"
            f"<span class='metric-pill' style='background:rgba(255,92,115,0.12);color:{theme['danger']}'>Conf {tf_confidence}%</span>"
            f"<span class='metric-pill' style='background:rgba(0,224,138,0.12);color:{theme['success']}'>Signal {html.escape(str(d.get('signal','N/A')))}</span>"
            f"<span class='metric-pill'>Contribution {contribution:+.1f}</span>"
            f"</div>"
            "</div>"
        )
    st.markdown(f"<div class='dashboard-grid mtf-timeframe-grid'>{cards_html}</div>", unsafe_allow_html=True)

    ltf_noise = "Yes" if ltf_conflict_weight else "No"
    htf_bias_label = "Bullish" if htf_direction > 0 else "Bearish" if htf_direction < 0 else "Neutral"
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Alignment Matrix</div>"
        "<div class='compact-section-meta'>Indicator agreement by timeframe</div></div>"
        "<div class='mtf-summary-grid'>"
        + command_chip("Weighted Score", f"{weighted_score}/100", mtf_bias, bias_status)
        + command_chip("HTF Bias", htf_bias_label, "1H / 4H priority", status_from_bias(htf_bias_label))
        + command_chip("LTF Noise", ltf_noise, f"Conflict weight {ltf_conflict_weight}", "warning" if ltf_conflict_weight else "positive")
        + command_chip("Agreement", f"{agreement_pct}%", signal_agreement, "positive" if agreement_pct >= 70 else "warning" if agreement_pct >= 48 else "negative")
        + "</div>",
        unsafe_allow_html=True,
    )

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
            v = str(mtf[tf].get("details", {}).get(m, "N/A"))
            row_d.append(v)
            if any(label in v for label in ["Bull", "Oversold", "Strong Bull", "Uptrend"]):
                row_c.append(theme["heat_bull"])
            elif any(label in v for label in ["Bear", "Overbought", "Strong Bear", "Downtrend"]):
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
                [f"{MTF_LABELS.get(tf, tf)} · {'W' + str(tf_weight(tf)) if tf_weight(tf) else 'Ref'}" for tf in tfs],
                *[[heat_data[i][j] for i in range(len(tfs))] for j in range(len(metrics))],
            ],
            fill_color=[theme["card_bg"]] + [[heat_colors[i][j] for i in range(len(tfs))] for j in range(len(metrics))],
            font=dict(color=theme["text"], size=11),
            align="center",
            height=32,
        ),
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=6, b=0),
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

    theme_name = normalize_theme_name(cfg.get("theme", "Default") if isinstance(cfg, dict) else "Default")
    theme = THEME_TOKENS.get(THEME_TOKEN_MAP.get(theme_name, "Institutional Dark"), THEME_TOKENS["Institutional Dark"])

    sig = str(signal_result.get("signal", "HOLD") or "HOLD").upper()
    conf = float(signal_result.get("confidence", 0.0) or 0.0)
    sc = float(signal_result.get("score", 0.0) or 0.0)
    bull = int(signal_result.get("bull_signals", 0) or 0)
    bear = int(signal_result.get("bear_signals", 0) or 0)
    norm = float(signal_result.get("normalized_score", 0.0) or 0.0)
    color = signal_color(sig)

    inst_label = str(signal_result.get("institutional_bias", "Neutral") or "Neutral")
    inst_score = float(signal_result.get("institutional_bias_score", 0.0) or 0.0)
    market_regime = str(signal_result.get("market_regime", "N/A") or "N/A")
    trend_strength = float(signal_result.get("trend_strength", 0.0) or 0.0)
    trend_category = str(signal_result.get("trend_strength_label") or "")
    if not trend_category:
        if trend_strength >= 0.9:
            trend_category = "Extreme"
        elif trend_strength >= 0.66:
            trend_category = "Strong"
        elif trend_strength >= 0.33:
            trend_category = "Moderate"
        else:
            trend_category = "Weak"
    risk_level = str(signal_result.get("risk_level", "N/A") or "N/A")
    reasons = [str(r) for r in signal_result.get("reasons", []) if r]

    def clamp(value, low=0.0, high=100.0):
        return max(low, min(high, value))

    def status_from_text(value: str) -> str:
        text = str(value).lower()
        if any(word in text for word in ["buy", "bull", "up", "long", "low", "positive", "above", "strong"]):
            return "positive"
        if any(word in text for word in ["sell", "bear", "down", "short", "high", "negative", "below", "weak"]):
            return "negative"
        return "warning"

    ai_direction_score = int(round(clamp((norm + 1.0) * 50)))
    ai_bias = "BUY" if ai_direction_score >= 65 else "SELL" if ai_direction_score <= 35 else "HOLD"
    bias_status = status_from_text(ai_bias)
    confidence_pct = int(round(clamp(conf * 100)))
    trend_status = "positive" if trend_strength >= 0.66 else "warning" if trend_strength >= 0.33 else "negative"
    risk_status = "negative" if risk_level == "High" else "positive" if risk_level == "Low" else "warning"

    rf = ml_result.get("rf", {}) if isinstance(ml_result, dict) else {}
    xgb = ml_result.get("xgb", {}) if isinstance(ml_result, dict) else {}
    rf_prob = rf.get("probability") if isinstance(rf, dict) else None
    xgb_prob = xgb.get("probability") if isinstance(xgb, dict) else None
    ml_prob = ml_result.get("combined_probability") if isinstance(ml_result, dict) else None
    ml_direction = str(ml_result.get("direction", "N/A") if isinstance(ml_result, dict) else "N/A")

    def model_direction(prob, pred=None):
        if pred is not None:
            try:
                return "UP" if int(pred) == 1 else "DOWN"
            except (TypeError, ValueError):
                pass
        if prob is None:
            return "N/A"
        return "UP" if float(prob) >= 0.5 else "DOWN"

    rf_dir = model_direction(rf_prob, rf.get("prediction") if isinstance(rf, dict) else None)
    xgb_dir = model_direction(xgb_prob, xgb.get("prediction") if isinstance(xgb, dict) else None)
    available_model_dirs = [d for d in [rf_dir, xgb_dir] if d != "N/A"]
    if len(available_model_dirs) >= 2:
        model_agreement_pct = 100 if len(set(available_model_dirs)) == 1 else 50
    elif len(available_model_dirs) == 1:
        model_agreement_pct = 50
    else:
        model_agreement_pct = 0
    model_agreement = "Agree" if model_agreement_pct == 100 else "Partial" if model_agreement_pct == 50 else "Unavailable"
    if ml_direction == "N/A" and ml_prob is not None:
        ml_direction = "UP" if float(ml_prob) >= 0.5 else "DOWN"
    ml_bias_aligned = (ai_bias == "BUY" and ml_direction == "UP") or (ai_bias == "SELL" and ml_direction == "DOWN") or ai_bias == "HOLD"

    def ai_chip(label, value, sub, status=None):
        cls = status or status_from_text(value)
        return (
            f"<div class='ai-chip status-{cls}'>"
            f"<span>{render_info_label(str(label))}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(sub))}</em>"
            "</div>"
        )

    st.markdown(
        "<div class='ai-command-head'>"
        "<div class='ai-command-title'>AI Signal Command Center</div>"
        f"<div class='ai-command-meta'>{html.escape(symbol)} / display-only synthesis from existing signal outputs</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ai-command-card'><div class='ai-command-grid'>"
        f"<div class='ai-core status-{bias_status}'>"
        f"<div class='ai-label'>{render_info_label('AI Bias')}</div>"
        f"<div class='ai-value'>{html.escape(ai_bias)}</div>"
        f"<div class='ai-sub'>Confidence {confidence_pct}% · Score {ai_direction_score}/100</div>"
        "<div class='ai-gauge'><div class='ai-gauge-track'>"
        f"<span class='ai-gauge-marker' style='left:{ai_direction_score}%'></span>"
        "</div><div class='ai-gauge-scale'><span>SELL</span><span>HOLD</span><span>BUY</span></div></div>"
        "</div>"
        "<div class='ai-health-grid'>"
        + ai_chip("AI Bias", ai_bias, f"Raw {sc:+.2f}", bias_status)
        + ai_chip("Confidence", f"{confidence_pct}%", f"Norm {norm:+.3f}", "positive" if confidence_pct >= 70 else "warning" if confidence_pct >= 50 else "negative")
        + ai_chip("Model Agreement", f"{model_agreement_pct}%", model_agreement, "positive" if model_agreement_pct == 100 else "warning" if model_agreement_pct else "negative")
        + ai_chip("Market Regime", market_regime, f"Institutional {inst_label}")
        + ai_chip("Trend Strength", trend_category, f"{trend_strength:.3f}", trend_status)
        + ai_chip("Risk Level", risk_level, f"Bull {bull} / Bear {bear}", risk_status)
        + "</div></div></div>",
        unsafe_allow_html=True,
    )

    factor_aliases = [
        ("Supertrend", ["supertrend"]),
        ("MACD", ["macd"]),
        ("RSI", ["rsi"]),
        ("VWAP", ["vwap"]),
        ("Fear & Greed", ["fear", "greed"]),
        ("Market regime", ["market regime", "regime"]),
    ]
    factor_entries = []
    seen_factors = set()
    for idx, reason in enumerate(reasons):
        lower = reason.lower()
        for label, keys in factor_aliases:
            if label not in seen_factors and any(key in lower for key in keys):
                strength = len(reasons) - idx
                factor_entries.append((strength, label, reason))
                seen_factors.add(label)
                break
    if ml_direction != "N/A":
        ml_strength = int(round(abs(float(ml_prob or 0.5) - 0.5) * 100)) + 1
        ml_reason = f"ML prediction {ml_direction} ({float(ml_prob or 0.5) * 100:.1f}% up probability)"
        factor_entries.append((ml_strength, "ML prediction", ml_reason))
    factor_entries.sort(key=lambda item: item[0], reverse=True)
    if not factor_entries:
        factor_entries.append((1, "Signal", "Mixed signal inputs"))
    reason_chips = "".join(
        f"<span class='ai-reason-chip'>{html.escape(label)}: {html.escape(reason)}</span>"
        for _, label, reason in factor_entries[:7]
    )
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Why this AI signal?</div>"
        "<div class='compact-section-meta'>Strongest available factors by signal-engine order and model confidence</div></div>"
        f"<div class='ai-reasons'>{reason_chips}</div>",
        unsafe_allow_html=True,
    )

    if ai_bias == "BUY":
        preferred_direction = "Long"
        preferred_setup = "Long continuation" if trend_strength >= 0.5 else "Pullback confirmation"
        entry_context = "Buy signal with supportive model" if ml_bias_aligned else "Buy signal with ML conflict"
        confirmation_needed = "MACD / Supertrend follow-through"
        invalidation = "AI bias falls back to HOLD"
    elif ai_bias == "SELL":
        preferred_direction = "Short"
        preferred_setup = "Short continuation" if trend_strength >= 0.5 else "Bounce rejection"
        entry_context = "Sell signal with supportive model" if ml_bias_aligned else "Sell signal with ML conflict"
        confirmation_needed = "MACD / Supertrend follow-through"
        invalidation = "AI bias recovers to HOLD"
    else:
        preferred_direction = "Wait"
        preferred_setup = "Wait for confirmation"
        entry_context = "Mixed signal stack"
        confirmation_needed = "AI score clears BUY or SELL zone"
        invalidation = "New directional break"

    def action_card(label, value, sub, status=None):
        cls = status or status_from_text(value)
        return (
            f"<div class='ai-action-item status-{cls}'>"
            f"<span>{render_info_label(str(label))}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(sub))}</em>"
            "</div>"
        )

    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>AI Trade Setup Generator</div>"
        "<div class='compact-section-meta'>Display-only setup from current AI signal values</div></div>"
        "<div class='ai-action-grid'>"
        + action_card("Preferred Direction", preferred_direction, ai_bias, bias_status)
        + action_card("Preferred Setup", preferred_setup, trend_category)
        + action_card("Entry Context", entry_context, f"Model {model_agreement.lower()}")
        + action_card("Confirmation Needed", confirmation_needed, market_regime)
        + action_card("Invalidation", invalidation, f"Score {ai_direction_score}/100")
        + action_card("Risk Level", risk_level, f"Confidence {confidence_pct}%", risk_status)
        + "</div>",
        unsafe_allow_html=True,
    )

    def fmt_prob(prob):
        return "N/A" if prob is None else f"{float(prob) * 100:.1f}% up"

    def accuracy_text(model):
        meta = (model.get("meta", {}) if isinstance(model, dict) else {}) or {}
        val = meta.get("test_accuracy", meta.get("train_accuracy"))
        return "Accuracy N/A" if val is None else f"Accuracy {float(val) * 100:.1f}%"

    def consensus_card(label, value, sub, status=None):
        cls = status or status_from_text(value)
        return (
            f"<div class='ai-consensus-card status-{cls}'>"
            f"<span>{render_info_label(str(label))}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(sub))}</em>"
            "</div>"
        )

    ensemble_label = f"{ml_direction}" if ml_direction != "N/A" else "Unavailable"
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Model Consensus Panel</div>"
        "<div class='compact-section-meta'>Random Forest / XGBoost / Ensemble agreement</div></div>"
        "<div class='ai-consensus-grid'>"
        + consensus_card("Random Forest", rf_dir, f"{fmt_prob(rf_prob)} · {accuracy_text(rf)}")
        + consensus_card("XGBoost", xgb_dir, f"{fmt_prob(xgb_prob)} · {accuracy_text(xgb)}")
        + consensus_card("Ensemble / ML Consensus", ensemble_label, fmt_prob(ml_prob))
        + consensus_card("Agreement", f"{model_agreement_pct}%", model_agreement, "positive" if model_agreement_pct == 100 else "warning" if model_agreement_pct else "negative")
        + "</div>",
        unsafe_allow_html=True,
    )

    technical_hits = sum(1 for r in reasons if any(k in r.lower() for k in ["rsi", "macd", "ema", "vwap", "supertrend", "bollinger", "sar"]))
    trend_score = int(round(clamp(trend_strength * 100)))
    technical_score = int(round(clamp(50 + (bull - bear) * 7 + technical_hits * 3)))
    ml_score = int(round(clamp((float(ml_prob or 0.5)) * 100)))
    fg_value = fg.get("value", 50) if isinstance(fg, dict) else 50
    try:
        fg_num = int(float(fg_value))
    except (TypeError, ValueError):
        fg_num = 50
    sentiment_score = int(round(clamp(100 - abs(fg_num - 50) * 1.2)))

    def breakdown_status(score_value):
        return "positive" if score_value >= 65 else "negative" if score_value <= 35 else "warning"

    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>AI Confidence Breakdown</div>"
        "<div class='compact-section-meta'>Deterministic read from existing AI inputs</div></div>"
        "<div class='ai-breakdown-grid'>"
        + consensus_card("Technical Contribution", f"{technical_score}/100", f"{technical_hits} technical factors", breakdown_status(technical_score))
        + consensus_card("Trend Contribution", f"{trend_score}/100", trend_category, breakdown_status(trend_score))
        + consensus_card("ML Contribution", f"{ml_score}/100", ensemble_label, breakdown_status(ml_score))
        + consensus_card("Sentiment Contribution", f"{sentiment_score}/100", f"Fear & Greed {fg_num}", breakdown_status(sentiment_score))
        + "</div>",
        unsafe_allow_html=True,
    )

    css_state = "buy" if sig == "BUY" else ("sell" if sig == "SELL" else "hold")
    signal_status_class = card_status_class(signal=sig)
    dominance_total = bull + bear
    if dominance_total > 0:
        bull_pct = bull / dominance_total * 100
    else:
        bull_pct = ob.get("buy_pct", 50) if isinstance(ob, dict) else 50
    bear_pct = max(0.0, 100 - bull_pct)
    rclass = "risk-low" if risk_level == "Low" else ("risk-medium" if risk_level == "Medium" else "risk-high")

    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Signal Summary</div>"
        "<div class='compact-section-meta'>Existing AI signal output</div></div>",
        unsafe_allow_html=True,
    )
    with st.container():
        left, right = st.columns([1, 2])
        with left:
            st.markdown(
                f"<div class='signal-card {css_state} {signal_status_class}'><div style='display:flex;align-items:center;justify-content:space-between'>"
                f"<div style='display:flex;flex-direction:column;align-items:flex-start'>"
                f"<div class='signal-badge' style='background:{color}'>{html.escape(sig)}</div>"
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
            st.markdown(
                f"<div class='signal-card {card_status_class(signal=sig)}'><div class='signal-row'>"
                f"<div style='flex:1'><div class='signal-meta'><b>Institutional:</b> {html.escape(inst_label)} ({inst_score:+.3f})</div></div>"
                f"<div style='flex:1'><div class='signal-meta'><b>Regime:</b> {html.escape(market_regime)}</div></div>"
                f"<div style='flex:1'><div class='signal-meta'><b>Risk:</b> <span class='risk-badge {rclass}'>{html.escape(risk_level)}</span></div></div>"
                f"</div>"
                f"<div style='height:8px'></div>"
                f"<div class='signal-row'>"
                f"<div style='flex:1'><div class='signal-item'><b>Trend:</b> {html.escape(trend_category)}</div></div>"
                f"<div style='flex:1'><div class='signal-item'><b>Bull:</b> {bull}</div></div>"
                f"<div style='flex:1'><div class='signal-item'><b>Bear:</b> {bear}</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### Signal Reasoning (top factors by weight)")
    for r in reasons[:12]:
        icon = "POS" if any(w in r.lower() for w in
                            ["bull", "oversold", "above", "positive", "discount", "dominant bid", "strong buy"]) \
               else "NEG" if any(w in r.lower() for w in
                                  ["bear", "overbought", "below", "negative", "premium", "dominant ask", "strong sell"]) \
               else "NEU"
        st.caption(f"{icon} · {r}")

    st.divider()
    st.subheader("ML Predictions")
    if render_ml_prediction_state(ml_result):
        direction = ml_result.get("direction", "?")
        prob = ml_result.get("combined_probability", 0.5)
        dir_c = "#00E08A" if direction == "UP" else "#FF5C73"
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.markdown(
            f"<div class='terminal-card {card_status_class(signal=direction)}' style='text-align:center'>"
            f"<div class='metric-label'>{render_info_label('ML Consensus')}</div>"
            f"<div style='font-size:1.8em;color:{dir_c};font-weight:700'>"
            f"{('UP' if direction=='UP' else 'DOWN')} {direction}</div></div>",
            unsafe_allow_html=True)
        mc2.metric("Up Probability", f"{prob*100:.1f}%")
        rf = ml_result.get("rf", {}) or {}
        rfm = rf.get("meta", {}) or {}
        mc3.metric("RF Accuracy", f"{rfm.get('test_accuracy', rfm.get('train_accuracy',0))*100:.1f}%")
        xgb = ml_result.get("xgb", {}) or {}
        xgbm = xgb.get("meta", {}) or {}
        mc4.metric("XGB Accuracy", f"{xgbm.get('test_accuracy', xgbm.get('train_accuracy',0))*100:.1f}%")

        fi = ml_result.get("feature_importance", {})
        if fi:
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance")
            fig = px.bar(fi_df.tail(10), x="Importance", y="Feature", orientation="h",
                         title="Top Feature Importances", color="Importance",
                         color_continuous_scale="teal")
            fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width="stretch")

        cache_status = ml_result.get("_cache_status")
        elapsed_ms = ml_result.get("_elapsed_ms")
        if cache_status == "cached":
            st.caption("ML prediction reused from the current session cache.")
        elif elapsed_ms is not None:
            st.caption(f"ML prediction refreshed in {elapsed_ms / 1000:.1f}s.")

# ── Tab 7: Backtest ───────────────────────────────────────────────────────────

def render_backtest(df, cfg, symbol):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        render_empty_state("Backtest requires market data. Select a valid symbol and timeframe.")
        return
    st.markdown(render_section_header("Strategy Backtester", "Simulate trade performance with premium risk controls"), unsafe_allow_html=True)
    init_widget_from_query("bt_run_sl", "bt_run_sl", cfg["stop_loss_pct"] * 100, lambda v: qp_float("bt_run_sl", cfg["stop_loss_pct"] * 100, 0.5, 10.0))
    init_widget_from_query("bt_run_tp", "bt_run_tp", cfg["take_profit_pct"] * 100, lambda v: qp_float("bt_run_tp", cfg["take_profit_pct"] * 100, 1.0, 20.0))
    init_widget_from_query("bt_run_cap", "bt_run_cap", float(cfg["capital"]), lambda v: qp_float("bt_run_cap", float(cfg["capital"]), 5.0, 1_000_000.0))
    init_widget_from_query("bt_run_pos", "bt_run_pos", int(cfg["bt_pos_size"] * 100), lambda v: qp_int("bt_run_pos", int(cfg["bt_pos_size"] * 100), 5, 50, 5))

    bc1, bc2 = st.columns(2)
    with bc1:
        bt_sl_pct = st.slider("Stop Loss %", 0.5, 10.0, step=0.5, key="bt_run_sl")
        bt_tp_pct = st.slider("Take Profit %", 1.0, 20.0, step=0.5, key="bt_run_tp")
    with bc2:
        bt_cap = st.number_input("Starting Capital ($)", 5.0, 1_000_000.0, step=1.0, format="%.2f", key="bt_run_cap")
        if bt_cap < 5:
            st.error("Minimum capital is $5.00")
            bt_cap = 5.0
        bt_pos_pct = st.slider("Position Size %", 5, 50, step=5, key="bt_run_pos")

    qp_set("bt_run_sl", bt_sl_pct)
    qp_set("bt_run_tp", bt_tp_pct)
    qp_set("bt_run_cap", bt_cap)
    qp_set("bt_run_pos", bt_pos_pct)
    bt_sl = bt_sl_pct / 100
    bt_tp = bt_tp_pct / 100
    bt_pos = bt_pos_pct / 100

    bt_key = backtest_cache_key(symbol, cfg.get("timeframe", "1h"), cfg["limit"], bt_cap, bt_sl, bt_tp, bt_pos)
    bt_cache = st.session_state.setdefault("bt_result_cache", {})

    if st.button("Run Backtest", type="primary"):
        status = st.empty()
        with status.container():
            if bt_key in bt_cache:
                render_compact_state("Loading cached result...", "Backtest settings unchanged")
            else:
                render_compact_state("Calculating...", "Preparing full dataset and strategy results")
        start = time.perf_counter()
        bt_r = bt_cache.get(bt_key)
        if bt_r is None:
            bt_r = load_backtest_result(
                symbol, cfg.get("timeframe", "1h"), cfg["limit"],
                bt_cap, bt_sl, bt_tp, bt_pos,
            )
            bt_cache[bt_key] = bt_r
            trim_session_cache("bt_result_cache")
        st.session_state["bt_result"] = bt_r
        st.session_state["bt_result_key"] = bt_key
        status.empty()
        elapsed = time.perf_counter() - start
        st.success(f"Done in {elapsed:.1f}s")

    if st.session_state.get("bt_result_key") != bt_key:
        if bt_key in bt_cache:
            st.session_state["bt_result"] = bt_cache[bt_key]
            st.session_state["bt_result_key"] = bt_key
            render_compact_state("Loading cached result...", "Backtest settings unchanged")
        else:
            st.info("Configure parameters above and click **Run Backtest**.")
            return

    bt_result = st.session_state.get("bt_result")
    if not isinstance(bt_result, dict):
        render_empty_state("Backtest result unavailable.")
        return

    m = bt_result["metrics"]
    if m["total_trades"] == 0:
        st.warning("No trades generated. Try different SL/TP or more candles.")
        return

    trades = bt_result.get("trades", []) or []
    td_raw = pd.DataFrame(trades)
    eq_raw = bt_result.get("equity_curve", pd.DataFrame())
    eq = eq_raw.reset_index() if isinstance(eq_raw, pd.DataFrame) and not eq_raw.empty else pd.DataFrame(columns=["timestamp", "equity"])

    def clamp(value, low=0.0, high=100.0):
        return max(low, min(high, value))

    def finite_num(value, default=0.0):
        try:
            value = float(value)
            return value if np.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    def status_from_score(score):
        return "positive" if score >= 70 else "warning" if score >= 45 else "negative"

    def status_from_value(value):
        return "positive" if value > 0 else "negative" if value < 0 else "warning"

    def grade_from_score(score, labels=("Excellent", "Good", "Average", "Weak", "Danger")):
        if score >= 82:
            return labels[0]
        if score >= 65:
            return labels[1]
        if score >= 45:
            return labels[2]
        if score >= 25:
            return labels[3]
        return labels[4]

    total_trades = int(m.get("total_trades", 0) or 0)
    winning_trades = int(m.get("winning_trades", 0) or 0)
    losing_trades = int(m.get("losing_trades", 0) or 0)
    win_rate = finite_num(m.get("win_rate"), 0.0)
    total_return = finite_num(m.get("total_return"), 0.0)
    total_return_pct = finite_num(m.get("total_return_pct"), 0.0)
    final_capital = finite_num(m.get("final_capital"), bt_cap)
    profit_factor = finite_num(m.get("profit_factor"), 0.0)
    sharpe = finite_num(m.get("sharpe_ratio"), 0.0)
    max_dd = abs(finite_num(m.get("max_drawdown"), 0.0))
    avg_winner = finite_num(m.get("avg_win"), 0.0)
    avg_loser = finite_num(m.get("avg_loss"), 0.0)

    pnl_values = td_raw["pnl"].astype(float).tolist() if not td_raw.empty and "pnl" in td_raw else []
    pct_values = td_raw["pnl_pct"].astype(float).tolist() if not td_raw.empty and "pnl_pct" in td_raw else []
    avg_trade = float(np.mean(pnl_values)) if pnl_values else 0.0
    best_trade = max(pnl_values) if pnl_values else 0.0
    worst_trade = min(pnl_values) if pnl_values else 0.0
    largest_winner = max([p for p in pnl_values if p > 0], default=0.0)
    largest_loser = min([p for p in pnl_values if p <= 0], default=0.0)
    win_loss_ratio = winning_trades / losing_trades if losing_trades else float(winning_trades)
    gross_loss = abs(sum(p for p in pnl_values if p <= 0))
    recovery_factor = total_return / (bt_cap * max_dd / 100) if max_dd > 0 else (99.0 if total_return > 0 else 0.0)

    equity_values = eq["equity"].astype(float).tolist() if not eq.empty and "equity" in eq else []
    peak_equity = max(equity_values) if equity_values else final_capital
    current_equity = equity_values[-1] if equity_values else final_capital
    current_dd = abs(((current_equity - peak_equity) / peak_equity) * 100) if peak_equity else 0.0
    eq_returns = pd.Series(equity_values).pct_change().dropna() if len(equity_values) > 1 else pd.Series(dtype=float)
    equity_vol = float(eq_returns.std()) if len(eq_returns) else 0.0
    equity_stability = int(round(clamp(100 - max_dd * 4 - equity_vol * 4500)))
    equity_health = int(round(clamp(equity_stability * 0.45 + max(0, min(100, 50 + total_return_pct * 4)) * 0.35 + max(0, min(100, 50 + sharpe * 10)) * 0.20)))

    pf_score = clamp(profit_factor / 2.0 * 100) if profit_factor < 10 else 100
    sharpe_score = clamp(50 + sharpe * 12)
    drawdown_score = clamp(100 - max_dd * 8)
    trade_count_score = clamp(total_trades / 30 * 100)
    strategy_score = int(round(clamp(win_rate * 0.25 + pf_score * 0.25 + sharpe_score * 0.20 + drawdown_score * 0.20 + trade_count_score * 0.10)))
    health_score = int(round(clamp(win_rate * 0.28 + pf_score * 0.28 + sharpe_score * 0.22 + drawdown_score * 0.22)))
    confidence = int(round(clamp(35 + strategy_score * 0.45 + min(total_trades, 30) * 0.75)))

    verdict = "Excellent" if strategy_score >= 82 else "Good" if strategy_score >= 65 else "Average" if strategy_score >= 45 else "Poor" if strategy_score >= 25 else "Failing"
    health_label = grade_from_score(health_score, ("Excellent", "Good", "Average", "Weak", "Danger"))
    risk_level = "Low" if max_dd <= 5 and profit_factor >= 1.4 else "Moderate" if max_dd <= 12 and profit_factor >= 0.8 else "High"
    profitability_rating = "Strong" if total_return > 0 and profit_factor >= 1.3 else "Positive" if total_return > 0 else "Weak" if profit_factor >= 0.8 else "Negative"
    robustness_rating = "Robust" if total_trades >= 30 and sharpe > 0 and max_dd < 10 else "Developing" if total_trades >= 10 else "Thin Sample"
    trade_quality = "High" if avg_trade > 0 and win_loss_ratio >= 1 else "Mixed" if avg_trade >= -0.01 else "Poor"

    risk_score = int(round(clamp(drawdown_score * 0.45 + pf_score * 0.25 + max(0, min(100, 50 + total_return_pct * 4)) * 0.15 + (100 - bt_pos_pct * 1.5) * 0.15)))
    dd_severity = "Low" if max_dd <= 5 else "Moderate" if max_dd <= 12 else "High"
    capital_preservation = "Strong" if current_dd <= 3 and total_return >= 0 else "Stable" if current_dd <= 8 else "Weak"
    position_risk = "Elevated" if bt_pos_pct >= 35 else "Moderate" if bt_pos_pct >= 20 else "Controlled"
    risk_verdict = "Healthy" if risk_score >= 70 else "Caution" if risk_score >= 45 else "Danger"

    def fmt_money(value):
        return f"${value:,.2f}"

    def fmt_pct(value):
        return f"{value:+.2f}%"

    def fmt_pf(value):
        return "Inf" if value >= 99 else f"{value:.3f}"

    def bt_card(label, value, sub, status="warning", klass="bt-chip"):
        return (
            f"<div class='{klass} status-{status}'>"
            f"<span>{render_info_label(str(label))}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(sub))}</em>"
            "</div>"
        )

    st.markdown(
        "<div class='bt-command-head'>"
        "<div class='bt-command-title'>Backtest Command Center</div>"
        f"<div class='bt-command-meta'>{html.escape(symbol)} / {html.escape(str(cfg.get('timeframe', '1h')))} / {total_trades} trades</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='bt-command-card'><div class='bt-command-grid'>"
        f"<div class='bt-core status-{status_from_score(strategy_score)}'>"
        f"<div class='bt-label'>{render_info_label('Strategy Verdict')}</div>"
        f"<div class='bt-value'>{html.escape(verdict)}</div>"
        f"<div class='bt-sub'>Strategy score {strategy_score}/100 · Confidence {confidence}%</div>"
        "<div class='bt-gauge'><div class='bt-gauge-track'>"
        f"<span class='bt-gauge-marker' style='left:{strategy_score}%'></span>"
        "</div><div class='bt-gauge-scale'><span>Failing</span><span>Average</span><span>Excellent</span></div></div>"
        "</div>"
        "<div class='bt-health-grid'>"
        + bt_card("Strategy Score", f"{strategy_score}/100", verdict, status_from_score(strategy_score))
        + bt_card("Confidence", f"{confidence}%", f"Sample {total_trades} trades", status_from_score(confidence))
        + bt_card("Risk Level", risk_level, f"Max DD {max_dd:.2f}%", "negative" if risk_level == "High" else "positive" if risk_level == "Low" else "warning")
        + bt_card("Profitability", profitability_rating, f"PF {fmt_pf(profit_factor)}")
        + bt_card("Robustness", robustness_rating, f"Sharpe {sharpe:.2f}")
        + bt_card("Trade Quality", trade_quality, f"Avg {fmt_money(avg_trade)}", "positive" if trade_quality == "High" else "negative" if trade_quality == "Poor" else "warning")
        + "</div></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Performance Summary</div><div class='compact-section-meta'>Premium strategy KPIs</div></div>", unsafe_allow_html=True)
    kpis = [
        ("Net Profit", fmt_money(total_return), fmt_pct(total_return_pct), status_from_value(total_return)),
        ("Total Return %", fmt_pct(total_return_pct), "Return on capital", status_from_value(total_return_pct)),
        ("Final Capital", fmt_money(final_capital), f"Start {fmt_money(bt_cap)}", status_from_value(final_capital - bt_cap)),
        ("Win Rate", f"{win_rate:.1f}%", f"{winning_trades}W / {losing_trades}L", "positive" if win_rate >= 50 else "negative"),
        ("Profit Factor", fmt_pf(profit_factor), "Gross profit / loss", "positive" if profit_factor >= 1.3 else "warning" if profit_factor >= 0.8 else "negative"),
        ("Sharpe Ratio", f"{sharpe:.3f}", "Risk-adjusted returns", "positive" if sharpe > 0.5 else "warning" if sharpe >= 0 else "negative"),
        ("Max Drawdown", f"-{max_dd:.2f}%", "Peak-to-trough", "positive" if max_dd <= 5 else "warning" if max_dd <= 12 else "negative"),
        ("Recovery Factor", f"{recovery_factor:.2f}", "Net / drawdown", "positive" if recovery_factor >= 1 else "warning" if recovery_factor >= 0 else "negative"),
        ("Average Trade", fmt_money(avg_trade), "Mean closed PnL", status_from_value(avg_trade)),
        ("Best Trade", fmt_money(best_trade), "Largest win", status_from_value(best_trade)),
        ("Worst Trade", fmt_money(worst_trade), "Largest loss", status_from_value(worst_trade)),
        ("Total Trades", str(total_trades), "Closed positions", "positive" if total_trades >= 30 else "warning" if total_trades >= 10 else "negative"),
    ]
    st.markdown("<div class='bt-kpi-grid'>" + "".join(bt_card(*item, klass="bt-kpi") for item in kpis) + "</div>", unsafe_allow_html=True)

    diagnosis = []
    if max_dd > 10:
        diagnosis.append("High drawdown")
    elif max_dd <= 5:
        diagnosis.append("Controlled drawdown")
    if win_rate < 40:
        diagnosis.append("Low win rate")
    elif win_rate >= 55:
        diagnosis.append("Strong win rate")
    if avg_winner > abs(avg_loser) and avg_winner > 0:
        diagnosis.append("Strong average winner")
    if profit_factor < 1:
        diagnosis.append("Weak profit factor")
    elif profit_factor >= 1.5:
        diagnosis.append("Healthy profit factor")
    if total_trades < 10:
        diagnosis.append("Too few trades")
    if equity_stability >= 70:
        diagnosis.append("Stable equity curve")
    if equity_vol > 0.01 or sharpe < 0:
        diagnosis.append("Volatile returns")
    if not diagnosis:
        diagnosis.append("Balanced but inconclusive profile")
    st.markdown(
        "<div class='compact-section-head'><div class='compact-section-title'>Why did the strategy perform this way?</div>"
        "<div class='compact-section-meta'>Generated from existing backtest metrics</div></div>"
        "<div class='bt-reasons'>" + "".join(f"<span class='bt-reason-chip'>{html.escape(x)}</span>" for x in diagnosis[:8]) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Trade Analytics</div><div class='compact-section-meta'>Closed trade quality</div></div>", unsafe_allow_html=True)
    trade_cards = [
        ("Winning Trades", str(winning_trades), f"{win_rate:.1f}% win rate", "positive" if winning_trades else "warning"),
        ("Losing Trades", str(losing_trades), f"{100-win_rate:.1f}% loss rate", "negative" if losing_trades else "positive"),
        ("Win/Loss Ratio", f"{win_loss_ratio:.2f}", "Wins per loss", "positive" if win_loss_ratio >= 1 else "negative"),
        ("Average Winner", fmt_money(avg_winner), "Mean winning PnL", status_from_value(avg_winner)),
        ("Average Loser", fmt_money(avg_loser), "Mean losing PnL", status_from_value(avg_loser)),
        ("Largest Winner", fmt_money(largest_winner), "Best closed trade", status_from_value(largest_winner)),
        ("Largest Loser", fmt_money(largest_loser), "Worst closed trade", status_from_value(largest_loser)),
    ]
    st.markdown("<div class='bt-analytics-grid'>" + "".join(bt_card(*item, klass="bt-analytics-card") for item in trade_cards) + "</div>", unsafe_allow_html=True)

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Equity Analysis</div><div class='compact-section-meta'>Portfolio path and drawdown health</div></div>", unsafe_allow_html=True)
    equity_cards = [
        ("Peak Equity", fmt_money(peak_equity), "Highest curve value", "positive"),
        ("Current Equity", fmt_money(current_equity), "Latest curve value", status_from_value(current_equity - bt_cap)),
        ("Drawdown %", f"-{current_dd:.2f}%", "Current from peak", "positive" if current_dd <= 3 else "warning" if current_dd <= 8 else "negative"),
        ("Equity Stability", f"{equity_stability}/100", "Volatility adjusted", status_from_score(equity_stability)),
        ("Equity Health Score", f"{equity_health}/100", grade_from_score(equity_health), status_from_score(equity_health)),
    ]
    st.markdown("<div class='bt-analytics-grid'>" + "".join(bt_card(*item, klass="bt-analytics-card") for item in equity_cards) + "</div>", unsafe_allow_html=True)

    if len(eq):
        eq_plot = eq.copy()
        eq_plot["peak"] = eq_plot["equity"].cummax()
        eq_plot["drawdown"] = (eq_plot["equity"] - eq_plot["peak"]) / eq_plot["peak"] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_plot["timestamp"], y=eq_plot["equity"], fill="tozeroy", fillcolor="rgba(0,224,138,0.12)", line=dict(color="#00E08A", width=2), name="Equity"))
        fig.add_trace(go.Scatter(x=eq_plot["timestamp"], y=eq_plot["peak"], line=dict(color="#FFB84D", width=1, dash="dot"), name="Peak Equity"))
        fig.update_layout(height=260, title="Equity Curve", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(tickprefix="$", gridcolor="rgba(255,255,255,0.04)"), xaxis=dict(gridcolor="rgba(255,255,255,0.04)"), margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, width="stretch")

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Trade Distribution</div><div class='compact-section-meta'>PnL shape and outcomes</div></div>", unsafe_allow_html=True)
    chart_cols = st.columns(3)
    if pnl_values:
        with chart_cols[0]:
            fig_hist = px.histogram(pd.DataFrame({"PnL": pnl_values}), x="PnL", nbins=min(12, max(4, total_trades)), title="PnL Distribution Histogram")
            fig_hist.update_traces(marker_color="#00A3FF")
            fig_hist.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig_hist, width="stretch")
        with chart_cols[1]:
            fig_pie = go.Figure(go.Pie(labels=["Wins", "Losses"], values=[winning_trades, losing_trades], marker=dict(colors=["#00E08A", "#FF5C73"]), hole=0.52))
            fig_pie.update_layout(height=230, title="Win vs Loss Pie Chart", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pie, width="stretch")
        with chart_cols[2]:
            outcomes = td_raw["exit_reason"].value_counts().reset_index() if "exit_reason" in td_raw else pd.DataFrame({"exit_reason": [], "count": []})
            outcomes.columns = ["Outcome", "Count"]
            fig_out = px.bar(outcomes, x="Outcome", y="Count", title="Trade Outcome Breakdown", color="Outcome")
            fig_out.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig_out, width="stretch")

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Strategy Health</div><div class='compact-section-meta'>Win rate, profit factor, Sharpe, drawdown</div></div>", unsafe_allow_html=True)
    health_cards = [
        ("Strategy Health Score", f"{health_score}/100", health_label, status_from_score(health_score)),
        ("Health Verdict", health_label, "Excellent / Good / Average / Weak / Danger", status_from_score(health_score)),
        ("Win Rate Input", f"{win_rate:.1f}%", "Health component", "positive" if win_rate >= 50 else "negative"),
        ("Drawdown Input", f"{drawdown_score:.0f}/100", f"Max DD {max_dd:.2f}%", status_from_score(drawdown_score)),
    ]
    st.markdown("<div class='bt-analytics-grid'>" + "".join(bt_card(*item, klass="bt-analytics-card") for item in health_cards) + "</div>", unsafe_allow_html=True)

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Risk Command Center</div><div class='compact-section-meta'>Capital defense and position risk</div></div>", unsafe_allow_html=True)
    risk_cards = [
        ("Risk Score", f"{risk_score}/100", risk_verdict, status_from_score(risk_score)),
        ("Drawdown Severity", dd_severity, f"Max DD {max_dd:.2f}%", "negative" if dd_severity == "High" else "positive" if dd_severity == "Low" else "warning"),
        ("Capital Preservation", capital_preservation, f"Current DD {current_dd:.2f}%", "positive" if capital_preservation == "Strong" else "warning" if capital_preservation == "Stable" else "negative"),
        ("Position Risk", position_risk, f"Position {bt_pos_pct}%", "negative" if position_risk == "Elevated" else "warning" if position_risk == "Moderate" else "positive"),
        ("Risk Verdict", risk_verdict, risk_level, status_from_score(risk_score)),
    ]
    st.markdown("<div class='bt-analytics-grid'>" + "".join(bt_card(*item, klass="bt-analytics-card") for item in risk_cards) + "</div>", unsafe_allow_html=True)

    if trades:
        td = pd.DataFrame(trades).copy()
        td.insert(0, "Trade #", range(1, len(td) + 1))
        td["Result"] = np.where(td["pnl"].astype(float) > 0, "+ Win", "- Loss")
        td["Risk Grade"] = np.select(
            [td["pnl_pct"].astype(float) <= -bt_sl * 0.95, td["pnl_pct"].astype(float) < 0, td["pnl_pct"].astype(float) > 0],
            ["High", "Moderate", "Low"],
            default="Moderate",
        )
        entry_times = []
        bt_df = bt_result.get("df")
        buy_times = []
        if isinstance(bt_df, pd.DataFrame) and "signal" in bt_df.columns:
            buy_times = list(bt_df.index[bt_df["signal"] == SIGNAL_BUY])
        cursor = 0
        for close_ts in pd.to_datetime(td["timestamp"]):
            entry_ts = pd.NaT
            while cursor < len(buy_times) and pd.to_datetime(buy_times[cursor]) <= close_ts:
                entry_ts = pd.to_datetime(buy_times[cursor])
                cursor += 1
                break
            entry_times.append(entry_ts)
        td["Entry Time"] = entry_times
        td["Duration"] = ["N/A" if pd.isna(x) else str(pd.to_datetime(y) - x) for x, y in zip(td["Entry Time"], pd.to_datetime(td["timestamp"]))]
        td["pnl_pct"] = (td["pnl_pct"].astype(float) * 100).round(2)
        td["pnl"] = td["pnl"].astype(float).round(2)
        td["entry"] = td["entry"].astype(float).round(6)
        td["exit"] = td["exit"].astype(float).round(6)
        st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Professional Trades Table</div><div class='compact-section-meta'>Existing trade log with display-only labels</div></div>", unsafe_allow_html=True)
        render_interactive_dataframe(
            td[["Trade #", "timestamp", "Duration", "entry", "exit", "pnl", "pnl_pct", "Result", "Risk Grade", "exit_reason"]].tail(50),
            signed_columns={"pnl", "pnl_pct", "Result"},
            column_config={
                "entry": st.column_config.NumberColumn("entry", format="%.6f"),
                "exit": st.column_config.NumberColumn("exit", format="%.6f"),
                "pnl": st.column_config.NumberColumn("pnl", format="$%.2f"),
                "pnl_pct": st.column_config.NumberColumn("pnl_pct", format="%+.2f%%"),
            },
            height=560,
        )

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

    sig = signal_result.get("signal", "HOLD")
    close = float(ind["close"])
    pos = risk["position_size"]
    ts = now_str("%Y-%m-%d %H:%M:%S WIB")

    def clamp(value, low=0.0, high=100.0):
        return max(low, min(high, value))

    def safe_float(value, default=0.0):
        try:
            value = float(value)
            return value if np.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    def status_from_score(score):
        return "positive" if score >= 70 else "warning" if score >= 45 else "negative"

    def status_from_value(value):
        return "positive" if value > 0 else "negative" if value < 0 else "warning"

    def fmt_money(value):
        return f"${value:,.2f}"

    def fmt_pct(value):
        return f"{value:+.2f}%"

    def pf_card(label, value, sub, status="warning", klass="pf-chip"):
        return (
            f"<div class='{klass} status-{status}'>"
            f"<span>{render_info_label(str(label))}</span>"
            f"<strong>{html.escape(str(value))}</strong>"
            f"<em>{html.escape(str(sub))}</em>"
            "</div>"
        )

    trades_df = pd.DataFrame(st.session_state.paper_trades)
    if not trades_df.empty:
        for col in ["price", "size", "units", "SL", "TP", "conf"]:
            if col in trades_df.columns:
                trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce").fillna(0.0)
        trades_df["time_dt"] = pd.to_datetime(trades_df.get("time", pd.Series(dtype=str)).astype(str).str.replace(" WIB", "", regex=False), errors="coerce")
        trades_df["direction"] = trades_df.get("signal", "HOLD").astype(str).str.upper()
        trades_df["entry"] = trades_df.get("price", 0.0).astype(float)
        trades_df["position_value"] = trades_df.get("size", 0.0).astype(float)
        trades_df["current_price"] = close
        side = np.where(trades_df["direction"].eq("SELL"), -1.0, 1.0)
        trades_df["unrealized_pnl"] = (close - trades_df["entry"]) * trades_df.get("units", 0.0).astype(float) * side
        trades_df["unrealized_pct"] = np.where(trades_df["entry"].abs() > 0, (close - trades_df["entry"]) / trades_df["entry"] * 100 * side, 0.0)
        trades_df["realized_pnl"] = pd.to_numeric(trades_df.get("pnl", 0.0), errors="coerce").fillna(0.0)
    else:
        trades_df = pd.DataFrame(columns=["time", "symbol", "direction", "entry", "position_value", "units", "current_price", "unrealized_pnl", "unrealized_pct", "realized_pnl", "time_dt"])

    active_positions = len(trades_df)
    used_margin = float(trades_df["position_value"].sum()) if active_positions else 0.0
    available_cash = max(0.0, float(capital) - used_margin)
    unrealized_pnl = float(trades_df["unrealized_pnl"].sum()) if active_positions else 0.0
    realized_pnl = float(trades_df["realized_pnl"].sum()) if active_positions else 0.0
    portfolio_value = available_cash + used_margin + unrealized_pnl + realized_pnl
    total_pnl = unrealized_pnl + realized_pnl
    total_return_pct = (total_pnl / max(float(capital), 1.0)) * 100
    exposure_pct = used_margin / max(float(capital), 1.0) * 100
    capital_utilization = min(100.0, exposure_pct)
    winning_positions = int((trades_df["unrealized_pnl"] > 0).sum()) if active_positions else 0
    losing_positions = int((trades_df["unrealized_pnl"] < 0).sum()) if active_positions else 0
    largest_position = float(trades_df["position_value"].max()) if active_positions else 0.0
    concentration = largest_position / max(used_margin, 1.0) * 100 if used_margin else 0.0

    now_ts = pd.Timestamp.now()
    def period_pnl(days):
        if not active_positions or "time_dt" not in trades_df:
            return 0.0
        return float(trades_df.loc[trades_df["time_dt"] >= now_ts - pd.Timedelta(days=days), "unrealized_pnl"].sum())

    daily_return = period_pnl(1) / max(float(capital), 1.0) * 100
    weekly_return = period_pnl(7) / max(float(capital), 1.0) * 100
    monthly_return = period_pnl(30) / max(float(capital), 1.0) * 100

    max_exposure = 50.0
    risk_utilization = exposure_pct / max_exposure * 100 if max_exposure else 0.0
    portfolio_drawdown = abs(min(0.0, total_return_pct))
    risk_score = int(round(clamp(100 - exposure_pct * 0.65 - concentration * 0.20 - portfolio_drawdown * 5)))
    portfolio_score = int(round(clamp(45 + total_return_pct * 4 + (100 - min(exposure_pct, 100)) * 0.18 + risk_score * 0.25 + min(active_positions, 8) * 2 - concentration * 0.08)))
    risk_label = "Low" if risk_score >= 75 else "Medium" if risk_score >= 55 else "High" if risk_score >= 30 else "Extreme"
    exposure_level = "Low" if exposure_pct < 20 else "Medium" if exposure_pct < 50 else "High" if exposure_pct < 80 else "Extreme"
    verdict = "Excellent" if portfolio_score >= 82 else "Good" if portfolio_score >= 65 else "Neutral" if portfolio_score >= 45 else "Weak" if portfolio_score >= 25 else "Dangerous"

    st.markdown(
        "<div class='pf-command-head'>"
        "<div class='pf-command-title'>Portfolio Command Center</div>"
        f"<div class='pf-command-meta'>{html.escape(symbol)} / {active_positions} active positions / Paper trading</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pf-command-card'><div class='pf-command-grid'>"
        f"<div class='pf-core status-{status_from_score(portfolio_score)}'>"
        f"<div class='pf-label'>{render_info_label('Portfolio Verdict')}</div>"
        f"<div class='pf-value'>{html.escape(verdict)}</div>"
        f"<div class='pf-sub'>Score {portfolio_score}/100 · Risk {risk_label}</div>"
        "<div class='pf-gauge'><div class='pf-gauge-track'>"
        f"<span class='pf-gauge-marker' style='left:{portfolio_score}%'></span>"
        "</div><div class='pf-gauge-scale'><span>Danger</span><span>Neutral</span><span>Excellent</span></div></div>"
        "</div>"
        "<div class='pf-health-grid'>"
        + pf_card("Portfolio Health", verdict, f"Score {portfolio_score}/100", status_from_score(portfolio_score))
        + pf_card("Portfolio Score", f"{portfolio_score}/100", "Composite account quality", status_from_score(portfolio_score))
        + pf_card("Portfolio Risk", risk_label, f"Risk score {risk_score}/100", "positive" if risk_label == "Low" else "warning" if risk_label == "Medium" else "negative")
        + pf_card("Exposure Level", exposure_level, f"{exposure_pct:.1f}% deployed", "positive" if exposure_level == "Low" else "warning" if exposure_level == "Medium" else "negative")
        + pf_card("Capital Utilization", f"{capital_utilization:.1f}%", f"Used {fmt_money(used_margin)}", "positive" if capital_utilization <= 40 else "warning" if capital_utilization <= 70 else "negative")
        + pf_card("Active Positions", str(active_positions), f"{winning_positions}W / {losing_positions}L", "positive" if winning_positions >= losing_positions else "negative" if losing_positions else "warning")
        + pf_card("Unrealized PnL", fmt_money(unrealized_pnl), fmt_pct(total_return_pct), status_from_value(unrealized_pnl))
        + pf_card("Realized PnL", fmt_money(realized_pnl), "Closed ledger PnL", status_from_value(realized_pnl))
        + "</div></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Account Summary</div><div class='compact-section-meta'>Capital, cash, margin, and period returns</div></div>", unsafe_allow_html=True)
    account_cards = [
        ("Total Capital", fmt_money(capital), "Configured account", "neutral"),
        ("Available Cash", fmt_money(available_cash), "Capital not allocated", "positive" if available_cash >= capital * 0.5 else "warning" if available_cash > 0 else "negative"),
        ("Used Margin", fmt_money(used_margin), f"{exposure_pct:.1f}% exposure", "positive" if exposure_pct <= 40 else "warning" if exposure_pct <= 70 else "negative"),
        ("Portfolio Value", fmt_money(portfolio_value), "Cash + positions + PnL", status_from_value(portfolio_value - capital)),
        ("Total Return %", fmt_pct(total_return_pct), "Open + realized", status_from_value(total_return_pct)),
        ("Daily Return %", fmt_pct(daily_return), "Logged today", status_from_value(daily_return)),
        ("Weekly Return %", fmt_pct(weekly_return), "Logged 7D", status_from_value(weekly_return)),
        ("Monthly Return %", fmt_pct(monthly_return), "Logged 30D", status_from_value(monthly_return)),
    ]
    st.markdown("<div class='pf-kpi-grid'>" + "".join(pf_card(*item, klass="pf-kpi") for item in account_cards) + "</div>", unsafe_allow_html=True)

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Risk Command Center</div><div class='compact-section-meta'>Exposure, concentration, and capital defense</div></div>", unsafe_allow_html=True)
    risk_cards = [
        ("Portfolio Risk Score", f"{risk_score}/100", risk_label, status_from_score(risk_score)),
        ("Current Exposure", f"{exposure_pct:.1f}%", fmt_money(used_margin), "positive" if exposure_pct <= 40 else "warning" if exposure_pct <= 70 else "negative"),
        ("Maximum Exposure", f"{max_exposure:.0f}%", "Display risk ceiling", "neutral"),
        ("Risk Utilization", f"{risk_utilization:.1f}%", "Current / max exposure", "positive" if risk_utilization <= 60 else "warning" if risk_utilization <= 100 else "negative"),
        ("Position Concentration", f"{concentration:.1f}%", "Largest / exposure", "positive" if concentration <= 35 else "warning" if concentration <= 60 else "negative"),
        ("Largest Position", fmt_money(largest_position), "Largest open allocation", "warning" if largest_position else "neutral"),
        ("Portfolio Drawdown", f"-{portfolio_drawdown:.2f}%", "Current account drawdown", "positive" if portfolio_drawdown <= 3 else "warning" if portfolio_drawdown <= 8 else "negative"),
    ]
    st.markdown("<div class='pf-analytics-grid'>" + "".join(pf_card(*item, klass="pf-analytics-card") for item in risk_cards) + "</div>", unsafe_allow_html=True)

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Position Management Center</div><div class='compact-section-meta'>Open paper positions marked to current price</div></div>", unsafe_allow_html=True)
    position_cards = [
        ("Open Positions", str(active_positions), "Logged paper entries", "positive" if active_positions else "warning"),
        ("Winning Positions", str(winning_positions), "Positive unrealized PnL", "positive" if winning_positions else "warning"),
        ("Losing Positions", str(losing_positions), "Negative unrealized PnL", "negative" if losing_positions else "positive"),
    ]
    st.markdown("<div class='pf-analytics-grid'>" + "".join(pf_card(*item, klass="pf-analytics-card") for item in position_cards) + "</div>", unsafe_allow_html=True)
    if active_positions:
        pos_table = trades_df.copy()
        pos_table["Symbol"] = pos_table.get("symbol", symbol)
        pos_table["Direction"] = pos_table["direction"]
        pos_table["Entry"] = pos_table["entry"].round(6)
        pos_table["Current Price"] = close
        pos_table["PnL $"] = pos_table["unrealized_pnl"].round(2)
        pos_table["PnL %"] = pos_table["unrealized_pct"].round(2)
        pos_table["Position Value"] = pos_table["position_value"].round(2)
        pos_table["Risk Grade"] = np.where(pos_table["Position Value"] / max(float(capital), 1.0) * 100 >= 25, "High", np.where(pos_table["Position Value"] / max(float(capital), 1.0) * 100 >= 10, "Medium", "Low"))
        pos_table["Status"] = np.where(pos_table["PnL $"] > 0, "Profit", np.where(pos_table["PnL $"] < 0, "Loss", "Flat"))
        render_interactive_dataframe(
            pos_table[["Symbol", "Direction", "Entry", "Current Price", "PnL $", "PnL %", "Position Value", "Risk Grade", "Status"]],
            signed_columns={"Direction", "PnL $", "PnL %", "Status"},
            column_config={
                "Entry": st.column_config.NumberColumn("Entry", format="%.6f"),
                "Current Price": st.column_config.NumberColumn("Current Price", format="%.6f"),
                "PnL $": st.column_config.NumberColumn("PnL $", format="$%.2f"),
                "PnL %": st.column_config.NumberColumn("PnL %", format="%+.2f%%"),
                "Position Value": st.column_config.NumberColumn("Position Value", format="$%.2f"),
            },
            height=280,
        )
    else:
        render_empty_state("No open paper positions logged yet.")

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Trade Journal</div><div class='compact-section-meta'>Professional ledger view with period filters</div></div>", unsafe_allow_html=True)
    journal_filter = st.radio("Journal Filter", ["Today", "Week", "Month", "All"], horizontal=True, key="pf_journal_filter", label_visibility="collapsed")
    if active_positions:
        journal = trades_df.copy()
        cutoff = None
        if journal_filter == "Today":
            cutoff = now_ts - pd.Timedelta(days=1)
        elif journal_filter == "Week":
            cutoff = now_ts - pd.Timedelta(days=7)
        elif journal_filter == "Month":
            cutoff = now_ts - pd.Timedelta(days=30)
        if cutoff is not None:
            journal = journal[journal["time_dt"] >= cutoff]
        journal = journal.reset_index(drop=True)
        journal.insert(0, "Trade #", range(1, len(journal) + 1))
        journal["Date"] = journal.get("time", "")
        journal["Symbol"] = journal.get("symbol", symbol)
        journal["Direction"] = journal["direction"]
        journal["Entry"] = journal["entry"].round(6)
        journal["Exit"] = close
        journal["PnL"] = journal["unrealized_pnl"].round(2)
        journal["PnL %"] = journal["unrealized_pct"].round(2)
        journal["Duration"] = ["N/A" if pd.isna(t) else str(now_ts - t).split(".")[0] for t in journal["time_dt"]]
        journal["Exit Reason"] = "Open"
        render_interactive_dataframe(
            journal[["Trade #", "Date", "Symbol", "Direction", "Entry", "Exit", "PnL", "PnL %", "Duration", "Exit Reason"]],
            signed_columns={"Direction", "PnL", "PnL %"},
            column_config={
                "Entry": st.column_config.NumberColumn("Entry", format="%.6f"),
                "Exit": st.column_config.NumberColumn("Exit", format="%.6f"),
                "PnL": st.column_config.NumberColumn("PnL", format="$%.2f"),
                "PnL %": st.column_config.NumberColumn("PnL %", format="%+.2f%%"),
            },
            height=360,
        )

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Portfolio Analytics</div><div class='compact-section-meta'>Display-only curves from paper ledger</div></div>", unsafe_allow_html=True)
    if active_positions:
        curve = trades_df.sort_values("time_dt").copy()
        curve["cum_pnl"] = curve["unrealized_pnl"].cumsum()
        curve["equity"] = float(capital) + curve["cum_pnl"]
        curve["growth"] = (curve["equity"] / max(float(capital), 1.0) - 1) * 100
        curve["exposure"] = curve["position_value"].cumsum() / max(float(capital), 1.0) * 100
        daily = curve.set_index("time_dt")["unrealized_pnl"].resample("D").sum().reset_index().rename(columns={"unrealized_pnl": "Daily PnL"})
        c1, c2 = st.columns(2)
        with c1:
            fig_eq = go.Figure(go.Scatter(x=curve["time_dt"], y=curve["equity"], fill="tozeroy", fillcolor="rgba(0,224,138,0.12)", line=dict(color="#00E08A", width=2), name="Equity"))
            fig_eq.update_layout(height=230, title="Equity Curve", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0), yaxis=dict(tickprefix="$"))
            st.plotly_chart(fig_eq, width="stretch")
        with c2:
            fig_growth = go.Figure(go.Scatter(x=curve["time_dt"], y=curve["growth"], line=dict(color="#00A3FF", width=2), name="Growth"))
            fig_growth.update_layout(height=230, title="Portfolio Growth", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0), yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_growth, width="stretch")
        c3, c4, c5 = st.columns(3)
        with c3:
            fig_daily = px.bar(daily, x="time_dt", y="Daily PnL", title="Daily PnL")
            fig_daily.update_traces(marker_color="#FFB84D")
            fig_daily.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_daily, width="stretch")
        with c4:
            fig_cum = go.Figure(go.Scatter(x=curve["time_dt"], y=curve["cum_pnl"], line=dict(color="#00E08A" if total_pnl >= 0 else "#FF5C73", width=2), name="Cumulative PnL"))
            fig_cum.update_layout(height=220, title="Cumulative PnL", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0), yaxis=dict(tickprefix="$"))
            st.plotly_chart(fig_cum, width="stretch")
        with c5:
            fig_exp = go.Figure(go.Scatter(x=curve["time_dt"], y=curve["exposure"], fill="tozeroy", fillcolor="rgba(255,184,77,0.12)", line=dict(color="#FFB84D", width=2), name="Exposure"))
            fig_exp.update_layout(height=220, title="Exposure History", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0), yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_exp, width="stretch")
    else:
        render_empty_state("Portfolio analytics will populate after trades are logged.")

    pnl_series = trades_df["unrealized_pnl"] if active_positions else pd.Series(dtype=float)
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    win_rate = len(wins) / max(len(pnl_series), 1) * 100 if active_positions else 0.0
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    returns = pnl_series / max(float(capital), 1.0)
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0.0
    recovery_factor = total_pnl / max(portfolio_drawdown / 100 * float(capital), 1.0) if portfolio_drawdown > 0 else (99.0 if total_pnl > 0 else 0.0)
    avg_winner = float(wins.mean()) if len(wins) else 0.0
    avg_loser = float(losses.mean()) if len(losses) else 0.0
    best_trade = float(pnl_series.max()) if active_positions else 0.0
    worst_trade = float(pnl_series.min()) if active_positions else 0.0

    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Performance Diagnostics</div><div class='compact-section-meta'>Paper ledger performance quality</div></div>", unsafe_allow_html=True)
    perf_cards = [
        ("Win Rate", f"{win_rate:.1f}%", f"{len(wins)} wins / {len(losses)} losses", "positive" if win_rate >= 50 else "negative" if active_positions else "warning"),
        ("Profit Factor", "Inf" if profit_factor >= 99 else f"{profit_factor:.2f}", "Gross profit / loss", "positive" if profit_factor >= 1.3 else "warning" if profit_factor >= 0.8 else "negative"),
        ("Sharpe Ratio", f"{sharpe:.2f}", "PnL consistency proxy", "positive" if sharpe > 0.5 else "warning" if sharpe >= 0 else "negative"),
        ("Recovery Factor", "Inf" if recovery_factor >= 99 else f"{recovery_factor:.2f}", "PnL / drawdown", "positive" if recovery_factor >= 1 else "warning" if recovery_factor >= 0 else "negative"),
        ("Average Winner", fmt_money(avg_winner), "Mean positive PnL", status_from_value(avg_winner)),
        ("Average Loser", fmt_money(avg_loser), "Mean negative PnL", status_from_value(avg_loser)),
        ("Best Trade", fmt_money(best_trade), "Largest paper gain", status_from_value(best_trade)),
        ("Worst Trade", fmt_money(worst_trade), "Largest paper loss", status_from_value(worst_trade)),
    ]
    st.markdown("<div class='pf-analytics-grid'>" + "".join(pf_card(*item, klass="pf-analytics-card") for item in perf_cards) + "</div>", unsafe_allow_html=True)

    diagnostics = []
    diagnostics.append("Strong profitability" if total_pnl > 0 and profit_factor >= 1.2 else "Weak profitability" if total_pnl < 0 else "Neutral profitability")
    diagnostics.append("Controlled risk" if risk_score >= 70 else "Overexposed" if exposure_pct > 60 else "Risk needs monitoring")
    diagnostics.append("Healthy portfolio" if portfolio_score >= 65 else "Portfolio deterioration" if portfolio_score < 35 else "Portfolio mixed")
    if concentration > 50:
        diagnostics.append("Position concentration high")
    st.markdown("<div class='pf-reasons'>" + "".join(f"<span class='pf-reason-chip'>{html.escape(x)}</span>" for x in diagnostics) + "</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='dashboard-grid paper-trade-signal-grid'>"
        + render_dashboard_card("Active Signal", sig, f"Price @ {fmt_price(close, symbol)}", status=sig.lower())
        + render_dashboard_card("Capital", f"${capital:,.2f}", "Paper trading balance", status="muted")
        + render_dashboard_card("Position Size", f"${pos['position_value']:,.2f}", f"{pos['units']:.6f} units", status="positive" if sig == SIGNAL_BUY else "negative" if sig == SIGNAL_SELL else "neutral")
        + "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Signal:** {sig_badge(sig)} at `{fmt_price(close, symbol)}`", unsafe_allow_html=True)
        st.markdown(f"**Capital:** `${capital:,.2f}`")
        if sig == SIGNAL_BUY:
            if st.button("Log BUY Trade", type="primary"):
                st.session_state.paper_trades.append({
                    "time": ts, "symbol": symbol, "signal": "BUY",
                    "price": close, "size": pos["position_value"],
                    "units": pos["units"], "SL": risk["stop_loss"],
                    "TP": risk["take_profit"], "conf": signal_result["confidence"],
                })
                st.success(f"Logged BUY {pos['units']:.6f} {symbol.split('/')[0]} @ {fmt_price(close, symbol)}")
        elif sig == SIGNAL_SELL:
            if st.button("Log SELL Trade", type="secondary"):
                st.session_state.paper_trades.append({
                    "time": ts, "symbol": symbol, "signal": "SELL",
                    "price": close, "size": pos["position_value"],
                    "units": pos["units"], "SL": risk["stop_loss"],
                    "TP": risk["take_profit"], "conf": signal_result["confidence"],
                })
        else:
            st.info("Signal is HOLD - no action.")
    with col2:
        if st.session_state.paper_trades:
            if st.button("Clear Log"):
                st.session_state.paper_trades = []
                st.rerun()

    st.divider()
    st.markdown("<div class='compact-section-head'><div class='compact-section-title'>Position Sizing Pro</div><div class='compact-section-meta'>Institutional allocation calculator</div></div>", unsafe_allow_html=True)
    init_widget_from_query("rc", "pf_cap", float(capital), lambda v: qp_float("pf_cap", float(capital), 5.0, 1_000_000.0))
    init_widget_from_query("re", "pf_entry", float(close), lambda v: qp_float("pf_entry", float(close), 0.000001, 1_000_000.0))
    init_widget_from_query("rsl", "pf_sl", float(risk["stop_loss"]), lambda v: qp_float("pf_sl", float(risk["stop_loss"]), 0.000001, 1_000_000.0))
    init_widget_from_query("rtp", "pf_tp", float(risk["take_profit"]), lambda v: qp_float("pf_tp", float(risk["take_profit"]), 0.000001, 1_000_000.0))
    init_widget_from_query("rrp", "pf_risk", 1.0, lambda v: qp_float("pf_risk", 1.0, 0.1, 5.0))
    init_widget_from_query("rmp", "pf_max", 25, lambda v: qp_int("pf_max", 25, 5, 50, 5))

    r1, r2 = st.columns(2)
    with r1:
        custom_cap = st.number_input("Capital ($)", 5.0, 1_000_000.0, step=1.0, format="%.2f", key="rc")
        if custom_cap < 5:
            st.error("Minimum capital is $5.00")
            custom_cap = 5.0
        custom_entry = st.number_input("Entry Price", 0.000001, 1_000_000.0, key="re", format="%.6f")
        custom_sl = st.number_input("Stop Loss", 0.000001, 1_000_000.0, key="rsl", format="%.6f")
    with r2:
        custom_tp = st.number_input("Take Profit", 0.000001, 1_000_000.0, key="rtp", format="%.6f")
        custom_risk_pct = st.slider("Risk per Trade %", 0.1, 5.0, step=0.1, key="rrp")
        custom_maxp_pct = st.slider("Max Position %", 5, 50, step=5, key="rmp")

    qp_set("pf_cap", custom_cap)
    qp_set("pf_entry", custom_entry)
    qp_set("pf_sl", custom_sl)
    qp_set("pf_tp", custom_tp)
    qp_set("pf_risk", custom_risk_pct)
    qp_set("pf_max", custom_maxp_pct)
    custom_risk = custom_risk_pct / 100
    custom_maxp = custom_maxp_pct / 100

    sz = load_position_size_cached(custom_cap, custom_entry, custom_sl, custom_risk, custom_maxp)
    rr_c = abs(custom_tp - custom_entry) / abs(custom_entry - custom_sl) if abs(custom_entry - custom_sl) > 0 else 0
    max_loss = sz["risk_amount"]
    expected_profit = abs(custom_tp - custom_entry) * sz["units"]
    suggested_alloc = sz["position_value"] / max(custom_cap, 1.0) * 100
    sizing_cards = [
        ("Units", f"{sz['units']:.6f}", "Suggested position size", "positive"),
        ("Position Value", fmt_money(sz["position_value"]), "Suggested capital allocation", "positive" if suggested_alloc <= custom_maxp_pct else "warning"),
        ("Risk Amount", fmt_money(sz["risk_amount"]), f"{custom_risk_pct:.1f}% risk", "warning"),
        ("Max Loss", fmt_money(max_loss), "At stop loss", "negative" if max_loss > 0 else "neutral"),
        ("Expected Profit", fmt_money(expected_profit), "At take profit", "positive"),
        ("Reward/Risk", f"1:{rr_c:.2f}", "TP / SL distance", "positive" if rr_c >= 1.5 else "warning"),
        ("Suggested Position Size", f"{sz['units']:.6f}", "Units from risk model", "positive"),
        ("Suggested Allocation", f"{suggested_alloc:.1f}%", f"Cap {custom_maxp_pct}%", "positive" if suggested_alloc <= custom_maxp_pct else "negative"),
    ]
    st.markdown("<div class='pf-kpi-grid'>" + "".join(pf_card(*item, klass="pf-kpi") for item in sizing_cards) + "</div>", unsafe_allow_html=True)


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
        height=125, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=22, b=0), font={"color": "white"},
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

    render_app_header(cfg["theme"])
    fg_value = fg.get("value", 50) if isinstance(fg, dict) else 50
    header_signal_result = generate_signal(ind, 0.0, advanced=adv, fg_value=fg_value)
    trade_decision = compute_trade_verdict(symbol, ind, adv, header_signal_result, cfg)
    render_trade_decision_card(trade_decision)
    active_tab = render_persistent_tabs()
    render_tab_density_css(active_tab)

    if active_tab == "overview":
        render_overview(tickers, cg_data, watchlist_symbols, ind_map, signal_map, fg)

    elif active_tab == "technical":
        render_technical(df, ind, adv, symbol, sr, cfg, fg)

    elif active_tab == "smart_money":
        if "smc" not in st.session_state:
            with st.spinner("Loading Smart Money data…"):
                st.session_state.smc = load_smc(symbol, timeframe, cfg["limit"])
        render_smart_money(df, st.session_state.smc, symbol)

    elif active_tab == "order_book":
        if "ob" not in st.session_state or st.session_state.get("ob_symbol") != symbol:
            with st.spinner("Fetching order book…"):
                st.session_state.ob = load_orderbook(symbol)
                st.session_state.ob_symbol = symbol
        elif st.session_state.ob.get("source") == "live":
            st.session_state.ob = {**st.session_state.ob, "source": "cached"}
        render_orderbook(st.session_state.ob, symbol)

    elif active_tab == "multi_tf":
        if "mtf" not in st.session_state or st.session_state.get("mtf_symbol") != symbol or st.session_state.get("mtf_timeframe") != timeframe:
            with st.spinner("Loading multi-timeframe analysis…"):
                st.session_state.mtf = load_mtf_data(symbol, timeframe)
                st.session_state.mtf_symbol = symbol
                st.session_state.mtf_timeframe = timeframe
        render_mtf(st.session_state.mtf, symbol, cfg["theme"])

    elif active_tab == "ai_signals":
        if not is_tab_rendered("ai_signals"):
            mark_tab_rendered("ai_signals")

        with st.spinner("Loading sentiment…"):
            sentiment = load_news_sentiment(symbol)
        sentiment_score = sentiment.get("score", 0.0)
        fg_val = fg.get("value", 50)
        smc_used = st.session_state.get("smc", smc)
        ob_used = st.session_state.get("ob", ob)
        mtf_used = st.session_state.get("mtf", mtf)
        mtf_overall = mtf_used.get("_overall", {}) if mtf_used else {}

        signal_result = generate_signal(
            ind, sentiment_score,
            advanced=adv, smc=smc_used, mtf_overall=mtf_overall,
            orderbook=ob_used, fg_value=fg_val,
        )
        risk = load_portfolio_risk(
            cfg["capital"], ind["close"],
            ind.get("atr", ind["close"] * 0.02),
            signal_result["confidence"],
            cfg["risk_tolerance"], cfg["risk_reward"],
        )
        ml_status = st.empty()
        with ml_status.container():
            if has_cached_ml_prediction(df, symbol, timeframe, cfg["limit"]):
                render_compact_state("Loading cached result…", "ML prediction")
            else:
                render_compact_state("Calculating…", "ML prediction")
        ml_result = get_cached_ml_prediction(df, symbol, timeframe, cfg["limit"])
        ml_status.empty()

        st.session_state["signal_result"] = signal_result
        render_ai_signals(ind, adv, smc_used, mtf_used, ob_used, sentiment, fg,
                  signal_result, ml_result, risk, symbol, cfg)

    elif active_tab == "news":
        if not is_tab_rendered("news"):
            mark_tab_rendered("news")
        render_news_tab(symbol)

    elif active_tab == "backtest":
        if not is_tab_rendered("backtest"):
            mark_tab_rendered("backtest")
        render_backtest(df, cfg, symbol)

    elif active_tab == "portfolio":
        if not is_tab_rendered("portfolio"):
            mark_tab_rendered("portfolio")

        portfolio_status = st.empty()
        if "signal_result" in st.session_state:
            signal_result = st.session_state.get("signal_result")
            with portfolio_status.container():
                render_compact_state("Loading cached result…", "Portfolio risk")
        else:
            with portfolio_status.container():
                render_compact_state("Refreshing signal…", "Portfolio inputs")
            sentiment = load_news_sentiment(symbol)
            sentiment_score = sentiment.get("score", 0.0)
            signal_result = generate_signal(ind, sentiment_score, advanced=adv, smc=st.session_state.get("smc", smc))
            st.session_state["signal_result"] = signal_result

        risk = load_portfolio_risk(
            cfg["capital"], ind["close"], ind.get("atr", ind["close"]*0.02),
            signal_result["confidence"], cfg["risk_tolerance"], cfg["risk_reward"],
        )
        portfolio_status.empty()
        render_portfolio(signal_result, ind, risk, symbol, cfg["capital"])

    st.caption(
        f"Binance (CCXT) · CoinGecko · alternative.me (Fear&Greed) · "
        f"{now_str('%Y-%m-%d %H:%M WIB')} · Paper Trading Only"
    )


if __name__ == "__main__":
    main()
