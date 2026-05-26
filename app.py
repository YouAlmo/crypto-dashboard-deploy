import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.market_data import fetch_ohlcv
from src.analysis.indicators import add_all_indicators
from src.analysis.signals import generate_signal
from src.analysis.mtf import fetch_mtf_analysis
from src.data.orderbook import fetch_order_book
from src.core.ai_engine import generate_ai_analysis

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SuperSignal Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# THEME
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0b1220;
    color: #e5e7eb;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}

.stSelectbox label,
.stSlider label,
.stCheckbox label {
    color: #f3f4f6 !important;
    font-weight: 600;
}

.metric-card {
    background: #111827;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #1f2937;
}

.signal-buy {
    color: #22c55e;
    font-weight: bold;
}

.signal-sell {
    color: #ef4444;
    font-weight: bold;
}

.signal-hold {
    color: #f59e0b;
    font-weight: bold;
}

.big-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sub-title {
    color: #9ca3af;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "symbol" not in st.session_state:
    st.session_state.symbol = "BTC/USDT"

if "timeframe" not in st.session_state:
    st.session_state.timeframe = "1h"

if "limit" not in st.session_state:
    st.session_state.limit = 250

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("# 🚀 SuperSignal Pro")

symbols = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "SUI/USDT",
    "AR/USDT",
    "ZEC/USDT",
    "FIL/USDT",
    "ALGO/USDT",
    "PYTH/USDT",
]

timeframes = [
    "2m",
    "3m",
    "5m",
    "10m",
    "15m",
    "1h",
    "4h",
    "1d",
    "1w",
    "1M",
]

# ---------- SESSION DEFAULTS ----------

defaults = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "limit": 250,
    "auto_refresh": False,
    "refresh_speed": 5,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- SYMBOL ----------

st.session_state.symbol = st.sidebar.selectbox(
    "Select Coin",
    symbols,
    key="symbol_select",
    index=symbols.index(st.session_state.symbol),
)

# ---------- TIMEFRAME ----------

st.session_state.timeframe = st.sidebar.selectbox(
    "Timeframe",
    timeframes,
    key="timeframe_select",
    index=timeframes.index(st.session_state.timeframe),
)

# ---------- LIMIT ----------

st.session_state.limit = st.sidebar.slider(
    "Candles",
    100,
    1000,
    value=st.session_state.limit,
    step=50,
    key="candles_slider",
)

# ---------- AUTO REFRESH ----------

st.session_state.auto_refresh = st.sidebar.checkbox(
    "Auto Refresh",
    value=st.session_state.auto_refresh,
    key="refresh_checkbox",
)

# ---------- REFRESH SPEED ----------

st.session_state.refresh_speed = st.sidebar.slider(
    "Refresh Speed",
    3,
    60,
    value=st.session_state.refresh_speed,
    key="refresh_speed_slider",
)

# =========================================================
# LOAD DATA
# =========================================================

with st.spinner("Loading market data..."):

    df = fetch_ohlcv(
        st.session_state.symbol,
        timeframe=st.session_state.timeframe,
        limit=st.session_state.limit
    )

    df = add_all_indicators(df)

    signal = generate_signal(df)

    mtf = fetch_mtf_analysis(st.session_state.symbol)

    orderbook = fetch_order_book(st.session_state.symbol)

    ai_analysis = generate_ai_analysis(df)

# =========================================================
# HEADER
# =========================================================

last = df.iloc[-1]

price = float(last["close"])

st.markdown(
    f"""
<div class="big-title">
🚀 SuperSignal Pro
</div>

<div class="sub-title">
Institutional Trading Intelligence Platform
</div>
""",
    unsafe_allow_html=True
)

# =========================================================
# TOP METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Price",
        f"${price:,.2f}"
    )

with m2:
    st.metric(
        "RSI",
        round(last["rsi"], 2)
    )

with m3:
    st.metric(
        "MACD",
        round(last["macd"], 4)
    )

with m4:

    verdict = signal["signal"]

    color_class = {
        "BUY": "signal-buy",
        "SELL": "signal-sell",
    }.get(verdict, "signal-hold")

    st.markdown(
        f"""
<div class="{color_class}">
AI SIGNAL: {verdict}
</div>
""",
        unsafe_allow_html=True
    )

# =========================================================
# PROFESSIONAL TRADING TERMINAL LAYOUT
# =========================================================

# ---------------------------------------------------------
# ROW 1
# ---------------------------------------------------------

left_col, right_col = st.columns([3.5, 1.2])

# =========================================================
# LEFT — MAIN CHART
# =========================================================

with left_col:

    st.markdown("## 📈 Market Structure")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22]
    )

    # ---------------- CANDLESTICK ----------------

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price"
        ),
        row=1,
        col=1
    )

    # ---------------- EMA 9 ----------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema_9"],
            name="EMA 9",
            line=dict(width=1.8)
        ),
        row=1,
        col=1
    )

    # ---------------- EMA 21 ----------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema_21"],
            name="EMA 21",
            line=dict(width=2)
        ),
        row=1,
        col=1
    )

    # ---------------- EMA 50 ----------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema_50"],
            name="EMA 50",
            line=dict(width=2.3)
        ),
        row=1,
        col=1
    )

    # ---------------- VOLUME ----------------

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            opacity=0.4
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=820,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# RIGHT — AI PANEL
# =========================================================

with right_col:

    signal = ai_analysis.get("signal", "HOLD")
    confidence = ai_analysis.get("confidence", 0)
    summary = ai_analysis.get("summary", "")
    reasons = ai_analysis.get("reasons", [])

    signal_color = {
        "BUY": "#22c55e",
        "SELL": "#ef4444",
        "HOLD": "#f59e0b",
    }.get(signal, "#ffffff")

    st.markdown(f"""
    <div style="
        background:#111827;
        padding:24px;
        border-radius:18px;
        border:1px solid #1f2937;
        margin-bottom:18px;
    ">
        <div style="
            font-size:32px;
            font-weight:800;
            color:{signal_color};
            margin-bottom:10px;
        ">
        {signal}
        </div>

        <div style="
            color:#d1d5db;
            font-size:18px;
        ">
        Confidence: {confidence}%
        </div>

    </div>
    """, unsafe_allow_html=True)

    # ---------------- SUMMARY ----------------

    st.markdown("### 🧠 AI Summary")

    st.markdown(f"""
    <div style="
        background:#111827;
        padding:18px;
        border-radius:16px;
        border:1px solid #1f2937;
        color:#d1d5db;
        margin-bottom:18px;
        line-height:1.7;
    ">
    {summary}
    </div>
    """, unsafe_allow_html=True)

    # ---------------- REASONS ----------------

    st.markdown("### 📊 AI Factors")

    for reason in reasons:

        st.markdown(f"""
        <div style="
            background:#111827;
            padding:14px;
            border-radius:14px;
            border:1px solid #1f2937;
            margin-bottom:12px;
            color:#e5e7eb;
        ">
        {reason}
        </div>
        """, unsafe_allow_html=True)

    # ---------------- RSI ----------------

    st.markdown("### ⚡ RSI")

    st.progress(
        min(
            max(last["rsi"] / 100, 0),
            1
        )
    )

# ---------------------------------------------------------
# ROW 2
# ---------------------------------------------------------

scanner_col, mtf_col = st.columns([2.4, 1.3])

# =========================================================
# MARKET SCANNER
# =========================================================

with scanner_col:

    st.markdown("## 🌍 Market Scanner")

    scanner_rows = []

    for tf, data in mtf.items():

        if tf.startswith("_"):
            continue

        scanner_rows.append({
            "TF": tf,
            "Signal": data.get("verdict", "N/A"),
            "Score": data.get("score", 0),
        })

    scanner_df = pd.DataFrame(scanner_rows)

    st.dataframe(
        scanner_df,
        use_container_width=True,
        height=320
    )

# =========================================================
# ORDERBOOK
# =========================================================

with mtf_col:

    st.markdown("## 📚 Order Flow")

    st.metric(
        "Bid Volume",
        round(orderbook["total_bid_vol"], 2)
    )

    st.metric(
        "Ask Volume",
        round(orderbook["total_ask_vol"], 2)
    )

    st.metric(
        "Spread %",
        round(orderbook["spread_pct"], 4)
    )

    st.metric(
        "Imbalance",
        round(orderbook["imbalance"], 4)
    )

# ---------------------------------------------------------
# ROW 3
# ---------------------------------------------------------

st.markdown("## 📊 Market Metrics")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Price",
        f"${price:,.2f}"
    )

with m2:
    st.metric(
        "RSI",
        round(last["rsi"], 2)
    )

with m3:
    st.metric(
        "MACD",
        round(last["macd"], 4)
    )

with m4:
    st.metric(
        "Volume Ratio",
        round(last.get("volume_ratio", 1), 2)
    )



    # =====================================================
    # HEADER CARD
    # =====================================================

    st.markdown(f"""
    <div style="
        background:{bg_color};
        padding:30px;
        border-radius:22px;
        border:1px solid rgba(255,255,255,0.08);
        margin-bottom:25px;
    ">

    <div style="
        font-size:42px;
        font-weight:800;
        color:{signal_color};
        margin-bottom:10px;
    ">
    {signal}
    </div>

    <div style="
        font-size:18px;
        color:#d1d5db;
        margin-bottom:8px;
    ">
    AI Confidence: {confidence}%
    </div>

    <div style="
        font-size:16px;
        color:#9ca3af;
    ">
    Score: {score}
    </div>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # SUMMARY
    # =====================================================

    st.markdown("## 🧠 AI Market Summary")

    st.markdown(f"""
    <div style="
        background:#111827;
        padding:22px;
        border-radius:18px;
        border:1px solid #1f2937;
        color:#e5e7eb;
        font-size:17px;
        line-height:1.7;
        margin-bottom:25px;
    ">
    {summary}
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # REASONS
    # =====================================================

    st.markdown("## 📊 AI Decision Factors")

    if len(reasons) == 0:

        st.warning("No AI reasons available.")

    else:

        cols = st.columns(2)

        for i, reason in enumerate(reasons):

            with cols[i % 2]:

                st.markdown(f"""
                <div style="
                    background:#111827;
                    padding:18px;
                    border-radius:16px;
                    border:1px solid #1f2937;
                    margin-bottom:15px;
                ">
                    <div style="
                        color:#e5e7eb;
                        font-size:16px;
                        font-weight:600;
                    ">
                    {reason}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # =====================================================
    # SIGNAL STRENGTH BAR
    # =====================================================

    st.markdown("## ⚡ Signal Strength")

    st.progress(min(confidence / 100, 1.0))

    # =====================================================
    # MARKET BIAS
    # =====================================================

    st.markdown("## 🌍 Market Bias")

    if signal == "BUY":

        bias = "Bullish Momentum"

    elif signal == "SELL":

        bias = "Bearish Momentum"

    else:

        bias = "Neutral / Sideways"

    st.markdown(f"""
    <div style="
        background:#111827;
        padding:20px;
        border-radius:18px;
        border:1px solid #1f2937;
        color:{signal_color};
        font-size:18px;
        font-weight:700;
    ">
    {bias}
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# ORDERBOOK TAB
# =========================================================

with tab4:

    st.subheader("Live Orderbook")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Bid Volume",
            round(orderbook["total_bid_vol"], 2)
        )

    with c2:
        st.metric(
            "Ask Volume",
            round(orderbook["total_ask_vol"], 2)
        )

    with c3:
        st.metric(
            "Spread %",
            round(orderbook["spread_pct"], 4)
        )

    ob_df = pd.DataFrame(orderbook["bids"])

    st.dataframe(
        ob_df,
        use_container_width=True
    )

# =========================================================
# AUTO REFRESH
# =========================================================

if st.session_state.auto_refresh:

    import time

    time.sleep(refresh_seconds)

    st.rerun()