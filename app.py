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

st.session_state.symbol = st.sidebar.selectbox(
    "Select Coin",
    symbols,
    index=symbols.index(st.session_state.symbol)
)

st.session_state.timeframe = st.sidebar.selectbox(
    "Timeframe",
    timeframes,
    index=timeframes.index(st.session_state.timeframe)
)

st.session_state.limit = st.sidebar.slider(
    "Candles",
    100,
    1000,
    st.session_state.limit,
    50
)

st.session_state.auto_refresh = st.sidebar.checkbox(
    "Auto Refresh",
    value=st.session_state.auto_refresh
)

refresh_seconds = st.sidebar.slider(
    "Refresh Speed",
    3,
    60,
    5
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
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Chart",
    "🤖 AI Analysis",
    "🌍 Multi-Timeframe",
    "📚 Orderbook"
])

# =========================================================
# CHART TAB
# =========================================================

with tab1:

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )

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

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema_9"],
            name="EMA 9"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema_21"],
            name="EMA 21"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema_50"],
            name="EMA 50"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume"
        ),
        row=2,
        col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=800,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# AI TAB
# =========================================================

with tab2:

    st.subheader("AI Trading Intelligence")

    st.json(ai_analysis)

# =========================================================
# MTF TAB
# =========================================================

with tab3:

    st.subheader("Multi Timeframe Analysis")

    rows = []

    for tf, data in mtf.items():

        if tf.startswith("_"):
            continue

        rows.append({
            "Timeframe": tf,
            "Signal": data["signal"],
            "Score": data["score"],
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True
    )

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