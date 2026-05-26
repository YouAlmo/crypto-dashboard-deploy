import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from streamlit_autorefresh import st_autorefresh

from src.data.market_data import (
    SYMBOLS,
    TIMEFRAMES,
    fetch_ohlcv,
)

from src.analysis.indicators import (
    add_all_indicators,
)

from src.analysis.signals import (
    generate_signal,
)

from src.analysis.mtf import (
    fetch_mtf_analysis,
)

from src.core.ai_engine import (
    run_ai_analysis,
    rank_opportunities,
)

from src.core.websocket_manager import (
    start_websocket,
    get_live_prices,
    websocket_alive,
)

from src.data.coingecko import (
    fetch_top20_markets,
    format_large_number,
)

from src.data.orderbook import (
    fetch_order_book,
)


# ─────────────────────────────────────────────────────────────
# Streamlit Config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SuperSignal Pro",
    page_icon="🚀",
    layout="wide",
)

st_autorefresh(
    interval=3000,
    key="supersignal-refresh",
)


# ─────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .stMetric {
        background: #161b22;
        padding: 12px;
        border-radius: 12px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Start WebSocket
# ─────────────────────────────────────────────────────────────

if not websocket_alive():

    start_websocket(SYMBOLS)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

st.sidebar.title("⚡ SuperSignal Pro")

symbol = st.sidebar.selectbox(
    "Select Coin",
    SYMBOLS,
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    list(TIMEFRAMES.keys()),
    index=7,
)

limit = st.sidebar.slider(
    "Candles",
    100,
    500,
    250,
)

show_orderbook = st.sidebar.checkbox(
    "Show Orderbook",
    value=True,
)

show_ai = st.sidebar.checkbox(
    "Show AI Analysis",
    value=True,
)

show_mtf = st.sidebar.checkbox(
    "Show MTF Analysis",
    value=True,
)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────

st.title("🚀 SuperSignal Pro")

st.caption(
    "Institutional AI Trading Intelligence Platform"
)


# ─────────────────────────────────────────────────────────────
# Load Market Data
# ─────────────────────────────────────────────────────────────

df = fetch_ohlcv(
    symbol=symbol,
    timeframe=timeframe,
    limit=limit,
)

df = add_all_indicators(df)

signal = generate_signal(df)

live_prices = get_live_prices()

live = live_prices.get(symbol, {})

last_price = live.get(
    "last",
    df["close"].iloc[-1],
)

change_pct = live.get(
    "change_percent",
    0,
)


# ─────────────────────────────────────────────────────────────
# Top Metrics
# ─────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Price",
        f"${last_price:,.4f}",
        f"{change_pct:.2f}%",
    )

with c2:

    st.metric(
        "Signal",
        signal["signal"],
        f"{signal['confidence']}%",
    )

with c3:

    st.metric(
        "Trend",
        signal["trend"],
        signal["strength"],
    )

with c4:

    st.metric(
        "Risk",
        signal["risk"],
    )


# ─────────────────────────────────────────────────────────────
# Candlestick Chart
# ─────────────────────────────────────────────────────────────

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["ema_9"],
        name="EMA 9",
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["ema_21"],
        name="EMA 21",
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["ema_50"],
        name="EMA 50",
    )
)

fig.update_layout(
    height=650,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ─────────────────────────────────────────────────────────────
# AI Analysis
# ─────────────────────────────────────────────────────────────

if show_ai:

    st.subheader("🧠 AI Analysis")

    ai = run_ai_analysis(
        symbol,
        df,
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        st.metric(
            "AI Score",
            ai["ai_score"],
        )

    with a2:

        st.metric(
            "Opportunity",
            ai["opportunity"],
        )

    with a3:

        st.metric(
            "Regime",
            ai["market_regime"],
        )

    with a4:

        st.metric(
            "Verdict",
            ai["verdict"],
        )

    st.progress(
        ai["ai_score"] / 100
    )


# ─────────────────────────────────────────────────────────────
# MTF Analysis
# ─────────────────────────────────────────────────────────────

if show_mtf:

    st.subheader("📊 Multi-Timeframe Analysis")

    mtf = fetch_mtf_analysis(symbol)

    rows = []

    for tf, data in mtf.items():

        if tf.startswith("_"):
            continue

        rows.append({

            "TF": data["label"],

            "Signal": data["signal"],

            "Score": data["score"],

            "Confidence": data["confidence"],

            "Trend": data["trend"],

            "RSI": round(data["rsi"], 2),

            "ADX": round(data["adx"], 2),

            "Momentum": round(
                data["momentum"],
                2,
            ),
        })

    mtf_df = pd.DataFrame(rows)

    st.dataframe(
        mtf_df,
        use_container_width=True,
    )

    overall = mtf["_overall"]

    st.success(
        f"""
        Overall Bias: {overall['market_bias']}
        | Alignment: {overall['alignment']}
        | Confidence: {overall['confidence']}%
        """
    )


# ─────────────────────────────────────────────────────────────
# Market Scanner
# ─────────────────────────────────────────────────────────────

st.subheader("🔥 AI Opportunity Scanner")

scanner_results = []

for sym in SYMBOLS:

    try:

        sdf = fetch_ohlcv(
            sym,
            timeframe="1h",
            limit=200,
        )

        sdf = add_all_indicators(sdf)

        ai = run_ai_analysis(
            sym,
            sdf,
        )

        scanner_results.append(ai)

    except Exception as e:

        print(e)

scanner_results = rank_opportunities(
    scanner_results
)

scanner_rows = []

for row in scanner_results:

    scanner_rows.append({

        "Symbol": row["symbol"],

        "AI Score": row["ai_score"],

        "Verdict": row["verdict"],

        "Opportunity": row["opportunity"],

        "Regime": row["market_regime"],
    })

scanner_df = pd.DataFrame(scanner_rows)

st.dataframe(
    scanner_df,
    use_container_width=True,
)


# ─────────────────────────────────────────────────────────────
# Market Cap Table
# ─────────────────────────────────────────────────────────────

st.subheader("🌍 Market Overview")

symbols, cg = fetch_top20_markets()

market_rows = []

for sym in symbols:

    coin = cg.get(sym, {})

    market_rows.append({

        "Coin": sym,

        "Market Cap": format_large_number(
            coin.get("market_cap", 0)
        ),

        "Volume": format_large_number(
            coin.get("total_volume", 0)
        ),

        "24H %": round(
            coin.get(
                "price_change_percentage_24h",
                0,
            ),
            2,
        ),
    })

market_df = pd.DataFrame(market_rows)

st.dataframe(
    market_df,
    use_container_width=True,
)


# ─────────────────────────────────────────────────────────────
# Orderbook
# ─────────────────────────────────────────────────────────────

if show_orderbook:

    st.subheader("📚 Live Orderbook")

    ob = fetch_order_book(symbol)

    o1, o2, o3 = st.columns(3)

    with o1:

        st.metric(
            "Bid Pressure",
            f"{ob['buy_pct']:.2f}%",
        )

    with o2:

        st.metric(
            "Ask Pressure",
            f"{ob['sell_pct']:.2f}%",
        )

    with o3:

        st.metric(
            "Spread %",
            f"{ob['spread_pct']:.4f}",
        )


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────

st.caption(
    "SuperSignal Pro • Real-Time AI Trading Terminal"
)