import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="CryptoAI Terminal",
    layout="wide",
    page_icon="🚀"
)

# =========================================
# AUTO REFRESH
# =========================================

st_autorefresh(interval=2000, key="refresh")

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050816;
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 20px;
}

.section-title {
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 10px;
    color: white;
}

[data-testid="stMetric"] {
    background-color: #0b1220;
    border: 1px solid #1e293b;
    padding: 15px;
    border-radius: 14px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================

st.markdown(
    '<p class="main-title">🚀 CryptoAI Terminal</p>',
    unsafe_allow_html=True
)

# =========================================
# FEAR & GREED INDEX
# =========================================

@st.cache_data(ttl=5)
def get_fear_greed():

    try:

        url = "https://api.alternative.me/fng/"

        response = requests.get(url, timeout=10)

        data = response.json()

        value = data["data"][0]["value"]

        label = data["data"][0]["value_classification"]

        return value, label

    except:

        return "N/A", "N/A"


fear_value, fear_label = get_fear_greed()

# =========================================
# GET MARKET DATA
# =========================================

@st.cache_data(ttl=5)
def get_market_data():

    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd"
        "&order=market_cap_desc"
        "&per_page=250"
        "&page=1"
        "&sparkline=false"
    )

    try:

        response = requests.get(url, timeout=20)

        data = response.json()

        rows = []

        for coin in data:

            rows.append({

                "Pair": f"{coin['symbol'].upper()}/USDT",

                "Name": coin["name"],

                "Price": coin["current_price"],

                "24h %": round(
                    coin["price_change_percentage_24h"] or 0,
                    2
                ),

                "Market Cap": coin["market_cap"],

                "Volume 24h": coin["total_volume"]

            })

        return pd.DataFrame(rows)

    except Exception as e:

        st.error(f"API Error: {e}")

        return pd.DataFrame()

# =========================================
# LOAD DATA
# =========================================

df = get_market_data()

# =========================================
# TOP METRICS
# =========================================

top7 = df.head(6)

metric_cols = st.columns(7)

for i in range(len(top7)):

    row = top7.iloc[i]

    with metric_cols[i]:

        st.metric(
            label=row["Pair"],
            value=f"${row['Price']:,.2f}",
            delta=f"{row['24h %']}%"
        )

# Fear & Greed
with metric_cols[6]:

    st.metric(
        label="Fear & Greed",
        value=fear_value,
        delta=fear_label
    )

# =========================================
# SEARCH
# =========================================

st.markdown(
    '<p class="section-title">🔎 Search Coin</p>',
    unsafe_allow_html=True
)

search = st.text_input(
    "Search by symbol or name",
    ""
)

# =========================================
# FILTER DATA
# =========================================

if search:

    df = df[
        df["Pair"].str.contains(search.upper())
        |
        df["Name"].str.contains(search, case=False)
    ]

# =========================================
# FORMAT DATA
# =========================================

display_df = df.copy()

def format_large_number(x):

    if x >= 1_000_000_000:
        return f"${x/1_000_000_000:.2f}B"

    if x >= 1_000_000:
        return f"${x/1_000_000:.2f}M"

    return f"${x:,.0f}"

display_df["Price"] = display_df["Price"].apply(
    lambda x: f"${x:,.6f}"
)

display_df["Market Cap"] = display_df["Market Cap"].apply(
    format_large_number
)

display_df["Volume 24h"] = display_df["Volume 24h"].apply(
    format_large_number
)

display_df["24h %"] = display_df["24h %"].apply(
    lambda x: f"{x}%"
)

# =========================================
# MARKET SCANNER
# =========================================

st.markdown(
    '<p class="section-title">📋 Market Scanner</p>',
    unsafe_allow_html=True
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# =========================================
# TOP GAINERS
# =========================================

st.markdown(
    '<p class="section-title">🚀 Top Gainers</p>',
    unsafe_allow_html=True
)

gainers = df.sort_values(
    by="24h %",
    ascending=False
).head(5)

st.dataframe(
    gainers,
    use_container_width=True,
    hide_index=True
)

# =========================================
# TOP LOSERS
# =========================================

st.markdown(
    '<p class="section-title">🔻 Top Losers</p>',
    unsafe_allow_html=True
)

losers = df.sort_values(
    by="24h %",
    ascending=True
).head(5)

st.dataframe(
    losers,
    use_container_width=True,
    hide_index=True
)

# =========================================
# FOOTER
# =========================================

st.caption(
    "Live crypto market data powered by CoinGecko API • Auto refresh every 30 seconds"
)