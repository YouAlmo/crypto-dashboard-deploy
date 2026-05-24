import requests
import streamlit as st
from typing import Dict, List, Tuple

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

STABLECOINS = {
    "usdt", "usdc", "dai", "busd", "tusd", "usdp", "frax", "lusd",
    "usdd", "fdusd", "pyusd", "eurs", "gusd", "usdn", "susd", "usd+",
    "crvusd", "mkr", "gho",
}

ALWAYS_INCLUDE_CG_IDS = {"arweave"}

FALLBACK_SYMBOLS = [
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


def _parse_coin(coin: dict) -> dict:
    return {
        "name": coin.get("name", ""),
        "market_cap": float(coin.get("market_cap") or 0),
        "total_volume": float(coin.get("total_volume") or 0),
        "current_price": float(coin.get("current_price") or 0),
        "circulating_supply": float(coin.get("circulating_supply") or 0),
        "max_supply": float(coin.get("max_supply") or 0),
        "price_change_percentage_24h": float(
            coin.get("price_change_percentage_24h") or 0
        ),
        "ath": float(coin.get("ath") or 0),
        "ath_change_percentage": float(
            coin.get("ath_change_percentage") or 0
        ),
        "market_cap_rank": int(coin.get("market_cap_rank") or 9999),
    }


def _fallback_market_data() -> Dict:
    out: Dict[str, dict] = {}
    for sym in FALLBACK_SYMBOLS:
        out[sym] = {
            "name": sym.split("/")[0],
            "market_cap": 0, "total_volume": 0, "current_price": 0,
            "circulating_supply": 0, "max_supply": None,
            "price_change_percentage_24h": 0,
            "ath": 0, "ath_change_percentage": 0, "market_cap_rank": 9999,
        }
    return out


# @st.cache_data(ttl=5, show_spinner=False)
def fetch_top20_markets() -> Tuple[List[str], Dict]:
    """
    Fetches top ~30 coins by market cap from CoinGecko, filters stablecoins,
    keeps top 20, always includes AR/USDT.

    Returns:
        symbols  — list of "XXX/USDT" strings, ordered by CoinGecko market-cap rank
        cg_data  — dict keyed by "XXX/USDT" symbol with market data
    """
    url = (
        f"{COINGECKO_BASE}/coins/markets"
        "?vs_currency=usd&order=market_cap_desc&per_page=25page=1"
        "&sparkline=false&price_change_percentage=24h"
    )
    try:
        resp = requests.get(url, timeout=12, headers={"Accept": "application/json"})
        if resp.status_code != 200:
            return FALLBACK_SYMBOLS[:], _fallback_market_data()

        raw: list = resp.json()

        print("RAW API DATA:")
        print(raw[0])
        non_stable = [
            c for c in raw
            if c.get("symbol", "").lower() not in STABLECOINS
        ]

        cg_data: Dict[str, dict] = {}
        symbols: List[str] = []

        allowed = set(FALLBACK_SYMBOLS)
        for coin in non_stable:
            sym = f"{coin['symbol'].upper()}/USDT"
            if sym in allowed:
                formatted = f"{coin['symbol'].upper()}/USDT"
                print(coin)
                cg_data[formatted] = {
                    "name": coin.get("name", ""),
                    "market_cap": float(coin.get("market_cap") or 0),
                    "total_volume": float(coin.get("total_volume") or 0),
                    "current_price": float(coin.get("current_price") or 0),
                    "circulating_supply": float(coin.get("circulating_supply") or 0),
                    "max_supply": float(coin.get("max_supply") or 0),
                    "price_change_percentage_24h": float(
                        coin.get("price_change_percentage_24h") or 0
                    ),
                    "ath": float(coin.get("ath") or 0),
                    "ath_change_percentage": float(
                        coin.get("ath_change_percentage") or 0
                    ),
                    "market_cap_rank": int(coin.get("market_cap_rank") or 9999),
                }

                symbols.append(formatted)



        seen: set = set()
        deduped: List[str] = []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                deduped.append(s)

        return deduped, cg_data

    except Exception:
        return FALLBACK_SYMBOLS[:], _fallback_market_data()


def fetch_coingecko_markets() -> Dict:
    """Backward-compat wrapper — returns market data dict keyed by USDT symbol."""
    _, data = fetch_top20_markets()
    return data


def get_coin_info(symbol: str) -> Dict:
    _, data = fetch_top20_markets()
    return data.get(symbol, {})


def format_large_number(n: float) -> str:
    if n >= 1e12:
        return f"${n / 1e12:.2f}T"
    if n >= 1e9:
        return f"${n / 1e9:.2f}B"
    if n >= 1e6:
        return f"${n / 1e6:.2f}M"
    return f"${n:,.0f}"


def format_supply(n: float, ticker: str = "") -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B {ticker}"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M {ticker}"
    return f"{n:,.0f} {ticker}"
