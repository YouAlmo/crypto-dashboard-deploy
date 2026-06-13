import requests
from typing import Dict, List, Tuple

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

STABLECOINS = {
    "usdt", "usdc", "dai", "busd", "tusd", "usdp", "frax",
    "lusd", "usdd", "fdusd", "pyusd", "eurs", "gusd",
    "usdn", "susd", "usd+", "crvusd", "mkr", "gho",
}

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
        "image": coin.get("image", ""),
        "market_cap": float(coin.get("market_cap") or 0),
        "total_volume": float(coin.get("total_volume") or 0),
        "current_price": float(coin.get("current_price") or 0),
        "circulating_supply": float(
            coin.get("circulating_supply") or 0
        ),
        "max_supply": float(
            coin.get("max_supply") or 0
        ) if coin.get("max_supply") else None,
        "price_change_percentage_24h": float(
            coin.get("price_change_percentage_24h") or 0
        ),
        "ath": float(coin.get("ath") or 0),
        "ath_change_percentage": float(
            coin.get("ath_change_percentage") or 0
        ),
        "market_cap_rank": int(
            coin.get("market_cap_rank") or 9999
        ),
    }


def _fallback_market_data() -> Dict[str, dict]:

    data = {}

    for sym in FALLBACK_SYMBOLS:
        data[sym] = {
            "name": sym.split("/")[0],
            "image": "",
            "market_cap": 0,
            "total_volume": 0,
            "current_price": 0,
            "circulating_supply": 0,
            "max_supply": None,
            "price_change_percentage_24h": 0,
            "ath": 0,
            "ath_change_percentage": 0,
            "market_cap_rank": 9999,
        }

    return data


def fetch_top20_markets() -> Tuple[List[str], Dict]:

    url = (
        f"{COINGECKO_BASE}/coins/markets"
        "?vs_currency=usd"
        "&order=market_cap_desc"
        "&per_page=250"
        "&page=1"
        "&sparkline=false"
        "&price_change_percentage=24h"
    )

    try:

        response = requests.get(
            url,
            timeout=15,
            headers={
                "Accept": "application/json"
            }
        )

        if response.status_code != 200:
            return FALLBACK_SYMBOLS[:], _fallback_market_data()

        raw = response.json()

        non_stable = [
            c for c in raw
            if c.get("symbol", "").lower() not in STABLECOINS
        ]

        market_lookup = {}

        for coin in non_stable:

            symbol = str(
                coin.get("symbol", "")
            ).upper()

            if not symbol:
                continue

            pair = f"{symbol}/USDT"

            if pair not in market_lookup:
                market_lookup[pair] = coin

        # معالجة العملات الخاصة
        for coin in non_stable:

            coin_id = str(
                coin.get("id", "")
            ).lower()

            if coin_id == "arweave":
                market_lookup["AR/USDT"] = coin

            elif coin_id == "pyth-network":
                market_lookup["PYTH/USDT"] = coin

        cg_data: Dict[str, dict] = {}
        symbols: List[str] = []

        for pair in FALLBACK_SYMBOLS:

            coin = market_lookup.get(pair)

            if coin:
                cg_data[pair] = _parse_coin(coin)

            else:
                cg_data[pair] = {
                    "name": pair.split("/")[0],
                    "image": "",
                    "market_cap": 0,
                    "total_volume": 0,
                    "current_price": 0,
                    "circulating_supply": 0,
                    "max_supply": None,
                    "price_change_percentage_24h": 0,
                    "ath": 0,
                    "ath_change_percentage": 0,
                    "market_cap_rank": 9999,
                }

            symbols.append(pair)

        return symbols, cg_data

    except Exception:
        return FALLBACK_SYMBOLS[:], _fallback_market_data()


def fetch_coingecko_markets() -> Dict:
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