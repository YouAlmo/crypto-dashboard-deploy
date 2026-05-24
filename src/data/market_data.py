import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional


SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "TRX/USDT", "LINK/USDT", "AVAX/USDT",
    "SUI/USDT", "XLM/USDT", "HBAR/USDT", "TON/USDT", "SHIB/USDT",
    "DOT/USDT", "LTC/USDT", "BCH/USDT", "UNI/USDT", "APT/USDT", "AR/USDT",
]

TIMEFRAMES = {
    "1m":  "1 minute",
    "3m":  "3 minutes",
    "5m":  "5 minutes",
    "15m": "15 minutes",
    "30m": "30 minutes",
    "1h":  "1 hour",
    "2h":  "2 hours",
    "4h":  "4 hours",
    "6h":  "6 hours",
    "8h":  "8 hours",
    "12h": "12 hours",
    "1d":  "1 day",
    "3d":  "3 days",
    "1w":  "1 week",
    "1M":  "1 month",
}


def get_exchange():
    exchanges = [
        ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }),
        ccxt.bybit({
            "enableRateLimit": True,
        }),
        ccxt.okx({
            "enableRateLimit": True,
        }),
    ]

    for ex in exchanges:
        try:
            ex.load_markets()
            print(f"Connected to {ex.id}")
            return ex
        except Exception as e:
            print(f"{ex.id} failed: {e}")

    raise Exception("No exchange available")


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500,
    exchange: Optional[ccxt.Exchange] = None,
) -> pd.DataFrame:

    if exchange is None:
        exchange = get_exchange()

    try:
        raw = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )

        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        df.set_index("timestamp", inplace=True)

        df = df.astype(float)

        df.sort_index(inplace=True)

        return df

    except Exception as e:
        print(f"OHLCV Error ({symbol}): {e}")
        return pd.DataFrame()


def fetch_ticker(symbol: str, exchange: Optional[ccxt.Exchange] = None) -> dict:
    if exchange is None:
        exchange = get_exchange()
    try:
        ticker = exchange.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "last": ticker.get("last", 0),
            "bid": ticker.get("bid", 0),
            "ask": ticker.get("ask", 0),
            "change": ticker.get("change", 0),
            "percentage": ticker.get("percentage", 0),
            "high": ticker.get("high", 0),
            "low": ticker.get("low", 0),
            "volume": ticker.get("baseVolume", 0),
            "quoteVolume": ticker.get("quoteVolume", 0),
            "timestamp": datetime.now(),
        }
    except Exception as e:
        print(f"Ticker Error: {e}")

        return {
            "symbol": symbol,
            "last": None,
            "bid": None,
            "ask": None,
            "change": None,
            "percentage": None,
            "high": None,
            "low": None,
            "volume": None,
            "quoteVolume": None,
            "timestamp": datetime.now(),
        }


def fetch_all_tickers(exchange: Optional[ccxt.Exchange] = None) -> dict:
    if exchange is None:
        exchange = get_exchange()
    return {s: fetch_ticker(s, exchange) for s in SYMBOLS}


def fetch_tickers_for(symbols: List[str], exchange: Optional[ccxt.Exchange] = None) -> dict:
    """Fetch live tickers for an arbitrary list of symbols."""
    if exchange is None:
        exchange = get_exchange()
    return {s: fetch_ticker(s, exchange) for s in symbols}


def _generate_synthetic_data(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    seed_map = {s: i + 42 for i, s in enumerate(SYMBOLS)}
    base_price_map = {
        "BTC/USDT": 65_000.0, "ETH/USDT": 3_500.0, "BNB/USDT": 580.0,
        "SOL/USDT": 180.0,   "XRP/USDT": 0.52,     "ADA/USDT": 0.45,
        "DOGE/USDT": 0.16,   "TRX/USDT": 0.12,     "LINK/USDT": 14.0,
        "AVAX/USDT": 36.0,   "SUI/USDT": 2.1,      "XLM/USDT": 0.11,
        "HBAR/USDT": 0.085,  "TON/USDT": 5.5,      "SHIB/USDT": 0.0000245,
        "DOT/USDT": 6.8,     "LTC/USDT": 82.0,     "BCH/USDT": 450.0,
        "UNI/USDT": 7.5,     "APT/USDT": 7.2,      "AR/USDT": 22.0,
    }
    rng = np.random.default_rng(seed_map.get(symbol, hash(symbol) % 1000))
    base = base_price_map.get(symbol, 1.0)

    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min",
                "1h": "1h", "4h": "4h", "1d": "1D"}
    freq = freq_map.get(timeframe, "1h")
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    idx = pd.date_range(end=end, periods=limit, freq=freq)

    returns = rng.normal(0.0002, 0.02, size=limit)
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.015, size=limit))
    low = close * (1 - rng.uniform(0, 0.015, size=limit))
    open_ = np.roll(close, 1)
    open_[0] = base
    volume = rng.uniform(1e6, 5e7, size=limit)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _synthetic_ticker(symbol: str) -> dict:
    base_map = {
        "BTC/USDT": 65_000.0, "ETH/USDT": 3_500.0, "BNB/USDT": 580.0,
        "SOL/USDT": 180.0,    "XRP/USDT": 0.52,    "ADA/USDT": 0.45,
        "DOGE/USDT": 0.16,    "TRX/USDT": 0.12,    "LINK/USDT": 14.0,
        "AVAX/USDT": 36.0,    "SUI/USDT": 2.1,     "XLM/USDT": 0.11,
        "HBAR/USDT": 0.085,   "TON/USDT": 5.5,     "SHIB/USDT": 0.0000245,
        "DOT/USDT": 6.8,      "LTC/USDT": 82.0,    "BCH/USDT": 450.0,
        "UNI/USDT": 7.5,      "APT/USDT": 7.2,     "AR/USDT": 22.0,
    }
    price = base_map.get(symbol, 1.0)
    return {
        "symbol": symbol,
        "last": price,
        "bid": price * 0.9998,
        "ask": price * 1.0002,
        "change": price * 0.015,
        "percentage": 1.5,
        "high": price * 1.03,
        "low": price * 0.97,
        "volume": 25_000.0,
        "quoteVolume": price * 25_000.0,
        "timestamp": datetime.now(),
    }
