import ccxt
import pandas as pd
import numpy as np

from datetime import datetime
from typing import List, Optional, Dict

from src.core.exchange_manager import get_exchange


# ─────────────────────────────────────────────────────────────
# Symbols
# ─────────────────────────────────────────────────────────────

SYMBOLS = [
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


# ─────────────────────────────────────────────────────────────
# Supported Timeframes
# ─────────────────────────────────────────────────────────────

TIMEFRAMES = {
    "1m": "1 Minute",
    "2m": "2 Minutes",
    "3m": "3 Minutes",
    "5m": "5 Minutes",
    "10m": "10 Minutes",
    "15m": "15 Minutes",
    "30m": "30 Minutes",
    "1h": "1 Hour",
    "2h": "2 Hours",
    "4h": "4 Hours",
    "6h": "6 Hours",
    "8h": "8 Hours",
    "12h": "12 Hours",
    "1d": "1 Day",
    "3d": "3 Days",
    "1w": "1 Week",
    "1M": "1 Month",
}


# ─────────────────────────────────────────────────────────────
# Fetch OHLCV
# ─────────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 300,
    exchange: Optional[ccxt.Exchange] = None,
) -> pd.DataFrame:

    if exchange is None:
        exchange = get_exchange()

    try:

        raw = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
        )

        if not raw:
            return _generate_synthetic_data(
                symbol,
                timeframe,
                limit,
            )

        df = pd.DataFrame(
            raw,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
        )

        df.set_index("timestamp", inplace=True)

        df = df.astype(float)

        df.sort_index(inplace=True)

        return df

    except Exception as e:

        print(f"OHLCV Error [{symbol} {timeframe}] -> {e}")

        return _generate_synthetic_data(
            symbol,
            timeframe,
            limit,
        )


# ─────────────────────────────────────────────────────────────
# Fetch Single Ticker
# ─────────────────────────────────────────────────────────────

def fetch_ticker(
    symbol: str,
    exchange: Optional[ccxt.Exchange] = None,
) -> Dict:

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

        print(f"Ticker Error [{symbol}] -> {e}")

        return _synthetic_ticker(symbol)


# ─────────────────────────────────────────────────────────────
# Fetch Multiple Tickers
# ─────────────────────────────────────────────────────────────

def fetch_tickers_for(
    symbols: List[str],
    exchange: Optional[ccxt.Exchange] = None,
) -> Dict:

    if exchange is None:
        exchange = get_exchange()

    results = {}

    for symbol in symbols:

        try:

            results[symbol] = fetch_ticker(
                symbol,
                exchange,
            )

        except Exception:

            results[symbol] = _synthetic_ticker(symbol)

    return results


def fetch_all_tickers(
    exchange: Optional[ccxt.Exchange] = None,
) -> Dict:

    return fetch_tickers_for(
        SYMBOLS,
        exchange,
    )


# ─────────────────────────────────────────────────────────────
# Synthetic OHLCV Fallback
# ─────────────────────────────────────────────────────────────

def _generate_synthetic_data(
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:

    seed_map = {
        s: i + 100
        for i, s in enumerate(SYMBOLS)
    }

    base_prices = {
        "BTC/USDT": 65000,
        "ETH/USDT": 3500,
        "SOL/USDT": 180,
        "XRP/USDT": 0.52,
        "BNB/USDT": 580,
        "DOGE/USDT": 0.16,
        "ADA/USDT": 0.45,
        "LINK/USDT": 14,
        "AVAX/USDT": 36,
        "SUI/USDT": 2.1,
        "AR/USDT": 22,
        "ZEC/USDT": 35,
        "FIL/USDT": 5,
        "ALGO/USDT": 0.18,
        "PYTH/USDT": 0.55,
    }

    rng = np.random.default_rng(
        seed_map.get(symbol, 999)
    )

    base = base_prices.get(symbol, 1.0)

    freq_map = {
        "1m": "1min",
        "2m": "2min",
        "3m": "3min",
        "5m": "5min",
        "10m": "10min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "1d": "1D",
        "1w": "1W",
    }

    freq = freq_map.get(timeframe, "1h")

    idx = pd.date_range(
        end=datetime.now(),
        periods=limit,
        freq=freq,
    )

    returns = rng.normal(
        0.0002,
        0.02,
        size=limit,
    )

    close = base * np.exp(
        np.cumsum(returns)
    )

    high = close * (
        1 + rng.uniform(0, 0.015, size=limit)
    )

    low = close * (
        1 - rng.uniform(0, 0.015, size=limit)
    )

    open_ = np.roll(close, 1)

    open_[0] = base

    volume = rng.uniform(
        100000,
        50000000,
        size=limit,
    )

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────
# Synthetic Ticker
# ─────────────────────────────────────────────────────────────

def _synthetic_ticker(symbol: str) -> Dict:

    base_prices = {
        "BTC/USDT": 65000,
        "ETH/USDT": 3500,
        "SOL/USDT": 180,
        "XRP/USDT": 0.52,
        "BNB/USDT": 580,
        "DOGE/USDT": 0.16,
        "ADA/USDT": 0.45,
        "LINK/USDT": 14,
        "AVAX/USDT": 36,
        "SUI/USDT": 2.1,
        "AR/USDT": 22,
        "ZEC/USDT": 35,
        "FIL/USDT": 5,
        "ALGO/USDT": 0.18,
        "PYTH/USDT": 0.55,
    }

    price = base_prices.get(symbol, 1)

    return {
        "symbol": symbol,
        "last": price,
        "bid": price * 0.999,
        "ask": price * 1.001,
        "change": 0,
        "percentage": 0,
        "high": price * 1.02,
        "low": price * 0.98,
        "volume": 0,
        "quoteVolume": 0,
        "timestamp": datetime.now(),
    }