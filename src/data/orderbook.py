"""
Live order book fetching from Binance via CCXT.
Computes bid/ask imbalance, buy/sell pressure, spread, and cumulative delta.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict

try:
    import ccxt
    _CCXT_OK = True
except ImportError:
    _CCXT_OK = False


@st.cache_resource(show_spinner=False)
def _get_exchange():
    if not _CCXT_OK:
        return None
    return ccxt.binance({
        "enableRateLimit": True,
        "timeout": 6000,
        "options": {"defaultType": "spot"},
    })


@st.cache_data(ttl=5, show_spinner=False)
def fetch_order_book(symbol: str, depth: int = 20) -> Dict:
    """Fetch live order book and compute analytics."""
    try:
        ex = _get_exchange()
        if ex is None:
            raise RuntimeError("ccxt not available")

        ob = ex.fetch_order_book(symbol, limit=depth)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        if not bids or not asks:
            return _synthetic_orderbook(symbol)

        bids = bids[:depth]
        asks = asks[:depth]

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread   = best_ask - best_bid
        spread_pct = spread / best_ask * 100 if best_ask else 0

        total_bid_vol = sum(b[1] for b in bids)
        total_ask_vol = sum(a[1] for a in asks)
        total_vol = total_bid_vol + total_ask_vol

        buy_pct  = total_bid_vol / total_vol * 100 if total_vol else 50
        sell_pct = total_ask_vol / total_vol * 100 if total_vol else 50
        imbalance = (total_bid_vol - total_ask_vol) / total_vol if total_vol else 0

        bid_df = pd.DataFrame(bids, columns=["price", "size"])
        ask_df = pd.DataFrame(asks, columns=["price", "size"])
        bid_df["cumulative"] = bid_df["size"].cumsum()
        ask_df["cumulative"] = ask_df["size"].cumsum()
        bid_df["side"] = "bid"
        ask_df["side"] = "ask"
        bid_df["value"] = bid_df["price"] * bid_df["size"]
        ask_df["value"] = ask_df["price"] * ask_df["size"]

        cum_delta = total_bid_vol - total_ask_vol

        return {
            "bids":          bid_df.to_dict("records"),
            "asks":          ask_df.to_dict("records"),
            "best_bid":      best_bid,
            "best_ask":      best_ask,
            "spread":        spread,
            "spread_pct":    spread_pct,
            "total_bid_vol": total_bid_vol,
            "total_ask_vol": total_ask_vol,
            "buy_pct":       buy_pct,
            "sell_pct":      sell_pct,
            "imbalance":     imbalance,
            "cum_delta":     cum_delta,
            "mid_price":     (best_bid + best_ask) / 2,
            "source":        "live",
        }
    except Exception:
        return _synthetic_orderbook(symbol)


def _synthetic_orderbook(symbol: str) -> Dict:
    """Synthetic order book for fallback."""
    base_prices = {
        "BTC/USDT": 65000, "ETH/USDT": 3500, "BNB/USDT": 580,
        "SOL/USDT": 180,   "XRP/USDT": 0.52, "AR/USDT": 22,
    }
    mid = base_prices.get(symbol, 100.0)
    spread = mid * 0.0003

    bids, asks = [], []
    for i in range(20):
        price_b = mid - (i + 1) * spread * 0.5
        price_a = mid + (i + 1) * spread * 0.5
        size_b  = np.random.uniform(0.5, 5.0)
        size_a  = np.random.uniform(0.5, 5.0)
        bids.append({"price": round(price_b, 6), "size": round(size_b, 4),
                     "cumulative": 0, "side": "bid", "value": price_b * size_b})
        asks.append({"price": round(price_a, 6), "size": round(size_a, 4),
                     "cumulative": 0, "side": "ask", "value": price_a * size_a})

    cum = 0
    for b in bids:
        cum += b["size"]
        b["cumulative"] = round(cum, 4)
    cum = 0
    for a in asks:
        cum += a["size"]
        a["cumulative"] = round(cum, 4)

    total_b = sum(b["size"] for b in bids)
    total_a = sum(a["size"] for a in asks)
    total   = total_b + total_a

    return {
        "bids":          bids,
        "asks":          asks,
        "best_bid":      bids[0]["price"],
        "best_ask":      asks[0]["price"],
        "spread":        spread,
        "spread_pct":    spread / mid * 100,
        "total_bid_vol": total_b,
        "total_ask_vol": total_a,
        "buy_pct":       total_b / total * 100,
        "sell_pct":      total_a / total * 100,
        "imbalance":     (total_b - total_a) / total,
        "cum_delta":     total_b - total_a,
        "mid_price":     mid,
        "source":        "synthetic",
    }
