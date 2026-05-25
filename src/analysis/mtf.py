import concurrent.futures
import numpy as np
import pandas as pd
import streamlit as st

from typing import Dict, List

from src.data.market_data import fetch_ohlcv
from src.analysis.indicators import (
    add_all_indicators,
)
from src.analysis.signals import (
    generate_signal,
)


# ─────────────────────────────────────────────────────────────
# Supported Timeframes
# ─────────────────────────────────────────────────────────────

MTF_TIMEFRAMES: List[str] = [
    "1m",
    "2m",
    "3m",
    "5m",
    "10m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
    "1w",
    "1M",
]


# ─────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────

MTF_LABELS: Dict[str, str] = {
    "1m": "1 Min",
    "2m": "2 Min",
    "3m": "3 Min",
    "5m": "5 Min",
    "10m": "10 Min",
    "15m": "15 Min",
    "30m": "30 Min",
    "1h": "1 Hour",
    "4h": "4 Hour",
    "1d": "Daily",
    "1w": "Weekly",
    "1M": "Monthly",
}


# ─────────────────────────────────────────────────────────────
# Timeframe Weights
# ─────────────────────────────────────────────────────────────

TIMEFRAME_WEIGHTS = {
    "1m": 1,
    "2m": 1,
    "3m": 1,
    "5m": 2,
    "10m": 2,
    "15m": 3,
    "30m": 4,
    "1h": 5,
    "4h": 8,
    "1d": 12,
    "1w": 18,
    "1M": 25,
}


# ─────────────────────────────────────────────────────────────
# Analyze One Timeframe
# ─────────────────────────────────────────────────────────────

def _analyze_single_tf(
    symbol: str,
    timeframe: str,
) -> Dict:

    try:

        limit = _dynamic_limit(timeframe)

        df = fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

        if df is None or df.empty:
            raise ValueError("empty dataframe")

        df = add_all_indicators(df)

        signal = generate_signal(df)

        last = df.iloc[-1]

        result = {

            "timeframe": timeframe,

            "label": MTF_LABELS.get(
                timeframe,
                timeframe,
            ),

            "signal": signal["signal"],

            "score": signal["score"],

            "confidence": signal["confidence"],

            "strength": signal["strength"],

            "trend": signal["trend"],

            "color": signal["color"],

            "risk": signal["risk"],

            "weight": TIMEFRAME_WEIGHTS.get(
                timeframe,
                1,
            ),

            "close": float(
                last.get("close", 0)
            ),

            "rsi": float(
                last.get("rsi", 50)
            ),

            "adx": float(
                last.get("adx", 0)
            ),

            "volume_ratio": float(
                last.get("volume_ratio", 1)
            ),

            "momentum": float(
                last.get("momentum_pct", 0)
            ),

            "ema_9": float(
                last.get("ema_9", 0)
            ),

            "ema_21": float(
                last.get("ema_21", 0)
            ),

            "ema_50": float(
                last.get("ema_50", 0)
            ),

            "ema_200": float(
                last.get("ema_200", 0)
            ),

            "reasons": signal["reasons"],
        }

        return result

    except Exception as e:

        print(
            f"MTF Error [{symbol} {timeframe}] -> {e}"
        )

        return {

            "timeframe": timeframe,

            "label": timeframe,

            "signal": "N/A",

            "score": 0,

            "confidence": 0,

            "strength": "UNKNOWN",

            "trend": "Neutral",

            "color": "#555555",

            "risk": "UNKNOWN",

            "weight": 1,

            "close": 0,

            "rsi": 50,

            "adx": 0,

            "volume_ratio": 1,

            "momentum": 0,

            "ema_9": 0,

            "ema_21": 0,

            "ema_50": 0,

            "ema_200": 0,

            "reasons": [],
        }


# ─────────────────────────────────────────────────────────────
# Dynamic Candle Limits
# ─────────────────────────────────────────────────────────────

def _dynamic_limit(
    timeframe: str,
) -> int:

    limits = {

        "1m": 120,
        "2m": 120,
        "3m": 120,
        "5m": 150,
        "10m": 150,
        "15m": 180,
        "30m": 180,
        "1h": 220,
        "4h": 250,
        "1d": 300,
        "1w": 200,
        "1M": 120,
    }

    return limits.get(
        timeframe,
        200,
    )


# ─────────────────────────────────────────────────────────────
# Main MTF Engine
# ─────────────────────────────────────────────────────────────

@st.cache_data(
    ttl=20,
    show_spinner=False,
)
def fetch_mtf_analysis(
    symbol: str,
) -> Dict:

    results = {}

    # ─────────────────────────────────────
    # Parallel Processing
    # ─────────────────────────────────────

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=12
    ) as executor:

        futures = {

            executor.submit(
                _analyze_single_tf,
                symbol,
                tf,
            ): tf

            for tf in MTF_TIMEFRAMES
        }

        for future in concurrent.futures.as_completed(
            futures
        ):

            tf = futures[future]

            try:

                results[tf] = future.result()

            except Exception:

                results[tf] = {

                    "timeframe": tf,

                    "signal": "N/A",

                    "score": 0,

                    "confidence": 0,

                    "color": "#555555",
                }

    # ─────────────────────────────────────
    # Overall Weighted Analysis
    # ─────────────────────────────────────

    overall = _calculate_overall_bias(
        results
    )

    results["_overall"] = overall

    return results


# ─────────────────────────────────────────────────────────────
# Overall Market Bias
# ─────────────────────────────────────────────────────────────

def _calculate_overall_bias(
    results: Dict,
) -> Dict:

    weighted_score = 0

    total_weight = 0

    bullish_count = 0

    bearish_count = 0

    total_confidence = 0

    for tf, data in results.items():

        if tf.startswith("_"):
            continue

        score = data.get("score", 0)

        confidence = data.get(
            "confidence",
            0,
        )

        weight = data.get(
            "weight",
            1,
        )

        weighted_score += (
            score * weight
        )

        total_weight += weight

        total_confidence += confidence

        if score > 0:
            bullish_count += 1

        elif score < 0:
            bearish_count += 1

    if total_weight == 0:

        return {

            "signal": "N/A",

            "score": 0,

            "confidence": 0,

            "color": "#555555",

            "market_bias": "Neutral",
        }

    normalized_score = (
        weighted_score / total_weight
    )

    avg_confidence = (
        total_confidence
        / max(len(results), 1)
    )

    # ─────────────────────────────────────
    # Final Market Bias
    # ─────────────────────────────────────

    if normalized_score >= 40:

        signal = "STRONG BUY"

        color = "#00c853"

        market_bias = "Strong Bullish"

    elif normalized_score >= 15:

        signal = "BUY"

        color = "#2ecc71"

        market_bias = "Bullish"

    elif normalized_score <= -40:

        signal = "STRONG SELL"

        color = "#d50000"

        market_bias = "Strong Bearish"

    elif normalized_score <= -15:

        signal = "SELL"

        color = "#ff5252"

        market_bias = "Bearish"

    else:

        signal = "HOLD"

        color = "#f39c12"

        market_bias = "Neutral"

    alignment = _calculate_alignment(
        bullish_count,
        bearish_count,
    )

    return {

        "signal": signal,

        "score": round(
            normalized_score,
            2,
        ),

        "confidence": int(
            avg_confidence
        ),

        "color": color,

        "market_bias": market_bias,

        "bullish_count": bullish_count,

        "bearish_count": bearish_count,

        "alignment": alignment,
    }


# ─────────────────────────────────────────────────────────────
# Alignment Quality
# ─────────────────────────────────────────────────────────────

def _calculate_alignment(
    bullish: int,
    bearish: int,
) -> str:

    total = bullish + bearish

    if total == 0:
        return "UNKNOWN"

    dominant = max(
        bullish,
        bearish,
    )

    ratio = dominant / total

    if ratio >= 0.9:
        return "EXTREME ALIGNMENT"

    if ratio >= 0.75:
        return "VERY STRONG"

    if ratio >= 0.6:
        return "STRONG"

    if ratio >= 0.5:
        return "MODERATE"

    return "WEAK"