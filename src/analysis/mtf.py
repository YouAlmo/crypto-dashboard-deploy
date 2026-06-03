"""
Multi-timeframe analysis.
For a given symbol, computes RSI, MACD trend, EMA alignment, and Supertrend
on 5m / 15m / 1h / 4h / 1d / 1w and returns an alignment matrix.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List

from src.data.market_data import fetch_ohlcv
from src.analysis.indicators import add_all_indicators


MTF_TIMEFRAMES: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
MTF_LABELS: Dict[str, str] = {
    "1m": "1 Min", "5m": "5 Min", "15m": "15 Min",
    "1h": "1H", "4h": "4H", "1d": "Daily",
}


def _classify_timeframe(ind: dict) -> dict:
    """Score a single timeframe based on trend, RSI, EMA alignment, and MACD."""
    score = 0
    details = {}

    rsi = ind.get("rsi", 50)
    if rsi < 30:
        rsi_score = 2
        details["rsi"] = "Oversold"
    elif rsi < 45:
        rsi_score = 1
        details["rsi"] = "Leaning Bull"
    elif rsi > 70:
        rsi_score = -2
        details["rsi"] = "Overbought"
    elif rsi > 55:
        rsi_score = -1
        details["rsi"] = "Leaning Bear"
    else:
        rsi_score = 0
        details["rsi"] = "Neutral"
    score += rsi_score

    macd = ind.get("macd", 0)
    macd_sig = ind.get("macd_signal", 0)
    macd_score = 1 if macd > macd_sig else -1
    details["macd"] = "Bullish" if macd_score > 0 else "Bearish"
    score += macd_score

    close = ind.get("close", 0)
    ema_50 = ind.get("ema_50", close)
    ema_200 = ind.get("ema_200", close)
    if close > ema_50 > ema_200:
        ema_score = 2
        details["ema"] = "Full Bull"
    elif close > ema_50:
        ema_score = 1
        details["ema"] = "Above EMA50"
    elif close < ema_50 < ema_200:
        ema_score = -2
        details["ema"] = "Full Bear"
    else:
        ema_score = -1
        details["ema"] = "Below EMA50"
    score += ema_score

    ema_9 = ind.get("ema_9", close)
    ema_21 = ind.get("ema_21", close)
    ema_9_21_score = 1 if ema_9 > ema_21 else -1
    details["ema_9_21"] = "Bullish" if ema_9_21_score > 0 else "Bearish"
    score += ema_9_21_score

    if ema_9 > ema_21 > ema_50 > ema_200:
        trend = "Strong Bull"
    elif close > ema_50 and ema_9 > ema_21:
        trend = "Bull"
    elif ema_9 < ema_21 < ema_50 < ema_200:
        trend = "Strong Bear"
    elif close < ema_50 and ema_9 < ema_21:
        trend = "Bear"
    else:
        trend = "Neutral"
    details["trend"] = trend

    momentum_score = score
    normalized_momentum = int(((momentum_score + 6) / 12) * 100)

    if score >= 5:
        verdict = "Strong Buy"
        color = "#1a7f37"
    elif score >= 2:
        verdict = "Buy"
        color = "#2ecc71"
    elif score <= -5:
        verdict = "Strong Sell"
        color = "#8b0000"
    elif score <= -2:
        verdict = "Sell"
        color = "#e74c3c"
    else:
        verdict = "Hold"
        color = "#f39c12"

    confidence = min(1.0, max(0.55, 0.55 + (abs(score) / 6) * 0.4))
    return {
        "score": score,
        "verdict": verdict,
        "signal": verdict,
        "color": color,
        "details": details,
        "momentum": normalized_momentum,
        "confidence": round(confidence, 2),
    }


@st.cache_data(ttl=5, show_spinner=False)
def fetch_mtf_analysis(symbol: str) -> Dict:
    """
    Fetch OHLCV on all MTF timeframes, compute indicators,
    and return multi-timeframe alignment details.
    """
    results = {}
    for tf in MTF_TIMEFRAMES:
        try:
            limit = 150 if tf == "1m" else 250
            df = fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = add_all_indicators(df)
            if len(df) == 0:
                raise ValueError("empty df")

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last

            def _f(col, default=0.0):
                v = last.get(col, default)
                return float(v) if pd.notna(v) else default

            ind = {
                "close":       _f("close"),
                "rsi":         _f("rsi", 50),
                "macd":        _f("macd"),
                "macd_signal": _f("macd_signal"),
                "ema_9":       _f("ema_9"),
                "ema_21":      _f("ema_21"),
                "ema_50":      _f("ema_50"),
                "ema_200":     _f("ema_200"),
                "bb_pct":      _f("bb_pct", 0.5),
            }
            results[tf] = _classify_timeframe(ind)
            results[tf]["indicators"] = ind
        except Exception:
            results[tf] = {
                "score": 0,
                "verdict": "N/A",
                "signal": "N/A",
                "color": "#555",
                "details": {},
                "indicators": {},
                "momentum": 0,
                "confidence": 0.0,
            }

    scores = [v["score"] for v in results.values() if v.get("verdict") != "N/A"]
    if scores:
        avg = np.mean(scores)
        if avg >= 4:
            overall = "Strong Buy"
            oc = "#1a7f37"
        elif avg >= 1.5:
            overall = "Buy"
            oc = "#2ecc71"
        elif avg <= -4:
            overall = "Strong Sell"
            oc = "#8b0000"
        elif avg <= -1.5:
            overall = "Sell"
            oc = "#e74c3c"
        else:
            overall = "Hold"
            oc = "#f39c12"
        bullish = sum(1 for v in results.values() if v.get("score", 0) >= 2)
        bearish = sum(1 for v in results.values() if v.get("score", 0) <= -2)
        hold = sum(1 for v in results.values() if abs(v.get("score", 0)) < 2)
        agreement = bullish - bearish
        confidence = round(min(1.0, max(0.55, 0.55 + abs(avg) / 6 * 0.45)), 2)
        results["_overall"] = {
            "verdict": overall,
            "color": oc,
            "avg_score": round(avg, 2),
            "bullish": bullish,
            "bearish": bearish,
            "hold": hold,
            "agreement": agreement,
            "confidence": confidence,
        }
    else:
        results["_overall"] = {
            "verdict": "N/A",
            "color": "#555",
            "avg_score": 0,
            "bullish": 0,
            "bearish": 0,
            "hold": 0,
            "agreement": 0,
            "confidence": 0.0,
        }

    return results
