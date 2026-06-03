"""
Signal engine — rule-based weighted scoring combining:
  Technical indicators, Advanced indicators, SMC, MTF, Order book, Sentiment, Fear & Greed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

SIGNAL_BUY  = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_HOLD = "HOLD"

# Weight table
_CATEGORY_WEIGHTS = {
    "rsi":       1.5,
    "macd":      1.5,
    "ema":       2.0,
    "bb":        1.0,
    "adv":       1.2,
    "smc":       1.5,
    "volume":    1.0,
    "sentiment": 0.8,
    "fg":        0.5,
    "ob":        0.8,
    "mtf":       1.5,
}


def _weighted(raw: float, weight: float) -> float:
    return raw * weight


def classify_strength(signal: str, confidence: float) -> str:
    if signal == SIGNAL_HOLD:
        return "Neutral"

    if confidence >= 0.9:
        return "Extreme"
    elif confidence >= 0.8:
        return "Very Strong"
    elif confidence >= 0.7:
        return "Strong"
    elif confidence >= 0.6:
        return "Moderate"

    return "Weak"


def classify_risk(confidence: float, bull_count: int, bear_count: int) -> str:
    disagreement = min(bull_count, bear_count)

    if confidence >= 0.82 and disagreement <= 2:
        return "Low"
    elif confidence >= 0.65:
        return "Medium"

    return "High"


def generate_signal(
    indicators: dict,
    sentiment_score: float = 0.0,
    advanced: dict = None,
    smc: dict = None,
    mtf_overall: dict = None,
    orderbook: dict = None,
    fg_value: int = 50,
) -> Dict:
    adv = advanced or {}
    smc = smc or {}
    ob  = orderbook or {}

    scores: List[float] = []
    reasons: List[str] = []
    max_possible = 0.0

    def _push(raw: float, cat: str, reason: str):
        w = _CATEGORY_WEIGHTS.get(cat, 1.0)
        scores.append(_weighted(raw, w))
        reasons.append(reason)

    # RSI
    rsi = indicators.get("rsi", 50)
    if rsi < 25:
        _push(3, "rsi", f"RSI extremely oversold ({rsi:.1f})")
    elif rsi < 35:
        _push(2, "rsi", f"RSI oversold ({rsi:.1f})")
    elif rsi < 45:
        _push(1, "rsi", f"RSI approaching oversold ({rsi:.1f})")
    elif rsi > 75:
        _push(-3, "rsi", f"RSI extremely overbought ({rsi:.1f})")
    elif rsi > 65:
        _push(-2, "rsi", f"RSI overbought ({rsi:.1f})")
    elif rsi > 55:
        _push(-1, "rsi", f"RSI approaching overbought ({rsi:.1f})")
    else:
        _push(0, "rsi", f"RSI neutral ({rsi:.1f})")

    max_possible += _CATEGORY_WEIGHTS["rsi"] * 3

    # MACD
    macd = indicators.get("macd", 0)
    macd_sig = indicators.get("macd_signal", 0)
    prev_macd = indicators.get("prev_macd", macd)

    if macd > macd_sig and prev_macd <= macd_sig:
        _push(2, "macd", "MACD bullish crossover")
    elif macd > macd_sig:
        _push(1, "macd", "MACD above signal line")
    elif macd < macd_sig and prev_macd >= macd_sig:
        _push(-2, "macd", "MACD bearish crossover")
    elif macd < macd_sig:
        _push(-1, "macd", "MACD below signal line")
    else:
        _push(0, "macd", "MACD neutral")

    max_possible += _CATEGORY_WEIGHTS["macd"] * 2

    # EMA
    close = indicators.get("close", 0)
    ema_9 = indicators.get("ema_9", close)
    ema_21 = indicators.get("ema_21", close)
    ema_50 = indicators.get("ema_50", close)
    ema_200 = indicators.get("ema_200", close)

    if indicators.get("ema_bullish_cross", False):
        _push(3, "ema", "EMA bullish crossover")
    elif indicators.get("ema_bearish_cross", False):
        _push(-3, "ema", "EMA bearish crossover")
    elif ema_9 > ema_21:
        _push(1, "ema", "EMA 9 above EMA 21")
    else:
        _push(-1, "ema", "EMA 9 below EMA 21")

    if close > ema_50 > ema_200:
        _push(2, "ema", "Full bullish EMA structure")
    elif close < ema_50 < ema_200:
        _push(-2, "ema", "Full bearish EMA structure")

    max_possible += _CATEGORY_WEIGHTS["ema"] * 6

    # Bollinger Bands
    bb_pct = indicators.get("bb_pct", 0.5)

    if bb_pct < 0.05:
        _push(2, "bb", "Price at lower Bollinger Band")
    elif bb_pct > 0.95:
        _push(-2, "bb", "Price at upper Bollinger Band")

    max_possible += _CATEGORY_WEIGHTS["bb"] * 2

    # Advanced indicators
    if adv:
        if adv.get("supertrend_dir", 0) == 1:
            _push(2, "adv", "Supertrend bullish")
        elif adv.get("supertrend_dir", 0) == -1:
            _push(-2, "adv", "Supertrend bearish")

        if adv.get("psar_bull", True):
            _push(1, "adv", "Parabolic SAR bullish")
        else:
            _push(-1, "adv", "Parabolic SAR bearish")

        if close > adv.get("vwap", close):
            _push(1, "adv", "Price above VWAP")
        else:
            _push(-1, "adv", "Price below VWAP")

        max_possible += _CATEGORY_WEIGHTS["adv"] * 8

    # SMC
    if smc:
        if smc.get("bos_bull"):
            _push(2, "smc", "Bullish BOS detected")

        if smc.get("bos_bear"):
            _push(-2, "smc", "Bearish BOS detected")

        if smc.get("choch_bull"):
            _push(2.5, "smc", "Bullish CHoCH reversal")

        if smc.get("choch_bear"):
            _push(-2.5, "smc", "Bearish CHoCH reversal")

        max_possible += _CATEGORY_WEIGHTS["smc"] * 8

    # MTF
    if mtf_overall:
        avg_score = mtf_overall.get("avg_score", 0)

        if avg_score >= 3:
            _push(3, "mtf", "Strong bullish multi-timeframe alignment")
        elif avg_score >= 1:
            _push(1.5, "mtf", "Bullish multi-timeframe alignment")
        elif avg_score <= -3:
            _push(-3, "mtf", "Strong bearish multi-timeframe alignment")
        elif avg_score <= -1:
            _push(-1.5, "mtf", "Bearish multi-timeframe alignment")

        max_possible += _CATEGORY_WEIGHTS["mtf"] * 3

    # Sentiment
    if sentiment_score > 0.4:
        _push(2, "sentiment", "Strong positive sentiment")
    elif sentiment_score < -0.4:
        _push(-2, "sentiment", "Strong negative sentiment")

    max_possible += _CATEGORY_WEIGHTS["sentiment"] * 2

    # Fear & Greed
    if fg_value < 20:
        _push(1, "fg", "Extreme fear")
    elif fg_value > 80:
        _push(-1, "fg", "Extreme greed")

    max_possible += _CATEGORY_WEIGHTS["fg"] * 1

    # Aggregate
    total = sum(scores)
    norm = total / max_possible if max_possible > 0 else 0
    norm = max(-1.0, min(1.0, norm))

    if total >= 6:
        signal = SIGNAL_BUY
        conf = min(0.97, 0.55 + abs(norm) * 0.42)
    elif total <= -6:
        signal = SIGNAL_SELL
        conf = min(0.97, 0.55 + abs(norm) * 0.42)
    else:
        signal = SIGNAL_HOLD
        conf = max(0.40, 0.65 - abs(norm) * 0.25)

    bull_count = sum(1 for s in scores if s > 0)
    bear_count = sum(1 for s in scores if s < 0)

    strength = classify_strength(signal, conf)
    risk_level = classify_risk(conf, bull_count, bear_count)

    reasons_sorted = sorted(
        zip(scores, reasons),
        key=lambda x: abs(x[0]),
        reverse=True,
    )

    return {
        "signal": signal,
        "score": round(total, 2),
        "confidence": round(conf, 3),
        "strength": strength,
        "risk_level": risk_level,
        "normalized_score": round(norm, 3),
        "reasons": [r for _, r in reasons_sorted],
        "bull_signals": bull_count,
        "bear_signals": bear_count,
    }


def generate_signals_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    signals = []

    for i in range(len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1] if i > 0 else row

        ind = {
            "rsi": row.get("rsi", 50),
            "macd": row.get("macd", 0),
            "macd_signal": row.get("macd_signal", 0),
            "prev_macd": prev_row.get("macd", 0),
            "close": row["close"],
            "ema_9": row.get("ema_9", row["close"]),
            "ema_21": row.get("ema_21", row["close"]),
            "ema_bullish_cross": bool(row.get("ema_bullish_cross", False)),
            "ema_bearish_cross": bool(row.get("ema_bearish_cross", False)),
            "ema_50": row.get("ema_50", row["close"]),
            "ema_200": row.get("ema_200", row["close"]),
            "bb_pct": row.get("bb_pct", 0.5),
        }

        result = generate_signal(ind)
        signals.append(result["signal"])

    df["signal"] = signals
    return df
