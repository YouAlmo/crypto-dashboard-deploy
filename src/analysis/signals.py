import pandas as pd
import numpy as np

from typing import Dict


# ─────────────────────────────────────────────────────────────
# Signal Thresholds
# ─────────────────────────────────────────────────────────────

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

ADX_STRONG = 25

VOLUME_STRONG = 1.5

CONFIDENCE_MAX = 100


# ─────────────────────────────────────────────────────────────
# Main Signal Engine
# ─────────────────────────────────────────────────────────────

def generate_signal(
    df: pd.DataFrame,
) -> Dict:

    if df is None or df.empty:
        return _neutral_signal()

    try:

        last = df.iloc[-1]

        score = 0

        confidence = 0

        reasons = []

        # ─────────────────────────────────────
        # Trend Regime
        # ─────────────────────────────────────

        close = float(last.get("close", 0))

        ema_9 = float(last.get("ema_9", close))

        ema_21 = float(last.get("ema_21", close))

        ema_50 = float(last.get("ema_50", close))

        ema_200 = float(last.get("ema_200", close))

        bullish_trend = (
            close > ema_9 > ema_21 > ema_50
        )

        bearish_trend = (
            close < ema_9 < ema_21 < ema_50
        )

        macro_bull = close > ema_200

        macro_bear = close < ema_200

        if bullish_trend:
            score += 25
            confidence += 15
            reasons.append("Bullish Trend")

        if bearish_trend:
            score -= 25
            confidence += 15
            reasons.append("Bearish Trend")

        if macro_bull:
            score += 10
            confidence += 5
            reasons.append("Above EMA200")

        if macro_bear:
            score -= 10
            confidence += 5
            reasons.append("Below EMA200")

        # ─────────────────────────────────────
        # RSI Analysis
        # ─────────────────────────────────────

        rsi = float(last.get("rsi", 50))

        if rsi <= RSI_OVERSOLD:
            score += 20
            confidence += 10
            reasons.append("RSI Oversold")

        elif rsi < 45:
            score += 5
            reasons.append("RSI Bullish Zone")

        elif rsi >= RSI_OVERBOUGHT:
            score -= 20
            confidence += 10
            reasons.append("RSI Overbought")

        elif rsi > 55:
            score -= 5
            reasons.append("RSI Bearish Zone")

        # ─────────────────────────────────────
        # MACD Momentum
        # ─────────────────────────────────────

        macd = float(last.get("macd", 0))

        macd_signal = float(
            last.get("macd_signal", 0)
        )

        macd_hist = float(
            last.get("macd_hist", 0)
        )

        if macd > macd_signal:

            score += 15

            confidence += 10

            reasons.append("MACD Bullish")

            if macd_hist > 0:
                score += 5

        else:

            score -= 15

            confidence += 10

            reasons.append("MACD Bearish")

            if macd_hist < 0:
                score -= 5

        # ─────────────────────────────────────
        # ADX Trend Strength
        # ─────────────────────────────────────

        adx = float(last.get("adx", 0))

        if adx >= ADX_STRONG:

            confidence += 15

            reasons.append("Strong Trend")

        else:

            confidence -= 5

            reasons.append("Weak Trend")

        # ─────────────────────────────────────
        # Volume Confirmation
        # ─────────────────────────────────────

        volume_ratio = float(
            last.get("volume_ratio", 1)
        )

        if volume_ratio >= VOLUME_STRONG:

            confidence += 10

            score += 5

            reasons.append("Strong Volume")

        # ─────────────────────────────────────
        # VWAP
        # ─────────────────────────────────────

        vwap = float(last.get("vwap", close))

        if close > vwap:

            score += 10

            reasons.append("Above VWAP")

        else:

            score -= 10

            reasons.append("Below VWAP")

        # ─────────────────────────────────────
        # Momentum
        # ─────────────────────────────────────

        momentum = float(
            last.get("momentum_pct", 0)
        )

        if momentum > 1.5:

            score += 10

            confidence += 5

            reasons.append("Positive Momentum")

        elif momentum < -1.5:

            score -= 10

            confidence += 5

            reasons.append("Negative Momentum")

        # ─────────────────────────────────────
        # Fake Breakout Filter
        # ─────────────────────────────────────

        bb_pct = float(last.get("bb_pct", 0.5))

        if bb_pct > 1.1:

            score -= 10

            confidence += 5

            reasons.append("Possible Exhaustion")

        elif bb_pct < -0.1:

            score += 10

            confidence += 5

            reasons.append("Potential Reversal")

        # ─────────────────────────────────────
        # Final Verdict
        # ─────────────────────────────────────

        confidence = int(
            max(
                0,
                min(CONFIDENCE_MAX, confidence)
            )
        )

        if score >= 45:

            signal = "STRONG BUY"

            color = "#00c853"

        elif score >= 20:

            signal = "BUY"

            color = "#2ecc71"

        elif score <= -45:

            signal = "STRONG SELL"

            color = "#d50000"

        elif score <= -20:

            signal = "SELL"

            color = "#ff5252"

        else:

            signal = "HOLD"

            color = "#f39c12"

        # ─────────────────────────────────────
        # Risk Estimate
        # ─────────────────────────────────────

        atr = float(last.get("atr", 0))

        risk_level = _estimate_risk(
            atr=atr,
            close=close,
            adx=adx,
        )

        return {

            "signal": signal,

            "score": round(score, 2),

            "confidence": confidence,

            "color": color,

            "risk": risk_level,

            "reasons": reasons,

            "trend": (
                "Bullish"
                if score > 0
                else "Bearish"
            ),

            "strength": _signal_strength(
                score,
                confidence,
            ),
        }

    except Exception as e:

        print(f"Signal Engine Error -> {e}")

        return _neutral_signal()


# ─────────────────────────────────────────────────────────────
# Signal Strength
# ─────────────────────────────────────────────────────────────

def _signal_strength(
    score: float,
    confidence: int,
) -> str:

    total = abs(score) + confidence

    if total >= 120:
        return "EXTREME"

    if total >= 90:
        return "VERY STRONG"

    if total >= 60:
        return "STRONG"

    if total >= 35:
        return "MODERATE"

    return "WEAK"


# ─────────────────────────────────────────────────────────────
# Risk Estimator
# ─────────────────────────────────────────────────────────────

def _estimate_risk(
    atr: float,
    close: float,
    adx: float,
) -> str:

    if close <= 0:
        return "UNKNOWN"

    volatility = (
        atr / close
    ) * 100

    if volatility >= 5:
        return "VERY HIGH"

    if volatility >= 3:
        return "HIGH"

    if volatility >= 1.5:
        return "MEDIUM"

    if adx >= 35:
        return "LOW"

    return "MEDIUM"


# ─────────────────────────────────────────────────────────────
# Neutral Fallback
# ─────────────────────────────────────────────────────────────

def _neutral_signal() -> Dict:

    return {

        "signal": "HOLD",

        "score": 0,

        "confidence": 0,

        "color": "#777777",

        "risk": "UNKNOWN",

        "reasons": [],

        "trend": "Neutral",

        "strength": "WEAK",
    }