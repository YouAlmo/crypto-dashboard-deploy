import numpy as np
import pandas as pd

from typing import Dict, List

from src.analysis.signals import (
    generate_signal,
)

from src.analysis.mtf import (
    fetch_mtf_analysis,
)


# ─────────────────────────────────────────────────────────────
# AI Weights
# ─────────────────────────────────────────────────────────────

WEIGHTS = {

    "trend": 0.25,

    "momentum": 0.15,

    "volume": 0.10,

    "volatility": 0.10,

    "confidence": 0.15,

    "mtf_alignment": 0.25,
}


# ─────────────────────────────────────────────────────────────
# Market Regimes
# ─────────────────────────────────────────────────────────────

MARKET_REGIMES = {

    "STRONG_TREND",

    "TREND",

    "RANGE",

    "VOLATILE",

    "ACCUMULATION",

    "DISTRIBUTION",
}


# ─────────────────────────────────────────────────────────────
# Main AI Analysis
# ─────────────────────────────────────────────────────────────

def run_ai_analysis(
    symbol: str,
    df: pd.DataFrame,
) -> Dict:

    if df is None or df.empty:

        return _empty_ai_result()

    try:

        signal = generate_signal(df)

        mtf = fetch_mtf_analysis(symbol)

        last = df.iloc[-1]

        # ─────────────────────────────────────
        # Core Metrics
        # ─────────────────────────────────────

        close = float(
            last.get("close", 0)
        )

        ema_50 = float(
            last.get("ema_50", close)
        )

        ema_200 = float(
            last.get("ema_200", close)
        )

        rsi = float(
            last.get("rsi", 50)
        )

        adx = float(
            last.get("adx", 0)
        )

        momentum = float(
            last.get("momentum_pct", 0)
        )

        volume_ratio = float(
            last.get("volume_ratio", 1)
        )

        atr = float(
            last.get("atr", 0)
        )

        # ─────────────────────────────────────
        # Trend Score
        # ─────────────────────────────────────

        trend_score = _calculate_trend_score(
            close,
            ema_50,
            ema_200,
            adx,
        )

        # ─────────────────────────────────────
        # Momentum Score
        # ─────────────────────────────────────

        momentum_score = _calculate_momentum_score(
            momentum,
            rsi,
        )

        # ─────────────────────────────────────
        # Volume Score
        # ─────────────────────────────────────

        volume_score = _calculate_volume_score(
            volume_ratio
        )

        # ─────────────────────────────────────
        # Volatility Score
        # ─────────────────────────────────────

        volatility_score = _calculate_volatility_score(
            atr,
            close,
        )

        # ─────────────────────────────────────
        # MTF Alignment
        # ─────────────────────────────────────

        mtf_score = _calculate_mtf_alignment(
            mtf
        )

        # ─────────────────────────────────────
        # Confidence
        # ─────────────────────────────────────

        confidence_score = float(
            signal.get(
                "confidence",
                0,
            )
        )

        # ─────────────────────────────────────
        # Final AI Score
        # ─────────────────────────────────────

        final_score = (

            trend_score
            * WEIGHTS["trend"]

            +

            momentum_score
            * WEIGHTS["momentum"]

            +

            volume_score
            * WEIGHTS["volume"]

            +

            volatility_score
            * WEIGHTS["volatility"]

            +

            confidence_score
            * WEIGHTS["confidence"]

            +

            mtf_score
            * WEIGHTS["mtf_alignment"]
        )

        final_score = round(
            min(100, max(0, final_score)),
            2,
        )

        # ─────────────────────────────────────
        # Market Regime
        # ─────────────────────────────────────

        regime = detect_market_regime(
            rsi,
            adx,
            atr,
            close,
        )

        # ─────────────────────────────────────
        # Opportunity Rating
        # ─────────────────────────────────────

        opportunity = classify_opportunity(
            final_score
        )

        # ─────────────────────────────────────
        # AI Verdict
        # ─────────────────────────────────────

        verdict = ai_verdict(
            final_score
        )

        return {

            "symbol": symbol,

            "ai_score": final_score,

            "verdict": verdict,

            "opportunity": opportunity,

            "market_regime": regime,

            "trend_score": round(
                trend_score,
                2,
            ),

            "momentum_score": round(
                momentum_score,
                2,
            ),

            "volume_score": round(
                volume_score,
                2,
            ),

            "volatility_score": round(
                volatility_score,
                2,
            ),

            "mtf_score": round(
                mtf_score,
                2,
            ),

            "confidence_score": round(
                confidence_score,
                2,
            ),

            "signal": signal,

            "mtf": mtf,
        }

    except Exception as e:

        print(
            f"AI Engine Error [{symbol}] -> {e}"
        )

        return _empty_ai_result()


# ─────────────────────────────────────────────────────────────
# Trend Score
# ─────────────────────────────────────────────────────────────

def _calculate_trend_score(
    close,
    ema50,
    ema200,
    adx,
):

    score = 50

    if close > ema50:
        score += 15

    if close > ema200:
        score += 20

    if ema50 > ema200:
        score += 15

    if adx >= 30:
        score += 15

    elif adx < 15:
        score -= 10

    return np.clip(score, 0, 100)


# ─────────────────────────────────────────────────────────────
# Momentum Score
# ─────────────────────────────────────────────────────────────

def _calculate_momentum_score(
    momentum,
    rsi,
):

    score = 50

    if momentum > 3:
        score += 30

    elif momentum > 1:
        score += 15

    elif momentum < -3:
        score -= 30

    elif momentum < -1:
        score -= 15

    if 45 <= rsi <= 65:
        score += 10

    if rsi > 80:
        score -= 20

    if rsi < 20:
        score += 10

    return np.clip(score, 0, 100)


# ─────────────────────────────────────────────────────────────
# Volume Score
# ─────────────────────────────────────────────────────────────

def _calculate_volume_score(
    volume_ratio,
):

    score = 50

    if volume_ratio >= 3:
        score += 40

    elif volume_ratio >= 2:
        score += 25

    elif volume_ratio >= 1.5:
        score += 15

    elif volume_ratio < 0.7:
        score -= 15

    return np.clip(score, 0, 100)


# ─────────────────────────────────────────────────────────────
# Volatility Score
# ─────────────────────────────────────────────────────────────

def _calculate_volatility_score(
    atr,
    close,
):

    if close <= 0:
        return 50

    volatility = (
        atr / close
    ) * 100

    if volatility <= 1:
        return 85

    if volatility <= 2:
        return 70

    if volatility <= 4:
        return 50

    if volatility <= 7:
        return 35

    return 20


# ─────────────────────────────────────────────────────────────
# MTF Alignment Score
# ─────────────────────────────────────────────────────────────

def _calculate_mtf_alignment(
    mtf: Dict,
):

    overall = mtf.get(
        "_overall",
        {},
    )

    signal = overall.get(
        "signal",
        "HOLD",
    )

    confidence = overall.get(
        "confidence",
        0,
    )

    alignment = overall.get(
        "alignment",
        "",
    )

    score = 50

    if signal == "STRONG BUY":
        score += 35

    elif signal == "BUY":
        score += 20

    elif signal == "STRONG SELL":
        score -= 35

    elif signal == "SELL":
        score -= 20

    if alignment == "EXTREME ALIGNMENT":
        score += 15

    elif alignment == "VERY STRONG":
        score += 10

    score += (
        confidence * 0.1
    )

    return np.clip(score, 0, 100)


# ─────────────────────────────────────────────────────────────
# Market Regime Detection
# ─────────────────────────────────────────────────────────────

def detect_market_regime(
    rsi,
    adx,
    atr,
    close,
):

    volatility = (
        atr / close
    ) * 100 if close > 0 else 0

    if adx >= 35:

        if volatility >= 5:
            return "VOLATILE"

        return "STRONG_TREND"

    if adx >= 25:
        return "TREND"

    if volatility >= 7:
        return "VOLATILE"

    if 45 <= rsi <= 55:
        return "RANGE"

    if rsi < 35:
        return "ACCUMULATION"

    if rsi > 70:
        return "DISTRIBUTION"

    return "RANGE"


# ─────────────────────────────────────────────────────────────
# Opportunity Classification
# ─────────────────────────────────────────────────────────────

def classify_opportunity(
    score,
):

    if score >= 85:
        return "ELITE"

    if score >= 70:
        return "HIGH"

    if score >= 55:
        return "GOOD"

    if score >= 40:
        return "MODERATE"

    return "LOW"


# ─────────────────────────────────────────────────────────────
# AI Verdict
# ─────────────────────────────────────────────────────────────

def ai_verdict(
    score,
):

    if score >= 85:
        return "STRONG BUY"

    if score >= 70:
        return "BUY"

    if score <= 20:
        return "STRONG SELL"

    if score <= 35:
        return "SELL"

    return "HOLD"


# ─────────────────────────────────────────────────────────────
# Opportunity Ranking
# ─────────────────────────────────────────────────────────────

def rank_opportunities(
    opportunities: List[Dict],
) -> List[Dict]:

    return sorted(

        opportunities,

        key=lambda x: x.get(
            "ai_score",
            0,
        ),

        reverse=True,
    )


# ─────────────────────────────────────────────────────────────
# Empty Result
# ─────────────────────────────────────────────────────────────

def _empty_ai_result():

    return {

        "symbol": "",

        "ai_score": 0,

        "verdict": "HOLD",

        "opportunity": "LOW",

        "market_regime": "UNKNOWN",

        "trend_score": 0,

        "momentum_score": 0,

        "volume_score": 0,

        "volatility_score": 0,

        "mtf_score": 0,

        "confidence_score": 0,

        "signal": {},

        "mtf": {},
    }