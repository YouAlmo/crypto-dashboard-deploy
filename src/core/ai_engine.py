import pandas as pd


def generate_ai_analysis(df: pd.DataFrame) -> dict:

    if df is None or df.empty:
        return {
            "signal": "HOLD",
            "confidence": 0,
            "summary": "No data available."
        }

    last = df.iloc[-1]

    score = 0
    reasons = []

    rsi = float(last.get("rsi", 50))
    macd = float(last.get("macd", 0))
    macd_signal = float(last.get("macd_signal", 0))

    ema9 = float(last.get("ema_9", 0))
    ema21 = float(last.get("ema_21", 0))
    close = float(last.get("close", 0))

    # RSI
    if rsi < 30:
        score += 2
        reasons.append("RSI Oversold")

    elif rsi > 70:
        score -= 2
        reasons.append("RSI Overbought")

    # MACD
    if macd > macd_signal:
        score += 1
        reasons.append("MACD Bullish")

    else:
        score -= 1
        reasons.append("MACD Bearish")

    # EMA
    if ema9 > ema21:
        score += 1
        reasons.append("EMA Bullish Cross")

    else:
        score -= 1
        reasons.append("EMA Bearish Cross")

    # Trend
    if close > ema9 > ema21:
        score += 2
        reasons.append("Strong Uptrend")

    elif close < ema9 < ema21:
        score -= 2
        reasons.append("Strong Downtrend")

    # Final Signal
    if score >= 4:
        signal = "BUY"

    elif score <= -4:
        signal = "SELL"

    else:
        signal = "HOLD"

    confidence = min(abs(score) * 20, 95)

    return {
        "signal": signal,
        "confidence": confidence,
        "score": score,
        "reasons": reasons,
        "summary": f"AI detected {signal} conditions with {confidence}% confidence."
    }