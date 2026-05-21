"""
Signal engine — rule-based weighted scoring combining:
  Technical indicators, Advanced indicators, SMC, MTF, Order book, Sentiment, Fear & Greed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

SIGNAL_BUY  = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_HOLD = "HOLD"

# Weight table: (max_score, weight_multiplier)
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

    scores:  List[float] = []
    reasons: List[str]   = []
    max_possible = 0.0

    def _push(raw: float, cat: str, reason: str):
        w = _CATEGORY_WEIGHTS.get(cat, 1.0)
        scores.append(_weighted(raw, w))
        max_possible_val = abs(raw) * w if abs(raw) > 0 else w * 2
        reasons.append(reason)
        return max_possible_val

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi = indicators.get("rsi", 50)
    if rsi < 25:
        v = _push(3, "rsi", f"RSI extremely oversold ({rsi:.1f})")
    elif rsi < 35:
        v = _push(2, "rsi", f"RSI oversold ({rsi:.1f})")
    elif rsi < 45:
        v = _push(1, "rsi", f"RSI approaching oversold ({rsi:.1f})")
    elif rsi > 75:
        v = _push(-3, "rsi", f"RSI extremely overbought ({rsi:.1f})")
    elif rsi > 65:
        v = _push(-2, "rsi", f"RSI overbought ({rsi:.1f})")
    elif rsi > 55:
        v = _push(-1, "rsi", f"RSI approaching overbought ({rsi:.1f})")
    else:
        v = _push(0, "rsi", f"RSI neutral ({rsi:.1f})")
    max_possible += _CATEGORY_WEIGHTS["rsi"] * 3

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd      = indicators.get("macd", 0)
    macd_sig  = indicators.get("macd_signal", 0)
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

    # ── EMA ───────────────────────────────────────────────────────────────────
    close   = indicators.get("close", 0)
    ema_9   = indicators.get("ema_9",   close)
    ema_21  = indicators.get("ema_21",  close)
    ema_50  = indicators.get("ema_50",  close)
    ema_200 = indicators.get("ema_200", close)
    bull_x  = indicators.get("ema_bullish_cross", False)
    bear_x  = indicators.get("ema_bearish_cross", False)

    if bull_x:
        _push(3, "ema", "🟢 EMA 9 bullish crossover above EMA 21")
    elif bear_x:
        _push(-3, "ema", "🔴 EMA 9 bearish crossover below EMA 21")
    elif ema_9 > ema_21:
        _push(1, "ema", "EMA 9 above EMA 21")
    else:
        _push(-1, "ema", "EMA 9 below EMA 21")

    if close > ema_50 > ema_200:
        _push(2, "ema", "Price above EMA50 > EMA200 (full bull)")
    elif close > ema_200:
        _push(1, "ema", "Price above EMA 200")
    elif close < ema_50 < ema_200:
        _push(-2, "ema", "Price below EMA50 < EMA200 (full bear)")
    else:
        _push(-1, "ema", "Price below EMA 200")

    if ema_50 > ema_200:
        _push(1, "ema", "Golden cross (EMA50 > EMA200)")
    else:
        _push(-1, "ema", "Death cross (EMA50 < EMA200)")
    max_possible += _CATEGORY_WEIGHTS["ema"] * 6

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_pct = indicators.get("bb_pct", 0.5)
    if bb_pct < 0.05:
        _push(2, "bb", "Price at lower BB (strongly oversold)")
    elif bb_pct < 0.2:
        _push(1, "bb", "Price in lower BB zone")
    elif bb_pct > 0.95:
        _push(-2, "bb", "Price at upper BB (strongly overbought)")
    elif bb_pct > 0.8:
        _push(-1, "bb", "Price in upper BB zone")
    else:
        _push(0, "bb", "Price in BB middle zone")
    max_possible += _CATEGORY_WEIGHTS["bb"] * 2

    # ── Advanced indicators ───────────────────────────────────────────────────
    if adv:
        supertrend_dir = adv.get("supertrend_dir", 0)
        if supertrend_dir == 1:
            _push(2, "adv", "Supertrend bullish")
        elif supertrend_dir == -1:
            _push(-2, "adv", "Supertrend bearish")

        adx = adv.get("adx", 25)
        if adx > 40:
            _push(0.5 * np.sign(supertrend_dir or 1), "adv",
                  f"ADX strong trend ({adx:.1f})")

        psar_bull = adv.get("psar_bull", True)
        if psar_bull:
            _push(1, "adv", "Parabolic SAR bullish")
        else:
            _push(-1, "adv", "Parabolic SAR bearish")

        vwap = adv.get("vwap", close)
        if close > vwap:
            _push(1, "adv", "Price above VWAP")
        else:
            _push(-1, "adv", "Price below VWAP")

        srsi_k = adv.get("stochrsi_k", 50)
        if srsi_k < 20:
            _push(1.5, "adv", f"Stoch RSI oversold ({srsi_k:.1f})")
        elif srsi_k > 80:
            _push(-1.5, "adv", f"Stoch RSI overbought ({srsi_k:.1f})")

        mfi = adv.get("mfi", 50)
        if mfi < 20:
            _push(1, "adv", f"MFI oversold ({mfi:.1f})")
        elif mfi > 80:
            _push(-1, "adv", f"MFI overbought ({mfi:.1f})")

        cmf = adv.get("cmf", 0)
        if cmf > 0.1:
            _push(1, "adv", f"CMF positive money flow ({cmf:.3f})")
        elif cmf < -0.1:
            _push(-1, "adv", f"CMF negative money flow ({cmf:.3f})")

        ich_a = adv.get("ich_senkou_a", 0)
        ich_b = adv.get("ich_senkou_b", 0)
        if close > max(ich_a, ich_b):
            _push(1.5, "adv", "Price above Ichimoku cloud (bullish)")
        elif close < min(ich_a, ich_b):
            _push(-1.5, "adv", "Price below Ichimoku cloud (bearish)")

        max_possible += _CATEGORY_WEIGHTS["adv"] * 12

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_ratio = indicators.get("volume_ratio", adv.get("rel_volume", 1.0))
    if vol_ratio > 2.0:
        _push(1, "volume", f"Very high volume ({vol_ratio:.1f}x avg)")
    elif vol_ratio > 1.5:
        _push(0.5, "volume", f"Above-average volume ({vol_ratio:.1f}x)")
    max_possible += _CATEGORY_WEIGHTS["volume"] * 1

    # ── SMC ───────────────────────────────────────────────────────────────────
    if smc:
        bos_bull  = smc.get("bos_bull", [])
        bos_bear  = smc.get("bos_bear", [])
        choch_b   = smc.get("choch_bull", [])
        choch_br  = smc.get("choch_bear", [])
        bull_fvg  = smc.get("bull_fvg", [])
        bear_fvg  = smc.get("bear_fvg", [])
        bull_ob   = smc.get("bull_ob", [])
        bear_ob   = smc.get("bear_ob", [])
        pd_zone   = smc.get("premium_discount", {})

        if bos_bull:
            _push(2, "smc", f"BOS bullish break ({len(bos_bull)} recent)")
        if bos_bear:
            _push(-2, "smc", f"BOS bearish break ({len(bos_bear)} recent)")
        if choch_b:
            _push(2.5, "smc", "CHoCH bullish reversal detected")
        if choch_br:
            _push(-2.5, "smc", "CHoCH bearish reversal detected")
        if bull_fvg:
            _push(1, "smc", f"{len(bull_fvg)} bullish FVG(s) below price")
        if bear_fvg:
            _push(-1, "smc", f"{len(bear_fvg)} bearish FVG(s) above price")
        if bull_ob:
            _push(1, "smc", f"Bullish order block support")
        if bear_ob:
            _push(-1, "smc", f"Bearish order block resistance")

        zone = pd_zone.get("current_zone", "")
        if zone == "Discount":
            _push(1.5, "smc", "Price in discount zone (buy zone)")
        elif zone == "Premium":
            _push(-1.5, "smc", "Price in premium zone (sell zone)")

        max_possible += _CATEGORY_WEIGHTS["smc"] * 10

    # ── Order book ────────────────────────────────────────────────────────────
    if ob:
        imbalance = ob.get("imbalance", 0)
        buy_pct   = ob.get("buy_pct", 50)
        if imbalance > 0.2:
            _push(1.5, "ob", f"Order book buy-side dominant ({buy_pct:.1f}%)")
        elif imbalance < -0.2:
            _push(-1.5, "ob", f"Order book sell-side dominant ({100-buy_pct:.1f}%)")
        max_possible += _CATEGORY_WEIGHTS["ob"] * 1.5

    # ── MTF alignment ─────────────────────────────────────────────────────────
    if mtf_overall:
        avg_score = mtf_overall.get("avg_score", 0)
        verdict   = mtf_overall.get("verdict", "Hold")
        if avg_score >= 3:
            _push(3, "mtf", f"MTF alignment: {verdict}")
        elif avg_score >= 1:
            _push(1.5, "mtf", f"MTF alignment: {verdict}")
        elif avg_score <= -3:
            _push(-3, "mtf", f"MTF alignment: {verdict}")
        elif avg_score <= -1:
            _push(-1.5, "mtf", f"MTF alignment: {verdict}")
        max_possible += _CATEGORY_WEIGHTS["mtf"] * 3

    # ── Sentiment ─────────────────────────────────────────────────────────────
    if sentiment_score > 0.4:
        _push(2, "sentiment", f"Strong positive sentiment ({sentiment_score:.2f})")
    elif sentiment_score > 0.15:
        _push(1, "sentiment", f"Positive sentiment ({sentiment_score:.2f})")
    elif sentiment_score < -0.4:
        _push(-2, "sentiment", f"Strong negative sentiment ({sentiment_score:.2f})")
    elif sentiment_score < -0.15:
        _push(-1, "sentiment", f"Negative sentiment ({sentiment_score:.2f})")
    max_possible += _CATEGORY_WEIGHTS["sentiment"] * 2

    # ── Fear & Greed ──────────────────────────────────────────────────────────
    if fg_value < 20:
        _push(1, "fg", f"Extreme Fear — contrarian buy signal ({fg_value})")
    elif fg_value > 80:
        _push(-1, "fg", f"Extreme Greed — contrarian sell signal ({fg_value})")
    max_possible += _CATEGORY_WEIGHTS["fg"] * 1

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total = sum(scores)
    norm  = total / max_possible if max_possible > 0 else 0
    norm  = max(-1.0, min(1.0, norm))

    if total >= 6:
        signal = SIGNAL_BUY
        conf   = min(0.97, 0.55 + abs(norm) * 0.42)
    elif total <= -6:
        signal = SIGNAL_SELL
        conf   = min(0.97, 0.55 + abs(norm) * 0.42)
    else:
        signal = SIGNAL_HOLD
        conf   = max(0.40, 0.65 - abs(norm) * 0.25)

    bull_count = sum(1 for s in scores if s > 0)
    bear_count = sum(1 for s in scores if s < 0)

    reasons_sorted = sorted(
        zip(scores, reasons), key=lambda x: abs(x[0]), reverse=True
    )

    return {
        "signal":           signal,
        "score":            round(total, 2),
        "confidence":       round(conf, 3),
        "normalized_score": round(norm, 3),
        "reasons":          [r for _, r in reasons_sorted],
        "bull_signals":     bull_count,
        "bear_signals":     bear_count,
    }


def generate_signals_series(df: pd.DataFrame) -> pd.DataFrame:
    """Apply signal generation row-by-row for backtesting."""
    df = df.copy()
    signals = []
    for i in range(len(df)):
        row      = df.iloc[i]
        prev_row = df.iloc[i - 1] if i > 0 else row
        ind = {
            "rsi":              row.get("rsi", 50),
            "macd":             row.get("macd", 0),
            "macd_signal":      row.get("macd_signal", 0),
            "prev_macd":        prev_row.get("macd", 0),
            "close":            row["close"],
            "ema_9":            row.get("ema_9",   row["close"]),
            "ema_21":           row.get("ema_21",  row["close"]),
            "prev_ema_9":       prev_row.get("ema_9",  row["close"]),
            "prev_ema_21":      prev_row.get("ema_21", row["close"]),
            "ema_bullish_cross": bool(row.get("ema_bullish_cross", False)),
            "ema_bearish_cross": bool(row.get("ema_bearish_cross", False)),
            "ema_50":           row.get("ema_50",  row["close"]),
            "ema_200":          row.get("ema_200", row["close"]),
            "bb_pct":           row.get("bb_pct", 0.5),
            "volume_ratio":     row.get("volume_ratio", 1.0),
        }
        result = generate_signal(ind)
        signals.append(result["signal"])
    df["signal"] = signals
    return df
