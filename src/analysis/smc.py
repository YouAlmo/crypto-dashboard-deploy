"""
Smart Money Concepts (SMC / ICT):
- Break of Structure (BOS) & Change of Character (CHoCH)
- Fair Value Gaps (FVG)
- Order Blocks
- Liquidity Zones (Equal Highs / Equal Lows)
- Premium / Discount Zones
- Supply and Demand Zones
"""

import pandas as pd
import numpy as np
from typing import Dict, List


# ── Swing detection ───────────────────────────────────────────────────────────

def _swing_highs(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.Series:
    highs = df["high"]
    result = pd.Series(False, index=df.index)
    for i in range(left, len(df) - right):
        window_left  = highs.iloc[i - left : i]
        window_right = highs.iloc[i + 1 : i + right + 1]
        if highs.iloc[i] >= window_left.max() and highs.iloc[i] >= window_right.max():
            result.iloc[i] = True
    return result


def _swing_lows(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.Series:
    lows = df["low"]
    result = pd.Series(False, index=df.index)
    for i in range(left, len(df) - right):
        window_left  = lows.iloc[i - left : i]
        window_right = lows.iloc[i + 1 : i + right + 1]
        if lows.iloc[i] <= window_left.min() and lows.iloc[i] <= window_right.min():
            result.iloc[i] = True
    return result


# ── BOS / CHoCH ───────────────────────────────────────────────────────────────

def detect_bos_choch(df: pd.DataFrame, swing: int = 3) -> Dict:
    """Detect Break of Structure and Change of Character."""
    sh = _swing_highs(df, swing, swing)
    sl = _swing_lows(df, swing, swing)

    sh_prices = df.loc[sh, "high"].tolist()
    sl_prices = df.loc[sl, "low"].tolist()
    sh_idx    = df.index[sh].tolist()
    sl_idx    = df.index[sl].tolist()

    bos_bull: List[dict] = []
    bos_bear: List[dict] = []
    choch_bull: List[dict] = []
    choch_bear: List[dict] = []

    if len(sh_prices) >= 2:
        for i in range(1, len(sh_prices)):
            close_after = df.loc[df.index > sh_idx[i - 1], "close"]
            level = sh_prices[i - 1]
            broke = close_after[close_after > level]
            if not broke.empty:
                entry = {"level": level, "time": sh_idx[i - 1], "break_time": broke.index[0]}
                if sh_prices[i] < sh_prices[i - 1]:
                    choch_bear.append(entry)
                else:
                    bos_bull.append(entry)

    if len(sl_prices) >= 2:
        for i in range(1, len(sl_prices)):
            close_after = df.loc[df.index > sl_idx[i - 1], "close"]
            level = sl_prices[i - 1]
            broke = close_after[close_after < level]
            if not broke.empty:
                entry = {"level": level, "time": sl_idx[i - 1], "break_time": broke.index[0]}
                if sl_prices[i] > sl_prices[i - 1]:
                    choch_bull.append(entry)
                else:
                    bos_bear.append(entry)

    return {
        "bos_bull":    bos_bull[-3:],
        "bos_bear":    bos_bear[-3:],
        "choch_bull":  choch_bull[-3:],
        "choch_bear":  choch_bear[-3:],
        "swing_highs": list(zip(sh_idx[-8:], sh_prices[-8:])),
        "swing_lows":  list(zip(sl_idx[-8:], sl_prices[-8:])),
    }


# ── Fair Value Gaps ───────────────────────────────────────────────────────────

def detect_fvg(df: pd.DataFrame, min_gap_pct: float = 0.1) -> Dict:
    """
    Bullish FVG: candle[i-2].high < candle[i].low  (gap between -2 high and current low)
    Bearish FVG: candle[i-2].low  > candle[i].high
    """
    bull_fvg: List[dict] = []
    bear_fvg: List[dict] = []

    for i in range(2, len(df)):
        c0_high = df["high"].iloc[i - 2]
        c2_low  = df["low"].iloc[i]
        c0_low  = df["low"].iloc[i - 2]
        c2_high = df["high"].iloc[i]
        mid_close = df["close"].iloc[i - 1]
        ts = df.index[i]

        if c2_low > c0_high:
            gap_pct = (c2_low - c0_high) / c0_high * 100
            if gap_pct >= min_gap_pct:
                bull_fvg.append({
                    "top": c2_low, "bottom": c0_high,
                    "mid": (c2_low + c0_high) / 2,
                    "time": ts, "gap_pct": round(gap_pct, 3),
                    "type": "bullish",
                })

        if c0_low > c2_high:
            gap_pct = (c0_low - c2_high) / c2_high * 100
            if gap_pct >= min_gap_pct:
                bear_fvg.append({
                    "top": c0_low, "bottom": c2_high,
                    "mid": (c0_low + c2_high) / 2,
                    "time": ts, "gap_pct": round(gap_pct, 3),
                    "type": "bearish",
                })

    # Filter: only unfilled FVGs (price hasn't entered the gap since formation)
    current_price = df["close"].iloc[-1]
    active_bull = [g for g in bull_fvg if current_price > g["bottom"]][-5:]
    active_bear = [g for g in bear_fvg if current_price < g["top"]][-5:]

    return {"bull_fvg": active_bull, "bear_fvg": active_bear}


# ── Order Blocks ──────────────────────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Bullish OB: Last bearish candle before a significant bullish move.
    Bearish OB: Last bullish candle before a significant bearish move.
    """
    recent = df.iloc[-lookback:] if len(df) > lookback else df
    bull_ob: List[dict] = []
    bear_ob: List[dict] = []
    threshold = 0.01

    for i in range(1, len(recent) - 2):
        c_open  = recent["open"].iloc[i]
        c_close = recent["close"].iloc[i]
        c_high  = recent["high"].iloc[i]
        c_low   = recent["low"].iloc[i]
        ts = recent.index[i]

        next_open  = recent["open"].iloc[i + 1]
        next_close = recent["close"].iloc[i + 1]
        move_pct   = abs(next_close - next_open) / next_open if next_open else 0

        if move_pct < threshold:
            continue

        is_bearish_candle = c_close < c_open
        next_is_bullish   = next_close > next_open

        if is_bearish_candle and next_is_bullish:
            bull_ob.append({
                "top": c_high, "bottom": c_low, "open": c_open, "close": c_close,
                "time": ts, "type": "bullish",
            })

        is_bullish_candle = c_close > c_open
        next_is_bearish   = next_close < next_open

        if is_bullish_candle and next_is_bearish:
            bear_ob.append({
                "top": c_high, "bottom": c_low, "open": c_open, "close": c_close,
                "time": ts, "type": "bearish",
            })

    current_price = df["close"].iloc[-1]
    active_bull = [b for b in bull_ob if current_price > b["bottom"] * 0.95][-4:]
    active_bear = [b for b in bear_ob if current_price < b["top"]   * 1.05][-4:]

    return {"bull_ob": active_bull, "bear_ob": active_bear}


# ── Liquidity Zones ───────────────────────────────────────────────────────────

def detect_liquidity_zones(df: pd.DataFrame, tolerance: float = 0.002, min_touches: int = 2) -> Dict:
    """Find equal highs and equal lows (liquidity pools)."""
    highs = df["high"].values
    lows  = df["low"].values
    idx   = df.index

    equal_highs: List[dict] = []
    equal_lows:  List[dict] = []

    used_h = set()
    for i in range(len(highs)):
        if i in used_h:
            continue
        cluster = [i]
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) / highs[i] <= tolerance:
                cluster.append(j)
                used_h.add(j)
        if len(cluster) >= min_touches:
            equal_highs.append({
                "level": float(np.mean([highs[k] for k in cluster])),
                "touches": len(cluster),
                "times": [idx[k] for k in cluster],
            })

    used_l = set()
    for i in range(len(lows)):
        if i in used_l:
            continue
        cluster = [i]
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) / (lows[i] + 1e-10) <= tolerance:
                cluster.append(j)
                used_l.add(j)
        if len(cluster) >= min_touches:
            equal_lows.append({
                "level": float(np.mean([lows[k] for k in cluster])),
                "touches": len(cluster),
                "times": [idx[k] for k in cluster],
            })

    current = df["close"].iloc[-1]
    above_h = sorted([h for h in equal_highs if h["level"] > current], key=lambda x: x["level"])
    below_l = sorted([l for l in equal_lows  if l["level"] < current], key=lambda x: -x["level"])

    return {
        "equal_highs_above": above_h[:4],
        "equal_lows_below":  below_l[:4],
    }


# ── Premium / Discount Zones ──────────────────────────────────────────────────

def detect_premium_discount(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """Uses the swing range to define premium (top 50%) and discount (bottom 50%)."""
    recent = df.iloc[-lookback:] if len(df) > lookback else df
    range_high = recent["high"].max()
    range_low  = recent["low"].min()
    rng = range_high - range_low
    if rng == 0:
        return {}

    equilibrium = (range_high + range_low) / 2
    premium_start  = equilibrium + rng * 0.25   # 75% of range
    discount_start = equilibrium - rng * 0.25   # 25% of range
    current = df["close"].iloc[-1]

    if current >= premium_start:
        zone = "Premium"
    elif current <= discount_start:
        zone = "Discount"
    else:
        zone = "Equilibrium"

    return {
        "range_high":      range_high,
        "range_low":       range_low,
        "equilibrium":     equilibrium,
        "premium_start":   premium_start,
        "discount_start":  discount_start,
        "current_zone":    zone,
        "fib_50":          equilibrium,
        "fib_618":         range_low + rng * 0.618,
        "fib_382":         range_low + rng * 0.382,
    }


# ── Supply and Demand Zones ───────────────────────────────────────────────────

def detect_supply_demand(df: pd.DataFrame, lookback: int = 80, threshold: float = 0.015) -> Dict:
    """Strong candles define supply/demand zones."""
    recent = df.iloc[-lookback:] if len(df) > lookback else df
    supply: List[dict] = []
    demand: List[dict] = []

    for i in range(len(recent) - 1):
        o = recent["open"].iloc[i]
        c = recent["close"].iloc[i]
        h = recent["high"].iloc[i]
        l = recent["low"].iloc[i]
        body_pct = abs(c - o) / o if o else 0

        if body_pct < threshold:
            continue

        if c > o:
            demand.append({"top": c, "bottom": o, "high": h, "low": l, "time": recent.index[i]})
        else:
            supply.append({"top": o, "bottom": c, "high": h, "low": l, "time": recent.index[i]})

    current = df["close"].iloc[-1]
    active_demand = sorted([d for d in demand if d["top"] < current], key=lambda x: -x["top"])[:3]
    active_supply = sorted([s for s in supply if s["bottom"] > current], key=lambda x: x["bottom"])[:3]

    return {"demand_zones": active_demand, "supply_zones": active_supply}


# ── Master function ───────────────────────────────────────────────────────────

def analyze_smc(df: pd.DataFrame) -> Dict:
    """Run all SMC analyses and return combined dict."""
    if len(df) < 20:
        return {}
    try:
        result = {}
        result.update(detect_bos_choch(df))
        result.update(detect_fvg(df))
        result.update(detect_order_blocks(df))
        result.update(detect_liquidity_zones(df))
        pd_zones = detect_premium_discount(df)
        result["premium_discount"] = pd_zones
        result.update(detect_supply_demand(df))
        result["current_price"] = float(df["close"].iloc[-1])
        return result
    except Exception:
        return {}
