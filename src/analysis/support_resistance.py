import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def find_support_resistance(
    df: pd.DataFrame,
    window: int = 20,
    tolerance_pct: float = 0.015,
    max_levels: int = 5,
) -> Dict:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    resistance_raw = []
    support_raw = []

    for i in range(window, n - window):
        if high[i] == max(high[i - window : i + window + 1]):
            resistance_raw.append(high[i])
        if low[i] == min(low[i - window : i + window + 1]):
            support_raw.append(low[i])

    current_price = close[-1]

    def cluster_levels(levels: List[float], tol: float) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        group = [levels[0]]
        for lv in levels[1:]:
            if (lv - group[0]) / group[0] <= tol:
                group.append(lv)
            else:
                clusters.append(float(np.mean(group)))
                group = [lv]
        clusters.append(float(np.mean(group)))
        return clusters

    resistance_levels = cluster_levels(resistance_raw, tolerance_pct)
    support_levels = cluster_levels(support_raw, tolerance_pct)

    resistance_above = sorted(
        [r for r in resistance_levels if r > current_price * 1.001]
    )[:max_levels]
    support_below = sorted(
        [s for s in support_levels if s < current_price * 0.999], reverse=True
    )[:max_levels]

    nearest_resistance = resistance_above[0] if resistance_above else current_price * 1.05
    nearest_support = support_below[0] if support_below else current_price * 0.95

    sr_ratio = (nearest_resistance - current_price) / (current_price - nearest_support) if (current_price - nearest_support) > 0 else 1.0

    pivot = (df["high"].iloc[-1] + df["low"].iloc[-1] + df["close"].iloc[-1]) / 3
    r1 = 2 * pivot - df["low"].iloc[-1]
    s1 = 2 * pivot - df["high"].iloc[-1]
    r2 = pivot + (df["high"].iloc[-1] - df["low"].iloc[-1])
    s2 = pivot - (df["high"].iloc[-1] - df["low"].iloc[-1])

    return {
        "resistance": resistance_above,
        "support": support_below,
        "nearest_resistance": round(nearest_resistance, 4),
        "nearest_support": round(nearest_support, 4),
        "resistance_pct": round((nearest_resistance - current_price) / current_price * 100, 2),
        "support_pct": round((current_price - nearest_support) / current_price * 100, 2),
        "sr_ratio": round(sr_ratio, 2),
        "pivot": round(pivot, 4),
        "pivot_r1": round(r1, 4),
        "pivot_r2": round(r2, 4),
        "pivot_s1": round(s1, 4),
        "pivot_s2": round(s2, 4),
        "current_price": current_price,
    }
