import pandas as pd
import numpy as np


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df = add_rsi(df)
    df = add_macd(df)
    df = add_ema(df, [9, 21, 50, 200])
    df = add_ema_crossover(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_sma(df)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.DataFrame:
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.DataFrame:
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_ema(df: pd.DataFrame, periods: list, col: str = "close") -> pd.DataFrame:
    for p in periods:
        df[f"ema_{p}"] = df[col].ewm(span=p, adjust=False).mean()
    return df


def add_ema_crossover(df: pd.DataFrame) -> pd.DataFrame:
    if "ema_9" not in df.columns or "ema_21" not in df.columns:
        return df
    # Keep as local bool Series — never re-read from DataFrame to avoid
    # pandas silently converting bool→float64 on column assignment.
    above: pd.Series = (df["ema_9"] > df["ema_21"]).fillna(False).astype(bool)
    prev: pd.Series = above.shift(1).fillna(False).astype(bool)
    df["ema9_above_ema21"] = above
    df["ema_bullish_cross"] = (~prev) & above
    df["ema_bearish_cross"] = prev & (~above)
    return df


def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close"
) -> pd.DataFrame:
    sma = df[col].rolling(window=period).mean()
    std = df[col].rolling(window=period).std()
    df["bb_upper"] = sma + std_dev * std
    df["bb_middle"] = sma
    df["bb_lower"] = sma - std_dev * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct"] = (df[col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=period - 1, min_periods=period).mean()
    return df


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["volume_sma"] = df["volume"].rolling(window=period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]
    return df


def get_current_indicator_values(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    return {
        "rsi": round(last.get("rsi", 50), 2),
        "macd": round(last.get("macd", 0), 4),
        "macd_signal": round(last.get("macd_signal", 0), 4),
        "macd_hist": round(last.get("macd_hist", 0), 4),
        "ema_9": round(last.get("ema_9", last["close"]), 4),
        "ema_21": round(last.get("ema_21", last["close"]), 4),
        "ema_50": round(last.get("ema_50", last["close"]), 4),
        "ema_200": round(last.get("ema_200", last["close"]), 4),
        "bb_upper": round(last.get("bb_upper", last["close"]), 4),
        "bb_middle": round(last.get("bb_middle", last["close"]), 4),
        "bb_lower": round(last.get("bb_lower", last["close"]), 4),
        "bb_pct": round(last.get("bb_pct", 0.5), 4),
        "atr": round(last.get("atr", 0), 4),
        "close": round(last["close"], 4),
        "volume_ratio": round(last.get("volume_ratio", 1.0), 2),
        "prev_rsi": round(prev.get("rsi", 50), 2),
        "prev_macd": round(prev.get("macd", 0), 4),
        "prev_ema_9": round(prev.get("ema_9", last["close"]), 4),
        "prev_ema_21": round(prev.get("ema_21", last["close"]), 4),
        "ema_bullish_cross": bool(last.get("ema_bullish_cross", False)),
        "ema_bearish_cross": bool(last.get("ema_bearish_cross", False)),
        "ema9_above_ema21": bool(last.get("ema9_above_ema21", False)),
    }
