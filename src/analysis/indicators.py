import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# Core Indicator Pipeline
# ─────────────────────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    df = add_rsi(df)

    df = add_macd(df)

    df = add_ema(
        df,
        periods=[9, 21, 50, 100, 200],
    )

    df = add_ema_crossover(df)

    df = add_bollinger_bands(df)

    df = add_atr(df)

    df = add_adx(df)

    df = add_vwap(df)

    df = add_volume_profile(df)

    df = add_volume_sma(df)

    df = add_momentum(df)

    return df


# ─────────────────────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────────────────────

def add_rsi(
    df: pd.DataFrame,
    period: int = 14,
    col: str = "close",
) -> pd.DataFrame:

    delta = df[col].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

    df["rsi"] = df["rsi"].fillna(50)

    return df


# ─────────────────────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────────────────────

def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.DataFrame:

    ema_fast = df[col].ewm(
        span=fast,
        adjust=False,
    ).mean()

    ema_slow = df[col].ewm(
        span=slow,
        adjust=False,
    ).mean()

    df["macd"] = ema_fast - ema_slow

    df["macd_signal"] = df["macd"].ewm(
        span=signal,
        adjust=False,
    ).mean()

    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    return df


# ─────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────

def add_ema(
    df: pd.DataFrame,
    periods: list,
    col: str = "close",
) -> pd.DataFrame:

    for p in periods:

        df[f"ema_{p}"] = df[col].ewm(
            span=p,
            adjust=False,
        ).mean()

    return df


# ─────────────────────────────────────────────────────────────
# EMA Crossovers
# ─────────────────────────────────────────────────────────────

def add_ema_crossover(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if (
        "ema_9" not in df.columns
        or "ema_21" not in df.columns
    ):
        return df

    above = (
        df["ema_9"] > df["ema_21"]
    ).fillna(False)

    prev = above.shift(1).fillna(False)

    df["ema9_above_ema21"] = above

    df["ema_bullish_cross"] = (
        (~prev) & above
    )

    df["ema_bearish_cross"] = (
        prev & (~above)
    )

    return df


# ─────────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────────

def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    col: str = "close",
) -> pd.DataFrame:

    sma = df[col].rolling(
        window=period
    ).mean()

    std = df[col].rolling(
        window=period
    ).std()

    df["bb_upper"] = sma + (std_dev * std)

    df["bb_middle"] = sma

    df["bb_lower"] = sma - (std_dev * std)

    df["bb_width"] = (
        (df["bb_upper"] - df["bb_lower"])
        / df["bb_middle"]
    )

    df["bb_pct"] = (
        (df[col] - df["bb_lower"])
        / (df["bb_upper"] - df["bb_lower"])
    )

    return df


# ─────────────────────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────────────────────

def add_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:

    hl = df["high"] - df["low"]

    hc = (
        df["high"]
        - df["close"].shift(1)
    ).abs()

    lc = (
        df["low"]
        - df["close"].shift(1)
    ).abs()

    tr = pd.concat(
        [hl, hc, lc],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    return df


# ─────────────────────────────────────────────────────────────
# ADX
# ─────────────────────────────────────────────────────────────

def add_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:

    up_move = (
        df["high"].diff()
    )

    down_move = (
        -df["low"].diff()
    )

    plus_dm = np.where(
        (up_move > down_move)
        & (up_move > 0),
        up_move,
        0,
    )

    minus_dm = np.where(
        (down_move > up_move)
        & (down_move > 0),
        down_move,
        0,
    )

    tr1 = (
        df["high"] - df["low"]
    )

    tr2 = (
        df["high"]
        - df["close"].shift()
    ).abs()

    tr3 = (
        df["low"]
        - df["close"].shift()
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(
        period
    ).mean()

    plus_di = (
        100
        * pd.Series(plus_dm).rolling(period).mean()
        / atr
    )

    minus_di = (
        100
        * pd.Series(minus_dm).rolling(period).mean()
        / atr
    )

    dx = (
        (plus_di - minus_di).abs()
        / (plus_di + minus_di)
    ) * 100

    df["adx"] = dx.rolling(period).mean()

    return df


# ─────────────────────────────────────────────────────────────
# VWAP
# ─────────────────────────────────────────────────────────────

def add_vwap(
    df: pd.DataFrame,
) -> pd.DataFrame:

    typical_price = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    cumulative_vp = (
        typical_price * df["volume"]
    ).cumsum()

    cumulative_volume = (
        df["volume"]
    ).cumsum()

    df["vwap"] = (
        cumulative_vp / cumulative_volume
    )

    return df


# ─────────────────────────────────────────────────────────────
# Volume SMA
# ─────────────────────────────────────────────────────────────

def add_volume_sma(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:

    df["volume_sma"] = (
        df["volume"]
        .rolling(window=period)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        / df["volume_sma"]
    )

    return df


# ─────────────────────────────────────────────────────────────
# Momentum
# ─────────────────────────────────────────────────────────────

def add_momentum(
    df: pd.DataFrame,
    period: int = 10,
) -> pd.DataFrame:

    df["momentum"] = (
        df["close"]
        - df["close"].shift(period)
    )

    df["momentum_pct"] = (
        df["close"].pct_change(period)
    ) * 100

    return df


# ─────────────────────────────────────────────────────────────
# Volume Profile
# ─────────────────────────────────────────────────────────────

def add_volume_profile(
    df: pd.DataFrame,
    bins: int = 20,
) -> pd.DataFrame:

    try:

        price_bins = pd.cut(
            df["close"],
            bins=bins,
        )

        volume_profile = (
            df.groupby(price_bins)["volume"]
            .sum()
        )

        poc = volume_profile.idxmax()

        df["poc"] = (
            poc.mid
            if hasattr(poc, "mid")
            else df["close"].mean()
        )

    except Exception:

        df["poc"] = df["close"].mean()

    return df


# ─────────────────────────────────────────────────────────────
# Current Snapshot
# ─────────────────────────────────────────────────────────────

def get_current_indicator_values(
    df: pd.DataFrame,
) -> dict:

    if df is None or df.empty:
        return {}

    last = df.iloc[-1]

    prev = (
        df.iloc[-2]
        if len(df) > 1
        else last
    )

    return {

        "close": round(
            last.get("close", 0),
            4,
        ),

        "rsi": round(
            last.get("rsi", 50),
            2,
        ),

        "adx": round(
            last.get("adx", 0),
            2,
        ),

        "macd": round(
            last.get("macd", 0),
            4,
        ),

        "macd_signal": round(
            last.get("macd_signal", 0),
            4,
        ),

        "macd_hist": round(
            last.get("macd_hist", 0),
            4,
        ),

        "ema_9": round(
            last.get("ema_9", 0),
            4,
        ),

        "ema_21": round(
            last.get("ema_21", 0),
            4,
        ),

        "ema_50": round(
            last.get("ema_50", 0),
            4,
        ),

        "ema_100": round(
            last.get("ema_100", 0),
            4,
        ),

        "ema_200": round(
            last.get("ema_200", 0),
            4,
        ),

        "atr": round(
            last.get("atr", 0),
            4,
        ),

        "vwap": round(
            last.get("vwap", 0),
            4,
        ),

        "poc": round(
            last.get("poc", 0),
            4,
        ),

        "momentum": round(
            last.get("momentum", 0),
            4,
        ),

        "momentum_pct": round(
            last.get("momentum_pct", 0),
            2,
        ),

        "volume_ratio": round(
            last.get("volume_ratio", 1),
            2,
        ),

        "bb_upper": round(
            last.get("bb_upper", 0),
            4,
        ),

        "bb_middle": round(
            last.get("bb_middle", 0),
            4,
        ),

        "bb_lower": round(
            last.get("bb_lower", 0),
            4,
        ),

        "bb_pct": round(
            last.get("bb_pct", 0.5),
            4,
        ),

        "ema_bullish_cross": bool(
            last.get(
                "ema_bullish_cross",
                False,
            )
        ),

        "ema_bearish_cross": bool(
            last.get(
                "ema_bearish_cross",
                False,
            )
        ),

        "ema9_above_ema21": bool(
            last.get(
                "ema9_above_ema21",
                False,
            )
        ),

        "prev_rsi": round(
            prev.get("rsi", 50),
            2,
        ),

        "prev_macd": round(
            prev.get("macd", 0),
            4,
        ),
    }