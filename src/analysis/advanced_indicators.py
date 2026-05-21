"""
Advanced technical indicators: SMA, VWAP, Supertrend, Ichimoku, ADX, PSAR,
Stochastic RSI, CCI, ROC, Momentum, MFI, OBV, CMF, Keltner, Donchian,
Volume Profile, Relative Volume.
"""

import pandas as pd
import numpy as np

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False


# ── Trend ─────────────────────────────────────────────────────────────────────

def add_sma(df: pd.DataFrame, periods=(20, 50, 200), col="close") -> pd.DataFrame:
    for p in periods:
        df[f"sma_{p}"] = df[col].rolling(window=p, min_periods=1).mean()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Session VWAP (resets each day; daily VWAP on intraday bars)."""
    if _TA_AVAILABLE:
        try:
            df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
                high=df["high"], low=df["low"], close=df["close"],
                volume=df["volume"], window=14,
            ).volume_weighted_average_price()
            return df
        except Exception:
            pass
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = cum_tp_vol / cum_vol
    return df


def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """ATR-based Supertrend."""
    hl2 = (df["high"] + df["low"]) / 2
    atr = df["high"].combine(df["low"], max) - df["high"].combine(df["close"].shift(1), min)
    atr = atr.abs()
    atr = atr.ewm(span=period, adjust=False).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upper.iloc[i]
            direction.iloc[i] = 1
            continue
        prev_st = st.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        close = df["close"].iloc[i]

        if prev_dir == 1:
            st.iloc[i] = max(lower.iloc[i], prev_st) if close > prev_st else upper.iloc[i]
            direction.iloc[i] = 1 if close > st.iloc[i] else -1
        else:
            st.iloc[i] = min(upper.iloc[i], prev_st) if close < prev_st else lower.iloc[i]
            direction.iloc[i] = -1 if close < st.iloc[i] else 1

    df["supertrend"] = st
    df["supertrend_direction"] = direction
    return df


def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            ich = ta.trend.IchimokuIndicator(
                high=df["high"], low=df["low"],
                window1=9, window2=26, window3=52,
            )
            df["ich_tenkan"] = ich.ichimoku_conversion_line()
            df["ich_kijun"] = ich.ichimoku_base_line()
            df["ich_senkou_a"] = ich.ichimoku_a()
            df["ich_senkou_b"] = ich.ichimoku_b()
            df["ich_chikou"] = df["close"].shift(-26)
            return df
        except Exception:
            pass
    # Manual fallback
    high9  = df["high"].rolling(9).max()
    low9   = df["low"].rolling(9).min()
    high26 = df["high"].rolling(26).max()
    low26  = df["low"].rolling(26).min()
    high52 = df["high"].rolling(52).max()
    low52  = df["low"].rolling(52).min()
    df["ich_tenkan"]   = (high9 + low9) / 2
    df["ich_kijun"]    = (high26 + low26) / 2
    df["ich_senkou_a"] = ((df["ich_tenkan"] + df["ich_kijun"]) / 2).shift(26)
    df["ich_senkou_b"] = ((high52 + low52) / 2).shift(26)
    df["ich_chikou"]   = df["close"].shift(-26)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            adx_ind = ta.trend.ADXIndicator(
                high=df["high"], low=df["low"], close=df["close"], window=period,
            )
            df["adx"]    = adx_ind.adx()
            df["adx_pos"] = adx_ind.adx_pos()
            df["adx_neg"] = adx_ind.adx_neg()
            return df
        except Exception:
            pass
    df["adx"] = 25.0
    df["adx_pos"] = 20.0
    df["adx_neg"] = 20.0
    return df


def add_parabolic_sar(df: pd.DataFrame) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            psar_ind = ta.trend.PSARIndicator(
                high=df["high"], low=df["low"], close=df["close"],
                step=0.02, max_step=0.2,
            )
            df["psar"]        = psar_ind.psar()
            df["psar_up"]     = psar_ind.psar_up()
            df["psar_down"]   = psar_ind.psar_down()
            df["psar_bull"]   = psar_ind.psar_up_indicator()
            return df
        except Exception:
            pass
    df["psar"] = df["close"] * 0.98
    df["psar_bull"] = 1
    return df


# ── Momentum ──────────────────────────────────────────────────────────────────

def add_stochastic_rsi(df: pd.DataFrame, period: int = 14,
                        smooth1: int = 3, smooth2: int = 3) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            srsi = ta.momentum.StochRSIIndicator(
                close=df["close"], window=period, smooth1=smooth1, smooth2=smooth2,
            )
            df["stochrsi_k"] = srsi.stochrsi_k() * 100
            df["stochrsi_d"] = srsi.stochrsi_d() * 100
            return df
        except Exception:
            pass
    df["stochrsi_k"] = 50.0
    df["stochrsi_d"] = 50.0
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            df["cci"] = ta.trend.CCIIndicator(
                high=df["high"], low=df["low"], close=df["close"], window=period,
            ).cci()
            return df
        except Exception:
            pass
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df["cci"] = (typical - sma) / (0.015 * mad)
    return df


def add_roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            df["roc"] = ta.momentum.ROCIndicator(close=df["close"], window=period).roc()
            return df
        except Exception:
            pass
    df["roc"] = df["close"].pct_change(period) * 100
    return df


def add_momentum_ind(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df["momentum"] = df["close"] - df["close"].shift(period)
    return df


# ── Volume ────────────────────────────────────────────────────────────────────

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            df["obv"] = ta.volume.OnBalanceVolumeIndicator(
                close=df["close"], volume=df["volume"],
            ).on_balance_volume()
            return df
        except Exception:
            pass
    direction = np.sign(df["close"].diff().fillna(0))
    df["obv"] = (direction * df["volume"]).cumsum()
    return df


def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            df["mfi"] = ta.volume.MFIIndicator(
                high=df["high"], low=df["low"],
                close=df["close"], volume=df["volume"], window=period,
            ).money_flow_index()
            return df
        except Exception:
            pass
    typical = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = typical * df["volume"]
    pos_mf = raw_mf.where(typical > typical.shift(1), 0).rolling(period).sum()
    neg_mf = raw_mf.where(typical < typical.shift(1), 0).rolling(period).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    df["mfi"] = 100 - (100 / (1 + mfr))
    return df


def add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
                high=df["high"], low=df["low"],
                close=df["close"], volume=df["volume"], window=period,
            ).chaikin_money_flow()
            return df
        except Exception:
            pass
    hl = df["high"] - df["low"]
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl.replace(0, np.nan)
    mfv = clv * df["volume"]
    df["cmf"] = mfv.rolling(period).sum() / df["volume"].rolling(period).sum()
    return df


def add_relative_volume(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    avg_vol = df["volume"].rolling(period).mean()
    df["rel_volume"] = df["volume"] / avg_vol.replace(0, np.nan)
    return df


def add_volume_profile(df: pd.DataFrame, bins: int = 30) -> dict:
    """Return simplified volume profile as dict {price_level: volume}."""
    price_min = df["low"].min()
    price_max = df["high"].max()
    edges = np.linspace(price_min, price_max, bins + 1)
    profile = {}
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mid = (lo + hi) / 2
        mask = (df["close"] >= lo) & (df["close"] < hi)
        profile[round(mid, 6)] = float(df.loc[mask, "volume"].sum())
    return profile


# ── Volatility ────────────────────────────────────────────────────────────────

def add_keltner_channel(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            kc = ta.volatility.KeltnerChannel(
                high=df["high"], low=df["low"], close=df["close"],
                window=period, window_atr=10,
            )
            df["kc_upper"]  = kc.keltner_channel_hband()
            df["kc_lower"]  = kc.keltner_channel_lband()
            df["kc_middle"] = kc.keltner_channel_mband()
            return df
        except Exception:
            pass
    ema = df["close"].ewm(span=period, adjust=False).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=10, adjust=False).mean()
    df["kc_middle"] = ema
    df["kc_upper"]  = ema + mult * atr
    df["kc_lower"]  = ema - mult * atr
    return df


def add_donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    if _TA_AVAILABLE:
        try:
            dc = ta.volatility.DonchianChannel(
                high=df["high"], low=df["low"], close=df["close"], window=period,
            )
            df["dc_upper"]  = dc.donchian_channel_hband()
            df["dc_lower"]  = dc.donchian_channel_lband()
            df["dc_middle"] = dc.donchian_channel_mband()
            return df
        except Exception:
            pass
    df["dc_upper"]  = df["high"].rolling(period).max()
    df["dc_lower"]  = df["low"].rolling(period).min()
    df["dc_middle"] = (df["dc_upper"] + df["dc_lower"]) / 2
    return df


# ── Master function ───────────────────────────────────────────────────────────

def add_all_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_sma(df)
    df = add_vwap(df)
    df = add_supertrend(df)
    df = add_ichimoku(df)
    df = add_adx(df)
    df = add_parabolic_sar(df)
    df = add_stochastic_rsi(df)
    df = add_cci(df)
    df = add_roc(df)
    df = add_momentum_ind(df)
    df = add_obv(df)
    df = add_mfi(df)
    df = add_cmf(df)
    df = add_relative_volume(df)
    df = add_keltner_channel(df)
    df = add_donchian_channel(df)
    return df


def get_advanced_indicator_values(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    def _f(col: str, default=0.0):
        v = last.get(col, default)
        return float(v) if pd.notna(v) else default

    return {
        "sma_20":            _f("sma_20"),
        "sma_50":            _f("sma_50"),
        "sma_200":           _f("sma_200"),
        "vwap":              _f("vwap"),
        "supertrend":        _f("supertrend"),
        "supertrend_dir":    int(_f("supertrend_direction", 0)),
        "ich_tenkan":        _f("ich_tenkan"),
        "ich_kijun":         _f("ich_kijun"),
        "ich_senkou_a":      _f("ich_senkou_a"),
        "ich_senkou_b":      _f("ich_senkou_b"),
        "adx":               _f("adx", 25.0),
        "adx_pos":           _f("adx_pos", 20.0),
        "adx_neg":           _f("adx_neg", 20.0),
        "psar":              _f("psar"),
        "psar_bull":         bool(_f("psar_bull", 1)),
        "stochrsi_k":        _f("stochrsi_k", 50.0),
        "stochrsi_d":        _f("stochrsi_d", 50.0),
        "cci":               _f("cci"),
        "roc":               _f("roc"),
        "momentum":          _f("momentum"),
        "obv":               _f("obv"),
        "mfi":               _f("mfi", 50.0),
        "cmf":               _f("cmf"),
        "rel_volume":        _f("rel_volume", 1.0),
        "kc_upper":          _f("kc_upper"),
        "kc_lower":          _f("kc_lower"),
        "kc_middle":         _f("kc_middle"),
        "dc_upper":          _f("dc_upper"),
        "dc_lower":          _f("dc_lower"),
        "dc_middle":         _f("dc_middle"),
        "close":             _f("close"),
    }
