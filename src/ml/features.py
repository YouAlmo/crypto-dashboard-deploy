import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lag in [1, 2, 3, 5, 10]:
        df[f"return_{lag}"] = df["close"].pct_change(lag)
        df[f"volume_change_{lag}"] = df["volume"].pct_change(lag)

    df["rsi_norm"] = (df["rsi"] - 50) / 50
    df["macd_norm"] = df["macd"] / df["close"]
    df["macd_hist_norm"] = df["macd_hist"] / df["close"]

    df["ema50_dist"] = (df["close"] - df["ema_50"]) / df["ema_50"]
    df["ema200_dist"] = (df["close"] - df["ema_200"]) / df["ema_200"]
    df["ema_cross"] = (df["ema_50"] > df["ema_200"]).astype(int)

    df["bb_pct_feat"] = df["bb_pct"]
    df["bb_width_norm"] = df["bb_width"]

    df["atr_pct"] = df["atr"] / df["close"]
    df["volume_ratio_feat"] = df["volume_ratio"].clip(0, 5)

    df["high_low_pct"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]

    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
    df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)

    return df


FEATURE_COLS = [
    "return_1", "return_2", "return_3", "return_5", "return_10",
    "volume_change_1", "volume_change_3",
    "rsi_norm", "macd_norm", "macd_hist_norm",
    "ema50_dist", "ema200_dist", "ema_cross",
    "bb_pct_feat", "bb_width_norm",
    "atr_pct", "volume_ratio_feat",
    "high_low_pct", "close_open_pct",
    "rsi_overbought", "rsi_oversold", "macd_bullish",
]


def prepare_training_data(df: pd.DataFrame, horizon: int = 5) -> tuple:
    df = build_features(df)
    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    X = df[FEATURE_COLS].values
    y = df["target"].values
    return X, y, df


def prepare_prediction_features(df: pd.DataFrame) -> tuple:
    df = build_features(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) == 0:
        return None, None
    X = df[FEATURE_COLS].values
    return X, df
