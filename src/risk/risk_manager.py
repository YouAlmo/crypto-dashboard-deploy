import numpy as np
import pandas as pd
from typing import Dict, Optional


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float = 0.01,
    max_position_pct: float = 0.25,
) -> Dict:
    risk_amount = capital * risk_pct
    price_risk = abs(entry_price - stop_loss_price)
    if price_risk == 0:
        return {"units": 0, "position_value": 0, "risk_amount": 0, "error": "Invalid stop loss"}

    units = risk_amount / price_risk
    position_value = units * entry_price
    max_position_value = capital * max_position_pct
    if position_value > max_position_value:
        position_value = max_position_value
        units = position_value / entry_price

    return {
        "units": round(units, 6),
        "position_value": round(position_value, 2),
        "risk_amount": round(risk_amount, 2),
        "position_pct": round(position_value / capital * 100, 2),
        "stop_loss_price": round(stop_loss_price, 4),
    }


def calculate_stop_loss(
    entry_price: float,
    atr: float,
    direction: str = "long",
    atr_multiplier: float = 2.0,
    fixed_pct: Optional[float] = None,
) -> float:
    if fixed_pct is not None:
        if direction == "long":
            return entry_price * (1 - fixed_pct)
        else:
            return entry_price * (1 + fixed_pct)

    sl_distance = atr * atr_multiplier
    if direction == "long":
        return entry_price - sl_distance
    else:
        return entry_price + sl_distance


def calculate_take_profit(
    entry_price: float,
    stop_loss_price: float,
    risk_reward_ratio: float = 2.0,
    direction: str = "long",
) -> float:
    risk = abs(entry_price - stop_loss_price)
    reward = risk * risk_reward_ratio
    if direction == "long":
        return entry_price + reward
    else:
        return entry_price - reward


def calculate_trailing_stop(
    current_price: float,
    highest_price: float,
    trail_pct: float = 0.03,
    direction: str = "long",
) -> float:
    if direction == "long":
        return highest_price * (1 - trail_pct)
    else:
        return highest_price * (1 + trail_pct)


def assess_risk(
    capital: float,
    entry_price: float,
    atr: float,
    signal_confidence: float = 0.5,
    risk_tolerance: str = "moderate",
) -> Dict:
    risk_pct_map = {
        "conservative": 0.005,
        "moderate": 0.01,
        "aggressive": 0.02,
    }
    max_pos_map = {
        "conservative": 0.10,
        "moderate": 0.20,
        "aggressive": 0.35,
    }

    risk_pct = risk_pct_map.get(risk_tolerance, 0.01)
    max_pos_pct = max_pos_map.get(risk_tolerance, 0.20)

    sl_price = calculate_stop_loss(entry_price, atr, atr_multiplier=2.0)
    tp_price = calculate_take_profit(entry_price, sl_price, risk_reward_ratio=2.0)
    sizing = calculate_position_size(capital, entry_price, sl_price, risk_pct, max_pos_pct)

    adjusted_confidence = min(signal_confidence * 1.2, 1.0)

    return {
        "stop_loss": round(sl_price, 4),
        "take_profit": round(tp_price, 4),
        "sl_pct": round((entry_price - sl_price) / entry_price * 100, 2),
        "tp_pct": round((tp_price - entry_price) / entry_price * 100, 2),
        "risk_reward": 2.0,
        "position_size": sizing,
        "signal_confidence": round(signal_confidence, 3),
        "adjusted_confidence": round(adjusted_confidence, 3),
        "risk_tolerance": risk_tolerance,
        "max_daily_loss_limit": round(capital * 0.05, 2),
    }


def calculate_portfolio_var(
    positions: list,
    confidence: float = 0.95,
    lookback: int = 252,
) -> Dict:
    if not positions:
        return {"var": 0.0, "cvar": 0.0}

    total_value = sum(p.get("value", 0) for p in positions)
    weighted_vol = sum(
        p.get("volatility", 0.02) * p.get("weight", 0) for p in positions
    )

    z_score = 1.645 if confidence == 0.95 else 2.326
    var = total_value * weighted_vol * z_score
    cvar = var * 1.4

    return {
        "var": round(var, 2),
        "cvar": round(cvar, 2),
        "var_pct": round(var / total_value * 100, 2) if total_value > 0 else 0,
        "confidence": confidence,
    }
