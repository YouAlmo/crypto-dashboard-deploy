"""Lazy ML prediction service used by the Streamlit UI."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict

import pandas as pd


ML_UNAVAILABLE_MESSAGE = (
    "ML model unavailable. Prediction engine requires sufficient historical data "
    "or model dependencies."
)


def run_ml_prediction(symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Run ML predictions lazily so app startup remains fast."""
    if df is None or df.empty:
        return {"available": False, "reason": "Historical market data is unavailable."}

    try:
        from src.ml.models import train_and_predict

        ml_df = df.tail(min(len(df), 320)).copy()
        df_json = ml_df.to_json(date_format="iso")
        df_hash = hashlib.sha256(df_json.encode("utf-8")).hexdigest()
        result = train_and_predict(df_hash, df_json, symbol)
    except Exception as exc:
        return {"available": False, "reason": f"ML model unavailable: {exc}"}

    if not isinstance(result, dict):
        return {"available": False, "reason": "ML model returned no prediction output."}
    if result.get("error"):
        return {"available": False, "reason": result["error"]}

    result["available"] = result.get("combined_probability") is not None
    if not result["available"]:
        result["reason"] = "Model completed, but prediction confidence was unavailable."
    return result


def run_ml_prediction_from_app_context(
    symbol: str,
    cfg: dict,
    load_full_data: Callable[..., pd.DataFrame] | None,
) -> Dict[str, Any]:
    """Load a compact dataset from app context and execute ML on demand."""
    if load_full_data is None:
        return {"available": False, "reason": "Market data loader is unavailable."}

    timeframe = cfg.get("timeframe", "1h") if isinstance(cfg, dict) else "1h"
    limit = min(int(cfg.get("limit", 320)), 320) if isinstance(cfg, dict) else 320

    try:
        df = load_full_data(symbol, timeframe, limit)
    except Exception as exc:
        return {"available": False, "reason": f"Historical market data unavailable: {exc}"}

    return run_ml_prediction(symbol, df)
