import requests
import streamlit as st
from typing import Dict


FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=7"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_fear_greed_index() -> Dict:
    try:
        resp = requests.get(FEAR_GREED_URL, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            entries = data.get("data", [])
            if entries:
                latest = entries[0]
                history = [
                    {"value": int(e["value"]), "classification": e["value_classification"]}
                    for e in entries
                ]
                return {
                    "value": int(latest["value"]),
                    "classification": latest["value_classification"],
                    "history": history,
                    "source": "live",
                }
    except Exception:
        pass

    return {
        "value": 55,
        "classification": "Greed",
        "history": [
            {"value": 55, "classification": "Greed"},
            {"value": 48, "classification": "Neutral"},
            {"value": 62, "classification": "Greed"},
            {"value": 38, "classification": "Fear"},
            {"value": 71, "classification": "Greed"},
            {"value": 44, "classification": "Fear"},
            {"value": 58, "classification": "Greed"},
        ],
        "source": "fallback",
    }


def get_fg_color(value: int) -> str:
    if value <= 24:
        return "#e74c3c"
    if value <= 44:
        return "#e67e22"
    if value <= 55:
        return "#f1c40f"
    if value <= 74:
        return "#2ecc71"
    return "#27ae60"


def get_fg_emoji(classification: str) -> str:
    mapping = {
        "Extreme Fear": "😱",
        "Fear": "😨",
        "Neutral": "😐",
        "Greed": "😊",
        "Extreme Greed": "🤑",
    }
    return mapping.get(classification, "😐")
