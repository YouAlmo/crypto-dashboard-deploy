import re
import time
from typing import List, Dict
import requests
import streamlit as st


CRYPTO_KEYWORDS = {
    "BTC/USDT": ["bitcoin", "btc", "bitcoin price", "bitcoin market"],
    "ETH/USDT": ["ethereum", "eth", "ether", "ethereum price"],
    "SOL/USDT": ["solana", "sol", "solana price", "solana blockchain"],
}

SAMPLE_HEADLINES = {
    "BTC/USDT": [
        "Bitcoin surges past $65,000 as institutional demand grows",
        "BlackRock Bitcoin ETF sees record inflows this week",
        "Bitcoin network hashrate reaches all-time high",
        "Analysts predict Bitcoin could reach $100k by year end",
        "Bitcoin consolidates after recent rally, bulls remain cautious",
        "SEC approves additional Bitcoin ETFs, market reacts positively",
        "Bitcoin whale wallets accumulate aggressively at current levels",
        "Crypto market sees broad sell-off amid regulatory concerns",
    ],
    "ETH/USDT": [
        "Ethereum developers finalize next upgrade roadmap",
        "ETH staking yields attract institutional capital",
        "Ethereum layer-2 TVL surpasses $40 billion milestone",
        "DeFi protocols report record volumes on Ethereum",
        "Ethereum competitors gain market share amid high gas fees",
        "Ethereum foundation increases grants for ecosystem development",
        "ETH burns accelerate after recent network activity spike",
        "Ethereum price lags Bitcoin in latest market cycle",
    ],
    "SOL/USDT": [
        "Solana DeFi ecosystem reaches record $8 billion TVL",
        "Solana NFT marketplace volume overtakes Ethereum briefly",
        "Solana network experiences brief outage, quickly resolved",
        "Major DEX launches exclusively on Solana blockchain",
        "Solana mobile phone Saga 2 pre-orders exceed expectations",
        "Solana validators vote on major protocol upgrade",
        "SOL price rallies as memecoin activity spikes on-chain",
        "Institutional interest in Solana grows amid ETF speculation",
    ],
}


@st.cache_resource(show_spinner=False)
def _load_sentiment_pipeline():
    try:
        from transformers import pipeline
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    except Exception:
        return None


def analyze_sentiment(texts: List[str]) -> List[Dict]:
    pipe = _load_sentiment_pipeline()
    results = []
    for text in texts:
        if pipe is not None:
            try:
                out = pipe(text[:512])[0]
                label = out["label"]
                score = out["score"]
                sentiment = "positive" if label == "POSITIVE" else "negative"
                compound = score if sentiment == "positive" else -score
            except Exception:
                compound, sentiment = _rule_based_sentiment(text)
        else:
            compound, sentiment = _rule_based_sentiment(text)

        results.append(
            {
                "text": text,
                "sentiment": sentiment,
                "score": compound,
            }
        )
    return results


def _rule_based_sentiment(text: str):
    positive_words = [
        "surge", "rally", "gain", "record", "high", "bull", "positive",
        "approval", "grow", "milestone", "launch", "upgrade", "adopt",
        "institutional", "inflow", "accumulate",
    ]
    negative_words = [
        "crash", "drop", "fall", "bear", "sell-off", "concern", "lag",
        "outage", "fee", "regulation", "ban", "hack", "loss", "risk",
        "caution", "delay",
    ]
    lower = text.lower()
    pos = sum(1 for w in positive_words if w in lower)
    neg = sum(1 for w in negative_words if w in lower)
    if pos > neg:
        return 0.6, "positive"
    elif neg > pos:
        return -0.6, "negative"
    return 0.0, "neutral"


def get_news_sentiment(symbol: str) -> Dict:
    headlines = SAMPLE_HEADLINES.get(symbol, [])
    if not headlines:
        return {"overall": "neutral", "score": 0.0, "articles": [], "headline_count": 0}

    results = analyze_sentiment(headlines)
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    if avg_score > 0.15:
        overall = "positive"
    elif avg_score < -0.15:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "score": round(avg_score, 4),
        "articles": results,
        "headline_count": len(results),
    }
