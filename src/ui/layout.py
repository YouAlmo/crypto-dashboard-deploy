import streamlit as st
def get_theme_css():
    return """
    <style>

    .stApp {
        background: linear-gradient(
            180deg,
            #020617 0%,
            #071226 100%
        );
        color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1600px;
    }

    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    div[data-baseweb="tab-list"] {
        gap: 12px;
        background: transparent;
    }

    button[data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 12px 20px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: 0.2s ease;
    }

    button[data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.08);
        transform: translateY(-1px);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border-radius: 18px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    </style>
    """
def render_header():
    st.markdown("""
    <div style='margin-bottom:20px;'>
        <h1 style='font-size:48px;'>📈 SuperSignal</h1>
        <p style='opacity:0.7; font-size:18px;'>
            Institutional-grade crypto intelligence platform
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_tabs():
    return st.tabs([
        "📊 Overview",
        "📈 Technical",
        "💰 Smart Money",
        "📖 Order Book",
        "⏰ Multi-TF",
        "🤖 AI Signals",
        "🧪 Backtest",
        "📋 Portfolio",
    ])