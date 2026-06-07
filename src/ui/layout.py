import streamlit as st
from src.ui_states import install_streamlit_state_patches

install_streamlit_state_patches(st)

def get_theme_css():
    return """
    <style>
    .stApp {
        background: transparent;
        color: inherit;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1600px;
    }
    h1, h2, h3 {
        font-weight: 700 !important;
    }
    </style>
    """

def render_header():
    st.markdown("""
    <div class='app-header'>
        <div class='brand-mark'>SS</div>
        <div>
            <h1>SuperSignal</h1>
            <p>Institutional-grade crypto intelligence platform</p>
        </div>
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