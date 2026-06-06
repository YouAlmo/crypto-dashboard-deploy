import streamlit as st
import plotly.graph_objects as go


def render_price_chart(fig):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )