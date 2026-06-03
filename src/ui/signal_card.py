import streamlit as st


def render_signal_card(signal_result: dict):
    signal = signal_result.get("signal", "HOLD")
    confidence = signal_result.get("confidence", 0)
    strength = signal_result.get("strength", "Neutral")
    risk = signal_result.get("risk_level", "Medium")
    bulls = signal_result.get("bull_signals", 0)
    bears = signal_result.get("bear_signals", 0)

    color_map = {
        "BUY": "#00c853",
        "SELL": "#ff5252",
        "HOLD": "#ffd600",
    }

    color = color_map.get(signal, "#90a4ae")

    st.markdown(
        f"""
        <div style="
            background:#0d1117;
            border:1px solid rgba(255,255,255,0.08);
            border-radius:18px;
            padding:24px;
            margin-bottom:18px;
            box-shadow:0 0 25px rgba(0,0,0,0.35);
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                <div>
                    <div style="font-size:0.9rem;color:#8b949e;">Institutional AI Signal</div>
                    <div style="font-size:2.6rem;font-weight:900;color:{color};">{signal}</div>
                    <div style="font-size:1rem;color:{color};font-weight:600;">
                        {confidence * 100:.1f}% Confidence
                    </div>
                </div>

                <div style="display:grid;grid-template-columns:repeat(2,minmax(120px,1fr));gap:12px;min-width:320px;">
                    <div style="background:#161b22;padding:12px;border-radius:12px;">
                        <div style="color:#8b949e;font-size:.8rem;">Strength</div>
                        <div style="font-size:1.1rem;font-weight:700;color:white;">{strength}</div>
                    </div>

                    <div style="background:#161b22;padding:12px;border-radius:12px;">
                        <div style="color:#8b949e;font-size:.8rem;">Risk Level</div>
                        <div style="font-size:1.1rem;font-weight:700;color:white;">{risk}</div>
                    </div>

                    <div style="background:#161b22;padding:12px;border-radius:12px;">
                        <div style="color:#8b949e;font-size:.8rem;">Bullish Signals</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#00c853;">{bulls}</div>
                    </div>

                    <div style="background:#161b22;padding:12px;border-radius:12px;">
                        <div style="color:#8b949e;font-size:.8rem;">Bearish Signals</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#ff5252;">{bears}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(float(confidence))
