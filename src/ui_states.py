"""Professional UI state helpers and compatibility patches."""

from __future__ import annotations

import inspect
from typing import Any, Dict

import pandas as pd

from src.ml_service import ML_UNAVAILABLE_MESSAGE, run_ml_prediction_from_app_context


def render_empty_state(st: Any, message: str, icon: str = "") -> None:
    prefix = f"{icon} " if icon else ""
    st.markdown(
        "<div style='padding:18px 20px;border-radius:18px;"
        "border:1px solid rgba(255,255,255,0.08);"
        "background:rgba(255,255,255,0.05);"
        "color:var(--muted);font-size:0.95rem;'>"
        f"{prefix}{message}</div>",
        unsafe_allow_html=True,
    )


def render_ml_prediction_state(st: Any, result: Dict[str, Any] | None) -> None:
    if not result or result.get("available") is False:
        render_empty_state(st, ML_UNAVAILABLE_MESSAGE)
        reason = (result or {}).get("reason")
        if reason:
            st.caption(reason)
        return

    direction = result.get("direction", "?")
    probability = float(result.get("combined_probability", 0.5) or 0.5)
    direction_color = "#26a69a" if direction == "UP" else "#ef5350"
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(
        "<div class='terminal-card' style='text-align:center'>"
        "<div class='metric-label'>ML Consensus</div>"
        f"<div style='font-size:1.8em;color:{direction_color};font-weight:700'>"
        f"{('UP' if direction == 'UP' else 'DOWN')} {direction}</div></div>",
        unsafe_allow_html=True,
    )
    mc2.metric("Up Probability", f"{probability * 100:.1f}%")
    rf_meta = ((result.get("rf") or {}).get("meta") or {})
    xgb_meta = ((result.get("xgb") or {}).get("meta") or {})
    mc3.metric("RF Accuracy", f"{rf_meta.get('test_accuracy', rf_meta.get('train_accuracy', 0)) * 100:.1f}%")
    mc4.metric("XGB Accuracy", f"{xgb_meta.get('test_accuracy', xgb_meta.get('train_accuracy', 0)) * 100:.1f}%")

    feature_importance = result.get("feature_importance", {})
    if feature_importance:
        import plotly.express as px

        fi_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=["Feature", "Importance"],
        ).sort_values("Importance")
        fig = px.bar(
            fi_df.tail(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Feature Importances",
            color="Importance",
            color_continuous_scale="teal",
        )
        fig.update_layout(
            height=240,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, width="stretch")


def _find_render_ai_context() -> tuple[str | None, dict | None, Any]:
    frame = inspect.currentframe()
    while frame is not None:
        if frame.f_code.co_name == "render_ai_signals":
            return (
                frame.f_locals.get("symbol"),
                frame.f_locals.get("cfg"),
                frame.f_globals.get("load_full_data"),
            )
        frame = frame.f_back
    return None, None, None


def install_streamlit_state_patches(st: Any) -> None:
    """Suppress persistent skeleton blocks and replace stale ML warnings."""
    if getattr(st, "_supersignal_state_patches", False):
        return

    original_markdown = st.markdown
    original_warning = st.warning

    def markdown_patched(body, *args, **kwargs):
        if isinstance(body, str) and "animation:pulse" in body:
            return None
        return original_markdown(body, *args, **kwargs)

    def warning_patched(body, *args, **kwargs):
        if body == "ML: No predictions available":
            symbol, cfg, load_full_data = _find_render_ai_context()
            with st.spinner("Loading ML prediction..."):
                result = run_ml_prediction_from_app_context(symbol or "", cfg or {}, load_full_data)
            render_ml_prediction_state(st, result)
            return None
        return original_warning(body, *args, **kwargs)

    st.markdown = markdown_patched
    st.warning = warning_patched
    st._supersignal_state_patches = True
