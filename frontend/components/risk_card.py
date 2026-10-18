"""
Risk score card component: the large "SAFETY SCORE" display.
"""

import streamlit as st

RISK_LEVEL_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}

RISK_LEVEL_LABELS = {
    "LOW": "LOW RISK",
    "MEDIUM": "MODERATE RISK",
    "HIGH": "HIGH RISK",
    "CRITICAL": "CRITICAL RISK",
}


def get_risk_color(risk_level: str) -> str:
    """Map a risk level to a display color. Pure function, easily testable."""
    return RISK_LEVEL_COLORS.get(risk_level, "#6b7280")


def get_risk_label(risk_level: str) -> str:
    """Map a risk level to a display label. Pure function, easily testable."""
    return RISK_LEVEL_LABELS.get(risk_level, risk_level)


def render_risk_card(risk_score: int, risk_level: str, explanation: str = "") -> None:
    """
    Render the large safety-score card.

    Args:
        risk_score: 0-100 risk score
        risk_level: LOW / MEDIUM / HIGH / CRITICAL
        explanation: Optional AI-generated explanation text shown below the score
    """
    color = get_risk_color(risk_level)
    label = get_risk_label(risk_level)

    st.markdown(
        f"""
        <div style="
            background-color: {color}22;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 24px;
            text-align: center;
        ">
            <div style="font-size: 14px; letter-spacing: 2px; color: #6b7280; font-weight: 600;">
                SAFETY SCORE
            </div>
            <div style="font-size: 56px; font-weight: 800; color: {color}; line-height: 1.1;">
                {risk_score}<span style="font-size: 24px; color: #6b7280;"> / 100</span>
            </div>
            <div style="font-size: 18px; font-weight: 700; color: {color};">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if explanation:
        st.markdown("##### AI Explanation")
        st.write(explanation)
