"""
Risk explanation panel component — Phase 28.

Displays a transparent, per-factor breakdown of exactly which conditions
contributed how many points to the current risk score, instead of just
showing the final number.
"""

from typing import Any, Dict, Optional

import streamlit as st


def render_risk_explanation(explanation: Optional[Dict[str, Any]]) -> None:
    """
    Render the explainable-risk breakdown panel.

    Args:
        explanation: dict matching RiskScoreResult.explain() shape
    """
    st.markdown("##### Why This Score?")

    if not explanation:
        st.caption("No risk breakdown available yet.")
        return

    breakdown = explanation.get("factor_breakdown", [])

    if not breakdown:
        st.success(f"No risk factors currently active — score is {explanation.get('risk_score', 0)}/100.")
        return

    for factor in breakdown:
        st.write(f"**+{factor.get('points')} points** — {factor.get('description')}")

    if explanation.get("was_capped"):
        st.caption(
            f"Raw total was {explanation.get('raw_score_before_cap')}, "
            f"capped at 100 (maximum possible score)."
        )
