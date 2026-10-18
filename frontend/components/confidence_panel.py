"""
Confidence panel component — Phase 26.

Displays the confidence assessment for the current analysis, broken down
by which detection components contributed to it, so "how sure is this"
has a transparent, inspectable answer.
"""

from typing import Any, Dict, Optional

import streamlit as st

LABEL_COLORS = {
    "HIGH": "🟢",
    "MEDIUM": "🟡",
    "LOW": "🟠",
    "UNKNOWN": "⚪",
}


def render_confidence(confidence: Optional[Dict[str, Any]]) -> None:
    """
    Render the confidence assessment panel.

    Args:
        confidence: dict matching ConfidenceResult.to_dict() shape
    """
    st.markdown("##### Assessment Confidence")

    if not confidence or confidence.get("overall_confidence") is None:
        st.caption("Confidence unavailable this frame (no active detections to assess).")
        return

    overall = confidence["overall_confidence"]
    label = confidence.get("confidence_label", "UNKNOWN")
    icon = LABEL_COLORS.get(label, "⚪")

    st.markdown(f"{icon} **{label}** — overall confidence {overall:.0%}")

    components = confidence.get("components", [])
    if components:
        with st.expander("Confidence breakdown"):
            for c in components:
                conf_val = c.get("confidence")
                conf_text = f"{conf_val:.0%}" if conf_val is not None else "n/a"
                st.write(f"**{c.get('component')}**: {conf_text} — {c.get('note')}")
