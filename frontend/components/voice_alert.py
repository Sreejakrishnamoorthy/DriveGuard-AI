"""
Voice Alert component — Phase 19.

Reads the top-priority alert aloud using the browser's built-in
SpeechSynthesis API (client-side JavaScript), not a server-side TTS
engine. This is a deliberate architectural choice: Streamlit runs on the
server, but audio needs to play in the user's browser — a server-side
TTS library (e.g. pyttsx3) would need its own audio output device on
the SERVER, which doesn't make sense for a web app and is unreliable in
headless/deployed environments. The browser's speech API works
everywhere a browser does, with no extra Python dependency.
"""

import html as html_lib
from typing import Optional

import streamlit as st


def _speak_js(text: str, rate: float = 1.0) -> str:
    """Build the JS snippet that speaks `text` via the Web Speech API."""
    safe_text = html_lib.escape(text).replace("`", "'")
    return f"""
    <script>
    (function() {{
        try {{
            const synth = window.speechSynthesis;
            if (!synth) {{ return; }}
            synth.cancel();  // don't overlap with any still-speaking alert
            const utterance = new SpeechSynthesisUtterance(`{safe_text}`);
            utterance.rate = {rate};
            synth.speak(utterance);
        }} catch (e) {{
            console.warn("DriveGuard AI: speech synthesis unavailable", e);
        }}
    }})();
    </script>
    """


def render_voice_toggle(default: bool = False) -> bool:
    """Render the voice-alerts on/off toggle. Returns the current setting."""
    return st.checkbox("🔊 Voice Alerts", value=default, key="voice_alerts_enabled")


def speak_alert_if_new(alert_text: Optional[str], enabled: bool) -> None:
    """
    Speak `alert_text` aloud IF voice alerts are enabled AND this exact
    text hasn't already been spoken (tracked in session state, so a
    Streamlit rerun doesn't repeat the same alert on every script pass —
    only genuinely new alert text triggers speech).

    Args:
        alert_text: The message to speak, or None/empty to speak nothing
        enabled: Whether voice alerts are currently turned on
    """
    if not enabled or not alert_text:
        return

    last_spoken = st.session_state.get("_last_spoken_alert_text")
    if alert_text == last_spoken:
        return

    st.session_state["_last_spoken_alert_text"] = alert_text
    # st.iframe's src accepts raw HTML/JS and executes it — used here only
    # to run a tiny, fully self-generated (not user-supplied) script that
    # triggers browser speech synthesis; height kept minimal since there's
    # no visible content, just a side-effect.
    st.iframe(src=_speak_js(alert_text), height=1)
