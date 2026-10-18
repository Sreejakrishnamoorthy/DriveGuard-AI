"""
Dashboard page: upload road/driver videos, run analysis, and display
live results.

Processing runs IN-PROCESS using the src/ analyzer pipeline directly
(RoadAnalyzer + DriverAnalyzer + PhoneDetector + RiskCalculator +
AlertManager + ConfidenceScorer + IncidentRecorder + ExplanationService),
rather than proxying through the FastAPI backend. This keeps the core
"upload a video and see results" workflow self-contained — it works even
if the FastAPI/MongoDB backend isn't running. The History and Analytics
pages, by contrast, read from the backend API (which reads from
MongoDB) where persistence matters more than immediacy.
"""

import tempfile
import os
import time

import cv2
import streamlit as st

from frontend.components.video_panel import render_video_panel
from frontend.components.risk_card import render_risk_card
from frontend.components.metric_cards import render_driver_metrics, render_road_metrics
from frontend.components.alert_panel import render_alerts, get_top_priority_alert
from frontend.components.charts import render_event_summary_charts
from frontend.components.voice_alert import render_voice_toggle, speak_alert_if_new
from frontend.components.confidence_panel import render_confidence
from frontend.components.risk_explanation_panel import render_risk_explanation
from frontend.components.near_miss_panel import render_near_miss

from src.road_analysis.road_analyzer import RoadAnalyzer
from src.driver_monitoring.driver_analyzer import DriverAnalyzer
from src.driver_monitoring.phone_detector import PhoneDetector
from src.risk_engine.risk_calculator import RiskCalculator, RiskFactors
from src.risk_engine.alert_manager import AlertManager
from src.risk_engine.confidence_scorer import compute_confidence
from src.risk_engine.incident_recorder import IncidentRecorder
from src.llm.explanation_service import ExplanationService, RiskContext
from src.utils.config import Config


MAX_FRAMES_TO_PROCESS = 300  # cap for UI responsiveness in a single run


def _get_pipeline():
    """Lazily create and cache the analysis pipeline in session state so
    models load once per browser session, not once per script rerun."""
    if "pipeline" not in st.session_state:
        with st.spinner("Loading models (first run only)..."):
            st.session_state.pipeline = {
                "road_analyzer": RoadAnalyzer(confidence_threshold=Config.CONFIDENCE_THRESHOLD),
                "driver_analyzer": DriverAnalyzer(),
                "phone_detector": PhoneDetector(),
                "risk_calculator": RiskCalculator(),
                "alert_manager": AlertManager(),
                "explanation_service": ExplanationService(),
                "incident_recorder": IncidentRecorder(),
            }
    return st.session_state.pipeline


def _derive_risk_factors(driver_result, road_result, phone_detected: bool = False) -> RiskFactors:
    """Same derivation logic as backend/services/analysis_service.py, kept
    local to avoid a hard dependency from frontend -> backend package.
    Phase 25 — phone detection is folded into `distraction` via OR."""
    drowsiness = bool(driver_result and driver_result.drowsiness)
    head_pose_distracted = bool(driver_result and driver_result.attention == "DISTRACTED")
    distraction = head_pose_distracted or phone_detected
    lane_violation = bool(road_result and road_result.lane and road_result.lane.lane_departure)
    road_hazard = bool(road_result and road_result.hazards and road_result.hazards.hazard_detected)
    unsafe_distance = bool(road_result and road_result.following_distance == "UNSAFE")

    return RiskFactors(
        drowsiness=drowsiness,
        unsafe_following_distance=unsafe_distance,
        lane_violation=lane_violation,
        road_hazard=road_hazard,
        distraction=distraction,
    )


def _save_upload_to_tempfile(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def render() -> None:
    """Render the main dashboard page."""
    st.markdown("### Upload Videos")

    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        road_file = st.file_uploader("ROAD VIDEO", type=["mp4", "avi", "mov"], key="road_upload")
    with col_upload2:
        driver_file = st.file_uploader("DRIVER VIDEO", type=["mp4", "avi", "mov"], key="driver_upload")

    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
    start_clicked = col_btn1.button("▶ START ANALYSIS", type="primary", width='stretch')
    stop_clicked = col_btn2.button("■ STOP", width='stretch')
    voice_enabled = col_btn3.container().checkbox("🔊 Voice", value=False, key="voice_alerts_enabled_dash")

    if stop_clicked:
        st.session_state.stop_requested = True

    if start_clicked:
        if road_file is None and driver_file is None:
            st.error("Please upload at least one video (road or driver).")
            return
        st.session_state.stop_requested = False
        _run_analysis(road_file, driver_file)

    _render_latest_results(voice_enabled)


def _run_analysis(road_file, driver_file) -> None:
    """Process the uploaded video(s) frame by frame and update session state."""
    pipeline = _get_pipeline()
    road_analyzer: RoadAnalyzer = pipeline["road_analyzer"]
    driver_analyzer: DriverAnalyzer = pipeline["driver_analyzer"]
    phone_detector: PhoneDetector = pipeline["phone_detector"]
    risk_calculator: RiskCalculator = pipeline["risk_calculator"]
    alert_manager: AlertManager = pipeline["alert_manager"]
    explanation_service: ExplanationService = pipeline["explanation_service"]
    incident_recorder: IncidentRecorder = pipeline["incident_recorder"]

    road_path = _save_upload_to_tempfile(road_file) if road_file else None
    driver_path = _save_upload_to_tempfile(driver_file) if driver_file else None

    road_cap = cv2.VideoCapture(road_path) if road_path else None
    driver_cap = cv2.VideoCapture(driver_path) if driver_path else None

    fps = 30.0
    if road_cap is not None:
        fps = road_cap.get(cv2.CAP_PROP_FPS) or 30.0
    elif driver_cap is not None:
        fps = driver_cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps

    st.session_state.risk_history = []
    st.session_state.alert_history = []
    st.session_state.incidents_this_run = []

    progress = st.progress(0, text="Processing video...")
    road_panel = st.empty()
    driver_panel = st.empty()
    risk_panel = st.empty()

    t = 0.0
    frame_idx = 0

    try:
        while frame_idx < MAX_FRAMES_TO_PROCESS:
            if st.session_state.get("stop_requested"):
                st.warning("Analysis stopped by user.")
                break

            road_frame, driver_frame = None, None
            road_ok, driver_ok = False, False

            if road_cap is not None:
                road_ok, road_frame = road_cap.read()
            if driver_cap is not None:
                driver_ok, driver_frame = driver_cap.read()

            if road_cap is not None and not road_ok:
                road_cap = None
            if driver_cap is not None and not driver_ok:
                driver_cap = None

            if road_cap is None and driver_cap is None:
                break

            road_result = road_analyzer.analyze(road_frame, timestamp=t) if road_frame is not None else None
            driver_result = (
                driver_analyzer.analyze(driver_frame, timestamp=t) if driver_frame is not None else None
            )
            phone_result = phone_detector.detect(driver_frame) if driver_frame is not None else None
            phone_detected = bool(phone_result and phone_result.phone_detected)

            factors = _derive_risk_factors(driver_result, road_result, phone_detected)
            risk_result = risk_calculator.calculate(factors)

            near_miss = road_result.near_miss if road_result else None
            new_alerts = alert_manager.process(
                factors,
                risk_result,
                timestamp=t,
                near_miss_level=near_miss.near_miss_level if near_miss else None,
                near_miss_ttc_seconds=near_miss.time_to_collision_seconds if near_miss else None,
            )

            confidence_result = compute_confidence(road_result=road_result, driver_result=driver_result)

            st.session_state.risk_history.append(
                {"timestamp": time.time(), "risk_score": risk_result.risk_score, "risk_level": risk_result.risk_level}
            )
            for a in new_alerts:
                st.session_state.alert_history.append(a.to_dict())
                # Phase 27 — auto-record an incident for HIGH/CRITICAL alerts
                if IncidentRecorder.should_record(a.severity):
                    rec = incident_recorder.record(
                        alert_type=a.type,
                        severity=a.severity,
                        message=a.message,
                        risk_score=risk_result.risk_score,
                        risk_level=risk_result.risk_level,
                        factor_breakdown=risk_result.factor_breakdown,
                        road_status=road_result.to_dict() if road_result else None,
                        driver_status=driver_result.to_dict() if driver_result else None,
                        confidence=confidence_result.to_dict(),
                        road_frame=(
                            road_analyzer.draw_annotations(road_frame.copy(), road_result)
                            if road_result is not None else None
                        ),
                        driver_frame=(
                            driver_analyzer.draw_annotations(driver_frame.copy(), driver_result)
                            if driver_result is not None else None
                        ),
                        timestamp=t,
                    )
                    st.session_state.incidents_this_run.append(rec.incident_id)

            frame_idx += 1
            t += dt

            if frame_idx % 5 == 0 or frame_idx == 1:
                progress.progress(
                    min(frame_idx / MAX_FRAMES_TO_PROCESS, 1.0),
                    text=f"Processing frame {frame_idx}...",
                )
                if road_result is not None:
                    annotated_road = road_analyzer.draw_annotations(road_frame.copy(), road_result)
                    with road_panel.container():
                        render_video_panel("ROAD CAMERA", annotated_road)
                if driver_result is not None:
                    annotated_driver = driver_analyzer.draw_annotations(driver_frame.copy(), driver_result)
                    with driver_panel.container():
                        render_video_panel("DRIVER CAMERA", annotated_driver)
                with risk_panel.container():
                    render_risk_card(risk_result.risk_score, risk_result.risk_level)

            st.session_state.last_road_result = road_result
            st.session_state.last_driver_result = driver_result
            st.session_state.last_risk_result = risk_result
            st.session_state.last_confidence_result = confidence_result
            st.session_state.last_phone_detected = phone_detected

    finally:
        if road_cap is not None:
            road_cap.release()
        if driver_cap is not None:
            driver_cap.release()
        if road_path:
            os.remove(road_path)
        if driver_path:
            os.remove(driver_path)

    progress.progress(1.0, text="Done.")

    if st.session_state.get("last_risk_result"):
        rr = st.session_state.last_risk_result
        factors = _derive_risk_factors(
            st.session_state.get("last_driver_result"),
            st.session_state.get("last_road_result"),
            st.session_state.get("last_phone_detected", False),
        )
        context = RiskContext(
            risk_score=rr.risk_score,
            risk_level=rr.risk_level,
            drowsiness=factors.drowsiness,
            distraction=factors.distraction,
            lane_violation=factors.lane_violation,
            unsafe_following_distance=factors.unsafe_following_distance,
            road_hazard=factors.road_hazard,
        )
        st.session_state.last_explanation = explanation_service.generate_explanation(context).text

    st.session_state.analysis_complete = True


def _render_latest_results(voice_enabled: bool = False) -> None:
    """Render the most recent analysis results, if any exist in session state."""
    if not st.session_state.get("analysis_complete"):
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            render_video_panel("ROAD CAMERA")
        with col2:
            render_video_panel("DRIVER CAMERA")
        return

    st.markdown("---")

    risk_result = st.session_state.get("last_risk_result")
    driver_result = st.session_state.get("last_driver_result")
    road_result = st.session_state.get("last_road_result")
    confidence_result = st.session_state.get("last_confidence_result")
    phone_detected = st.session_state.get("last_phone_detected", False)
    explanation = st.session_state.get("last_explanation", "")
    alert_history = st.session_state.get("alert_history", [])

    if risk_result:
        render_risk_card(risk_result.risk_score, risk_result.risk_level, explanation)

    # Phase 19 — Voice Alerts: speak the single most urgent alert from this run
    top_alert = get_top_priority_alert(alert_history)
    if top_alert:
        speak_alert_if_new(top_alert.get("message"), voice_enabled)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if driver_result:
            render_driver_metrics(driver_result.to_dict())
            if phone_detected:
                st.warning("📱 Phone use detected in driver camera")
    with col2:
        if road_result:
            render_road_metrics(road_result.to_dict())
            render_near_miss(road_result.to_dict().get("near_miss"))

    st.markdown("---")
    render_alerts(alert_history)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        if risk_result:
            render_risk_explanation(risk_result.explain())
    with col4:
        if confidence_result:
            render_confidence(confidence_result.to_dict())

    incidents_this_run = st.session_state.get("incidents_this_run", [])
    if incidents_this_run:
        st.markdown("---")
        st.markdown(f"##### 📼 {len(incidents_this_run)} Incident(s) Recorded This Run")
        st.caption("High-severity moments were automatically saved. See the Incidents tab to review them.")

    st.markdown("---")
    render_event_summary_charts(alert_history)
