"""
Tests for temporal alert persistence and cooldown logic — verifying a
sustained condition produces occasional alerts, not one per frame.
"""

import pytest

from src.risk_engine.temporal_analyzer import TemporalAnalyzer
from src.risk_engine.alert_manager import AlertManager
from src.risk_engine.risk_calculator import RiskCalculator, RiskFactors


class TestTemporalAnalyzer:
    def test_brief_flicker_never_fires(self):
        """A condition active for less than the persistence window should never fire."""
        ta = TemporalAnalyzer(persistence_seconds=3.0, cooldown_seconds=10.0)
        t = 0.0
        dt = 1.0 / 30.0
        fired = False
        for _ in range(60):  # 2 seconds, under the 3-second threshold
            r = ta.update("LANE_DEPARTURE", True, timestamp=t)
            fired = fired or r.should_fire
            t += dt
        assert fired is False

    def test_sustained_condition_fires_once_not_per_frame(self):
        """The spec's exact scenario: ~90 frames (3s) of sustained drowsiness must fire ONCE, not 90 times."""
        ta = TemporalAnalyzer(persistence_seconds=3.0, cooldown_seconds=10.0)
        t = 0.0
        dt = 1.0 / 30.0
        fire_count = 0
        for _ in range(91):  # slightly past 3.0s to clear frame-boundary discretization
            r = ta.update("DROWSINESS", True, timestamp=t)
            if r.should_fire:
                fire_count += 1
            t += dt
        assert fire_count == 1

    def test_fires_again_after_cooldown(self):
        ta = TemporalAnalyzer(persistence_seconds=3.0, cooldown_seconds=10.0)
        t = 0.0
        dt = 1.0 / 30.0
        fire_count = 0
        for _ in range(int(30 * 14)):  # 14 seconds of sustained condition
            r = ta.update("DROWSINESS", True, timestamp=t)
            if r.should_fire:
                fire_count += 1
            t += dt
        # One at ~3s, one more after the 10s cooldown elapses
        assert fire_count == 2

    def test_condition_clearing_resets_persistence(self):
        ta = TemporalAnalyzer(persistence_seconds=1.0, cooldown_seconds=5.0)
        r1 = ta.update("HAZARD", True, timestamp=0.0)
        r2 = ta.update("HAZARD", False, timestamp=0.5)
        r3 = ta.update("HAZARD", True, timestamp=0.6)
        # Restarted at t=0.6; only 0.9s elapsed since restart at t=1.5 below
        r4 = ta.update("HAZARD", True, timestamp=1.5)
        assert r4.should_fire is False  # only 0.9s since the restart, not yet 1.0s

    def test_independent_event_types_tracked_separately(self):
        ta = TemporalAnalyzer(persistence_seconds=1.0, cooldown_seconds=5.0)
        ta.update("DROWSINESS", True, timestamp=0.0)
        r = ta.update("UNSAFE_DISTANCE", False, timestamp=0.0)
        assert r.is_active is False


class TestAlertManager:
    def test_sustained_scenario_produces_few_alerts_not_hundreds(self):
        """15 seconds (450 frames) of sustained risk should produce a handful of alerts, not 450."""
        calc = RiskCalculator()
        mgr = AlertManager(persistence_seconds=3.0, cooldown_seconds=10.0)
        t = 0.0
        dt = 1.0 / 30.0
        all_alerts = []
        for _ in range(int(30 * 15)):
            factors = RiskFactors(drowsiness=True, unsafe_following_distance=True)
            risk = calc.calculate(factors)
            all_alerts.extend(mgr.process(factors, risk, timestamp=t))
            t += dt
        assert 0 < len(all_alerts) < 10
        types_fired = {a.type for a in all_alerts}
        assert "DROWSINESS" in types_fired
        assert "UNSAFE_DISTANCE" in types_fired

    def test_high_risk_meta_alert_fires_for_elevated_scenarios(self):
        calc = RiskCalculator()
        mgr = AlertManager(
            persistence_seconds=3.0,
            cooldown_seconds=10.0,
            high_risk_persistence_seconds=2.0,
            high_risk_cooldown_seconds=15.0,
        )
        t = 0.0
        dt = 1.0 / 30.0
        all_alerts = []
        for _ in range(int(30 * 5)):
            factors = RiskFactors(drowsiness=True, unsafe_following_distance=True, lane_violation=True)
            risk = calc.calculate(factors)
            all_alerts.extend(mgr.process(factors, risk, timestamp=t))
            t += dt
        types_fired = [a.type for a in all_alerts]
        assert "HIGH_RISK" in types_fired

    def test_alert_structure_matches_spec(self):
        """Each alert must contain type, severity, message, timestamp."""
        calc = RiskCalculator()
        mgr = AlertManager(persistence_seconds=0.01, cooldown_seconds=10.0)
        factors = RiskFactors(drowsiness=True)
        risk = calc.calculate(factors)
        mgr.process(factors, risk, timestamp=0.0)
        alerts = mgr.process(factors, risk, timestamp=0.02)
        assert len(alerts) == 1
        alert_dict = alerts[0].to_dict()
        assert set(alert_dict.keys()) == {"type", "severity", "message", "timestamp"}
        assert alert_dict["type"] == "DROWSINESS"
