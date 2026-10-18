"""
Tests for the risk calculation engine, matching the exact examples and
boundaries given in the project specification.
"""

import pytest

from src.risk_engine.risk_calculator import RiskCalculator, RiskFactors
from src.risk_engine.risk_rules import get_risk_level


@pytest.fixture
def calc() -> RiskCalculator:
    return RiskCalculator()


class TestRiskScoring:
    """Exact numeric examples from the project spec."""

    def test_no_factors_active(self, calc):
        result = calc.calculate(RiskFactors())
        assert result.risk_score == 0
        assert result.risk_level == "LOW"

    def test_drowsiness_only(self, calc):
        result = calc.calculate(RiskFactors(drowsiness=True))
        assert result.risk_score == 30

    def test_unsafe_distance_only(self, calc):
        result = calc.calculate(RiskFactors(unsafe_following_distance=True))
        assert result.risk_score == 25

    def test_drowsiness_and_unsafe_distance(self, calc):
        result = calc.calculate(RiskFactors(drowsiness=True, unsafe_following_distance=True))
        assert result.risk_score == 55

    def test_drowsiness_unsafe_distance_and_lane_violation(self, calc):
        result = calc.calculate(
            RiskFactors(drowsiness=True, unsafe_following_distance=True, lane_violation=True)
        )
        assert result.risk_score == 75
        assert result.risk_level == "HIGH"

    def test_score_never_exceeds_100(self, calc):
        result = calc.calculate(
            RiskFactors(
                drowsiness=True,
                unsafe_following_distance=True,
                lane_violation=True,
                road_hazard=True,
                distraction=True,
            )
        )
        # Raw sum is 30+25+20+15+20 = 110, must be capped
        assert result.raw_score_before_cap == 110
        assert result.risk_score == 100
        assert result.risk_level == "CRITICAL"

    def test_road_hazard_only(self, calc):
        result = calc.calculate(RiskFactors(road_hazard=True))
        assert result.risk_score == 15

    def test_distraction_only(self, calc):
        result = calc.calculate(RiskFactors(distraction=True))
        assert result.risk_score == 20

    def test_lane_violation_only(self, calc):
        result = calc.calculate(RiskFactors(lane_violation=True))
        assert result.risk_score == 20


class TestRiskLevelClassification:
    """Risk level boundaries, exactly as specified: LOW 0-30, MEDIUM 31-60, HIGH 61-80, CRITICAL 81-100."""

    @pytest.mark.parametrize(
        "score,expected_level",
        [
            (0, "LOW"),
            (15, "LOW"),
            (30, "LOW"),
            (31, "MEDIUM"),
            (45, "MEDIUM"),
            (60, "MEDIUM"),
            (61, "HIGH"),
            (70, "HIGH"),
            (80, "HIGH"),
            (81, "CRITICAL"),
            (100, "CRITICAL"),
        ],
    )
    def test_boundary(self, score, expected_level):
        assert get_risk_level(score) == expected_level


class TestRiskWeightsConfigurable:
    """Weights must be configurable, not hardcoded."""

    def test_custom_weights_change_score(self):
        from src.risk_engine.risk_rules import RiskWeights

        custom_weights = RiskWeights(drowsiness=50)
        calc = RiskCalculator(weights=custom_weights)
        result = calc.calculate(RiskFactors(drowsiness=True))
        assert result.risk_score == 50
