"""
Tests for the LLM explanation service: must always return a usable
explanation, even when no LLM backend is configured or reachable.
"""

import time

from src.llm.explanation_service import ExplanationService, RiskContext


class TestExplanationFallback:
    def test_unreachable_ollama_falls_back_fast(self):
        svc = ExplanationService(provider="ollama", ollama_base_url="http://localhost:11434", timeout_seconds=3.0)
        ctx = RiskContext(risk_score=82, risk_level="CRITICAL", drowsiness=True, lane_violation=True)

        start = time.time()
        result = svc.generate_explanation(ctx)
        elapsed = time.time() - start

        assert result.source == "fallback"
        assert len(result.text) > 0
        assert elapsed < 5.0  # must fail fast on connection refused, not hang

    def test_unconfigured_openai_compatible_falls_back(self):
        svc = ExplanationService(provider="openai_compatible", timeout_seconds=1.0)
        result = svc.generate_explanation(RiskContext(risk_score=40, risk_level="MEDIUM", distraction=True))
        assert result.source == "fallback"

    def test_unknown_provider_falls_back_without_crashing(self):
        svc = ExplanationService(provider="totally_not_a_real_provider")
        result = svc.generate_explanation(RiskContext(risk_score=20, risk_level="LOW"))
        assert result.source == "fallback"

    def test_fallback_mentions_active_factors(self):
        svc = ExplanationService(provider="ollama", timeout_seconds=1.0)
        ctx = RiskContext(risk_score=75, risk_level="HIGH", drowsiness=True, unsafe_following_distance=True)
        result = svc.generate_explanation(ctx)
        text_lower = result.text.lower()
        assert "drowsy" in text_lower or "drowsiness" in text_lower

    def test_fallback_with_no_active_factors(self):
        svc = ExplanationService(provider="ollama", timeout_seconds=1.0)
        result = svc.generate_explanation(RiskContext(risk_score=0, risk_level="LOW"))
        assert len(result.text) > 0
        assert result.source == "fallback"

    def test_never_raises_regardless_of_input(self):
        svc = ExplanationService(provider="ollama", timeout_seconds=1.0)
        # Should not raise even for a maximal, all-factors-active context
        result = svc.generate_explanation(
            RiskContext(
                risk_score=100,
                risk_level="CRITICAL",
                drowsiness=True,
                distraction=True,
                lane_violation=True,
                unsafe_following_distance=True,
                road_hazard=True,
            )
        )
        assert isinstance(result.text, str)
        assert len(result.text) > 0
