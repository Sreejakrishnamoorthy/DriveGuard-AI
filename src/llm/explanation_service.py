"""
LLM Explanation Service.

Converts STRUCTURED risk-analysis results into a short, human-readable
explanation. The LLM never sees raw video or images — only the already-
computed numeric/categorical outputs from the risk engine. This keeps
computer vision entirely out of the LLM's hands, per project requirements.

The core risk pipeline does not depend on this module. If no LLM backend
is configured, unreachable, or errors out, generate_explanation() always
returns a deterministic, template-based explanation instead of raising —
callers never need to handle an LLM failure specially.
"""

from dataclasses import dataclass
from typing import Optional
import requests

from src.utils.config import Config
from src.utils.logger import logger


@dataclass
class ExplanationResult:
    """Result of an explanation request."""

    text: str
    source: str  # "ollama", "openai_compatible", or "fallback"

    def to_dict(self) -> dict:
        return {"explanation": self.text, "source": self.source}


@dataclass
class RiskContext:
    """
    Structured input to the explanation service. Every field here is
    already-computed output from the risk engine / analyzers — no raw
    frames, images, or video ever pass through this class.
    """

    risk_score: int
    risk_level: str
    drowsiness: bool = False
    distraction: bool = False
    lane_violation: bool = False
    unsafe_following_distance: bool = False
    road_hazard: bool = False

    def to_prompt_facts(self) -> str:
        """Render the context as short factual bullet points for a prompt."""
        facts = [f"Risk score: {self.risk_score}/100 ({self.risk_level})"]
        if self.drowsiness:
            facts.append("Driver shows signs of drowsiness")
        if self.distraction:
            facts.append("Driver appears distracted (head turned away from the road)")
        if self.lane_violation:
            facts.append("Vehicle is departing its lane")
        if self.unsafe_following_distance:
            facts.append("Following distance to the vehicle ahead is unsafe")
        if self.road_hazard:
            facts.append("A potential road hazard was detected ahead")
        if len(facts) == 1:
            facts.append("No specific risk factors are currently active")
        return "\n".join(f"- {f}" for f in facts)


class ExplanationService:
    """
    Generates a short human-readable explanation of the current risk state.

    Tries the configured LLM backend (Ollama or an OpenAI-compatible API)
    first; on ANY failure (not configured, unreachable, timeout, malformed
    response), falls back to a deterministic template — the explanation
    service always returns something usable.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        """
        Initialize the explanation service.

        All arguments default to Config values (environment-configurable);
        pass explicit values mainly for testing.
        """
        self.provider = (provider or Config.LLM_PROVIDER).lower()
        self.ollama_base_url = ollama_base_url or Config.OLLAMA_BASE_URL
        self.ollama_model = ollama_model or Config.OLLAMA_MODEL
        self.api_base_url = api_base_url or Config.LLM_API_BASE_URL
        self.api_key = api_key or Config.LLM_API_KEY
        self.api_model = api_model or Config.LLM_API_MODEL
        self.timeout_seconds = timeout_seconds or Config.LLM_REQUEST_TIMEOUT_SECONDS

        logger.info(f"ExplanationService initialized (provider={self.provider})")

    def _build_prompt(self, context: RiskContext) -> str:
        """Build the LLM prompt from structured facts only."""
        return (
            "You are a driving safety assistant. Based ONLY on the following "
            "structured sensor facts, write ONE short paragraph (2-3 sentences) "
            "explaining the current driving risk to the driver in plain language. "
            "Be direct and factual. Do not invent details not listed below.\n\n"
            f"{context.to_prompt_facts()}\n\n"
            "Explanation:"
        )

    def _try_ollama(self, context: RiskContext) -> Optional[str]:
        """
        Attempt to generate an explanation via a local Ollama server.

        Returns None (never raises) if Ollama is unreachable, times out,
        or returns an unexpected response — callers fall back automatically.
        """
        try:
            url = f"{self.ollama_base_url.rstrip('/')}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": self._build_prompt(context),
                "stream": False,
            }
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            return text or None

        except requests.exceptions.ConnectionError:
            logger.warning(f"ExplanationService: Ollama unreachable at {self.ollama_base_url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"ExplanationService: Ollama request timed out after {self.timeout_seconds}s")
            return None
        except Exception as e:
            logger.warning(f"ExplanationService: Ollama request failed: {e}")
            return None

    def _try_openai_compatible(self, context: RiskContext) -> Optional[str]:
        """
        Attempt to generate an explanation via an OpenAI-compatible
        /v1/chat/completions endpoint. Returns None on any failure.
        """
        if not self.api_base_url or not self.api_model:
            logger.warning(
                "ExplanationService: LLM_API_BASE_URL/LLM_API_MODEL not configured "
                "for openai_compatible provider."
            )
            return None

        try:
            url = f"{self.api_base_url.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.api_model,
                "messages": [{"role": "user", "content": self._build_prompt(context)}],
                "max_tokens": 200,
                "temperature": 0.3,
            }
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
            return text or None

        except requests.exceptions.ConnectionError:
            logger.warning(f"ExplanationService: LLM API unreachable at {self.api_base_url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"ExplanationService: LLM API request timed out after {self.timeout_seconds}s")
            return None
        except Exception as e:
            logger.warning(f"ExplanationService: LLM API request failed: {e}")
            return None

    def _deterministic_fallback(self, context: RiskContext) -> str:
        """
        Build a deterministic, template-based explanation with no LLM
        dependency at all. This is what most users will see unless they've
        set up Ollama or another provider, so it must read naturally on
        its own, not like a degraded placeholder.
        """
        active_factors = []
        if context.drowsiness:
            active_factors.append("the driver appears drowsy")
        if context.unsafe_following_distance:
            active_factors.append("the vehicle is following too closely behind traffic ahead")
        if context.lane_violation:
            active_factors.append("the vehicle is drifting out of its lane")
        if context.distraction:
            active_factors.append("the driver appears distracted from the road")
        if context.road_hazard:
            active_factors.append("a potential hazard was detected on the road ahead")

        level_phrase = {
            "LOW": "Driving risk is currently low.",
            "MEDIUM": "Driving risk is currently moderate.",
            "HIGH": "Elevated driving risk detected.",
            "CRITICAL": "Critical driving risk detected.",
        }.get(context.risk_level, f"Driving risk level: {context.risk_level}.")

        if not active_factors:
            return f"{level_phrase} No specific risk factors are currently active."

        if len(active_factors) == 1:
            factor_text = active_factors[0]
        elif len(active_factors) == 2:
            factor_text = f"{active_factors[0]} while {active_factors[1]}"
        else:
            factor_text = ", ".join(active_factors[:-1]) + f", and {active_factors[-1]}"

        return f"{level_phrase} Currently, {factor_text}."

    def generate_explanation(self, context: RiskContext) -> ExplanationResult:
        """
        Generate a short human-readable explanation of the current risk state.

        Tries the configured provider first; ALWAYS returns a usable result
        (falling back to a deterministic template) rather than raising, so
        callers never need special-case error handling for the LLM.

        Args:
            context: RiskContext with structured (non-visual) risk facts

        Returns:
            ExplanationResult with the explanation text and which source
            produced it ("ollama", "openai_compatible", or "fallback")
        """
        text: Optional[str] = None
        source = "fallback"

        if self.provider == "ollama":
            text = self._try_ollama(context)
            if text:
                source = "ollama"
        elif self.provider == "openai_compatible":
            text = self._try_openai_compatible(context)
            if text:
                source = "openai_compatible"
        else:
            logger.warning(f"ExplanationService: unknown provider '{self.provider}', using fallback.")

        if not text:
            text = self._deterministic_fallback(context)
            source = "fallback"

        return ExplanationResult(text=text, source=source)
