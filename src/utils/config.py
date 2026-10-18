"""
Configuration management for DriveGuard AI.

Loads configuration from environment variables and provides
a centralized configuration object.
"""

import os
from dotenv import load_dotenv
from typing import Literal

# Load environment variables from .env file
load_dotenv()


class Config:
    """Main configuration class for DriveGuard AI."""

    # ========================================================================
    # MongoDB Configuration
    # ========================================================================
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "driveguard")

    # ========================================================================
    # YOLO Model Configuration
    # ========================================================================
    MODEL_NAME: str = os.getenv("MODEL_NAME", "yolov8n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.40))
    DEVICE: str = os.getenv("DEVICE", "auto")
    INFERENCE_RESOLUTION: int = int(os.getenv("INFERENCE_RESOLUTION", 640))

    # ========================================================================
    # Video Processing Configuration
    # ========================================================================
    FRAME_SKIP: int = int(os.getenv("FRAME_SKIP", 1))
    TARGET_FPS: int = int(os.getenv("TARGET_FPS", 30))

    # ========================================================================
    # FastAPI Configuration
    # ========================================================================
    FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", 8000))
    FASTAPI_RELOAD: bool = os.getenv("FASTAPI_RELOAD", "true").lower() == "true"

    # ========================================================================
    # Streamlit Configuration
    # ========================================================================
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", 8501))
    STREAMLIT_LOGGER_LEVEL: str = os.getenv("STREAMLIT_LOGGER_LEVEL", "info")

    # ========================================================================
    # LLM Configuration (Explanation Service)
    # ========================================================================
    # "ollama" (local) or "openai_compatible" (any provider implementing the
    # OpenAI /v1/chat/completions API shape — many hosted and self-hosted
    # options do). The explanation service works with NEITHER configured;
    # it falls back to a deterministic, template-based explanation.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    LLM_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", 8.0))

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma2:2b")

    # Generic OpenAI-compatible provider (only used if LLM_PROVIDER=openai_compatible)
    LLM_API_BASE_URL: str = os.getenv("LLM_API_BASE_URL", "")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_API_MODEL: str = os.getenv("LLM_API_MODEL", "")

    # ========================================================================
    # Application Settings
    # ========================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # ========================================================================
    # Risk Engine Configuration (Weights for risk scoring)
    # ========================================================================
    RISK_WEIGHTS: dict = {
        "drowsiness": 30,
        "unsafe_following_distance": 25,
        "lane_violation": 20,
        "road_hazard": 15,
        "distraction": 20,
    }

    # ========================================================================
    # Risk Level Thresholds
    # ========================================================================
    RISK_THRESHOLDS: dict = {
        "LOW": (0, 30),
        "MEDIUM": (31, 60),
        "HIGH": (61, 80),
        "CRITICAL": (81, 100),
    }

    # ========================================================================
    # Paths
    # ========================================================================
    PROJECT_ROOT: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "outputs")

    @classmethod
    def validate_device(cls) -> str:
        """
        Validate and return the device to use for inference.

        Returns:
            str: 'cuda', 'cpu', or auto-detected device
        """
        if cls.DEVICE.lower() == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return cls.DEVICE.lower()

    @classmethod
    def get_risk_level(cls, score: float) -> str:
        """
        Get risk level based on score.

        Args:
            score: Risk score (0-100)

        Returns:
            str: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        for level, (min_val, max_val) in cls.RISK_THRESHOLDS.items():
            if min_val <= score <= max_val:
                return level
        return "CRITICAL"

    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUTS_DIR, exist_ok=True)

    @classmethod
    def print_config(cls) -> None:
        """Print current configuration for debugging."""
        print("\n" + "=" * 70)
        print("DriveGuard AI - Configuration")
        print("=" * 70)
        print(f"Device: {cls.validate_device()}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}")
        print(f"Inference Resolution: {cls.INFERENCE_RESOLUTION}")
        print(f"MongoDB URI: {cls.MONGODB_URI}")
        print(f"Ollama URL: {cls.OLLAMA_BASE_URL}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Debug Mode: {cls.DEBUG_MODE}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    Config.print_config()
