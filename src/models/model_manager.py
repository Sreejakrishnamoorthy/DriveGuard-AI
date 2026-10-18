"""
Model Manager for DriveGuard AI.

Handles loading, caching, and lifecycle management of pretrained models.
Ensures models are loaded only once and reused across the application.
"""

import os
from typing import Optional
import torch

from src.utils.config import Config
from src.utils.logger import logger


class ModelManager:
    """
    Centralized model management.

    Responsibilities:
    - Load models once and cache them
    - Detect available devices (CUDA/CPU)
    - Provide models to all modules
    - Handle model loading errors gracefully
    """

    # Singleton pattern: store instances
    _instance: Optional["ModelManager"] = None
    _yolo_model = None
    _face_mesh_model = None
    _device: Optional[str] = None

    def __new__(cls) -> "ModelManager":
        """Ensure only one ModelManager instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize model manager."""
        if self._initialized:
            return

        self._initialized = True
        logger.info("Initializing ModelManager...")

        # Detect device
        self._device = self._detect_device()
        logger.info(f"Device selected: {self._device}")

        # Create models directory
        Config.create_directories()

    @staticmethod
    def _detect_device() -> str:
        """
        Detect available device for inference.

        Returns:
            str: Device name ('cuda' or 'cpu')
        """
        try:
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                return device
            else:
                logger.info("CUDA not available. Using CPU.")
                return "cpu"
        except Exception as e:
            logger.warning(f"Error detecting device: {e}. Defaulting to CPU.")
            return "cpu"

    @property
    def device(self) -> str:
        """Get the current device."""
        return self._device or "cpu"

    def load_yolo_model(self, model_name: Optional[str] = None) -> object:
        """
        Load YOLO model.

        Uses caching: if already loaded, returns cached model without reloading.

        Args:
            model_name: Model name (e.g., 'yolov8n.pt'). Defaults to Config.MODEL_NAME

        Returns:
            YOLO model object

        Raises:
            Exception: If model loading fails
        """
        if self._yolo_model is not None:
            logger.debug("Returning cached YOLO model.")
            return self._yolo_model

        if model_name is None:
            model_name = Config.MODEL_NAME

        logger.info(f"Loading YOLO model: {model_name}")

        try:
            from ultralytics import YOLO

            # Download model if not present
            model_path = os.path.join(Config.MODELS_DIR, model_name)

            # YOLO will auto-download if needed
            model = YOLO(model_name)
            model.to(self.device)

            self._yolo_model = model
            logger.info(f"YOLO model loaded successfully: {model_name}")
            logger.info(f"Model device: {model.device}")

            return model

        except ImportError:
            logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def get_yolo_model(self) -> object:
        """
        Get YOLO model (load if not already loaded).

        Returns:
            YOLO model object
        """
        if self._yolo_model is None:
            self.load_yolo_model()
        return self._yolo_model

    def is_yolo_loaded(self) -> bool:
        """Check if YOLO model is loaded."""
        return self._yolo_model is not None

    def get_face_mesh(self, refine_landmarks: bool = True) -> Optional[object]:
        """
        Get a shared MediaPipe FaceMesh instance (load if not already loaded).

        Driver-monitoring modules (EyeDetector, AttentionDetector) both need
        face landmarks from the same underlying model. Sharing one instance
        here means one inference call per frame instead of two, which
        matters for real-time performance.

        Args:
            refine_landmarks: Whether to include iris landmarks (needed for
                eye-aspect-ratio calculations). Only affects the FIRST call
                that actually creates the model; later calls reuse whatever
                was created first regardless of this argument.

        Returns:
            A mediapipe FaceMesh instance, or None if mediapipe isn't installed.
        """
        if self._face_mesh_model is not None:
            return self._face_mesh_model

        try:
            import mediapipe as mp

            self._face_mesh_model = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("ModelManager: shared MediaPipe FaceMesh model loaded.")
            return self._face_mesh_model

        except ImportError:
            logger.warning(
                "mediapipe package not installed. Driver-monitoring features "
                "requiring face landmarks will be unavailable."
            )
            return None
        except Exception as e:
            logger.error(f"ModelManager: failed to load FaceMesh model: {e}")
            return None

    def is_face_mesh_loaded(self) -> bool:
        """Check if the shared FaceMesh model is loaded."""
        return self._face_mesh_model is not None

    def clear_cache(self) -> None:
        """
        Clear all cached models from memory.

        Useful for cleanup or reloading.
        """
        logger.info("Clearing model cache...")
        if self._yolo_model is not None:
            del self._yolo_model
            self._yolo_model = None
        if self._face_mesh_model is not None:
            del self._face_mesh_model
            self._face_mesh_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared.")

    def print_status(self) -> None:
        """Print model manager status."""
        print("\n" + "=" * 70)
        print("Model Manager Status")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"YOLO Model Loaded: {self.is_yolo_loaded()}")
        if self.is_yolo_loaded():
            print(f"YOLO Model: {self._yolo_model.model_name if hasattr(self._yolo_model, 'model_name') else 'N/A'}")
        print("=" * 70 + "\n")


# Global singleton instance
model_manager = ModelManager()
