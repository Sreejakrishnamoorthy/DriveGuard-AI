"""
Logging configuration for DriveGuard AI.

Provides a centralized logger that can be used throughout the application.
"""

import logging
import os
from datetime import datetime

from src.utils.config import Config


def setup_logger(name: str = "driveguard") -> logging.Logger:
    """
    Set up and configure a logger for DriveGuard AI.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    # Set log level
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    log_dir = os.path.join(Config.OUTPUTS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"driveguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Create a global logger instance
logger = setup_logger()
