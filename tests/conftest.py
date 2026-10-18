"""
Shared pytest fixtures for DriveGuard AI tests.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A plain gray frame with no detectable content, for edge-case tests."""
    return np.full((480, 640, 3), 80, dtype=np.uint8)


@pytest.fixture(scope="session")
def real_test_image_path() -> str:
    """
    Path to a real downloaded test photo (people + a bus), used to verify
    detection models produce genuine, non-trivial output on real content.
    Downloaded once per test session; skips dependent tests if unreachable.
    """
    import urllib.request

    # tempfile.gettempdir() (not a hardcoded "/tmp/...") so this works
    # correctly on Windows too, where /tmp is not a standard directory.
    path = os.path.join(tempfile.gettempdir(), "driveguard_test_bus.jpg")
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
                path,
            )
        except Exception as e:
            pytest.skip(f"Could not download real test image (no network access): {e}")
    return path


@pytest.fixture
def mongomock_client():
    """An in-memory MongoDB-compatible client for testing DatabaseService without a real server."""
    mongomock = pytest.importorskip("mongomock")
    return mongomock.MongoClient()
