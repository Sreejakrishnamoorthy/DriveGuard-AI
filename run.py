#!/usr/bin/env python3
"""
DriveGuard AI - Application Launcher

Provides a single entry point for running available parts of the system.

IMPORTANT: This project is being built in phases. This script only exposes
commands for what has actually been implemented so far — it will not
pretend to launch a backend or dashboard that doesn't exist yet.

Current build status: All 18 phases complete (see `python run.py status`).

Usage:
    python run.py detect <video_path>        # Phase 1-2: YOLO vehicle detection only
    python run.py road-analysis <video_path> # Phase 3-6: full road analysis pipeline
    python run.py backend                    # Launch FastAPI backend (Phase 14)
    python run.py frontend                   # Launch Streamlit dashboard (Phase 16)
    python run.py test                       # Run the full pytest suite (Phase 17)
    python run.py status                     # Show what's implemented so far
"""

import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

PHASES = [
    ("Phase 1", "Project setup + environment", True),
    ("Phase 2", "YOLO vehicle detection", True),
    ("Phase 3", "ByteTrack vehicle tracking", True),
    ("Phase 4", "Lane detection (OpenCV fallback)", True),
    ("Phase 5", "Road hazard detection (pluggable + heuristic fallback)", True),
    ("Phase 6", "Road Analyzer integration", True),
    ("Phase 7", "Driver face detection", True),
    ("Phase 8", "Eye/drowsiness detection", True),
    ("Phase 9", "Attention detection", True),
    ("Phase 10", "Driver Monitoring integration", True),
    ("Phase 11", "Risk Engine", True),
    ("Phase 12", "Temporal analysis + alerts", True),
    ("Phase 13", "LLM explanation service", True),
    ("Phase 14", "FastAPI backend", True),
    ("Phase 15", "MongoDB integration", True),
    ("Phase 16", "Streamlit dashboard", True),
    ("Phase 17", "Integration testing (149 pytest tests)", True),
    ("Phase 18", "Documentation", True),
    ("Phase 19", "Voice Safety Alerts", True),
    ("Phase 20", "Alert Prioritization", True),
    ("Phase 23", "Near-Miss Detection", True),
    ("Phase 25", "Phone/Distraction Detection", True),
    ("Phase 26", "Confidence Scores", True),
    ("Phase 27", "Incident Reconstruction", True),
    ("Phase 28", "Explainable Risk", True),
]


def print_status() -> None:
    print("\n" + "=" * 70)
    print("DriveGuard AI - Build Status")
    print("=" * 70)
    for phase, description, done in PHASES:
        mark = "✅" if done else "⏳"
        print(f"  {mark} {phase}: {description}")
    print("=" * 70 + "\n")


def run_detection(args: list) -> None:
    if not args:
        print("Usage: python run.py detect <video_path>")
        return
    script = PROJECT_ROOT / "test_phase1_detection.py"
    subprocess.run([sys.executable, str(script), args[0]])


def run_road_analysis(args: list) -> None:
    if not args:
        print("Usage: python run.py road-analysis <video_path>")
        return
    script = PROJECT_ROOT / "test_phase3_6_road_analyzer.py"
    subprocess.run([sys.executable, str(script), args[0]])


def run_backend() -> None:
    backend_main = PROJECT_ROOT / "backend" / "main.py"
    if not backend_main.exists() or backend_main.stat().st_size == 0:
        print(
            "\n❌ Backend is not implemented yet (Phase 14).\n"
            "Run `python run.py status` to see what's currently available.\n"
        )
        return
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload"]
    )


def run_frontend() -> None:
    frontend_app = PROJECT_ROOT / "frontend" / "app.py"
    if not frontend_app.exists() or frontend_app.stat().st_size == 0:
        print(
            "\n❌ Streamlit dashboard is not implemented yet (Phase 16).\n"
            "Run `python run.py status` to see what's currently available.\n"
        )
        return
    subprocess.run(["streamlit", "run", str(frontend_app)])


def run_tests() -> None:
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print_status()
        return

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "detect":
        run_detection(args)
    elif command == "road-analysis":
        run_road_analysis(args)
    elif command == "backend":
        run_backend()
    elif command == "frontend":
        run_frontend()
    elif command == "test":
        run_tests()
    elif command == "status":
        print_status()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
