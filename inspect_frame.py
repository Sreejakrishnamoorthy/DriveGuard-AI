#!/usr/bin/env python3
"""
DriveGuard AI - ROI Calibration Helper

Extracts a sample frame from your video and overlays horizontal ratio
guide lines, so you can visually determine the correct roi_bottom_ratio
(and roi_top_ratio) to pass to RoadAnalyzer / test_phase3_6_road_analyzer.py.

Why this matters:
    Lane detection and hazard detection only analyze a region of the frame
    between roi_top_ratio and roi_bottom_ratio (as fractions of frame
    height). If your camera is mounted low (e.g. on the hood/dashboard)
    and the hood is visible in your footage, the default roi_bottom_ratio
    of 1.0 will cause the detectors to analyze hood reflections instead of
    the actual road — producing nonsense results.

Usage:
    python inspect_frame.py <video_path> [frame_number]

Example:
    python inspect_frame.py data/road_videos/your_video.mp4 400

Output:
    Saves outputs/roi_calibration_frame.jpg with horizontal lines at every
    10% of frame height, labeled with their ratio value. Open that image,
    find the line just above where your hood/dashboard begins, and use
    that ratio as roi_bottom_ratio.
"""

import sys
import os
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.config import Config


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python inspect_frame.py <video_path> [frame_number]\n")
        return

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"\n❌ Error: Video file not found: {video_path}")
        return

    frame_number = int(sys.argv[2]) if len(sys.argv) > 2 else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n❌ Error: Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = frame_number if frame_number is not None else total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"\n❌ Error: Could not read frame {target} from video.")
        return

    height, width = frame.shape[:2]

    for pct in range(10, 100, 10):
        y = int(height * pct / 100)
        color = (0, 255, 255)
        cv2.line(frame, (0, y), (width, y), color, 2)
        cv2.putText(
            frame,
            f"{pct/100:.2f}",
            (10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

    Config.create_directories()
    output_path = os.path.join(Config.OUTPUTS_DIR, "roi_calibration_frame.jpg")
    cv2.imwrite(output_path, frame)

    print("\n" + "=" * 70)
    print("ROI Calibration Frame Saved")
    print("=" * 70)
    print(f"Frame {target}/{total_frames} saved to: {output_path}")
    print(f"Resolution: {width}x{height}")
    print()
    print("Open that image. Each yellow line is labeled with its ratio")
    print("(0.10 - 0.90) as a fraction of frame height.")
    print()
    print("- Find the line just ABOVE where your hood/dashboard begins.")
    print("  Use that value as roi_bottom_ratio.")
    print("- Find the line roughly at the horizon (where the road meets")
    print("  the sky/background). Use that value as roi_top_ratio.")
    print()
    print("Then run:")
    print(f"  python test_phase3_6_road_analyzer.py {video_path} <roi_top_ratio> <roi_bottom_ratio>")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
