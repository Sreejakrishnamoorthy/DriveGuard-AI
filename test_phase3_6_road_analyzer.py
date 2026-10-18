#!/usr/bin/env python3
"""
DriveGuard AI - PHASE 3-6 Test Script

Tests the full road-analysis pipeline: vehicle detection + ByteTrack
tracking + lane detection + hazard detection, integrated via RoadAnalyzer.

Usage:
    python test_phase3_6_road_analyzer.py <video_path> [roi_top_ratio] [roi_bottom_ratio]

Example:
    python test_phase3_6_road_analyzer.py data/road_videos/sample.mp4
    python test_phase3_6_road_analyzer.py data/road_videos/sample.mp4 0.35 0.56

IMPORTANT - roi_bottom_ratio:
    If your camera is mounted low enough that the car's hood/dashboard is
    visible in the lower part of the frame (common with hood/dash-mounted
    dashcams), the default roi_bottom_ratio=1.0 will cause lane and hazard
    detection to analyze hood reflections instead of the road. Extract a
    sample frame from your video and check what fraction of the frame
    height the hood occupies, then pass a roi_bottom_ratio just above that
    point (e.g. if the hood starts at ~56% down the frame, use 0.56).
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logger import logger
from src.utils.video_utils import VideoProcessor, VideoWriter
from src.road_analysis.road_analyzer import RoadAnalyzer


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python test_phase3_6_road_analyzer.py <video_path> [roi_top_ratio] [roi_bottom_ratio]\n")
        print("This will run: vehicle detection -> ByteTrack tracking ->")
        print("lane detection -> hazard detection -> combined annotation,")
        print("and save the result to outputs/.\n")
        print("If your dashcam is hood/dash-mounted and the hood is visible")
        print("in your footage, pass roi_bottom_ratio to exclude it, e.g.:")
        print("  python test_phase3_6_road_analyzer.py data/road_videos/your.mp4 0.35 0.56\n")
        return

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"\n❌ Error: Video file not found: {video_path}")
        return

    roi_top_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    roi_bottom_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    Config.print_config()
    print(f"Using roi_top_ratio={roi_top_ratio}, roi_bottom_ratio={roi_bottom_ratio}")
    print("(If lane/hazard results look wrong, your camera mount may need different values —")
    print(" see the note in this script's usage text.)\n")

    logger.info("Initializing RoadAnalyzer (Phases 3-6)...")
    analyzer = RoadAnalyzer(
        confidence_threshold=Config.CONFIDENCE_THRESHOLD,
        roi_top_ratio=roi_top_ratio,
        roi_bottom_ratio=roi_bottom_ratio,
    )

    video = VideoProcessor(video_path)
    width, height = video.get_resolution()
    fps = video.get_fps()
    frame_count = video.get_frame_count()

    output_path = os.path.join(
        Config.OUTPUTS_DIR, f"road_analysis_{Path(video_path).stem}.mp4"
    )
    writer = VideoWriter(output_path, fps, (width, height))

    print("\n" + "=" * 70)
    print("Processing Video: Road Analysis Pipeline (Phases 3-6)")
    print("=" * 70)

    frame_idx = 0
    stats = {
        "frames_processed": 0,
        "frames_with_vehicles": 0,
        "unique_tracking_ids": set(),
        "lane_departure_frames": 0,
        "hazard_frames": 0,
        "following_distance_counts": {"SAFE": 0, "MODERATE": 0, "UNSAFE": 0},
    }

    while True:
        ok, frame = video.read_frame()
        if not ok:
            break
        frame_idx += 1

        result = analyzer.analyze(frame)
        annotated = analyzer.draw_annotations(frame, result)
        writer.write_frame(annotated)

        stats["frames_processed"] += 1
        if result.vehicles:
            stats["frames_with_vehicles"] += 1
        for v in result.vehicles:
            stats["unique_tracking_ids"].add(v.tracking_id)
        if result.lane and result.lane.lane_departure:
            stats["lane_departure_frames"] += 1
        if result.hazards and result.hazards.hazard_detected:
            stats["hazard_frames"] += 1
        stats["following_distance_counts"][result.following_distance] += 1

        if frame_idx % 30 == 0:
            print(f"  Progress: {frame_idx}/{frame_count} frames")

    video.close()
    writer.close()

    print("\n" + "=" * 70)
    print("Road Analysis Results")
    print("=" * 70)
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Frames with vehicles detected: {stats['frames_with_vehicles']}")
    print(f"Unique tracking IDs assigned: {len(stats['unique_tracking_ids'])}")
    print(f"Frames with lane departure warning: {stats['lane_departure_frames']}")
    print(f"Frames with hazard flagged: {stats['hazard_frames']}")
    print(f"Following distance breakdown: {stats['following_distance_counts']}")
    print(f"\nHazard detector mode: {analyzer.hazard_detector.mode}")
    if analyzer.hazard_detector.mode == "heuristic_fallback":
        print(
            "  NOTE: No pretrained hazard model configured (HAZARD_MODEL_PATH).\n"
            "  Hazard results are from an experimental OpenCV heuristic only."
        )

    print(f"\nOutput video saved: {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
