#!/usr/bin/env python3
"""
DriveGuard AI - PHASE 1 Test Script

Tests basic vehicle detection using YOLO on a video file.

This script:
1. Loads a video file
2. Runs YOLO vehicle detection on each frame
3. Draws bounding boxes with confidence scores
4. Saves output video to outputs/
5. Prints detection statistics

Usage:
    python test_phase1_detection.py <video_path>

Example:
    python test_phase1_detection.py data/road_videos/sample.mp4
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logger import logger
from src.utils.video_utils import VideoProcessor, VideoWriter
from src.road_analysis.vehicle_detector import VehicleDetector


def main():
    """Main test function."""

    # Print configuration
    Config.print_config()

    # Get video path from command line
    if len(sys.argv) < 2:
        print("\n" + "=" * 70)
        print("DriveGuard AI - PHASE 1: Vehicle Detection Test")
        print("=" * 70)
        print("\nUsage: python test_phase1_detection.py <video_path>")
        print("\nExample:")
        print("  python test_phase1_detection.py data/road_videos/sample.mp4")
        print("\nThe script will:")
        print("  1. Load the video")
        print("  2. Run YOLO vehicle detection on each frame")
        print("  3. Draw bounding boxes")
        print("  4. Save output video to outputs/")
        print("=" * 70 + "\n")
        return

    video_path = sys.argv[1]

    # Check if file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        print(f"\n❌ Error: Video file not found: {video_path}")
        return

    try:
        # ====================================================================
        # Step 1: Initialize components
        # ====================================================================
        logger.info("=" * 70)
        logger.info("DriveGuard AI - PHASE 1: Vehicle Detection Test")
        logger.info("=" * 70)

        logger.info("Step 1: Initializing components...")
        detector = VehicleDetector(confidence_threshold=Config.CONFIDENCE_THRESHOLD)
        detector.load_model()

        # ====================================================================
        # Step 2: Open video
        # ====================================================================
        logger.info("Step 2: Opening video file...")
        video = VideoProcessor(video_path)

        width, height = video.get_resolution()
        fps = video.get_fps()
        frame_count = video.get_frame_count()
        duration = video.get_duration_seconds()

        logger.info(f"Video info:")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Duration: {duration:.2f} seconds")

        # ====================================================================
        # Step 3: Prepare output
        # ====================================================================
        logger.info("Step 3: Preparing output video writer...")
        output_filename = (
            f"detection_{Path(video_path).stem}_{Config.MODEL_NAME.replace('.pt', '')}.mp4"
        )
        output_path = os.path.join(Config.OUTPUTS_DIR, output_filename)

        writer = VideoWriter(output_path, fps, (width, height))

        # ====================================================================
        # Step 4: Process frames
        # ====================================================================
        logger.info("Step 4: Processing frames...")
        frame_idx = 0
        detection_stats = {
            "total_frames": 0,
            "frames_with_detections": 0,
            "total_detections": 0,
            "class_counts": {},
        }

        print("\n" + "=" * 70)
        print("Processing Video")
        print("=" * 70)

        while True:
            success, frame = video.read_frame()

            if not success:
                break

            frame_idx += 1

            # Detect vehicles
            detections = detector.detect(frame)

            # Draw detections
            annotated_frame = detector.draw_detections(frame, detections)

            # Update statistics
            detection_stats["total_frames"] += 1
            if detections:
                detection_stats["frames_with_detections"] += 1

            for detection in detections:
                detection_stats["total_detections"] += 1
                class_name = detection.class_name
                if class_name not in detection_stats["class_counts"]:
                    detection_stats["class_counts"][class_name] = 0
                detection_stats["class_counts"][class_name] += 1

            # Write frame
            writer.write_frame(annotated_frame)

            # Print progress
            if frame_idx % 30 == 0:
                progress = (frame_idx / frame_count) * 100
                print(
                    f"  Progress: {frame_idx}/{frame_count} frames ({progress:.1f}%) "
                    f"| Detections this frame: {len(detections)}"
                )

        video.close()
        writer.close()

        # ====================================================================
        # Step 5: Print results
        # ====================================================================
        print("\n" + "=" * 70)
        print("Detection Results")
        print("=" * 70)

        print(f"\nTotal frames processed: {detection_stats['total_frames']}")
        print(f"Frames with detections: {detection_stats['frames_with_detections']}")
        print(f"Total objects detected: {detection_stats['total_detections']}")

        if detection_stats["class_counts"]:
            print("\nDetections by class:")
            for class_name, count in sorted(
                detection_stats["class_counts"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {class_name}: {count}")

        print(f"\nOutput video saved: {output_path}")
        print(f"Output size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

        print("\n" + "=" * 70)
        print("✅ Phase 1 Test Completed Successfully!")
        print("=" * 70 + "\n")

        logger.info("Phase 1 test completed successfully!")

    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("See logs for details.")


if __name__ == "__main__":
    main()
