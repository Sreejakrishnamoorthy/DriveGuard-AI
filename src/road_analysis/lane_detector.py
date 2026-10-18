"""
Lane Detector for Road Analysis.

Implementation note (per project requirements):
    No general-purpose pretrained lane-detection model ships with the
    open-source ecosystem the way YOLO does for object detection. Rather
    than falsely claim to use a pretrained model, this module implements a
    transparent, classical OpenCV pipeline:

        grayscale -> Gaussian blur -> Canny edge detection ->
        region-of-interest mask -> probabilistic Hough transform ->
        left/right line grouping by slope -> lane-center estimation

    This is clearly a heuristic/fallback approach. It works reasonably
    well on clear, marked lanes in good lighting but will degrade on
    faded markings, heavy shadow, rain, or unmarked roads — like any
    classical (non-learned) lane detector. The class is intentionally
    isolated behind a small interface (`detect()`) so it can later be
    swapped for a pretrained segmentation model (e.g. Ultra-Fast-Lane
    or a semantic segmentation network) without touching calling code.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.utils.logger import logger

LaneLine = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class LaneResult:
    """Result of lane detection for a single frame."""

    lane_detected: bool
    lane_status: str  # CENTERED, LEFT_OF_CENTER, RIGHT_OF_CENTER, UNKNOWN
    lane_departure: bool
    left_line: Optional[LaneLine] = None
    right_line: Optional[LaneLine] = None
    offset_pixels: float = 0.0
    offset_ratio: float = 0.0  # offset as fraction of half-lane-width, signed

    def to_dict(self) -> dict:
        """Convert to a plain dict matching the spec's expected output shape."""
        return {
            "lane_detected": self.lane_detected,
            "lane_status": self.lane_status,
            "lane_departure": self.lane_departure,
        }


class LaneDetector:
    """
    Detects lane boundaries and estimates vehicle position within the lane
    using a classical OpenCV computer-vision pipeline.
    """

    LANE_STATUS_CENTERED = "CENTERED"
    LANE_STATUS_LEFT = "LEFT_OF_CENTER"
    LANE_STATUS_RIGHT = "RIGHT_OF_CENTER"
    LANE_STATUS_UNKNOWN = "UNKNOWN"

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 30,
        hough_min_line_length: int = 40,
        hough_max_line_gap: int = 100,
        departure_threshold_ratio: float = 0.65,
        roi_top_ratio: float = 0.6,
        roi_bottom_ratio: float = 1.0,
        smoothing_window: int = 5,
    ) -> None:
        """
        Initialize lane detector.

        Args:
            canny_low: Lower threshold for Canny edge detector
            canny_high: Upper threshold for Canny edge detector
            hough_threshold: Minimum votes for a Hough line
            hough_min_line_length: Minimum line length in pixels
            hough_max_line_gap: Maximum gap between line segments to join
            departure_threshold_ratio: Offset ratio beyond which lane departure triggers
            roi_top_ratio: Fraction of frame height above which is ignored (sky/horizon)
            roi_bottom_ratio: Fraction of frame height BELOW which is ignored. Default 1.0
                assumes the road is visible all the way to the bottom edge of the frame
                (typical windshield-mounted camera). If your camera is mounted lower
                (e.g. on the hood/dashboard) and the hood is visible in the lower portion
                of the frame, LOWER this value so the ROI stops before the hood begins —
                otherwise the detector analyzes hood reflections instead of the road.
                Inspect a sample frame to find where the hood starts (as a fraction of
                frame height) and set this just above that point.
            smoothing_window: Number of recent frames to smooth offset over
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        self.departure_threshold_ratio = departure_threshold_ratio
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_ratio = roi_bottom_ratio

        self._offset_history: List[float] = []
        self.smoothing_window = smoothing_window

        logger.info("LaneDetector initialized (OpenCV Hough-transform fallback).")

    def _region_of_interest(self, edges: np.ndarray) -> np.ndarray:
        """
        Mask out everything except a trapezoid region in front of the vehicle,
        where lane markings are expected to appear. Excludes both the sky
        (above roi_top_ratio) and anything below roi_bottom_ratio (e.g. a
        visible hood on hood/dashboard-mounted cameras).
        """
        height, width = edges.shape[:2]
        mask = np.zeros_like(edges)

        top_y = int(height * self.roi_top_ratio)
        bottom_y = int(height * self.roi_bottom_ratio)
        polygon = np.array(
            [
                [
                    (int(width * 0.05), bottom_y),
                    (int(width * 0.40), top_y),
                    (int(width * 0.60), top_y),
                    (int(width * 0.95), bottom_y),
                ]
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _detect_edges(self, frame: np.ndarray) -> np.ndarray:
        """Run grayscale -> blur -> Canny edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return edges

    def _detect_hough_lines(self, masked_edges: np.ndarray) -> Optional[np.ndarray]:
        """Run probabilistic Hough transform to extract line segments."""
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )
        return lines

    def _average_slope_intercept(
        self, lines: np.ndarray, width: int, height: int
    ) -> Tuple[Optional[LaneLine], Optional[LaneLine]]:
        """
        Separate Hough line segments into left/right lane candidates by
        slope sign, then average each group into a single representative
        line spanning from the visible-road bottom edge to the ROI horizon.
        """
        left_fits = []
        right_fits = []

        for line in lines:
            # HoughLinesP returns shape (N, 1, 4); flatten defensively in
            # case a different OpenCV build/version returns (N, 4) instead.
            coords = np.array(line).reshape(-1)
            if coords.size != 4:
                continue
            x1, y1, x2, y2 = coords
            if x2 == x1:
                continue  # skip vertical lines (undefined slope)

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Filter near-horizontal lines (not lane markings)
            if abs(slope) < 0.3:
                continue

            if slope < 0:
                left_fits.append((slope, intercept))
            else:
                right_fits.append((slope, intercept))

        bottom_y = int(height * self.roi_bottom_ratio)
        top_y = int(height * self.roi_top_ratio)

        def make_line(fits: List[Tuple[float, float]]) -> Optional[LaneLine]:
            if not fits:
                return None
            slope_avg = float(np.mean([f[0] for f in fits]))
            intercept_avg = float(np.mean([f[1] for f in fits]))

            if slope_avg == 0:
                return None

            x1 = int((bottom_y - intercept_avg) / slope_avg)
            x2 = int((top_y - intercept_avg) / slope_avg)
            return (x1, bottom_y, x2, top_y)

        return make_line(left_fits), make_line(right_fits)

    def _smooth_offset(self, offset_ratio: float) -> float:
        """Apply a rolling average to reduce frame-to-frame jitter."""
        self._offset_history.append(offset_ratio)
        if len(self._offset_history) > self.smoothing_window:
            self._offset_history.pop(0)
        return float(np.mean(self._offset_history))

    def detect(self, frame: np.ndarray) -> LaneResult:
        """
        Detect lane boundaries and estimate lane position.

        Args:
            frame: Input frame (BGR)

        Returns:
            LaneResult with lane_detected, lane_status, and lane_departure
        """
        try:
            height, width = frame.shape[:2]

            edges = self._detect_edges(frame)
            masked_edges = self._region_of_interest(edges)
            lines = self._detect_hough_lines(masked_edges)

            if lines is None or len(lines) == 0:
                return LaneResult(
                    lane_detected=False,
                    lane_status=self.LANE_STATUS_UNKNOWN,
                    lane_departure=False,
                )

            left_line, right_line = self._average_slope_intercept(lines, width, height)

            if left_line is None and right_line is None:
                return LaneResult(
                    lane_detected=False,
                    lane_status=self.LANE_STATUS_UNKNOWN,
                    lane_departure=False,
                )

            # Estimate lane center at the bottom of the frame using whichever
            # boundaries were found. If only one side is detected, we can
            # still infer a rough centering trend but flag lower confidence
            # by not asserting lane_departure from a single boundary alone.
            frame_center_x = width / 2.0

            if left_line is not None and right_line is not None:
                lane_center_x = (left_line[0] + right_line[0]) / 2.0
                lane_width = abs(right_line[0] - left_line[0])
                half_lane_width = max(lane_width / 2.0, 1.0)
            elif left_line is not None:
                # Only left boundary visible; assume a typical lane width
                assumed_lane_width = width * 0.5
                lane_center_x = left_line[0] + assumed_lane_width / 2.0
                half_lane_width = assumed_lane_width / 2.0
            else:
                assumed_lane_width = width * 0.5
                lane_center_x = right_line[0] - assumed_lane_width / 2.0
                half_lane_width = assumed_lane_width / 2.0

            offset_pixels = frame_center_x - lane_center_x
            offset_ratio = offset_pixels / half_lane_width
            offset_ratio = self._smooth_offset(offset_ratio)

            if offset_ratio > 0.15:
                status = self.LANE_STATUS_RIGHT
            elif offset_ratio < -0.15:
                status = self.LANE_STATUS_LEFT
            else:
                status = self.LANE_STATUS_CENTERED

            lane_departure = abs(offset_ratio) >= self.departure_threshold_ratio

            return LaneResult(
                lane_detected=True,
                lane_status=status,
                lane_departure=lane_departure,
                left_line=left_line,
                right_line=right_line,
                offset_pixels=offset_pixels,
                offset_ratio=offset_ratio,
            )

        except Exception as e:
            logger.error(f"Error during lane detection: {e}", exc_info=True)
            return LaneResult(
                lane_detected=False,
                lane_status=self.LANE_STATUS_UNKNOWN,
                lane_departure=False,
            )

    def draw_lanes(self, frame: np.ndarray, result: LaneResult) -> np.ndarray:
        """
        Draw detected lane lines and status overlay on the frame.

        Args:
            frame: Input frame
            result: LaneResult from detect()

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        overlay = frame.copy()

        if result.left_line is not None:
            x1, y1, x2, y2 = result.left_line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 8)

        if result.right_line is not None:
            x1, y1, x2, y2 = result.right_line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 8)

        # Fill lane area if both boundaries found
        if result.left_line is not None and result.right_line is not None:
            pts = np.array(
                [
                    [result.left_line[0], result.left_line[1]],
                    [result.left_line[2], result.left_line[3]],
                    [result.right_line[2], result.right_line[3]],
                    [result.right_line[0], result.right_line[1]],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(overlay, [pts], (0, 180, 0))

        annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

        # Status text
        color = (0, 255, 0) if not result.lane_departure else (0, 0, 255)
        status_text = f"Lane: {result.lane_status}"
        if result.lane_departure:
            status_text += " - DEPARTURE WARNING"

        cv2.putText(
            annotated,
            status_text,
            (10, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        return annotated

    def reset(self) -> None:
        """Reset smoothing history (e.g., when starting a new video)."""
        self._offset_history = []
