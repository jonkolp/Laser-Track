"""
Laser Pointer Tracker
=====================
Real-time laser pointer detection and tracking using OpenCV and NumPy.

Features:
- Configurable HSV thresholding with optional red-hue wrap support (0-10 & 170-180).
- Subpixel centroid calculation using image spatial moments.
- Noise filtering with minimum contour area thresholding.
- Motion history / trajectory trail visualization.
- Flexible input support (webcam index or video file) with FPS pacing.
"""

from collections import deque
import argparse
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class LaserTracker:
    """Detects and tracks a laser pointer dot within video frames."""

    def __init__(
        self,
        lower_hsv: Tuple[int, int, int] = (160, 100, 100),
        upper_hsv: Tuple[int, int, int] = (180, 255, 255),
        lower_hsv_aux: Optional[Tuple[int, int, int]] = (0, 100, 100),
        upper_hsv_aux: Optional[Tuple[int, int, int]] = (10, 255, 255),
        min_area: float = 5.0,
        trail_length: int = 32,
    ) -> None:
        """
        Initialize tracker parameters.

        Args:
            lower_hsv: Primary lower HSV bound (H: 0-179, S: 0-255, V: 0-255).
            upper_hsv: Primary upper HSV bound.
            lower_hsv_aux: Auxiliary lower HSV bound for wrapped hue (e.g., red 0-10).
            upper_hsv_aux: Auxiliary upper HSV bound for wrapped hue.
            min_area: Minimum contour area in pixels to eliminate noise.
            trail_length: Number of past centroid points to retain for motion trail.
        """
        self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
        self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)

        self.lower_hsv_aux = np.array(lower_hsv_aux, dtype=np.uint8) if lower_hsv_aux else None
        self.upper_hsv_aux = np.array(upper_hsv_aux, dtype=np.uint8) if upper_hsv_aux else None

        self.min_area = min_area
        self.trail: deque = deque(maxlen=trail_length)

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray, np.ndarray]:
        """
        Process a single frame to detect the laser spot.

        Args:
            frame: Input BGR image from camera or video stream.

        Returns:
            center: (x, y) coordinates of the detected laser, or None if not found.
            processed_frame: Frame with visual overlays, bounding box, and trail.
            laser_mask: Mask visualization highlighting the detected laser location.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Primary color threshold mask
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Combine with auxiliary mask if defined (useful for red hue wrap around 0/180)
        if self.lower_hsv_aux is not None and self.upper_hsv_aux is not None:
            aux_mask = cv2.inRange(hsv, self.lower_hsv_aux, self.upper_hsv_aux)
            mask = cv2.bitwise_or(mask, aux_mask)

        # Morphological operations to remove small speckles and close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find external contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center: Optional[Tuple[int, int]] = None
        laser_mask = np.zeros_like(frame)
        output_frame = frame.copy()

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area >= self.min_area:
                # Compute centroid using spatial moments
                moments = cv2.moments(largest_contour)
                if moments["m00"] != 0:
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                else:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)

                center = (center_x, center_y)

                # Draw bounding box and target crosshair
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(output_frame, center, 4, (0, 0, 255), -1)

                # Draw laser indicator on laser_mask
                cv2.circle(laser_mask, center, 20, (255, 255, 255), -1)

                # Display coordinates on frame
                cv2.putText(
                    output_frame,
                    f"X: {center_x}, Y: {center_y} (Area: {int(area)})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

        # Update motion trail
        self.trail.appendleft(center)

        # Draw motion trail on output frame
        for i in range(1, len(self.trail)):
            if self.trail[i - 1] is None or self.trail[i] is None:
                continue
            thickness = int(np.sqrt(len(self.trail) / float(i + 1)) * 2)
            cv2.line(output_frame, self.trail[i - 1], self.trail[i], (0, 255, 255), thickness)

        return center, output_frame, laser_mask


def run_tracker(
    source: str = "0",
    width: int = 640,
    height: int = 480,
    lower_hsv: Tuple[int, int, int] = (160, 100, 100),
    upper_hsv: Tuple[int, int, int] = (180, 255, 255),
    enable_red_wrap: bool = True,
    min_area: float = 5.0,
    trail_length: int = 32,
    output_path: Optional[str] = None,
) -> None:
    """Run the laser tracking loop with display windows."""
    # Resolve video source (integer index for webcam or string path for file)
    src: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'.")
        return

    # Video timing configuration
    is_video_file = not str(source).isdigit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if (is_video_file and fps and fps > 0) else 0.0

    aux_lower = (0, 100, 100) if enable_red_wrap else None
    aux_upper = (10, 255, 255) if enable_red_wrap else None

    tracker = LaserTracker(
        lower_hsv=lower_hsv,
        upper_hsv=upper_hsv,
        lower_hsv_aux=aux_lower,
        upper_hsv_aux=aux_upper,
        min_area=min_area,
        trail_length=trail_length,
    )

    writer: Optional[cv2.VideoWriter] = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (width, height))

    print("Starting Laser Tracker... Press 'q' or 'ESC' in any window to exit.")

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()

            if not ret:
                if is_video_file:
                    print("Reached end of video stream.")
                else:
                    print("Error: Could not read frame from camera.")
                break

            frame = cv2.resize(frame, (width, height))
            center, output_frame, laser_mask = tracker.process_frame(frame)

            if writer:
                writer.write(output_frame)

            cv2.imshow("Laser Tracker - Original & Overlay", output_frame)
            cv2.imshow("Laser Tracker - Target Mask", laser_mask)

            # Frame pacing for video playback
            elapsed = time.time() - start_time
            wait_time = max(1, int((frame_delay - elapsed) * 1000)) if is_video_file else 1

            key = cv2.waitKey(wait_time) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Laser tracker finished cleanly.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Laser Pointer Tracker with OpenCV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--source",
        default="0",
        help="Input video source: webcam index (e.g. '0') or path to video file.",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Frame width to resize input stream."
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Frame height to resize input stream."
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=5.0,
        help="Minimum contour area to consider as laser spot.",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=32,
        help="Length of motion history trail (0 to disable).",
    )
    parser.add_argument(
        "--lower-hsv",
        nargs=3,
        type=int,
        default=[160, 100, 100],
        metavar=("H", "S", "V"),
        help="Primary lower HSV bounds.",
    )
    parser.add_argument(
        "--upper-hsv",
        nargs=3,
        type=int,
        default=[180, 255, 255],
        metavar=("H", "S", "V"),
        help="Primary upper HSV bounds.",
    )
    parser.add_argument(
        "--no-red-wrap",
        action="store_true",
        help="Disable auxiliary HSV check for low red hues (0-10).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path to save tracked output video (.mp4).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tracker(
        source=args.source,
        width=args.width,
        height=args.height,
        lower_hsv=tuple(args.lower_hsv),
        upper_hsv=tuple(args.upper_hsv),
        enable_red_wrap=not args.no_red_wrap,
        min_area=args.min_area,
        trail_length=args.trail,
        output_path=args.output,
    )
