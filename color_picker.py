"""
HSV Color Picker & Calibration Tool
====================================
Interactive HSV color calibration tool for computer vision projects.

Features:
- Left-click on any pixel to inspect BGR and HSV values in real-time.
- Automatically calculates recommended lower and upper HSV threshold bounds.
- Displays an on-screen HUD with the sampled color swatch and values.
- Pause/Resume playback with SPACEBAR to easily sample fast-moving targets.
- Supports both live webcam and recorded video files.
"""

import argparse
from typing import Optional, Tuple
import cv2
import numpy as np


class ColorPicker:
    """Manages color picking state and mouse interactions."""

    def __init__(self, h_tol: int = 10, s_tol: int = 40, v_tol: int = 50) -> None:
        self.h_tol = h_tol
        self.s_tol = s_tol
        self.v_tol = v_tol

        self.last_clicked_pos: Optional[Tuple[int, int]] = None
        self.last_bgr: Optional[Tuple[int, int, int]] = None
        self.last_hsv: Optional[Tuple[int, int, int]] = None
        self.suggested_lower: Optional[Tuple[int, int, int]] = None
        self.suggested_upper: Optional[Tuple[int, int, int]] = None

    def on_mouse_click(self, event: int, x: int, y: int, flags: int, param: dict) -> None:
        """Mouse callback handling click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param.get("frame")
            hsv = param.get("hsv")
            if frame is None or hsv is None:
                return

            h, w = frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                bgr_val = tuple(map(int, frame[y, x]))
                hsv_val = tuple(map(int, hsv[y, x]))

                self.last_clicked_pos = (x, y)
                self.last_bgr = (bgr_val[0], bgr_val[1], bgr_val[2])
                self.last_hsv = (hsv_val[0], hsv_val[1], hsv_val[2])

                # Calculate suggested bounds with clamping
                h_val, s_val, v_val = hsv_val
                low_h = max(0, h_val - self.h_tol)
                high_h = min(179, h_val + self.h_tol)
                low_s = max(20, s_val - self.s_tol)
                high_s = min(255, s_val + self.s_tol)
                low_v = max(50, v_val - self.v_tol)
                high_v = min(255, v_val + self.v_tol)

                self.suggested_lower = (low_h, low_s, low_v)
                self.suggested_upper = (high_h, high_s, high_v)

                print("\n" + "=" * 50)
                print(f"Sampled pixel at (X: {x}, Y: {y})")
                print(f"  • BGR: {self.last_bgr}")
                print(f"  • HSV: {self.last_hsv} (H: {h_val}, S: {s_val}, V: {v_val})")
                print(f"Suggested lower bound: np.array([{low_h}, {low_s}, {low_v}])")
                print(f"Suggested upper bound: np.array([{high_h}, {high_s}, {high_v}])")
                print("=" * 50)

    def draw_hud(self, display_frame: np.ndarray, is_paused: bool) -> None:
        """Draw HUD instructions, clicked crosshair, and sampled info."""
        h, w = display_frame.shape[:2]

        # Draw status banner at top
        banner_color = (0, 0, 180) if is_paused else (40, 40, 40)
        cv2.rectangle(display_frame, (0, 0), (w, 40), banner_color, -1)
        status_text = "PAUSED (Press SPACE to resume)" if is_paused else "RUNNING (Press SPACE to pause)"
        cv2.putText(
            display_frame,
            f"{status_text} | Left-Click to pick | 'q' to quit",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        # Draw clicked point crosshairs and info box
        if self.last_clicked_pos and self.last_hsv and self.last_bgr:
            x, y = self.last_clicked_pos

            # Crosshair circle
            cv2.circle(display_frame, (x, y), 8, (0, 255, 255), 2)
            cv2.drawMarker(display_frame, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

            # Floating information card
            info_h, info_w = 70, 260
            card_x = min(x + 15, w - info_w - 10)
            card_y = min(y + 15, h - info_h - 10)

            # Background card
            cv2.rectangle(
                display_frame,
                (card_x, card_y),
                (card_x + info_w, card_y + info_h),
                (20, 20, 20),
                -1,
            )
            cv2.rectangle(
                display_frame,
                (card_x, card_y),
                (card_x + info_w, card_y + info_h),
                (0, 255, 255),
                1,
            )

            # Color swatch
            cv2.rectangle(
                display_frame,
                (card_x + 8, card_y + 8),
                (card_x + 38, card_y + 38),
                self.last_bgr,
                -1,
            )
            cv2.rectangle(
                display_frame,
                (card_x + 8, card_y + 8),
                (card_x + 38, card_y + 38),
                (255, 255, 255),
                1,
            )

            # Text values
            cv2.putText(
                display_frame,
                f"HSV: {self.last_hsv}",
                (card_x + 48, card_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            if self.suggested_lower and self.suggested_upper:
                cv2.putText(
                    display_frame,
                    f"Low : {self.suggested_lower}",
                    (card_x + 48, card_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (180, 255, 180),
                    1,
                )
                cv2.putText(
                    display_frame,
                    f"High: {self.suggested_upper}",
                    (card_x + 48, card_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (180, 255, 180),
                    1,
                )


def run_picker(source: str = "0", width: int = 640, height: int = 480) -> None:
    """Run interactive color calibration window."""
    src: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'.")
        return

    picker = ColorPicker()
    window_name = "HSV Color Picker"
    cv2.namedWindow(window_name)

    shared_context = {"frame": None, "hsv": None}
    cv2.setMouseCallback(window_name, picker.on_mouse_click, shared_context)

    is_paused = False
    current_frame = None

    print("\n" + "=" * 60)
    print("  HSV Color Picker Started")
    print("  • Left-click: Sample HSV and generate bounds")
    print("  • Spacebar:   Pause / Resume video")
    print("  • 'c':        Clear selection")
    print("  • 'q' or ESC: Quit")
    print("=" * 60 + "\n")

    try:
        while True:
            if not is_paused or current_frame is None:
                ret, frame = cap.read()
                if not ret:
                    if not str(source).isdigit():
                        # Loop video file for convenience
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break
                    else:
                        break
                current_frame = cv2.resize(frame, (width, height))

            hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
            shared_context["frame"] = current_frame
            shared_context["hsv"] = hsv

            display = current_frame.copy()
            picker.draw_hud(display, is_paused)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(30 if not is_paused else 100) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord(" "):  # Spacebar toggles pause
                is_paused = not is_paused
            elif key == ord("c"):  # Clear
                picker.last_clicked_pos = None
                picker.last_bgr = None
                picker.last_hsv = None
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for color picker."""
    parser = argparse.ArgumentParser(
        description="Interactive HSV Color Picker and Threshold Calibrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--source",
        default="0",
        help="Input video source: webcam index (e.g. '0') or video file path.",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Display frame width."
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Display frame height."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_picker(source=args.source, width=args.width, height=args.height)
