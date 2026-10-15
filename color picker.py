"""
Color Picker Tool (Interactive HSV Inspector)
==============================================
Utility script for obtaining HSV color boundaries for computer vision segmentation.

Usage:
    Left-click anywhere on the video frame to print the exact HSV values
    and suggested lower/upper bounds for tracker configuration.
"""

import os
import cv2
import numpy as np


# Global context to share frame data with the OpenCV mouse callback
current_hsv = None


def get_hsv_value(event: int, x: int, y: int, flags: int, param: object) -> None:
    """Callback triggered on mouse interaction."""
    global current_hsv
    if event == cv2.EVENT_LBUTTONDOWN and current_hsv is not None:
        h, w = current_hsv.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            pixel_hsv = current_hsv[y, x]
            h_val, s_val, v_val = int(pixel_hsv[0]), int(pixel_hsv[1]), int(pixel_hsv[2])

            # Provide suggested ranges (+/- 10 for H, +/- 40 for S, +/- 50 for V)
            low_h = max(0, h_val - 10)
            high_h = min(179, h_val + 10)
            low_s = max(30, s_val - 40)
            high_s = min(255, s_val + 40)
            low_v = max(50, v_val - 50)
            high_v = min(255, v_val + 50)

            print(f"\n[Sample] Clicked at (X: {x}, Y: {y})")
            print(f"  HSV Values: [H={h_val}, S={s_val}, V={v_val}]")
            print(f"  Suggested Lower Bound: np.array([{low_h}, {low_s}, {low_v}])")
            print(f"  Suggested Upper Bound: np.array([{high_h}, {high_s}, {high_v}])")


def main() -> None:
    global current_hsv

    # Initialize video capture (webcam 0 with fallback to sample video)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        fallback_video = "Laser assignment.mp4"
        if os.path.exists(fallback_video):
            print(f"Webcam 0 unavailable. Falling back to '{fallback_video}'...")
            cap = cv2.VideoCapture(fallback_video)
        else:
            print("Error: Could not open camera or video.")
            return

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", get_hsv_value)

    print("Running Color Picker...")
    print("  • Left-click on the video window to inspect HSV values.")
    print("  • Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # If playing from video file, loop back to the start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        frame = cv2.resize(frame, (640, 480))
        current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Informative overlay
        cv2.putText(
            frame,
            "Left-Click to sample HSV | 'q' to quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Frame", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
