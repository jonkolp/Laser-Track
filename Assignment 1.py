"""
Assignment 1: Real-time Laser Tracking
======================================
Computer Vision assignment script for detecting and tracking a laser pointer dot.

This script processes video frames from a connected webcam (or falls back to the
included sample video), segments the laser spot in the HSV color space, calculates
centroid coordinates, and displays real-time tracking overlays.
"""

import os
import cv2
import numpy as np


def main() -> None:
    # Attempt to open default webcam (0); fallback to sample video if camera is unavailable
    video_source: int | str = 0
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        fallback_video = "Laser assignment.mp4"
        if os.path.exists(fallback_video):
            print(f"Webcam 0 not detected. Falling back to '{fallback_video}'...")
            cap = cv2.VideoCapture(fallback_video)
        else:
            print("Error: Could not open video source or camera.")
            return

    # HSV thresholds for red laser detection
    # Note: OpenCV HSV ranges: H: 0-179, S: 0-255, V: 0-255
    lower_color = np.array([160, 100, 100])
    upper_color = np.array([180, 255, 255])

    # Minimum contour area (pixels) to filter out spurious noise
    min_contour_area = 5.0

    print("Running Assignment 1 Laser Tracker... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished video stream or unable to read frame.")
            break

        # Standardize frame resolution
        frame = cv2.resize(frame, (640, 480))

        # Convert BGR frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to isolate laser colors
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Morphological opening to clean small noise artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find external contours of detected regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize blank mask for drawing the laser point
        laser_mask = np.zeros_like(frame)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area >= min_contour_area:
                # Compute centroid via spatial moments
                m = cv2.moments(largest_contour)
                if m["m00"] != 0:
                    center_x = int(m["m10"] / m["m00"])
                    center_y = int(m["m01"] / m["m00"])
                else:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)

                # Draw bounding box and centroid indicator
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                # Draw solid circle on the laser mask window
                cv2.circle(laser_mask, (center_x, center_y), 20, (255, 255, 255), -1)

                # Display coordinates on the original frame
                cv2.putText(
                    frame,
                    f"X: {center_x}, Y: {center_y}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        # Display windows
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Laser Tracker", laser_mask)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean release of resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
