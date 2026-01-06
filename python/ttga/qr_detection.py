# Copyright 2026 Marc-Antoine Desjardins
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QR code detection module using PyBoof for MicroQR detection.

This module provides QR code detection capabilities using the PyBoof library,
with support for MicroQR codes and drawing detection results.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
import pyboof as pb


@dataclass
class QRDetection:
    """Represents a detected QR code.

    Attributes:
        message: The decoded message from the QR code.
        corners: List of corner points [(x, y), ...] in clockwise order.
        bounds: Bounding box as (x, y, width, height).
    """
    message: str
    corners: list[tuple[float, float]]
    bounds: tuple[int, int, int, int]


class QRDetector:
    """QR code detector using PyBoof.

    This class provides QR code detection capabilities (including MicroQR) using
    the PyBoof library, which wraps BoofCV for computer vision tasks.

    Example:
        >>> detector = QRDetector()
        >>> image = cv.imread("image.jpg")
        >>> detections = detector.detect(image)
        >>> for detection in detections:
        ...     print(f"Detected: {detection.message}")
    """

    def __init__(self) -> None:
        """Initialize the QR detector."""
        # Create MicroQR detector
        self._detector = pb.FactoryFiducial(np.uint8).microqr()

    def detect(self, image: np.ndarray) -> list[QRDetection]:
        """Detect MicroQR codes in an image.

        Args:
            image: Input image as numpy array (BGR or grayscale).

        Returns:
            List of QRDetection objects for all detected QR codes.

        Example:
            >>> detector = QRDetector()
            >>> image = cv.imread("qr_image.jpg")
            >>> detections = detector.detect(image)
            >>> print(f"Found {len(detections)} QR code(s)")
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        # Ensure the array is contiguous, uint8, and copied
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        # Convert to PyBoof image
        boof_image = pb.ndarray_to_boof(gray)

        # Detect QR codes
        self._detector.detect(boof_image)

        detections = []
        for qr in self._detector.detections:
            # Get message
            message = qr.message

            # Get corner points from polygon vertexes
            corners = []
            for vertex in qr.bounds.vertexes:
                corners.append((float(vertex.x), float(vertex.y)))

            # Calculate bounding box
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            bounds = (x_min, y_min, x_max - x_min, y_max - y_min)

            detections.append(QRDetection(
                message=message,
                corners=corners,
                bounds=bounds
            ))

        return detections


def draw_qr_detections(
    image: np.ndarray,
    detections: list[QRDetection],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    draw_text: bool = True
) -> np.ndarray:
    """Draw detected QR codes on an image.

    Draws a polygon around each detected QR code and optionally displays
    the decoded message.

    Args:
        image: Input image to draw on (will be copied).
        detections: List of QRDetection objects to draw.
        color: BGR color tuple for drawing (default: green).
        thickness: Line thickness in pixels (default: 2).
        draw_text: Whether to draw the decoded message (default: True).

    Returns:
        New image with detections drawn.

    Example:
        >>> detector = QRDetector()
        >>> image = cv.imread("qr_image.jpg")
        >>> detections = detector.detect(image)
        >>> result = draw_qr_detections(image, detections)
        >>> cv.imwrite("result.jpg", result)
    """
    # Create a copy to avoid modifying the original
    output = image.copy()

    for detection in detections:
        # Draw polygon around QR code
        corners_array = np.array(detection.corners, dtype=np.int32)
        cv.polylines(output, [corners_array], isClosed=True, color=color, thickness=thickness)

        # Draw text if requested
        if draw_text:
            # Try to parse as integer for display
            try:
                value = int(detection.message)
                text = str(value)
            except ValueError:
                text = detection.message

            # Position text at top-left corner with offset
            x, y, w, h = detection.bounds
            text_pos = (x, y - 10 if y > 20 else y + h + 20)

            # Draw text background for better visibility
            (text_w, text_h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv.rectangle(
                output,
                (text_pos[0] - 5, text_pos[1] - text_h - 5),
                (text_pos[0] + text_w + 5, text_pos[1] + 5),
                (0, 0, 0),
                -1
            )

            # Draw text
            cv.putText(
                output,
                text,
                text_pos,
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv.LINE_AA
            )

    return output
