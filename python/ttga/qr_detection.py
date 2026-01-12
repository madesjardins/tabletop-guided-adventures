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
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np
import pyboof as pb
from PySide6 import QtCore

if TYPE_CHECKING:
    from .zone import Zone


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


class QRDetector(QtCore.QObject):
    """QR code detector using PyBoof with periodic detection and signal emission.

    This class provides QR code detection capabilities (including MicroQR) using
    the PyBoof library, which wraps BoofCV for computer vision tasks. It can
    automatically detect QR codes at a specified refresh rate and emit signals
    with the detection results.

    Signals:
        detections_updated: Emitted with list of QRDetection objects when new detections are found.

    Example:
        >>> detector = QRDetector(zone, refresh_rate=5)
        >>> detector.detections_updated.connect(lambda dets: print(f"Found {len(dets)} QR codes"))
        >>> detector.start()
    """

    detections_updated = QtCore.Signal(list)  # list[QRDetection]

    def __init__(self, zone: Zone, camera_manager, refresh_rate: int = 5) -> None:
        """Initialize the QR detector.

        Args:
            zone: Zone object to get images from for detection.
            camera_manager: CameraManager instance to access cameras.
            refresh_rate: Detection refresh rate in Hz (detections per second).
        """
        super().__init__()
        self.zone = zone
        self.camera_manager = camera_manager
        self.refresh_rate = refresh_rate

        # Create MicroQR detector
        self._detector = pb.FactoryFiducial(np.uint8).microqr()

        # Timer for periodic detection
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._on_timer)
        self._running = False

    def start(self) -> None:
        """Start periodic QR code detection."""
        if not self._running:
            self._running = True
            interval_ms = int(1000 / self.refresh_rate)
            self._timer.start(interval_ms)

    def stop(self) -> None:
        """Stop periodic QR code detection."""
        if self._running:
            self._running = False
            self._timer.stop()

    def is_running(self) -> bool:
        """Check if detector is currently running.

        Returns:
            True if detector is running, False otherwise.
        """
        return self._running

    def set_refresh_rate(self, refresh_rate: int) -> None:
        """Set the detection refresh rate.

        Args:
            refresh_rate: New refresh rate in Hz.
        """
        self.refresh_rate = refresh_rate
        if self._running:
            interval_ms = int(1000 / self.refresh_rate)
            self._timer.setInterval(interval_ms)

    @QtCore.Slot()
    def _on_timer(self) -> None:
        """Timer callback for periodic detection."""
        # Get latest image from zone (cropped to ROI)
        image = self.zone.get_latest_camera_image_cropped(self.camera_manager)
        if image is None:
            return

        # Detect QR codes
        detections = self.detect(image)

        # Emit signal with detections
        self.detections_updated.emit(detections)

    def detect(self, image: np.ndarray) -> list[QRDetection]:
        """Detect MicroQR codes in an image.

        Args:
            image: Input image as numpy array (BGR or grayscale).

        Returns:
            List of QRDetection objects for all detected QR codes.

        Example:
            >>> detector = QRDetector(zone)
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
