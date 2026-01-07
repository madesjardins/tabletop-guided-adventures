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

"""Camera feed module for capturing webcam frames with frame rate control.

This module provides a CameraFeed class that captures frames from a webcam
using OpenCV and emits them via PySide6 signals with proper frame rate limiting.
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2 as cv
import numpy as np
from PySide6.QtCore import QObject, Signal


DEFAULT_CAPTURE_PROPERTIES: dict[int, int | float] = {
    cv.CAP_PROP_HW_ACCELERATION: cv.VIDEO_ACCELERATION_ANY,
    cv.CAP_PROP_FRAME_WIDTH: 1920,
    cv.CAP_PROP_FRAME_HEIGHT: 1080,
    cv.CAP_PROP_FPS: 30,
    cv.CAP_PROP_AUTOFOCUS: 1,
    cv.CAP_PROP_FOCUS: 0,
    cv.CAP_PROP_AUTO_EXPOSURE: 1,
    cv.CAP_PROP_EXPOSURE: -5,
    cv.CAP_PROP_FOURCC: cv.VideoWriter_fourcc(*'YUY2'),
    cv.CAP_PROP_BRIGHTNESS: 128,
    cv.CAP_PROP_CONTRAST: 128,
    cv.CAP_PROP_GAIN: 128,
    cv.CAP_PROP_SATURATION: 128,
    cv.CAP_PROP_SHARPNESS: 128,
    cv.CAP_PROP_ZOOM: 100
}


def get_frame_with_text(
    text: str,
    width: int = 960,
    height: int = 540,
    position: Optional[tuple[int, int]] = None,
    font: int = cv.FONT_HERSHEY_PLAIN,
    fontsize: int = 8,
    color: Optional[tuple[int, int, int]] = None,
    bg_color: Optional[tuple[int, int, int]] = None
) -> np.ndarray:
    """Create a frame with text rendered on it.

    Args:
        text: The text to render on the frame.
        width: Frame width in pixels (default: 960).
        height: Frame height in pixels (default: 540).
        position: Lower left corner of text as (x, y). If None, text is centered
            vertically and aligned left with 10px margin (default: None).
        font: OpenCV font type (default: cv.FONT_HERSHEY_PLAIN).
        fontsize: Font size/scale (default: 8).
        color: Text color as RGB tuple (0-255). If None, uses white (default: None).
        bg_color: Background color as RGB tuple (0-255). If None, uses black (default: None).

    Returns:
        Numpy array representing the frame with text.

    Example:
        >>> frame = get_frame_with_text("Error: Camera not found", color=(255, 0, 0))
    """
    if color is None:
        color = (255, 255, 255)

    if bg_color is None:
        bg_color = (0, 0, 0)

    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

    if position is None:
        text_size = cv.getTextSize(text, font, fontsize, thickness=2)[0]
        x = 10
        y = (height + text_size[1]) // 2
        position = (x, y)

    cv.putText(
        frame,
        text,
        position,
        font,
        fontsize,
        color,
        thickness=2,
        lineType=cv.LINE_AA
    )

    return frame


class CameraFeed(QObject):
    """Webcam frame capture with PySide6 signal emission and frame rate control.

    This class manages webcam capture using OpenCV, emitting frames via Qt signals
    while maintaining a maximum frame rate. Multiple instances can run simultaneously
    with different device IDs.

    Signals:
        frame_captured: Emitted when a frame is available. Passes (frame: np.ndarray, success: bool).
            success=True for actual camera frames, False for error message frames.
        error_occurred: Emitted when an error occurs, passes error message string.

    Attributes:
        device_id: The camera device ID to capture from.
        capture_api: OpenCV capture API backend.

    Example:
        >>> feed = CameraFeed(device_id=0)
        >>> feed.frame_captured.connect(on_frame_received)
        >>> feed.start()
        >>> # ... later ...
        >>> feed.stop()
        >>> feed.release()
    """

    frame_captured = Signal(np.ndarray, bool)
    error_occurred = Signal(str)

    def __init__(
        self,
        device_id: int = 0,
        capture_api: int = cv.CAP_DSHOW,
        capture_properties: Optional[dict[int, int | float]] = None,
        camera_info: Optional[dict[str, any]] = None,
        parent: Optional[QObject] = None
    ) -> None:
        """Initialize the camera feed.

        Args:
            device_id: Camera device ID (default: 0 for primary camera).
            capture_api: OpenCV capture API backend (default: cv.CAP_DSHOW).
            capture_properties: Optional dictionary of OpenCV capture properties.
                If provided, these override the defaults. Missing properties will
                use values from DEFAULT_CAPTURE_PROPERTIES (default: None).
            camera_info: Optional camera info dict with 'index', 'name', 'path' keys.
            parent: Optional parent QObject for Qt object hierarchy.
        """
        super().__init__(parent)

        self.device_id: int = device_id
        self.capture_api: int = capture_api
        self.camera_info: Optional[dict[str, any]] = camera_info

        self._capture: Optional[cv.VideoCapture] = None
        self._is_running: bool = False
        self._capture_thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()

        self._last_frame_resolution: tuple[int, int] = (0, 0)

        self._capture_properties: dict[int, int | float] = DEFAULT_CAPTURE_PROPERTIES.copy()
        if capture_properties is not None:
            self._capture_properties.update(capture_properties)

    def _initialize_capture(self) -> bool:
        """Initialize the video capture device with configured properties.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            self._capture = cv.VideoCapture(self.device_id, self.capture_api)

            if not self._capture.isOpened():
                error_msg = f"Failed to open camera device {self.device_id}"
                self.error_occurred.emit(error_msg)
                error_frame = get_frame_with_text(
                    error_msg,
                    width=self._capture_properties.get(cv.CAP_PROP_FRAME_WIDTH, 1920),
                    height=self._capture_properties.get(cv.CAP_PROP_FRAME_HEIGHT, 1080),
                    color=(255, 0, 0)
                )
                self.frame_captured.emit(error_frame, False)
                return False

            for prop, value in self._capture_properties.items():
                self._capture.set(prop, value)

            return True

        except Exception as e:
            error_msg = f"Error initializing camera: {e}"
            self.error_occurred.emit(error_msg)
            error_frame = get_frame_with_text(
                error_msg,
                width=self._capture_properties.get(cv.CAP_PROP_FRAME_WIDTH, 1920),
                height=self._capture_properties.get(cv.CAP_PROP_FRAME_HEIGHT, 1080),
                color=(255, 0, 0)
            )
            self.frame_captured.emit(error_frame, False)
            return False

    def _capture_loop(self) -> None:
        """Main capture loop running in a separate thread.

        Captures frames as fast as possible and emits them via the frame_captured signal.
        """
        while self._is_running:
            if self._capture is None or not self._capture.isOpened():
                error_msg = "Camera capture is not available"
                self.error_occurred.emit(error_msg)
                error_frame = get_frame_with_text(
                    error_msg,
                    width=self._capture_properties.get(cv.CAP_PROP_FRAME_WIDTH, 1920),
                    height=self._capture_properties.get(cv.CAP_PROP_FRAME_HEIGHT, 1080),
                    color=(255, 0, 0)
                )
                self.frame_captured.emit(error_frame, False)
                break

            ret, frame = self._capture.read()

            if not ret:
                error_msg = "Failed to read frame from camera"
                self.error_occurred.emit(error_msg)
                error_frame = get_frame_with_text(
                    error_msg,
                    width=self._capture_properties.get(cv.CAP_PROP_FRAME_WIDTH, 1920),
                    height=self._capture_properties.get(cv.CAP_PROP_FRAME_HEIGHT, 1080),
                    color=(255, 0, 0)
                )
                self.frame_captured.emit(error_frame, False)
                continue

            # Update frame resolution
            with self._lock:
                height, width = frame.shape[:2]
                self._last_frame_resolution = (width, height)

            self.frame_captured.emit(frame, True)

    def start(self) -> bool:
        """Start capturing frames from the camera.

        Initializes the video capture device and starts the capture thread.

        Returns:
            True if started successfully, False otherwise.

        Example:
            >>> feed = CameraFeed(device_id=0)
            >>> if feed.start():
            ...     print("Camera started successfully")
        """
        with self._lock:
            if self._is_running:
                return True

            if not self._initialize_capture():
                return False

            self._is_running = True

            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name=f"CameraFeed-{self.device_id}"
            )
            self._capture_thread.start()

            return True

    def stop(self) -> None:
        """Stop capturing frames from the camera.

        Stops the capture thread but does not release the camera device.
        Use release() to fully clean up resources.

        Example:
            >>> feed.stop()
        """
        with self._lock:
            if not self._is_running:
                return

            self._is_running = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

    def release(self) -> None:
        """Release the camera device and clean up resources.

        Stops capture if running and releases the OpenCV VideoCapture object.

        Example:
            >>> feed.release()
        """
        self.stop()

        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def is_running(self) -> bool:
        """Check if the camera feed is currently running.

        Returns:
            True if capturing frames, False otherwise.

        Example:
            >>> if feed.is_running():
            ...     print("Camera is active")
        """
        with self._lock:
            return self._is_running

    def get_frame_resolution(self) -> tuple[int, int]:
        """Get the current frame resolution (width, height).

        Returns:
            Tuple of (width, height) in pixels, or (0, 0) if not available.

        Example:
            >>> width, height = feed.get_frame_resolution()
            >>> print(f"Resolution: {width}x{height}")
        """
        with self._lock:
            if self._last_frame_resolution != (0, 0):
                return self._last_frame_resolution

            if self._capture is not None and self._capture.isOpened():
                width = int(self._capture.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(self._capture.get(cv.CAP_PROP_FRAME_HEIGHT))
                return (width, height)

            return (0, 0)

    def update_capture_property(self, prop: int, value: int | float) -> bool:
        """Update a single capture property.

        The property will be applied immediately if the camera is running,
        or stored for the next start() call.

        Args:
            prop: OpenCV capture property constant (e.g., cv.CAP_PROP_BRIGHTNESS).
            value: New value for the property.

        Returns:
            True if property was updated successfully, False otherwise.

        Example:
            >>> feed.update_capture_property(cv.CAP_PROP_BRIGHTNESS, 150)
        """
        with self._lock:
            self._capture_properties[prop] = value

            if self._capture is not None and self._capture.isOpened():
                return self._capture.set(prop, value)

            return True

    def update_capture_properties(self, properties: dict[int, int | float]) -> None:
        """Update multiple capture properties at once.

        Properties will be applied immediately if the camera is running,
        or stored for the next start() call.

        Args:
            properties: Dictionary mapping property constants to values.

        Example:
            >>> feed.update_capture_properties({
            ...     cv.CAP_PROP_BRIGHTNESS: 150,
            ...     cv.CAP_PROP_CONTRAST: 140
            ... })
        """
        with self._lock:
            self._capture_properties.update(properties)

            if self._capture is not None and self._capture.isOpened():
                for prop, value in properties.items():
                    self._capture.set(prop, value)

    def get_capture_property(self, prop: int) -> Optional[int | float]:
        """Get the current value of a capture property.

        Args:
            prop: OpenCV capture property constant.

        Returns:
            Current property value, or None if not available.

        Example:
            >>> brightness = feed.get_capture_property(cv.CAP_PROP_BRIGHTNESS)
        """
        with self._lock:
            if self._capture is not None and self._capture.isOpened():
                return self._capture.get(prop)

            return self._capture_properties.get(prop)

    def __enter__(self) -> CameraFeed:
        """Context manager entry.

        Returns:
            The CameraFeed instance.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self.release()
