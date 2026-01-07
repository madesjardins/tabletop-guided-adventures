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

"""Camera module for managing individual camera instances.

This module contains the Camera class which wraps a CameraFeed and manages
frame buffering, calibration data, and camera control.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore

from .camera_feed import CameraFeed


class Camera(QtCore.QObject):
    """Camera class managing a camera feed with frame buffering.

    This class wraps a CameraFeed and provides frame buffering, calibration
    data storage, and convenient camera control methods.

    Attributes:
        name: Camera name identifier.
        camera_feed: CameraFeed instance.
        calibration_data: Camera calibration data (None if not calibrated).
    """

    def __init__(
        self,
        name: str,
        backend: int,
        device_id: int,
        parent: QtCore.QObject | None = None
    ) -> None:
        """Initialize the camera.

        Args:
            name: Camera name identifier.
            backend: OpenCV capture API backend.
            device_id: Camera device ID.
            parent: Parent QObject.
        """
        super().__init__(parent)

        self.name = name
        self.calibration_data = None
        self._backend = backend
        self._device_id = device_id

        # Create camera feed
        self.camera_feed = CameraFeed(device_id, backend)

        # Frame buffer - holds 3 frames
        self._frame_buffer: list[np.ndarray | None] = [None, None, None]
        self._buffer_index = 0
        self._current_index = 0

        # Connect frame captured signal
        self.camera_feed.frame_captured.connect(self._on_frame_captured)

    @QtCore.Slot(np.ndarray)
    def _on_frame_captured(self, frame: np.ndarray) -> None:
        """Handle frame captured from camera feed.

        Args:
            frame: Captured frame.
        """
        # Increment and wrap buffer index
        self._buffer_index = (self._buffer_index + 1) % 3

        # Store frame in buffer
        self._frame_buffer[self._buffer_index] = frame.copy()

        # Update current index
        self._current_index = self._buffer_index

    def get_frame(self) -> np.ndarray | None:
        """Get the current frame from the buffer.

        Returns:
            Current frame or None if no frame available.
        """
        return self._frame_buffer[self._current_index]

    def start(self) -> None:
        """Start the camera feed."""
        self.camera_feed.start()

    def stop(self) -> None:
        """Stop the camera feed."""
        self.camera_feed.stop()

    def release(self) -> None:
        """Release the camera feed resources."""
        self.camera_feed.release()

    def get_property(self, prop_id: int) -> float:
        """Get a camera property value.

        Args:
            prop_id: OpenCV property ID.

        Returns:
            Property value.
        """
        result = self.camera_feed.get_capture_property(prop_id)
        return result if result is not None else 0.0

    def set_property(self, prop_id: int, value: float) -> bool:
        """Set a camera property value.

        Args:
            prop_id: OpenCV property ID.
            value: Property value to set.

        Returns:
            True if property was set successfully.
        """
        return self.camera_feed.update_capture_property(prop_id, value)

    def get_backend(self) -> int:
        """Get the camera backend API.

        Returns:
            OpenCV capture API backend.
        """
        return self._backend

    def get_device_id(self) -> int:
        """Get the camera device ID.

        Returns:
            Camera device ID.
        """
        return self._device_id
