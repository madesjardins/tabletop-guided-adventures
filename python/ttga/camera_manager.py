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

"""Camera manager module for managing multiple cameras.

This module contains the CameraManager class which manages all active cameras
and provides camera enumeration functionality.
"""

from __future__ import annotations

from cv2_enumerate_cameras import enumerate_cameras
from PySide6 import QtCore

from .camera import Camera


def enumerate_available_cameras(
    backend: int,
    used_device_ids: set[int] | None = None
) -> list[dict]:
    """Enumerate available cameras for a given backend.

    Args:
        backend: OpenCV capture API backend.
        used_device_ids: Set of device IDs already in use (optional).

    Returns:
        List of camera info dictionaries with 'index', 'name', and 'path' keys.
    """
    if used_device_ids is None:
        used_device_ids = set()

    # Get all cameras for this backend
    all_cameras = enumerate_cameras(backend)

    # Filter out cameras that are already in use and convert to dict
    available_cameras = []
    for cam in all_cameras:
        if cam.index not in used_device_ids:
            available_cameras.append({
                'index': cam.index,
                'name': cam.name,
                'path': cam.path
            })

    return available_cameras


class CameraManager(QtCore.QObject):
    """Manager for all active cameras.

    This class manages the lifecycle of all cameras in the application,
    including adding, removing, and accessing cameras by name.

    Signals:
        camera_added: Emitted when a camera is added (camera_name: str).
        camera_removed: Emitted when a camera is removed (camera_name: str).
    """

    camera_added = QtCore.Signal(str)
    camera_removed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """Initialize the camera manager.

        Args:
            parent: Parent QObject.
        """
        super().__init__(parent)

        self._cameras: dict[str, Camera] = {}

    def add_camera(self, name: str, backend: int, device_id: int) -> Camera:
        """Add a new camera.

        Args:
            name: Camera name (must be unique).
            backend: OpenCV capture API backend.
            device_id: Camera device ID.

        Returns:
            Created Camera instance.

        Raises:
            ValueError: If camera name already exists.
        """
        if name in self._cameras:
            raise ValueError(f"Camera with name '{name}' already exists")

        # Create camera
        camera = Camera(name, backend, device_id, parent=self)

        # Store camera
        self._cameras[name] = camera

        # Emit signal
        self.camera_added.emit(name)

        return camera

    def remove_camera(self, name: str) -> None:
        """Remove a camera.

        Args:
            name: Camera name to remove.

        Raises:
            KeyError: If camera name doesn't exist.
        """
        if name not in self._cameras:
            raise KeyError(f"Camera '{name}' not found")

        # Get camera
        camera = self._cameras[name]

        # Release camera resources
        camera.release()

        # Remove from dictionary
        del self._cameras[name]

        # Emit signal
        self.camera_removed.emit(name)

    def get_camera(self, name: str) -> Camera:
        """Get a camera by name.

        Args:
            name: Camera name.

        Returns:
            Camera instance.

        Raises:
            KeyError: If camera name doesn't exist.
        """
        return self._cameras[name]

    def get_camera_names(self) -> list[str]:
        """Get all camera names.

        Returns:
            List of camera names.
        """
        return list(self._cameras.keys())

    def has_camera(self, name: str) -> bool:
        """Check if a camera exists.

        Args:
            name: Camera name.

        Returns:
            True if camera exists.
        """
        return name in self._cameras

    def get_used_device_ids(self, backend: int) -> set[int]:
        """Get device IDs currently in use for a given backend.

        Args:
            backend: OpenCV capture API backend.

        Returns:
            Set of device IDs in use for this backend.
        """
        used_ids = set()
        for camera in self._cameras.values():
            if camera.get_backend() == backend:
                used_ids.add(camera.get_device_id())
        return used_ids

    def release_all(self) -> None:
        """Release all cameras."""
        for camera in list(self._cameras.values()):
            camera.release()
        self._cameras.clear()
