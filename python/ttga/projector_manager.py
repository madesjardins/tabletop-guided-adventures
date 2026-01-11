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

"""Projector manager for handling multiple projectors."""

from PySide6 import QtCore

from .projector import Projector


class ProjectorManager(QtCore.QObject):
    """Manager for handling multiple projectors.

    Signals:
        projector_added: Emitted when a projector is added (projector_name).
        projector_removed: Emitted when a projector is removed (projector_name).
    """

    projector_added = QtCore.Signal(str)
    projector_removed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """Initialize the projector manager.

        Args:
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._projectors: dict[str, Projector] = {}

    def add_projector(self, name: str, resolution: tuple[int, int], fps: int = 15) -> Projector:
        """Add a new projector.

        Args:
            name: Unique name for the projector.
            resolution: Tuple of (width, height) for the projector resolution.
            fps: Refresh rate in frames per second (default: 15).

        Returns:
            The created Projector instance.

        Raises:
            ValueError: If a projector with this name already exists.
        """
        if name in self._projectors:
            raise ValueError(f"Projector '{name}' already exists")

        projector = Projector(name, resolution)
        projector.set_fps(fps)
        self._projectors[name] = projector
        self.projector_added.emit(name)
        return projector

    def remove_projector(self, name: str) -> None:
        """Remove a projector.

        Args:
            name: Name of the projector to remove.

        Raises:
            KeyError: If projector doesn't exist.
        """
        if name not in self._projectors:
            raise KeyError(f"Projector '{name}' not found")

        projector = self._projectors[name]

        # Close dialog if open
        if projector.dialog is not None:
            projector.dialog.close()
            projector.dialog = None

        del self._projectors[name]
        self.projector_removed.emit(name)

    def get_projector(self, name: str) -> Projector:
        """Get a projector by name.

        Args:
            name: Name of the projector.

        Returns:
            The Projector instance.

        Raises:
            KeyError: If projector doesn't exist.
        """
        return self._projectors[name]

    def get_all_projectors(self) -> list[Projector]:
        """Get all projectors.

        Returns:
            List of all Projector instances.
        """
        return list(self._projectors.values())

    def projector_exists(self, name: str) -> bool:
        """Check if a projector exists.

        Args:
            name: Name to check.

        Returns:
            True if projector exists, False otherwise.
        """
        return name in self._projectors

    def serialize_projectors(self) -> list[dict]:
        """Serialize all projectors to a list of dictionaries.

        Returns:
            List of projector data dictionaries.
        """
        return [projector.to_dict() for projector in self._projectors.values()]

    def clear_all(self) -> None:
        """Remove all projectors."""
        projector_names = list(self._projectors.keys())
        for name in projector_names:
            self.remove_projector(name)
