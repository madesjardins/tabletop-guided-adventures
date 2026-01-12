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

"""Projector class for managing virtual projector displays."""

from PySide6 import QtWidgets


class Projector:
    """Represents a virtual projector with a name and resolution.

    Attributes:
        name: Unique name for the projector.
        resolution: Tuple of (width, height) for the projector resolution.
    """

    def __init__(self, name: str, resolution: tuple[int, int], fps: int = 60) -> None:
        """Initialize a projector.

        Args:
            name: Unique name for the projector.
            resolution: Tuple of (width, height) for the projector resolution.
            fps: Refresh rate in frames per second (default: 60).
        """
        self.name = name
        self.resolution = resolution
        self.fps = fps
        self.dialog: QtWidgets.QDialog | None = None

    def set_fps(self, fps: int) -> None:
        """Set the projector refresh rate.

        Args:
            fps: Refresh rate in frames per second.
        """
        self.fps = fps

    def to_dict(self) -> dict:
        """Serialize projector to dictionary.

        Returns:
            Dictionary containing projector data.
        """
        return {
            'name': self.name,
            'resolution': self.resolution,
            'fps': self.fps
        }

    @staticmethod
    def from_dict(data: dict) -> 'Projector':
        """Deserialize projector from dictionary.

        Args:
            data: Dictionary containing projector data.

        Returns:
            Projector instance.
        """
        return Projector(
            name=data['name'],
            resolution=tuple(data['resolution']),
            fps=data.get('fps', 60)  # Default to 60 for backward compatibility
        )
