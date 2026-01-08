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

    def __init__(self, name: str, resolution: tuple[int, int]) -> None:
        """Initialize a projector.

        Args:
            name: Unique name for the projector.
            resolution: Tuple of (width, height) for the projector resolution.
        """
        self.name = name
        self.resolution = resolution
        self.dialog: QtWidgets.QDialog | None = None

    def to_dict(self) -> dict:
        """Serialize projector to dictionary.

        Returns:
            Dictionary containing projector data.
        """
        return {
            'name': self.name,
            'resolution': self.resolution
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
            resolution=tuple(data['resolution'])
        )
