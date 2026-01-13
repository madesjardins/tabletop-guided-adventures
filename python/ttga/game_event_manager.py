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

"""Abstract base class for game event managers.

This module provides the GameEventManager abstract base class that games
must subclass to handle game-specific events like QR detections and speech commands.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

from PySide6 import QtCore

if TYPE_CHECKING:
    from .qr_detection import QRDetection


class QObjectABCMeta(type(QtCore.QObject), ABCMeta):
    """Combined metaclass for QObject and ABC to avoid metaclass conflicts."""
    pass


class GameEventManager(QtCore.QObject, metaclass=QObjectABCMeta):
    """Abstract base class for processing game events.

    Games should subclass this to implement custom event handling for
    QR code detections and speech recognition results.

    Example:
        >>> class MyGameEventManager(GameEventManager):
        ...     def process_game_detection(self, detections, zone_name):
        ...         # Custom QR detection handling
        ...         pass
        ...     def process_game_speech(self, text):
        ...         # Custom speech handling
        ...         pass
    """

    @abstractmethod
    @QtCore.Slot(list, str)
    def process_game_detection(self, detections: list[QRDetection], zone_name: str) -> None:
        """Process QR code detections.

        Args:
            detections: List of QRDetection objects in camera ROI coordinates.
            zone_name: Name of the zone where detections occurred.
        """
        pass

    @abstractmethod
    @QtCore.Slot(str)
    def process_game_speech(self, text: str) -> None:
        """Process speech recognition results.

        Args:
            text: Recognized speech text.
        """
        pass
