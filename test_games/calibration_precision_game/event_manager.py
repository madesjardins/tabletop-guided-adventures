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

"""Event manager for Calibration Precision game."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore

from ttga.game_event_manager import GameEventManager

if TYPE_CHECKING:
    from ttga.qr_detection import QRDetection


class CalibrationPrecisionEventManager(GameEventManager):
    """Event manager for Calibration Precision game.

    This game doesn't use QR detection or speech, so these are no-op implementations.
    """

    def __init__(self, game, parent: QtCore.QObject | None = None) -> None:
        """Initialize the calibration precision event manager.

        Args:
            game: Game instance for accessing overlays and zone data.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self.game = game

    @QtCore.Slot(list, str)
    def process_game_detection(self, detections: list[QRDetection], zone_name: str) -> None:
        """Process QR code detections (not used in this game).

        Args:
            detections: List of QRDetection objects in camera ROI coordinates.
            zone_name: Name of the zone where detections occurred.
        """
        pass

    @QtCore.Slot(str)
    def process_game_speech(self, text: str) -> None:
        """Process speech recognition results (not used in this game).

        Args:
            text: Recognized speech text.
        """
        pass
