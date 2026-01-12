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

"""Game event manager for handling game-related events.

This module provides the GameEventManager class that games can use to
process events like QR detections and speech commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PySide6 import QtCore

if TYPE_CHECKING:
    from .qr_detection import QRDetection


class GameEventManager(QtCore.QObject):
    """Manager for processing game events.

    This class provides methods for handling various game events such as
    QR code detections and speech recognition results. Games can subclass
    this or use it directly to process events.

    Example:
        >>> event_manager = GameEventManager(game_instance)
        >>> qr_detector.detections_updated.connect(
        ...     lambda dets: event_manager.process_game_detection(dets, zone_name)
        ... )
        >>> speech_recognizer.final_result.connect(event_manager.process_game_speech)
    """

    def __init__(self, game: Any = None, parent: QtCore.QObject | None = None) -> None:
        """Initialize the game event manager.

        Args:
            game: Game instance for accessing overlays and zone data.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self.game = game

    @QtCore.Slot(list, str)
    def process_game_detection(self, detections: list[QRDetection], zone_name: str) -> None:
        """Process QR code detections and update overlay visualization.

        Args:
            detections: List of QRDetection objects in camera ROI coordinates.
            zone_name: Name of the zone where detections occurred.
        """
        if not self.game:
            print("[GameEventManager] No game instance")
            return

        # Get zone
        zone = self.game.core.zone_manager.get_zone(zone_name)
        if not zone:
            print(f"[GameEventManager] Zone '{zone_name}' not found in zone_manager")
            return

        # Check if zone has camera mapping
        has_camera = zone.camera_mapping and zone.camera_mapping.is_calibrated
        has_projector = zone.projector_mapping and zone.projector_mapping.is_calibrated

        if not has_camera:
            print(f"[GameEventManager] Zone '{zone_name}' camera mapping not calibrated")
            return

        # Get camera overlay
        camera_overlay = self.game.camera_overlays.get(zone_name)
        if camera_overlay is None:
            print(f"[GameEventManager] Zone '{zone_name}' not in camera_overlays")
            return

        # Get projector overlay if available
        projector_overlay = self.game.projector_overlays.get(zone_name) if has_projector else None

        # Clear overlays (make transparent)
        camera_overlay[:] = 0
        if projector_overlay is not None:
            projector_overlay[:] = 0

        if detections:
            # Calculate circle radius in pixels (0.6 inches * pixels per unit)
            radius_px = int(0.6 * zone.resolution)

            for detection in detections:
                # Calculate center from corners (in camera ROI coordinates)
                corners = np.array(detection.corners, dtype=np.float32)
                center_roi = np.mean(corners, axis=0)

                # Convert from camera ROI coordinates to game coordinates (in pixels)
                try:
                    center_game_px = zone.camera_to_game((center_roi[0], center_roi[1]))

                    # Use game pixel coordinates directly for overlay (already in pixels)
                    center_px = (
                        int(center_game_px[0]),
                        int(center_game_px[1])
                    )

                    # Draw circle on camera overlay (BGRA format: green with full alpha)
                    cv2.circle(camera_overlay, center_px, radius_px, (0, 255, 0, 255), 2)

                    # Draw circle on projector overlay if available
                    if projector_overlay is not None:
                        cv2.circle(projector_overlay, center_px, radius_px, (0, 255, 0, 255), 2)

                except Exception as e:
                    print(f"[GameEventManager] Error converting QR detection to game coordinates: {e}")

    @QtCore.Slot(str)
    def process_game_speech(self, text: str) -> None:
        """Process speech recognition results.

        Override this method in subclasses to implement custom speech handling.

        Args:
            text: Recognized speech text.

        Example:
            >>> def process_game_speech(self, text):
            ...     if "start" in text.lower():
            ...         self.start_game()
        """
        print(f"[GameEventManager] Received speech: {text}")
