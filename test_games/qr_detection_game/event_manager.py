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

"""Event manager for QR Detection game."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6 import QtCore

from ttga.game_event_manager import GameEventManager

if TYPE_CHECKING:
    from ttga.qr_detection import QRDetection


class QRDetectionEventManager(GameEventManager):
    """Event manager for QR Detection game.

    Handles QR code detections by drawing circles on camera and projector overlays.
    """

    def __init__(self, game, parent: QtCore.QObject | None = None) -> None:
        """Initialize the QR detection event manager.

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
            print("[QRDetectionEventManager] No game instance")
            return

        # Get zone
        zone = self.game.core.zone_manager.get_zone(zone_name)
        if not zone:
            print(f"[QRDetectionEventManager] Zone '{zone_name}' not found in zone_manager")
            return

        # Check if zone has camera mapping
        has_camera = zone.camera_mapping and zone.camera_mapping.is_calibrated
        has_projector = zone.projector_mapping and zone.projector_mapping.is_calibrated

        if not has_camera:
            print(f"[QRDetectionEventManager] Zone '{zone_name}' camera mapping not calibrated")
            return

        # Get camera overlay
        camera_overlay = self.game.camera_overlays.get(zone_name)
        if camera_overlay is None:
            print(f"[QRDetectionEventManager] Zone '{zone_name}' not in camera_overlays")
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

                    # Draw circle on projector overlay if available (white)
                    if projector_overlay is not None:
                        cv2.circle(projector_overlay, center_px, radius_px, (255, 255, 255, 255), 2)

                except Exception as e:
                    print(f"[QRDetectionEventManager] Error converting QR detection to game coordinates: {e}")

    @QtCore.Slot(str)
    def process_game_speech(self, text: str) -> None:
        """Process speech recognition results.

        Args:
            text: Recognized speech text.
        """
        print(f"[QRDetectionEventManager] Received speech: {text}")
