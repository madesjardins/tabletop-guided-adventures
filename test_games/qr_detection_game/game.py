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

"""QR Detection game demonstrating zone configuration and validation.

This game shows how to:
- Load configuration from YAML
- Define zone requirements
- Create a custom game dialog
- Validate zone mappings
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import yaml
from PySide6 import QtWidgets

if TYPE_CHECKING:
    from ttga.main_core import MainCore

from ttga.game_base import GameBase
from ttga.game_dialog import GameDialog, ZoneRequirement
from ttga.game_event_manager import GameEventManager
from ttga.qr_detection import QRDetector
from ttga.sound_mixer import Channel


class QRDetectionDialog(GameDialog):
    """Custom dialog for QR Detection game."""

    def __init__(
        self,
        core: MainCore,
        game_name: str,
        zone_requirements: list[ZoneRequirement],
        settings: dict,
        game_instance,
        parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        """Initialize the QR Detection dialog.

        Args:
            core: MainCore instance.
            game_name: Name of the game.
            zone_requirements: List of zone requirements.
            settings: Game settings from YAML.
            game_instance: Reference to the Game instance.
            parent: Parent widget.
        """
        self.settings = settings
        self.game_instance = game_instance
        super().__init__(core, game_name, zone_requirements, parent)

    def _create_main_tab(self) -> QtWidgets.QWidget:
        """Create the main tab with Start/Stop game buttons.

        Returns:
            Widget containing main game controls.
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Instructions
        instructions = QtWidgets.QLabel(
            "Start the game to begin QR code detection. "
            "Make sure zones are validated before starting."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        layout.addSpacing(20)

        # Game status
        self.game_status_label = QtWidgets.QLabel("Game Status: Not Started")
        self.game_status_label.setStyleSheet(
            "QLabel { font-weight: bold; padding: 10px; background-color: #f0f0f0; }"
        )
        layout.addWidget(self.game_status_label)

        layout.addSpacing(20)

        # Start/Stop buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.start_game_button = QtWidgets.QPushButton("Start Game")
        self.start_game_button.clicked.connect(self._on_start_game)
        button_layout.addWidget(self.start_game_button)

        self.stop_game_button = QtWidgets.QPushButton("Stop Game")
        self.stop_game_button.clicked.connect(self._on_stop_game)
        self.stop_game_button.setEnabled(False)
        button_layout.addWidget(self.stop_game_button)

        layout.addLayout(button_layout)

        layout.addStretch()

        return widget

    def _on_start_game(self) -> None:
        """Handle start game button click."""
        # Validate zones first
        self._on_validate_zones()

        # Check if validation passed
        if not self.is_validated():
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Failed",
                "Please fix zone validation errors before starting the game."
            )
            return

        # Get zone mapping
        zone_mapping = self.get_zone_mapping()

        # Start the game
        success = self.game_instance.start_game(zone_mapping)

        if success:
            self.game_status_label.setText("Game Status: Running")
            self.game_status_label.setStyleSheet(
                "QLabel { font-weight: bold; padding: 10px; background-color: #ccffcc; color: #008800; }"
            )
            self.start_game_button.setEnabled(False)
            self.stop_game_button.setEnabled(True)
        else:
            QtWidgets.QMessageBox.critical(
                self,
                "Game Start Failed",
                "Failed to start the game. Check console for errors."
            )

    def _on_stop_game(self) -> None:
        """Handle stop game button click."""
        self.game_instance.stop_game()

        self.game_status_label.setText("Game Status: Stopped")
        self.game_status_label.setStyleSheet(
            "QLabel { font-weight: bold; padding: 10px; background-color: #ffcccc; color: #cc0000; }"
        )
        self.start_game_button.setEnabled(True)
        self.stop_game_button.setEnabled(False)


class Game(GameBase):
    """QR Detection game with zone configuration and validation.

    This game demonstrates:
    - Loading configuration from YAML
    - Zone requirements with camera/projector mapping
    - Custom game dialog with settings
    - Zone validation with helpful error messages
    """

    # Class-level metadata for quick discovery
    GAME_NAME = "QR Detection"
    GAME_VERSION = "1.0.0"
    GAME_AUTHOR = "TTGA Team"
    GAME_DESCRIPTION = "A game that detects QR codes in a play area and tracks their positions"

    def __init__(self, core: MainCore) -> None:
        """Initialize the QR Detection game.

        Args:
            core: MainCore instance.
        """
        super().__init__(core)

        # Load configuration from YAML
        config_path = Path(__file__).parent / "game.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Parse zone requirements
        self.zone_requirements = []
        for zone_config in self.config.get('zones', []):
            req = ZoneRequirement(
                internal_name=zone_config['internal_name'],
                display_name=zone_config['display_name'],
                requires_camera=zone_config.get('requires_camera', False),
                requires_projector=zone_config.get('requires_projector', False),
                units=zone_config.get('units')
            )
            self.zone_requirements.append(req)

        self.dialog: Optional[QRDetectionDialog] = None
        self.event_manager: Optional[GameEventManager] = None
        self.qr_detectors: dict[str, QRDetector] = {}
        self.is_running = False

        # Overlay images for visualization (zone_name -> BGRA image)
        self.camera_overlays: dict[str, np.ndarray] = {}
        self.projector_overlays: dict[str, np.ndarray] = {}
        self.zone_mapping: dict[str, str] = {}  # internal_name -> actual zone_name

    def get_metadata(self) -> dict[str, str]:
        """Get game metadata from YAML configuration.

        Returns:
            Dictionary with game information.
        """
        return {
            'name': self.config.get('name', 'QR Detection'),
            'version': self.config.get('version', '1.0.0'),
            'author': self.config.get('author', 'Unknown'),
            'description': self.config.get('description', '')
        }

    def on_load(self) -> None:
        """Called when the game is loaded."""
        self.core.narrator.synthesize_and_play(
            "QR Detection game loaded. Use Game menu to show the configuration dialog.",
            channel=Channel.VOICE
        )

    def on_unload(self) -> None:
        """Called when the game is unloaded."""
        # Stop game if running
        if self.is_running:
            self.stop_game()

        # Close dialog if open
        if self.dialog:
            self.dialog.close()
            self.dialog = None

        self.core.narrator.synthesize_and_play(
            "QR Detection game unloaded.",
            channel=Channel.VOICE
        )

    def show_dialog(self, parent=None) -> None:
        """Show the QR Detection configuration dialog.

        Args:
            parent: Parent widget for the dialog.
        """
        if self.dialog is None:
            self.dialog = QRDetectionDialog(
                self.core,
                self.config.get('name', 'QR Detection'),
                self.zone_requirements,
                self.config.get('settings', {}),
                self,
                parent
            )

        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def start_game(self, zone_mapping: dict[str, str]) -> bool:
        """Start the game with validated zones.

        Args:
            zone_mapping: Dictionary mapping internal zone names to actual zone names.

        Returns:
            True if game started successfully, False otherwise.
        """
        if self.is_running:
            print("[QR Detection] Game is already running")
            return False

        print("[QR Detection] Starting game...")

        # Store zone mapping
        self.zone_mapping = zone_mapping.copy()

        # Initialize transparent overlay images for each zone
        for internal_name, zone_name in zone_mapping.items():
            if zone_name:
                zone = self.core.zone_manager.get_zone(zone_name)
                if zone:
                    # Create transparent BGRA image with zone dimensions
                    width_px = int(zone.width * zone.resolution)
                    height_px = int(zone.height * zone.resolution)

                    # Create camera overlay if zone has camera mapping
                    if zone.camera_mapping and zone.camera_mapping.enabled:
                        camera_overlay = np.zeros((height_px, width_px, 4), dtype=np.uint8)
                        self.camera_overlays[zone_name] = camera_overlay
                        print(f"[QR Detection] Created camera overlay for zone '{zone_name}' ({width_px}x{height_px})")

                    # Create projector overlay if zone has projector mapping
                    if zone.projector_mapping and zone.projector_mapping.enabled:
                        projector_overlay = np.zeros((height_px, width_px, 4), dtype=np.uint8)
                        self.projector_overlays[zone_name] = projector_overlay
                        print(f"[QR Detection] Created projector overlay for zone '{zone_name}' ({width_px}x{height_px})")

        # Create event manager with reference to game for overlay updates
        self.event_manager = GameEventManager(self)

        # Connect speech recognition
        self.core.speech_final_result.connect(self.event_manager.process_game_speech)

        # Create QR detectors for zones with enable_qr_detector
        for zone_config in self.config.get('zones', []):
            if zone_config.get('enable_qr_detector', False):
                internal_name = zone_config['internal_name']
                zone_name = zone_mapping.get(internal_name)

                if zone_name:
                    zone = self.core.zone_manager.get_zone(zone_name)
                    if zone:
                        # Create QR detector with refresh rate from MainCore
                        detector = QRDetector(zone, self.core.camera_manager, self.core.qr_code_refresh_rate)
                        # Pass zone_name to event manager via lambda
                        detector.detections_updated.connect(
                            lambda dets, zn=zone_name: self.event_manager.process_game_detection(dets, zn)
                        )
                        detector.start()
                        self.qr_detectors[internal_name] = detector
                        print(f"[QR Detection] Started QR detector for zone '{zone_name}' at {self.core.qr_code_refresh_rate} Hz")

        self.is_running = True
        print("[QR Detection] Game started successfully")
        return True

    def stop_game(self) -> None:
        """Stop the game and clean up resources."""
        if not self.is_running:
            return

        print("[QR Detection] Stopping game...")

        # Stop and disconnect QR detectors
        for detector in self.qr_detectors.values():
            detector.stop()
            if self.event_manager:
                detector.detections_updated.disconnect(self.event_manager.process_game_detection)

        self.qr_detectors.clear()

        # Disconnect speech recognition
        if self.event_manager:
            self.core.speech_final_result.disconnect(self.event_manager.process_game_speech)
            self.event_manager = None

        # Clear overlays
        self.camera_overlays.clear()
        self.projector_overlays.clear()
        self.zone_mapping.clear()

        self.is_running = False
        print("[QR Detection] Game stopped")

    def set_qr_detectors_refresh_rate(self, fps: int) -> None:
        """Set the refresh rate for all QR detectors.

        Args:
            fps: New refresh rate in frames per second.
        """
        for detector in self.qr_detectors.values():
            detector.set_refresh_rate(fps)
        print(f"[QR Detection] Updated QR detector refresh rate to {fps} Hz")

    def get_camera_overlay(self, zone_name: str):
        """Get the camera overlay image for a specific zone.

        Args:
            zone_name: Name of the zone to get overlay for.

        Returns:
            numpy.ndarray with shape (height, width, 4) in BGRA format with game coordinates,
            or None if no overlay for this zone.
        """
        return self.camera_overlays.get(zone_name)

    def get_projector_overlay(self, zone_name: str):
        """Get the projector overlay image for a specific zone.

        Args:
            zone_name: Name of the zone to get overlay for.

        Returns:
            numpy.ndarray with shape (height, width, 4) in BGRA format with game coordinates,
            or None if no overlay for this zone.
        """
        return self.projector_overlays.get(zone_name)
