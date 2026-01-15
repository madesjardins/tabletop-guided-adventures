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

"""Calibration Precision game for testing camera and projector calibration accuracy.

This game draws a white cross on the play area to help verify calibration precision.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import yaml
from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from ttga.main_core import MainCore

from ttga.game_base import GameBase
from ttga.game_dialog import GameDialog, ZoneRequirement

from .event_manager import CalibrationPrecisionEventManager


class CalibrationPrecisionDialog(GameDialog):
    """Custom dialog for Calibration Precision game."""

    # Signals for cross parameter changes
    cross_position_changed = QtCore.Signal(float, float)
    cross_length_changed = QtCore.Signal(float)
    cross_thickness_changed = QtCore.Signal(float)

    def __init__(
        self,
        core: MainCore,
        game_name: str,
        zone_requirements: list[ZoneRequirement],
        settings: dict,
        game_instance,
        parent=None
    ):
        """Initialize the calibration precision dialog.

        Args:
            core: MainCore instance.
            game_name: Name of the game.
            zone_requirements: List of zone requirements.
            settings: Game settings dictionary.
            game_instance: Reference to the Game instance.
            parent: Parent widget.
        """
        self.settings = settings
        self.game_instance = game_instance

        # Store zone dimensions for range limits
        self.zone_width = 0.0
        self.zone_height = 0.0

        super().__init__(core, game_name, zone_requirements, parent)

    def _create_main_tab(self) -> QtWidgets.QWidget:
        """Create the main tab with Start/Stop game buttons and cross controls.

        Returns:
            Widget containing main game controls.
        """
        print("[CalibrationPrecision] Creating main tab UI")
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Instructions
        instructions = QtWidgets.QLabel(
            "Start the game to display a calibration cross on the play area. "
            "Adjust the cross position, length, and thickness to test calibration precision."
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

        layout.addSpacing(20)

        # Create cross controls group
        controls_group = QtWidgets.QGroupBox("Cross Controls")
        controls_layout = QtWidgets.QFormLayout()

        # Cross position X
        self.cross_x_spinbox = QtWidgets.QDoubleSpinBox()
        self.cross_x_spinbox.setDecimals(4)
        self.cross_x_spinbox.setSingleStep(0.1)
        self.cross_x_spinbox.setEnabled(False)
        self.cross_x_spinbox.valueChanged.connect(self._on_cross_position_changed)
        controls_layout.addRow("Cross X (inches):", self.cross_x_spinbox)

        # Cross position Y
        self.cross_y_spinbox = QtWidgets.QDoubleSpinBox()
        self.cross_y_spinbox.setDecimals(4)
        self.cross_y_spinbox.setSingleStep(0.1)
        self.cross_y_spinbox.setEnabled(False)
        self.cross_y_spinbox.valueChanged.connect(self._on_cross_position_changed)
        controls_layout.addRow("Cross Y (inches):", self.cross_y_spinbox)

        # Cross length
        self.cross_length_spinbox = QtWidgets.QDoubleSpinBox()
        self.cross_length_spinbox.setDecimals(4)
        self.cross_length_spinbox.setMinimum(0.0)
        self.cross_length_spinbox.setMaximum(5.0)
        self.cross_length_spinbox.setValue(2.0)
        self.cross_length_spinbox.setSingleStep(0.1)
        self.cross_length_spinbox.setEnabled(False)
        self.cross_length_spinbox.valueChanged.connect(
            lambda v: self.cross_length_changed.emit(v)
        )
        controls_layout.addRow("Cross Length (inches):", self.cross_length_spinbox)

        # Cross thickness
        self.cross_thickness_spinbox = QtWidgets.QDoubleSpinBox()
        self.cross_thickness_spinbox.setDecimals(4)
        self.cross_thickness_spinbox.setMinimum(0.0)
        self.cross_thickness_spinbox.setMaximum(1.0)
        self.cross_thickness_spinbox.setValue(0.1)
        self.cross_thickness_spinbox.setSingleStep(0.01)
        self.cross_thickness_spinbox.setEnabled(False)
        self.cross_thickness_spinbox.valueChanged.connect(
            lambda v: self.cross_thickness_changed.emit(v)
        )
        controls_layout.addRow("Cross Thickness (inches):", self.cross_thickness_spinbox)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

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
                "Validation Required",
                "Please validate zones before starting the game."
            )
            return

        # Get zone mapping
        zone_mapping = self.get_zone_mapping()
        if not zone_mapping:
            QtWidgets.QMessageBox.warning(
                self,
                "No Zones",
                "Please map at least one zone before starting the game."
            )
            return

        # Start the game
        if self.game_instance.start_game(zone_mapping):
            self.game_status_label.setText("Game Status: Running")
            self.game_status_label.setStyleSheet(
                "QLabel { font-weight: bold; padding: 10px; background-color: #90EE90; }"
            )
            self.start_game_button.setEnabled(False)
            self.stop_game_button.setEnabled(True)

            # Enable cross controls
            self.cross_x_spinbox.setEnabled(True)
            self.cross_y_spinbox.setEnabled(True)
            self.cross_length_spinbox.setEnabled(True)
            self.cross_thickness_spinbox.setEnabled(True)
        else:
            QtWidgets.QMessageBox.critical(
                self,
                "Start Failed",
                "Failed to start the game. Check the console for errors."
            )

    def _on_stop_game(self) -> None:
        """Handle stop game button click."""
        self.game_instance.stop_game()
        self.game_status_label.setText("Game Status: Stopped")
        self.game_status_label.setStyleSheet(
            "QLabel { font-weight: bold; padding: 10px; background-color: #FFB6C1; }"
        )
        self.start_game_button.setEnabled(True)
        self.stop_game_button.setEnabled(False)

        # Disable cross controls
        self.cross_x_spinbox.setEnabled(False)
        self.cross_y_spinbox.setEnabled(False)
        self.cross_length_spinbox.setEnabled(False)
        self.cross_thickness_spinbox.setEnabled(False)

    def set_zone_dimensions(self, width: float, height: float) -> None:
        """Set the zone dimensions to update spinbox ranges.

        Args:
            width: Zone width in inches.
            height: Zone height in inches.
        """
        self.zone_width = width
        self.zone_height = height

        # Update spinbox ranges
        self.cross_x_spinbox.setMinimum(0.0)
        self.cross_x_spinbox.setMaximum(width)
        self.cross_x_spinbox.setValue(width / 2.0)

        self.cross_y_spinbox.setMinimum(0.0)
        self.cross_y_spinbox.setMaximum(height)
        self.cross_y_spinbox.setValue(height / 2.0)

    def _on_cross_position_changed(self) -> None:
        """Handle cross position change."""
        x = self.cross_x_spinbox.value()
        y = self.cross_y_spinbox.value()
        self.cross_position_changed.emit(x, y)


class Game(GameBase):
    """Calibration Precision game for testing calibration accuracy.

    This game draws a white cross on the play area to help verify that
    camera and projector calibrations are accurate.
    """

    def __init__(self, core: MainCore):
        """Initialize the calibration precision game.

        Args:
            core: MainCore instance.
        """
        super().__init__(core)

        # Load configuration
        config_path = Path(__file__).parent / "game.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Build zone requirements from config
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

        self.dialog: Optional[CalibrationPrecisionDialog] = None
        self.event_manager: Optional[CalibrationPrecisionEventManager] = None
        self.is_running = False

        # Cross parameters (in inches)
        self.cross_x = 0.0
        self.cross_y = 0.0
        self.cross_length = 2.0
        self.cross_thickness = 0.1

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
            'name': self.config.get('name', 'Unknown'),
            'version': self.config.get('version', '0.0.0'),
            'author': self.config.get('author', 'Unknown'),
            'description': self.config.get('description', '')
        }

    def on_load(self) -> None:
        """Called when the game is loaded."""
        print("[Calibration Precision] Game loaded")

    def on_unload(self) -> None:
        """Called when the game is unloaded."""
        if self.is_running:
            self.stop_game()
        print("[Calibration Precision] Game unloaded")

    def show_dialog(self, parent=None) -> None:
        """Show the Calibration Precision configuration dialog.

        Args:
            parent: Parent widget for the dialog.
        """
        if self.dialog is None:
            metadata = self.get_metadata()
            self.dialog = CalibrationPrecisionDialog(
                self.core,
                metadata['name'],
                self.zone_requirements,
                {},
                self,
                parent
            )

            # Connect dialog signals
            self.dialog.cross_position_changed.connect(self._on_cross_position_changed)
            self.dialog.cross_length_changed.connect(self._on_cross_length_changed)
            self.dialog.cross_thickness_changed.connect(self._on_cross_thickness_changed)

        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def start_game(self, zone_mapping: dict[str, str]) -> bool:
        """Start the game with the given zone mapping.

        Args:
            zone_mapping: Dictionary mapping internal zone names to actual zone names.

        Returns:
            True if game started successfully, False otherwise.
        """
        print("[Calibration Precision] Starting game...")
        self.zone_mapping = zone_mapping

        # Get the play area zone
        play_area_name = zone_mapping.get('play_area')
        if not play_area_name:
            print("[Calibration Precision] No play area zone mapped")
            return False

        zone = self.core.zone_manager.get_zone(play_area_name)
        if not zone:
            print(f"[Calibration Precision] Zone '{play_area_name}' not found")
            return False

        # Update dialog with zone dimensions
        if self.dialog:
            self.dialog.set_zone_dimensions(zone.width, zone.height)
            # Initialize cross position to center
            self.cross_x = zone.width / 2.0
            self.cross_y = zone.height / 2.0

        # Initialize overlays for each zone
        width_px = int(zone.width * zone.resolution)
        height_px = int(zone.height * zone.resolution)

        # Create camera overlay if zone has camera mapping
        if zone.camera_mapping and zone.camera_mapping.enabled:
            camera_overlay = np.zeros((height_px, width_px, 4), dtype=np.uint8)
            self.camera_overlays[play_area_name] = camera_overlay
            print(f"[Calibration Precision] Created camera overlay for zone '{play_area_name}' ({width_px}x{height_px})")

        # Create projector overlay if zone has projector mapping
        if zone.projector_mapping and zone.projector_mapping.enabled:
            projector_overlay = np.zeros((height_px, width_px, 4), dtype=np.uint8)
            self.projector_overlays[play_area_name] = projector_overlay
            print(f"[Calibration Precision] Created projector overlay for zone '{play_area_name}' ({width_px}x{height_px})")

        # Create event manager
        self.event_manager = CalibrationPrecisionEventManager(self)

        # Connect speech recognition
        self.core.speech_final_result.connect(self.event_manager.process_game_speech)

        self.is_running = True

        # Draw initial cross
        self._update_cross()

        print("[Calibration Precision] Game started successfully")
        return True

    def stop_game(self) -> None:
        """Stop the game and clean up resources."""
        if not self.is_running:
            return

        print("[Calibration Precision] Stopping game...")

        # Disconnect speech recognition
        if self.event_manager:
            try:
                self.core.speech_final_result.disconnect(self.event_manager.process_game_speech)
            except RuntimeError:
                pass
            self.event_manager = None

        # Clear overlays
        self.camera_overlays.clear()
        self.projector_overlays.clear()
        self.zone_mapping.clear()

        self.is_running = False
        print("[Calibration Precision] Game stopped")

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

    @QtCore.Slot(float, float)
    def _on_cross_position_changed(self, x: float, y: float) -> None:
        """Handle cross position change.

        Args:
            x: X position in inches.
            y: Y position in inches.
        """
        self.cross_x = x
        self.cross_y = y
        self._update_cross()

    @QtCore.Slot(float)
    def _on_cross_length_changed(self, length: float) -> None:
        """Handle cross length change.

        Args:
            length: Cross length in inches.
        """
        self.cross_length = length
        self._update_cross()

    @QtCore.Slot(float)
    def _on_cross_thickness_changed(self, thickness: float) -> None:
        """Handle cross thickness change.

        Args:
            thickness: Cross thickness in inches.
        """
        self.cross_thickness = thickness
        self._update_cross()

    def _update_cross(self) -> None:
        """Update the cross drawing on all overlays."""
        if not self.is_running:
            return

        play_area_name = self.zone_mapping.get('play_area')
        if not play_area_name:
            return

        zone = self.core.zone_manager.get_zone(play_area_name)
        if not zone:
            return

        # Get overlays
        camera_overlay = self.camera_overlays.get(play_area_name)
        projector_overlay = self.projector_overlays.get(play_area_name)

        # Clear overlays
        if camera_overlay is not None:
            camera_overlay[:] = 0
        if projector_overlay is not None:
            projector_overlay[:] = 0

        # Convert measurements to pixels
        center_x_px = int(self.cross_x * zone.resolution)
        center_y_px = int(self.cross_y * zone.resolution)
        half_length_px = int(self.cross_length / 2.0 * zone.resolution)
        thickness_px = max(1, int(self.cross_thickness * zone.resolution))

        # Draw horizontal line
        x1 = center_x_px - half_length_px
        x2 = center_x_px + half_length_px
        y = center_y_px

        # Draw vertical line
        y1 = center_y_px - half_length_px
        y2 = center_y_px + half_length_px
        x = center_x_px

        # White color in BGRA format
        color = (255, 255, 255, 255)

        # Draw on camera overlay (green)
        if camera_overlay is not None:
            cv2.line(camera_overlay, (x1, y), (x2, y), (0, 255, 0, 255), thickness_px)
            cv2.line(camera_overlay, (x, y1), (x, y2), (0, 255, 0, 255), thickness_px)

        # Draw on projector overlay (white)
        if projector_overlay is not None:
            cv2.line(projector_overlay, (x1, y), (x2, y), color, thickness_px)
            cv2.line(projector_overlay, (x, y1), (x, y2), color, thickness_px)
