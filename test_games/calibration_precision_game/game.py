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

    # Signals for grid parameter changes
    grid_division_changed = QtCore.Signal(float)
    grid_thickness_changed = QtCore.Signal(float)
    grid_offset_changed = QtCore.Signal(float)

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
            "Start the game to display a calibration grid on the play area. "
            "Adjust the division size, line thickness, and offset to test calibration precision."
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

        # Create grid controls group
        controls_group = QtWidgets.QGroupBox("Grid Controls")
        controls_layout = QtWidgets.QFormLayout()

        # Division size
        self.division_size_spinbox = QtWidgets.QDoubleSpinBox()
        self.division_size_spinbox.setDecimals(4)
        self.division_size_spinbox.setMinimum(0.1)
        self.division_size_spinbox.setMaximum(100.0)
        self.division_size_spinbox.setValue(1.0)
        self.division_size_spinbox.setSingleStep(0.1)
        self.division_size_spinbox.setEnabled(False)
        self.division_size_spinbox.valueChanged.connect(
            lambda v: self.grid_division_changed.emit(v)
        )
        controls_layout.addRow("Division Size (inches):", self.division_size_spinbox)

        # Line thickness
        self.line_thickness_spinbox = QtWidgets.QDoubleSpinBox()
        self.line_thickness_spinbox.setDecimals(4)
        self.line_thickness_spinbox.setMinimum(0.0001)
        self.line_thickness_spinbox.setMaximum(1.0)
        self.line_thickness_spinbox.setValue(0.0625)
        self.line_thickness_spinbox.setSingleStep(0.0625)
        self.line_thickness_spinbox.setEnabled(False)
        self.line_thickness_spinbox.valueChanged.connect(
            lambda v: self.grid_thickness_changed.emit(v)
        )
        controls_layout.addRow("Line Thickness (inches):", self.line_thickness_spinbox)

        # Offset
        self.offset_spinbox = QtWidgets.QDoubleSpinBox()
        self.offset_spinbox.setDecimals(4)
        self.offset_spinbox.setMinimum(0.0)
        self.offset_spinbox.setMaximum(100.0)
        self.offset_spinbox.setValue(0.5)
        self.offset_spinbox.setSingleStep(0.1)
        self.offset_spinbox.setEnabled(False)
        self.offset_spinbox.valueChanged.connect(
            lambda v: self.grid_offset_changed.emit(v)
        )
        controls_layout.addRow("Offset (inches):", self.offset_spinbox)

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

            # Enable grid controls
            self.division_size_spinbox.setEnabled(True)
            self.line_thickness_spinbox.setEnabled(True)
            self.offset_spinbox.setEnabled(True)
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

        # Disable grid controls
        self.division_size_spinbox.setEnabled(False)
        self.line_thickness_spinbox.setEnabled(False)
        self.offset_spinbox.setEnabled(False)

    def set_zone_dimensions(self, width: float, height: float) -> None:
        """Set the zone dimensions to update spinbox ranges.

        Args:
            width: Zone width in inches.
            height: Zone height in inches.
        """
        self.zone_width = width
        self.zone_height = height


class Game(GameBase):
    """Calibration Precision game for testing calibration accuracy.

    This game draws a calibration grid on the play area to help verify that
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

        # Grid parameters (in inches)
        self.division_size = 1.0
        self.line_thickness = 0.0625
        self.offset = 0.5

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

        # Close dialog if open
        if self.dialog:
            self.dialog.close()
            self.dialog = None

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
            self.dialog.grid_division_changed.connect(self._on_division_size_changed)
            self.dialog.grid_thickness_changed.connect(self._on_line_thickness_changed)
            self.dialog.grid_offset_changed.connect(self._on_offset_changed)

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

        # Draw initial grid
        self._update_grid()

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

    @QtCore.Slot(float)
    def _on_division_size_changed(self, size: float) -> None:
        """Handle division size change.

        Args:
            size: Division size in inches.
        """
        self.division_size = size
        self._update_grid()

    @QtCore.Slot(float)
    def _on_line_thickness_changed(self, thickness: float) -> None:
        """Handle line thickness change.

        Args:
            thickness: Line thickness in inches.
        """
        self.line_thickness = thickness
        self._update_grid()

    @QtCore.Slot(float)
    def _on_offset_changed(self, offset: float) -> None:
        """Handle offset change.

        Args:
            offset: Offset in inches.
        """
        self.offset = offset
        self._update_grid()

    def _update_grid(self) -> None:
        """Update the grid drawing on all overlays."""
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
        division_px = int(self.division_size * zone.resolution)
        thickness_px = max(1, int(self.line_thickness * zone.resolution))
        offset_px = int(self.offset * zone.resolution)

        width_px = int(zone.width * zone.resolution)
        height_px = int(zone.height * zone.resolution)

        # White color in BGRA format for projector
        white_color = (255, 255, 255, 255)
        # Green color in BGRA format for camera
        green_color = (0, 255, 0, 255)

        # Draw vertical lines
        x = offset_px
        while x < width_px:
            if camera_overlay is not None:
                cv2.line(camera_overlay, (x, 0), (x, height_px - 1), green_color, thickness_px)
            if projector_overlay is not None:
                cv2.line(projector_overlay, (x, 0), (x, height_px - 1), white_color, thickness_px)
            x += division_px

        # Draw horizontal lines
        y = offset_px
        while y < height_px:
            if camera_overlay is not None:
                cv2.line(camera_overlay, (0, y), (width_px - 1, y), green_color, thickness_px)
            if projector_overlay is not None:
                cv2.line(projector_overlay, (0, y), (width_px - 1, y), white_color, thickness_px)
            y += division_px
