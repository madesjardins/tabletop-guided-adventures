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

"""Base game dialog for game configuration and match management.

This module provides the GameDialog base class that all games can use to
create a configuration dialog with zone mapping, settings, and match management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6 import QtCore, QtWidgets

if TYPE_CHECKING:
    from .main_core import MainCore


class ZoneRequirement:
    """Represents a zone requirement for a game.

    Attributes:
        internal_name: Internal name used by the game to reference this zone.
        display_name: Display name shown to the user.
        requires_camera: Whether this zone requires camera mapping.
        requires_projector: Whether this zone requires projector mapping.
        units: Required units for the zone (e.g., 'mm', 'cm', 'inches').
    """

    def __init__(
        self,
        internal_name: str,
        display_name: str,
        requires_camera: bool = False,
        requires_projector: bool = False,
        units: Optional[str] = None
    ) -> None:
        """Initialize zone requirement.

        Args:
            internal_name: Internal name for the zone.
            display_name: Display name for the zone.
            requires_camera: Whether camera mapping is required.
            requires_projector: Whether projector mapping is required.
            units: Required units (None means any units are acceptable).
        """
        self.internal_name = internal_name
        self.display_name = display_name
        self.requires_camera = requires_camera
        self.requires_projector = requires_projector
        self.units = units


class GameDialog(QtWidgets.QDialog):
    """Base dialog for game configuration and match management.

    This dialog provides:
    - Zones tab: Map game zones to application zones with validation
    - Settings tab: Game-specific settings (override in subclass)
    - Match management: Start/stop matches

    Subclasses should:
    1. Call super().__init__() with zone requirements
    2. Override _create_settings_tab() to add custom settings
    3. Override _on_start_match() to handle match start
    """

    def __init__(
        self,
        core: MainCore,
        game_name: str,
        zone_requirements: list[ZoneRequirement],
        parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        """Initialize the game dialog.

        Args:
            core: MainCore instance.
            game_name: Name of the game.
            zone_requirements: List of required zones for the game.
            parent: Parent widget.
        """
        super().__init__(parent)

        self.core = core
        self.game_name = game_name
        self.zone_requirements = zone_requirements

        # Zone mapping: internal_name -> actual zone name
        self.zone_mapping: dict[str, Optional[str]] = {
            req.internal_name: None for req in zone_requirements
        }

        # UI elements
        self.zone_combos: dict[str, QtWidgets.QComboBox] = {}
        self.validate_button: Optional[QtWidgets.QPushButton] = None
        self.validation_label: Optional[QtWidgets.QLabel] = None
        self.start_match_button: Optional[QtWidgets.QPushButton] = None

        self._setup_ui()

        # Connect zone manager signals to update combos
        self.core.zone_manager.zone_added.connect(self._refresh_zone_combos)
        self.core.zone_manager.zone_removed.connect(self._refresh_zone_combos)

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle(f"{self.game_name} - Configuration")
        self.setMinimumSize(700, 500)

        # Make dialog non-closable
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowType.WindowCloseButtonHint
        )

        layout = QtWidgets.QVBoxLayout(self)

        # Tab widget
        tabs = QtWidgets.QTabWidget()

        # Main tab (can be overridden by subclass)
        main_tab = self._create_main_tab()
        if main_tab:
            tabs.addTab(main_tab, "Main")

        # Zones tab
        zones_tab = self._create_zones_tab()
        tabs.addTab(zones_tab, "Zones")

        # Settings tab (can be overridden by subclass)
        settings_tab = self._create_settings_tab()
        if settings_tab:
            tabs.addTab(settings_tab, "Settings")

        layout.addWidget(tabs)

    def _create_zones_tab(self) -> QtWidgets.QWidget:
        """Create the zones configuration tab.

        Returns:
            Widget containing zone mapping UI.
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Instructions
        instructions = QtWidgets.QLabel(
            "Map the game's required zones to your calibrated zones:"
        )
        layout.addWidget(instructions)

        # Zone mapping table
        if self.zone_requirements:
            form_layout = QtWidgets.QFormLayout()

            for req in self.zone_requirements:
                # Create label with requirements
                req_parts = []
                if req.requires_camera:
                    req_parts.append("Camera")
                if req.requires_projector:
                    req_parts.append("Projector")
                if req.units:
                    req_parts.append(f"Units: {req.units}")

                req_text = f"{req.display_name}"
                if req_parts:
                    req_text += f" ({', '.join(req_parts)})"

                label = QtWidgets.QLabel(req_text)

                # Create combo box
                combo = QtWidgets.QComboBox()
                combo.setMinimumWidth(200)
                self.zone_combos[req.internal_name] = combo

                form_layout.addRow(label, combo)

            layout.addLayout(form_layout)

            # Populate combos
            self._refresh_zone_combos()
        else:
            no_zones_label = QtWidgets.QLabel("This game does not require any zones.")
            layout.addWidget(no_zones_label)

        layout.addStretch()

        # Validation section
        validation_layout = QtWidgets.QHBoxLayout()

        self.validate_button = QtWidgets.QPushButton("Validate Zone Mapping")
        self.validate_button.clicked.connect(self._on_validate_zones)
        validation_layout.addWidget(self.validate_button)

        validation_layout.addStretch()

        layout.addLayout(validation_layout)

        # Validation message
        self.validation_label = QtWidgets.QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("QLabel { padding: 10px; }")
        layout.addWidget(self.validation_label)

        return widget

    def _create_main_tab(self) -> Optional[QtWidgets.QWidget]:
        """Create the main tab.

        Override this method in subclasses to add game-specific main UI
        (e.g., Start/Stop buttons, game status, etc.).

        Returns:
            Widget containing main UI, or None if no main tab needed.
        """
        return None

    def _create_settings_tab(self) -> Optional[QtWidgets.QWidget]:
        """Create the settings tab.

        Override this method in subclasses to add game-specific settings.

        Returns:
            Widget containing settings UI, or None if no settings needed.
        """
        return None

    def _refresh_zone_combos(self) -> None:
        """Refresh the zone combo boxes with available zones."""
        # Get all zones
        all_zones = self.core.zone_manager.get_all_zones()

        for req in self.zone_requirements:
            combo = self.zone_combos[req.internal_name]

            # Save current selection
            current_text = combo.currentText()

            # Clear and repopulate
            combo.clear()
            combo.addItem("-- Select Zone --", None)

            for zone in all_zones:
                combo.addItem(zone.name, zone.name)

            # Restore selection if still valid
            if current_text:
                index = combo.findText(current_text)
                if index >= 0:
                    combo.setCurrentIndex(index)

    @QtCore.Slot()
    def _on_validate_zones(self) -> None:
        """Validate the zone mapping."""
        errors = []
        warnings = []

        # Update zone mapping from combos
        for req in self.zone_requirements:
            combo = self.zone_combos[req.internal_name]
            selected_zone = combo.currentData()
            self.zone_mapping[req.internal_name] = selected_zone

        # Validate each requirement
        for req in self.zone_requirements:
            zone_name = self.zone_mapping[req.internal_name]

            if zone_name is None:
                errors.append(f"❌ {req.display_name}: No zone selected")
                continue

            # Get the zone
            zone = self.core.zone_manager.get_zone(zone_name)
            if zone is None:
                errors.append(f"❌ {req.display_name}: Zone '{zone_name}' not found")
                continue

            # Check camera requirement
            if req.requires_camera and zone.camera_mapping is None:
                errors.append(
                    f"❌ {req.display_name}: Zone '{zone_name}' requires camera mapping. "
                    f"Please calibrate the camera for this zone."
                )

            # Check projector requirement
            if req.requires_projector and zone.projector_mapping is None:
                errors.append(
                    f"❌ {req.display_name}: Zone '{zone_name}' requires projector mapping. "
                    f"Please calibrate the projector for this zone."
                )

            # Check units requirement
            if req.units and zone.unit != req.units:
                if zone.unit:
                    warnings.append(
                        f"⚠️ {req.display_name}: Zone '{zone_name}' uses '{zone.unit}' "
                        f"but game expects '{req.units}'. This may cause issues."
                    )
                else:
                    errors.append(
                        f"❌ {req.display_name}: Zone '{zone_name}' has no units set. "
                        f"Game requires '{req.units}'."
                    )

        # Check for duplicate zone assignments
        assigned_zones = [z for z in self.zone_mapping.values() if z is not None]
        if len(assigned_zones) != len(set(assigned_zones)):
            warnings.append(
                "⚠️ Multiple game zones are mapped to the same physical zone. "
                "This may be intentional, but verify it's correct."
            )

        # Display results
        if errors:
            self.validation_label.setStyleSheet(
                "QLabel { background-color: #ffcccc; color: #cc0000; padding: 10px; }"
            )
            self.validation_label.setText("\n".join(errors))
        elif warnings:
            self.validation_label.setStyleSheet(
                "QLabel { background-color: #ffffcc; color: #cc8800; padding: 10px; }"
            )
            self.validation_label.setText(
                "✅ Validation passed with warnings:\n" + "\n".join(warnings)
            )
        else:
            self.validation_label.setStyleSheet(
                "QLabel { background-color: #ccffcc; color: #008800; padding: 10px; }"
            )
            self.validation_label.setText("✅ All zones validated successfully!")

    def keyPressEvent(self, event) -> None:
        """Override key press event to disable Esc key closing the dialog.

        Args:
            event: Key press event.
        """
        if event.key() == QtCore.Qt.Key.Key_Escape:
            # Ignore Esc key
            event.ignore()
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)

    def get_zone_mapping(self) -> dict[str, Optional[str]]:
        """Get the current zone mapping.

        Returns:
            Dictionary mapping internal zone names to actual zone names.
        """
        return self.zone_mapping.copy()

    def is_validated(self) -> bool:
        """Check if zone mapping has been validated successfully.

        Returns:
            True if validation passed (with or without warnings), False otherwise.
        """
        # Re-run validation logic to check current state
        errors = []

        for req in self.zone_requirements:
            zone_name = self.zone_mapping.get(req.internal_name)

            if zone_name is None:
                errors.append(f"No zone selected for {req.display_name}")
                continue

            zone = self.core.zone_manager.get_zone(zone_name)
            if zone is None:
                errors.append(f"Zone '{zone_name}' not found")
                continue

            if req.requires_camera and zone.camera_mapping is None:
                errors.append(f"Zone '{zone_name}' missing camera mapping")

            if req.requires_projector and zone.projector_mapping is None:
                errors.append(f"Zone '{zone_name}' missing projector mapping")

        return len(errors) == 0
