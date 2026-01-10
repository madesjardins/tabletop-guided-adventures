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

"""Dialog for adding a new zone."""

from PySide6 import QtWidgets

from .zone_manager import ZoneManager


class AddZoneDialog(QtWidgets.QDialog):
    """Dialog for adding a new zone with a unique name."""

    def __init__(self, zone_manager: ZoneManager, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the add zone dialog.

        Args:
            zone_manager: Zone manager to check for name uniqueness.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.zone_manager = zone_manager
        self.setWindowTitle("Add Zone")

        # Create layout
        layout = QtWidgets.QVBoxLayout(self)

        # Zone name input
        form_layout = QtWidgets.QFormLayout()
        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setPlaceholderText("Enter zone name")
        self.name_input.textChanged.connect(self._validate_name)
        form_layout.addRow("Zone Name:", self.name_input)
        layout.addLayout(form_layout)

        # Error label
        self.error_label = QtWidgets.QLabel()
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.ok_button = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setEnabled(False)

        # Set initial focus
        self.name_input.setFocus()

    def _validate_name(self) -> None:
        """Validate the zone name."""
        name = self.name_input.text().strip()

        if not name:
            self.error_label.setVisible(False)
            self.ok_button.setEnabled(False)
            return

        if self.zone_manager.zone_exists(name):
            self.error_label.setText(f"Zone '{name}' already exists")
            self.error_label.setVisible(True)
            self.ok_button.setEnabled(False)
        else:
            self.error_label.setVisible(False)
            self.ok_button.setEnabled(True)

    def get_zone_name(self) -> str:
        """Get the entered zone name.

        Returns:
            The zone name.
        """
        return self.name_input.text().strip()
