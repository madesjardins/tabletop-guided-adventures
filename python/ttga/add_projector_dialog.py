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

"""Dialog for adding a new projector."""

from PySide6 import QtWidgets


# Available projector resolutions
PROJECTOR_RESOLUTIONS = [
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 800),
    (1920, 1080),
    (1920, 1200),
    (3840, 2160)
]


class AddProjectorDialog(QtWidgets.QDialog):
    """Dialog for adding a new projector."""

    def __init__(self, existing_names: list[str], parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the add projector dialog.

        Args:
            existing_names: List of existing projector names to validate against.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.existing_names = existing_names
        self.projector_name = ""
        self.projector_resolution = (1920, 1080)

        self.setWindowTitle("Add Projector")
        self.setModal(True)

        # Create layout
        layout = QtWidgets.QVBoxLayout(self)

        # Form layout for inputs
        form_layout = QtWidgets.QFormLayout()

        # Name input
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Enter unique projector name")
        self.name_edit.textChanged.connect(self._on_name_changed)
        form_layout.addRow("Projector Name:", self.name_edit)

        # Resolution combo box
        self.resolution_combo = QtWidgets.QComboBox()
        for width, height in PROJECTOR_RESOLUTIONS:
            self.resolution_combo.addItem(f"{width}x{height}", (width, height))
        # Set default to 1920x1080
        self.resolution_combo.setCurrentIndex(4)
        form_layout.addRow("Resolution:", self.resolution_combo)

        layout.addLayout(form_layout)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Store button reference to enable/disable
        self.ok_button = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setEnabled(False)

    def _on_name_changed(self, text: str) -> None:
        """Handle name text change to validate."""
        # Enable OK button only if name is non-empty and unique
        is_valid = bool(text.strip()) and text.strip() not in self.existing_names
        self.ok_button.setEnabled(is_valid)

    def accept(self) -> None:
        """Accept the dialog and store values."""
        self.projector_name = self.name_edit.text().strip()
        self.projector_resolution = self.resolution_combo.currentData()
        super().accept()
