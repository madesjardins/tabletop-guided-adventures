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

"""Add camera dialog module.

This module contains the AddCameraDialog class for adding new cameras
to the application.
"""

from __future__ import annotations

import cv2 as cv
from PySide6 import QtWidgets, QtCore, QtGui

from .camera_manager import enumerate_available_cameras


# Mapping of backend names to OpenCV constants
BACKEND_MAP = {
    "ANY": cv.CAP_ANY,
    "MSMF": cv.CAP_MSMF,
    "DSHOW": cv.CAP_DSHOW,
}


class AddCameraDialog(QtWidgets.QDialog):
    """Dialog for adding a new camera.

    This dialog allows the user to select a capture API backend, device,
    and provide a name for the new camera.
    """

    def __init__(
        self,
        used_device_ids_by_backend: dict[int, set[int]],
        existing_camera_names: set[str],
        parent: QtWidgets.QWidget | None = None
    ) -> None:
        """Initialize the add camera dialog.

        Args:
            used_device_ids_by_backend: Dictionary mapping backend to used device IDs.
            existing_camera_names: Set of existing camera names to prevent duplicates.
            parent: Parent widget.
        """
        super().__init__(parent)

        self.used_device_ids_by_backend = used_device_ids_by_backend
        self.existing_camera_names = existing_camera_names
        self.selected_backend: int | None = None
        self.selected_device_id: int | None = None
        self.selected_camera_info: dict | None = None
        self.camera_name: str | None = None

        self.setWindowTitle("Add Camera")
        self.setModal(True)
        self.resize(500, 300)

        self._setup_ui()
        self._populate_backends()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QtWidgets.QVBoxLayout(self)

        # Form layout for inputs
        form_layout = QtWidgets.QFormLayout()

        # Backend selection
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        form_layout.addRow("Capture API:", self.backend_combo)

        # Device selection
        self.device_combo = QtWidgets.QComboBox()
        form_layout.addRow("Device:", self.device_combo)

        # Camera name input with validator
        self.name_input = QtWidgets.QLineEdit()
        name_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^[a-zA-Z0-9_.\-]+$")
        )
        self.name_input.setValidator(name_validator)
        self.name_input.setPlaceholderText("e.g., main_camera")
        form_layout.addRow("Camera Name:", self.name_input)

        layout.addLayout(form_layout)

        # Status label
        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet("color: rgb(255, 100, 100);")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Spacer
        layout.addStretch()

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        self.continue_button = QtWidgets.QPushButton("Continue")
        self.continue_button.clicked.connect(self._on_continue)
        self.continue_button.setDefault(True)
        button_layout.addWidget(self.continue_button)

        layout.addLayout(button_layout)

    def _populate_backends(self) -> None:
        """Populate the backend combo box."""
        self.backend_combo.clear()

        for backend_name in BACKEND_MAP.keys():
            self.backend_combo.addItem(backend_name, BACKEND_MAP[backend_name])

        # Set DSHOW as default selection
        dshow_index = self.backend_combo.findText("DSHOW")
        if dshow_index >= 0:
            self.backend_combo.setCurrentIndex(dshow_index)

    @QtCore.Slot()
    def _on_backend_changed(self) -> None:
        """Handle backend selection change."""
        self.device_combo.clear()
        self.status_label.clear()

        if self.backend_combo.currentIndex() < 0:
            return

        # Get selected backend
        backend = self.backend_combo.currentData()

        # Get used device IDs for this backend
        used_ids = self.used_device_ids_by_backend.get(backend, set())

        # Enumerate available cameras
        try:
            available_cameras = enumerate_available_cameras(backend, used_ids)

            if not available_cameras:
                self.status_label.setText("No available cameras found for this backend.")
                self.continue_button.setEnabled(False)
                return

            # Populate device combo
            for camera_info in available_cameras:
                device_id = camera_info['index']
                device_name = camera_info.get('name', f"Camera {device_id}")
                self.device_combo.addItem(f"{device_id}: {device_name}", camera_info)

            self.continue_button.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"Error enumerating cameras: {str(e)}")
            self.continue_button.setEnabled(False)

    @QtCore.Slot()
    def _on_continue(self) -> None:
        """Handle continue button click."""
        # Validate inputs
        if self.backend_combo.currentIndex() < 0:
            self.status_label.setText("Please select a capture API.")
            return

        if self.device_combo.currentIndex() < 0:
            self.status_label.setText("Please select a device.")
            return

        camera_name = self.name_input.text().strip()
        if not camera_name:
            self.status_label.setText("Please enter a camera name.")
            return

        # Check for duplicate camera name
        if camera_name in self.existing_camera_names:
            self.status_label.setText(f"Camera name '{camera_name}' already exists. Please choose a different name.")
            return

        # Store selections
        self.selected_backend = self.backend_combo.currentData()
        camera_info = self.device_combo.currentData()
        self.selected_device_id = camera_info['index'] if camera_info else None
        self.selected_camera_info = camera_info
        self.camera_name = camera_name

        # Accept dialog
        self.accept()

    def get_camera_info(self) -> tuple[str, int, int, dict | None] | None:
        """Get the selected camera information.

        Returns:
            Tuple of (name, backend, device_id, camera_info) or None if dialog was cancelled.
        """
        if self.result() == QtWidgets.QDialog.DialogCode.Accepted:
            return (self.camera_name, self.selected_backend, self.selected_device_id, self.selected_camera_info)
        return None
