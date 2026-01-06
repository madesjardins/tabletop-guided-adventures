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

"""Main window module for the Tabletop Guided Adventures application.

This module contains the MainWindow class which provides the main
application interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtWidgets, QtCore, QtGui

from .viewport_widget import ViewportWidget
from .add_camera_dialog import AddCameraDialog, BACKEND_MAP

if TYPE_CHECKING:
    from .main_core import MainCore


class MainWindow(QtWidgets.QMainWindow):
    """Main application window.

    This window provides the main interface for the Tabletop Guided Adventures
    application, including camera management, settings, and viewport display.
    """

    def __init__(self, core: MainCore) -> None:
        """Initialize the main window.

        Args:
            core: Main core instance managing application state.
        """
        super().__init__()

        self.core = core

        self.setWindowTitle("Tabletop Guided Adventures")
        self.resize(1400, 900)

        self._setup_menu_bar()
        self._setup_ui()

        # Connect camera manager signals
        self.core.camera_manager.camera_added.connect(self._on_camera_added)
        self.core.camera_manager.camera_removed.connect(self._on_camera_removed)

    def _setup_menu_bar(self) -> None:
        """Set up the menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        # Quit action
        quit_action = QtGui.QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _setup_ui(self) -> None:
        """Set up the main user interface."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Main layout - 2x2 grid
        main_layout = QtWidgets.QGridLayout(central_widget)

        # Top-left: Camera list with buttons
        camera_group = self._create_camera_list_group()
        main_layout.addWidget(camera_group, 0, 0)

        # Top-right: Settings tabs (Zones, Voice, Sound, Advanced, Debug)
        settings_tabs = self._create_settings_tabs()
        main_layout.addWidget(settings_tabs, 0, 1)

        # Center-left: Camera tabs (Settings, Calibration, Snapshot)
        camera_tabs = self._create_camera_tabs()
        main_layout.addWidget(camera_tabs, 1, 0)

        # Center-right: Viewport
        viewport = self._create_viewport()
        main_layout.addWidget(viewport, 1, 1)

        # Set column stretches to give more space to the right side
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 2)

        # Set row stretches to give more space to the bottom
        main_layout.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 2)

    def _create_camera_list_group(self) -> QtWidgets.QGroupBox:
        """Create the camera list group with buttons.

        Returns:
            Group box containing camera list and control buttons.
        """
        group = QtWidgets.QGroupBox("Cameras")
        layout = QtWidgets.QVBoxLayout(group)

        # Camera list
        self.camera_list = QtWidgets.QListWidget()
        self.camera_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.camera_list.itemSelectionChanged.connect(self._on_camera_selection_changed)
        layout.addWidget(self.camera_list)

        # Buttons in 2x2 grid
        button_layout = QtWidgets.QGridLayout()

        self.add_camera_button = QtWidgets.QPushButton("Add")
        self.add_camera_button.clicked.connect(self._on_add_camera)
        button_layout.addWidget(self.add_camera_button, 0, 0)

        self.delete_camera_button = QtWidgets.QPushButton("Delete")
        self.delete_camera_button.clicked.connect(self._on_delete_camera)
        self.delete_camera_button.setEnabled(False)
        button_layout.addWidget(self.delete_camera_button, 0, 1)

        self.load_camera_button = QtWidgets.QPushButton("Load")
        self.load_camera_button.clicked.connect(self._on_load_camera)
        button_layout.addWidget(self.load_camera_button, 1, 0)

        self.save_camera_button = QtWidgets.QPushButton("Save")
        self.save_camera_button.clicked.connect(self._on_save_camera)
        button_layout.addWidget(self.save_camera_button, 1, 1)

        layout.addLayout(button_layout)

        return group

    def _create_settings_tabs(self) -> QtWidgets.QTabWidget:
        """Create the settings tab widget.

        Returns:
            Tab widget containing settings tabs.
        """
        tabs = QtWidgets.QTabWidget()

        # Zones tab
        zones_widget = QtWidgets.QWidget()
        zones_layout = QtWidgets.QVBoxLayout(zones_widget)
        zones_layout.addWidget(QtWidgets.QLabel("Zones configuration will be added here"))
        tabs.addTab(zones_widget, "Zones")

        # Voice tab
        voice_widget = QtWidgets.QWidget()
        voice_layout = QtWidgets.QVBoxLayout(voice_widget)
        voice_layout.addWidget(QtWidgets.QLabel("Voice settings will be added here"))
        tabs.addTab(voice_widget, "Voice")

        # Sound tab
        sound_widget = QtWidgets.QWidget()
        sound_layout = QtWidgets.QVBoxLayout(sound_widget)
        sound_layout.addWidget(QtWidgets.QLabel("Sound settings will be added here"))
        tabs.addTab(sound_widget, "Sound")

        # Advanced tab
        advanced_widget = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(advanced_widget)
        advanced_layout.addWidget(QtWidgets.QLabel("Advanced settings will be added here"))
        tabs.addTab(advanced_widget, "Advanced")

        # Debug tab
        debug_widget = QtWidgets.QWidget()
        debug_layout = QtWidgets.QVBoxLayout(debug_widget)
        debug_layout.addWidget(QtWidgets.QLabel("Debug information will be added here"))
        tabs.addTab(debug_widget, "Debug")

        return tabs

    def _create_camera_tabs(self) -> QtWidgets.QTabWidget:
        """Create the camera tab widget.

        Returns:
            Tab widget containing camera-specific tabs.
        """
        tabs = QtWidgets.QTabWidget()

        # Settings tab
        settings_widget = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_widget)
        settings_layout.addWidget(QtWidgets.QLabel("Camera settings will be added here"))
        tabs.addTab(settings_widget, "Settings")

        # Calibration tab
        calibration_widget = QtWidgets.QWidget()
        calibration_layout = QtWidgets.QVBoxLayout(calibration_widget)
        calibration_layout.addWidget(QtWidgets.QLabel("Camera calibration will be added here"))
        tabs.addTab(calibration_widget, "Calibration")

        # Snapshot tab
        snapshot_widget = QtWidgets.QWidget()
        snapshot_layout = QtWidgets.QVBoxLayout(snapshot_widget)
        snapshot_layout.addWidget(QtWidgets.QLabel("Camera snapshot will be added here"))
        tabs.addTab(snapshot_widget, "Snapshot")

        return tabs

    def _create_viewport(self) -> ViewportWidget:
        """Create the viewport widget.

        Returns:
            Viewport widget for displaying camera feeds.
        """
        self.viewport = ViewportWidget()
        return self.viewport

    @QtCore.Slot()
    def _on_add_camera(self) -> None:
        """Handle add camera button click."""
        # Get used device IDs by backend
        used_device_ids_by_backend = {}
        for backend_name, backend_id in BACKEND_MAP.items():
            used_device_ids_by_backend[backend_id] = self.core.camera_manager.get_used_device_ids(backend_id)

        # Open dialog
        dialog = AddCameraDialog(used_device_ids_by_backend, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            camera_info = dialog.get_camera_info()
            if camera_info:
                name, backend, device_id = camera_info

                # Check if name already exists
                if self.core.camera_manager.has_camera(name):
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Camera Exists",
                        f"A camera with the name '{name}' already exists."
                    )
                    return

                try:
                    # Add camera through camera manager
                    camera = self.core.camera_manager.add_camera(name, backend, device_id)

                    # Start camera feed
                    camera.start()

                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Error Adding Camera",
                        f"Failed to add camera: {str(e)}"
                    )

    @QtCore.Slot()
    def _on_delete_camera(self) -> None:
        """Handle delete camera button click."""
        # Get selected items
        selected_items = self.camera_list.selectedItems()
        if not selected_items:
            return

        # Get camera names
        camera_names = [item.text() for item in selected_items]

        # Confirm deletion
        if len(camera_names) == 1:
            message = f"Are you sure you want to delete camera '{camera_names[0]}'?"
        else:
            message = f"Are you sure you want to delete {len(camera_names)} cameras?"

        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Camera(s)",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # Remove cameras
            for camera_name in camera_names:
                try:
                    self.core.camera_manager.remove_camera(camera_name)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Error Deleting Camera",
                        f"Failed to delete camera '{camera_name}': {str(e)}"
                    )

    @QtCore.Slot()
    def _on_camera_selection_changed(self) -> None:
        """Handle camera list selection change."""
        # Enable delete button only if at least one camera is selected
        has_selection = len(self.camera_list.selectedItems()) > 0
        self.delete_camera_button.setEnabled(has_selection)

    @QtCore.Slot()
    def _on_load_camera(self) -> None:
        """Handle load camera button click."""
        pass

    @QtCore.Slot()
    def _on_save_camera(self) -> None:
        """Handle save camera button click."""
        pass

    @QtCore.Slot(str)
    def _on_camera_added(self, camera_name: str) -> None:
        """Handle camera added signal.

        Args:
            camera_name: Name of the added camera.
        """
        # Add to list widget
        self.camera_list.addItem(camera_name)

        # Select the new camera
        items = self.camera_list.findItems(camera_name, QtCore.Qt.MatchFlag.MatchExactly)
        if items:
            self.camera_list.setCurrentItem(items[0])

    @QtCore.Slot(str)
    def _on_camera_removed(self, camera_name: str) -> None:
        """Handle camera removed signal.

        Args:
            camera_name: Name of the removed camera.
        """
        # Remove from list widget
        items = self.camera_list.findItems(camera_name, QtCore.Qt.MatchFlag.MatchExactly)
        for item in items:
            row = self.camera_list.row(item)
            self.camera_list.takeItem(row)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window close event.

        Args:
            event: Close event.
        """
        # Release all camera resources
        self.core.release_all()

        # Accept the close event
        event.accept()
