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

import cv2 as cv
import numpy as np
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

        # Set up viewport callbacks and start timer
        self.viewport.set_get_frames_callback(
            self._get_selected_camera_frames,
            self._get_selected_camera_ids
        )
        self.viewport.start()

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
        camera_tabs.setFixedWidth(350)
        main_layout.addWidget(camera_tabs, 1, 0)

        # Center-right: Viewport
        viewport = self._create_viewport()
        main_layout.addWidget(viewport, 1, 1)

        # Set column stretches to give more space to the right side
        main_layout.setColumnStretch(0, 0)
        main_layout.setColumnStretch(1, 1)

        # Set row stretches to give more space to the bottom
        main_layout.setRowStretch(0, 1)
        main_layout.setRowStretch(1, 2)

    def _create_camera_list_group(self) -> QtWidgets.QGroupBox:
        """Create the camera list group with buttons.

        Returns:
            Group box containing camera list and control buttons.
        """
        group = QtWidgets.QGroupBox("Cameras")
        group.setFixedWidth(350)
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

        # Viewport refresh rate
        refresh_form = QtWidgets.QFormLayout()
        self.viewport_fps_spinbox = QtWidgets.QSpinBox()
        self.viewport_fps_spinbox.setRange(5, 60)
        self.viewport_fps_spinbox.setValue(30)
        self.viewport_fps_spinbox.setSuffix(" fps")
        self.viewport_fps_spinbox.valueChanged.connect(self._on_viewport_fps_changed)
        refresh_form.addRow("Viewport Refresh Rate:", self.viewport_fps_spinbox)
        advanced_layout.addLayout(refresh_form)

        advanced_layout.addStretch()
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
        settings_widget = self._create_camera_settings_widget()
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

    def _create_camera_settings_widget(self) -> QtWidgets.QWidget:
        """Create the camera settings widget.

        Returns:
            Widget containing camera property controls.
        """
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(widget)

        # Use grid layout for all properties
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1, 1)

        row = 0

        # Device ID (read-only)
        grid.addWidget(QtWidgets.QLabel("Device ID:"), row, 0)
        self.device_id_edit = QtWidgets.QLineEdit()
        self.device_id_edit.setReadOnly(True)
        self.device_id_edit.setPlaceholderText("No camera selected")
        grid.addWidget(self.device_id_edit, row, 1)
        row += 1

        # FourCC
        grid.addWidget(QtWidgets.QLabel("FourCC:"), row, 0)
        self.fourcc_combo = QtWidgets.QComboBox()
        self.fourcc_combo.addItems(["YUY2", "MJPG"])
        self.fourcc_combo.setCurrentText("YUY2")
        self.fourcc_combo.currentTextChanged.connect(self._on_fourcc_changed)
        grid.addWidget(self.fourcc_combo, row, 1)
        row += 1

        # Capture Resolution
        grid.addWidget(QtWidgets.QLabel("Resolution:"), row, 0)
        self.resolution_combo = QtWidgets.QComboBox()
        self.resolution_combo.addItems([
            "640x480", "960x540", "1280x720", "1920x1080",
            "2304x1536", "2560x1440", "3840x2160", "4096x2160"
        ])
        self.resolution_combo.setCurrentText("1920x1080")
        self.resolution_combo.currentTextChanged.connect(self._on_resolution_changed)
        grid.addWidget(self.resolution_combo, row, 1)
        row += 1

        # Exposure
        grid.addWidget(QtWidgets.QLabel("Exposure:"), row, 0)
        self.exposure_spinbox = QtWidgets.QSpinBox()
        self.exposure_spinbox.setRange(-10, 10)
        self.exposure_spinbox.setValue(5)
        self.exposure_spinbox.valueChanged.connect(self._on_exposure_changed)
        grid.addWidget(self.exposure_spinbox, row, 1)
        row += 1

        main_layout.addLayout(grid)

        # Add separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        main_layout.addWidget(separator)

        # Slider properties grid
        slider_grid = QtWidgets.QGridLayout()
        slider_grid.setColumnStretch(1, 1)

        slider_row = 0

        # Focus
        self.focus_slider, self.focus_reset = self._create_slider_with_reset(
            "Focus", 0, 255, 0, self._on_focus_changed, self._on_focus_reset
        )
        self._add_slider_to_grid(slider_grid, slider_row, "Focus:", self.focus_slider, self.focus_reset)
        slider_row += 1

        # Zoom
        self.zoom_slider, self.zoom_reset = self._create_slider_with_reset(
            "Zoom", 100, 500, 100, self._on_zoom_changed, self._on_zoom_reset
        )
        self._add_slider_to_grid(slider_grid, slider_row, "Zoom:", self.zoom_slider, self.zoom_reset)
        slider_row += 1

        # Brightness
        self.brightness_slider, self.brightness_reset = self._create_slider_with_reset(
            "Brightness", 0, 255, 128, self._on_brightness_changed, self._on_brightness_reset
        )
        self._add_slider_to_grid(slider_grid, slider_row, "Brightness:", self.brightness_slider, self.brightness_reset)
        slider_row += 1

        # Contrast
        self.contrast_slider, self.contrast_reset = self._create_slider_with_reset(
            "Contrast", 0, 255, 128, self._on_contrast_changed, self._on_contrast_reset
        )
        self._add_slider_to_grid(slider_grid, slider_row, "Contrast:", self.contrast_slider, self.contrast_reset)
        slider_row += 1

        # Gain
        self.gain_slider, self.gain_reset = self._create_slider_with_reset(
            "Gain", 0, 255, 128, self._on_gain_changed, self._on_gain_reset
        )
        self._add_slider_to_grid(slider_grid, slider_row, "Gain:", self.gain_slider, self.gain_reset)
        slider_row += 1

        # Saturation
        self.saturation_slider, self.saturation_reset = self._create_slider_with_reset(
            "Saturation", 0, 255, 128, self._on_saturation_changed, self._on_saturation_reset
        )
        self._add_slider_to_grid(slider_grid, slider_row, "Saturation:", self.saturation_slider, self.saturation_reset)
        slider_row += 1

        # Sharpness
        self.sharpness_slider, self.sharpness_reset = self._create_slider_with_reset(
            "Sharpness", 0, 255, 128, self._on_sharpness_changed, self._on_sharpness_reset
        )
        self._add_slider_to_grid(slider_grid, slider_row, "Sharpness:", self.sharpness_slider, self.sharpness_reset)
        slider_row += 1

        main_layout.addLayout(slider_grid)
        main_layout.addStretch()

        # Initially disable all controls
        self._set_camera_settings_enabled(False)

        return widget

    def _add_slider_to_grid(self, grid: QtWidgets.QGridLayout, row: int, label_text: str,
                            slider: QtWidgets.QSlider, reset_button: QtWidgets.QPushButton) -> None:
        """Add a slider row to the grid layout.


        Args:
            grid: Grid layout to add to.
            row: Row number.
            label_text: Label text.
            slider: Slider widget.
            reset_button: Reset button widget.
        """
        # Label
        label = QtWidgets.QLabel(label_text)
        grid.addWidget(label, row, 0)

        # Slider
        grid.addWidget(slider, row, 1)

        # Value label
        value_label = QtWidgets.QLabel(str(slider.value()))
        value_label.setMinimumWidth(40)
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        grid.addWidget(value_label, row, 2)

        # Vertical separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: white;")
        grid.addWidget(separator, row, 3)

        # Reset button
        grid.addWidget(reset_button, row, 4)

    def _create_slider_with_reset(
        self,
        name: str,
        min_val: int,
        max_val: int,
        default_val: int,
        value_changed_callback,
        reset_callback
    ) -> tuple[QtWidgets.QSlider, QtWidgets.QPushButton]:
        """Create a slider with associated reset button.

        Args:
            name: Property name.
            min_val: Minimum value.
            max_val: Maximum value.
            default_val: Default value.
            value_changed_callback: Callback for value changes.
            reset_callback: Callback for reset button.

        Returns:
            Tuple of (slider, reset_button).
        """
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setProperty("default_value", default_val)
        slider.valueChanged.connect(value_changed_callback)

        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.setMaximumWidth(60)
        reset_button.clicked.connect(reset_callback)

        return slider, reset_button

    def _create_viewport(self) -> ViewportWidget:
        """Create the viewport widget.

        Returns:
            Viewport widget for displaying camera feeds.
        """
        self.viewport = ViewportWidget()
        return self.viewport

    def _get_selected_camera_frames(self) -> list[np.ndarray]:
        """Get frames from selected cameras.

        Returns:
            List of frames from selected cameras.
        """
        frames = []
        selected_items = self.camera_list.selectedItems()

        for item in selected_items:
            camera_name = item.text()
            try:
                camera = self.core.camera_manager.get_camera(camera_name)
                frame = camera.get_frame()
                if frame is not None:
                    frames.append(frame)
            except KeyError:
                # Camera was removed
                pass

        return frames

    def _get_selected_camera_ids(self) -> list[int]:
        """Get IDs (hashes) of selected cameras for tracking selection changes.

        Returns:
            List of camera name hashes.
        """
        selected_items = self.camera_list.selectedItems()
        return [hash(item.text()) for item in selected_items]

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
        selected_items = self.camera_list.selectedItems()
        has_selection = len(selected_items) > 0

        # Enable delete button only if at least one camera is selected
        self.delete_camera_button.setEnabled(has_selection)

        # Update camera settings based on selection
        if len(selected_items) == 1:
            # Single camera selected - enable and load settings
            self._set_camera_settings_enabled(True)
            self._load_camera_settings(selected_items[0].text())
        else:
            # Multiple or no cameras selected - disable settings
            self._set_camera_settings_enabled(False)
            self.device_id_edit.clear()
            self.device_id_edit.setPlaceholderText("No camera selected" if len(selected_items) == 0 else "Multiple cameras selected")

    @QtCore.Slot(int)
    def _on_viewport_fps_changed(self, fps: int) -> None:
        """Handle viewport FPS change.

        Args:
            fps: New frames per second value.
        """
        self.viewport.set_fps(fps)

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
        # Stop viewport updates
        self.viewport.stop()

        # Release all camera resources
        self.core.release_all()

        # Accept the close event
        event.accept()

    def _set_camera_settings_enabled(self, enabled: bool) -> None:
        """Enable or disable camera settings controls.

        Args:
            enabled: True to enable, False to disable.
        """
        self.fourcc_combo.setEnabled(enabled)
        self.resolution_combo.setEnabled(enabled)
        self.exposure_spinbox.setEnabled(enabled)
        self.focus_slider.setEnabled(enabled)
        self.focus_reset.setEnabled(enabled)
        self.zoom_slider.setEnabled(enabled)
        self.zoom_reset.setEnabled(enabled)
        self.brightness_slider.setEnabled(enabled)
        self.brightness_reset.setEnabled(enabled)
        self.contrast_slider.setEnabled(enabled)
        self.contrast_reset.setEnabled(enabled)
        self.gain_slider.setEnabled(enabled)
        self.gain_reset.setEnabled(enabled)
        self.saturation_slider.setEnabled(enabled)
        self.saturation_reset.setEnabled(enabled)
        self.sharpness_slider.setEnabled(enabled)
        self.sharpness_reset.setEnabled(enabled)

    def _load_camera_settings(self, camera_name: str) -> None:
        """Load settings from a camera into the UI.

        Args:
            camera_name: Name of the camera to load settings from.
        """
        try:
            camera = self.core.camera_manager.get_camera(camera_name)

            # Block signals while updating to avoid triggering camera updates
            self.fourcc_combo.blockSignals(True)
            self.resolution_combo.blockSignals(True)
            self.exposure_spinbox.blockSignals(True)
            self.focus_slider.blockSignals(True)
            self.zoom_slider.blockSignals(True)
            self.brightness_slider.blockSignals(True)
            self.contrast_slider.blockSignals(True)
            self.gain_slider.blockSignals(True)
            self.saturation_slider.blockSignals(True)
            self.sharpness_slider.blockSignals(True)

            # Device ID
            self.device_id_edit.setText(str(camera.get_device_id()))

            # FourCC
            fourcc_int = int(camera.get_property(cv.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            if fourcc_str in ["YUY2", "MJPG"]:
                self.fourcc_combo.setCurrentText(fourcc_str)

            # Resolution
            width = int(camera.get_property(cv.CAP_PROP_FRAME_WIDTH))
            height = int(camera.get_property(cv.CAP_PROP_FRAME_HEIGHT))
            resolution_str = f"{width}x{height}"
            idx = self.resolution_combo.findText(resolution_str)
            if idx >= 0:
                self.resolution_combo.setCurrentIndex(idx)

            # Exposure
            exposure = int(camera.get_property(cv.CAP_PROP_EXPOSURE))
            self.exposure_spinbox.setValue(exposure)

            # Slider properties
            self.focus_slider.setValue(int(camera.get_property(cv.CAP_PROP_FOCUS)))
            self.zoom_slider.setValue(int(camera.get_property(cv.CAP_PROP_ZOOM)))
            self.brightness_slider.setValue(int(camera.get_property(cv.CAP_PROP_BRIGHTNESS)))
            self.contrast_slider.setValue(int(camera.get_property(cv.CAP_PROP_CONTRAST)))
            self.gain_slider.setValue(int(camera.get_property(cv.CAP_PROP_GAIN)))
            self.saturation_slider.setValue(int(camera.get_property(cv.CAP_PROP_SATURATION)))
            self.sharpness_slider.setValue(int(camera.get_property(cv.CAP_PROP_SHARPNESS)))

            # Unblock signals
            self.fourcc_combo.blockSignals(False)
            self.resolution_combo.blockSignals(False)
            self.exposure_spinbox.blockSignals(False)
            self.focus_slider.blockSignals(False)
            self.zoom_slider.blockSignals(False)
            self.brightness_slider.blockSignals(False)
            self.contrast_slider.blockSignals(False)
            self.gain_slider.blockSignals(False)
            self.saturation_slider.blockSignals(False)
            self.sharpness_slider.blockSignals(False)

        except KeyError:
            # Camera not found
            pass

    def _get_selected_camera(self):
        """Get the currently selected camera if exactly one is selected.

        Returns:
            Camera object or None.
        """
        selected_items = self.camera_list.selectedItems()
        if len(selected_items) == 1:
            try:
                return self.core.camera_manager.get_camera(selected_items[0].text())
            except KeyError:
                pass
        return None

    # Property change handlers
    @QtCore.Slot(str)
    def _on_fourcc_changed(self, fourcc: str) -> None:
        """Handle FourCC change."""
        camera = self._get_selected_camera()
        if camera:
            fourcc_int = sum([ord(c) << (8 * i) for i, c in enumerate(fourcc[:4])])
            camera.set_property(cv.CAP_PROP_FOURCC, float(fourcc_int))

    @QtCore.Slot(str)
    def _on_resolution_changed(self, resolution: str) -> None:
        """Handle resolution change."""
        camera = self._get_selected_camera()
        if camera:
            width, height = map(int, resolution.split('x'))
            camera.set_property(cv.CAP_PROP_FRAME_WIDTH, float(width))
            camera.set_property(cv.CAP_PROP_FRAME_HEIGHT, float(height))

    @QtCore.Slot(int)
    def _on_exposure_changed(self, value: int) -> None:
        """Handle exposure change."""
        camera = self._get_selected_camera()
        if camera:
            camera.set_property(cv.CAP_PROP_EXPOSURE, float(value))

    @QtCore.Slot(int)
    def _on_focus_changed(self, value: int) -> None:
        """Handle focus change."""
        camera = self._get_selected_camera()
        if camera:
            camera.set_property(cv.CAP_PROP_FOCUS, float(value))

    @QtCore.Slot()
    def _on_focus_reset(self) -> None:
        """Reset focus to default."""
        self.focus_slider.setValue(self.focus_slider.property("default_value"))

    @QtCore.Slot(int)
    def _on_zoom_changed(self, value: int) -> None:
        """Handle zoom change."""
        camera = self._get_selected_camera()
        if camera:
            camera.set_property(cv.CAP_PROP_ZOOM, float(value))

    @QtCore.Slot()
    def _on_zoom_reset(self) -> None:
        """Reset zoom to default."""
        self.zoom_slider.setValue(self.zoom_slider.property("default_value"))

    @QtCore.Slot(int)
    def _on_brightness_changed(self, value: int) -> None:
        """Handle brightness change."""
        camera = self._get_selected_camera()
        if camera:
            camera.set_property(cv.CAP_PROP_BRIGHTNESS, float(value))

    @QtCore.Slot()
    def _on_brightness_reset(self) -> None:
        """Reset brightness to default."""
        self.brightness_slider.setValue(self.brightness_slider.property("default_value"))

    @QtCore.Slot(int)
    def _on_contrast_changed(self, value: int) -> None:
        """Handle contrast change."""
        camera = self._get_selected_camera()
        if camera:
            camera.set_property(cv.CAP_PROP_CONTRAST, float(value))

    @QtCore.Slot()
    def _on_contrast_reset(self) -> None:
        """Reset contrast to default."""
        self.contrast_slider.setValue(self.contrast_slider.property("default_value"))

    @QtCore.Slot(int)
    def _on_gain_changed(self, value: int) -> None:
        """Handle gain change."""
        camera = self._get_selected_camera()
        if camera:
            camera.set_property(cv.CAP_PROP_GAIN, float(value))

    @QtCore.Slot()
    def _on_gain_reset(self) -> None:
        """Reset gain to default."""
        self.gain_slider.setValue(self.gain_slider.property("default_value"))

    @QtCore.Slot(int)
    def _on_saturation_changed(self, value: int) -> None:
        """Handle saturation change."""
        camera = self._get_selected_camera()
        if camera:
            camera.set_property(cv.CAP_PROP_SATURATION, float(value))

    @QtCore.Slot()
    def _on_saturation_reset(self) -> None:
        """Reset saturation to default."""
        self.saturation_slider.setValue(self.saturation_slider.property("default_value"))

    @QtCore.Slot(int)
    def _on_sharpness_changed(self, value: int) -> None:
        """Handle sharpness change."""
        camera = self._get_selected_camera()
        if camera:
            camera.set_property(cv.CAP_PROP_SHARPNESS, float(value))

    @QtCore.Slot()
    def _on_sharpness_reset(self) -> None:
        """Reset sharpness to default."""
        self.sharpness_slider.setValue(self.sharpness_slider.property("default_value"))
