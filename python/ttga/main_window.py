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

import json
import os
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui

from .constants import SAVED_CAMERAS_DIR_PATH
from .viewport_widget import ViewportWidget
from .add_camera_dialog import AddCameraDialog, BACKEND_MAP
from .camera_calibration import CalibrationView

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

        # Connect projector manager signals
        self.core.projector_manager.projector_added.connect(self._on_projector_added)
        self.core.projector_manager.projector_removed.connect(self._on_projector_removed)

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

        # Center-left: Camera tabs (Settings, Calibration, Snapshots)
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
        """Create the camera/projector list group with tabs.

        Returns:
            Group box containing camera and projector tabs.
        """
        group = QtWidgets.QGroupBox("Cameras & Projectors")
        group.setFixedWidth(350)
        layout = QtWidgets.QVBoxLayout(group)

        # Create tab widget
        tabs = QtWidgets.QTabWidget()

        # Cameras tab
        cameras_widget = QtWidgets.QWidget()
        cameras_layout = QtWidgets.QVBoxLayout(cameras_widget)

        self.camera_list = QtWidgets.QListWidget()
        self.camera_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.camera_list.itemSelectionChanged.connect(self._on_camera_selection_changed)
        cameras_layout.addWidget(self.camera_list)

        # Camera buttons in 2x2 grid
        camera_button_layout = QtWidgets.QGridLayout()

        self.add_camera_button = QtWidgets.QPushButton("Add")
        self.add_camera_button.clicked.connect(self._on_add_camera)
        camera_button_layout.addWidget(self.add_camera_button, 0, 0)

        self.delete_camera_button = QtWidgets.QPushButton("Delete")
        self.delete_camera_button.clicked.connect(self._on_delete_camera)
        self.delete_camera_button.setEnabled(False)
        camera_button_layout.addWidget(self.delete_camera_button, 0, 1)

        self.load_camera_button = QtWidgets.QPushButton("Load")
        self.load_camera_button.clicked.connect(self._on_load_camera)
        camera_button_layout.addWidget(self.load_camera_button, 1, 0)

        self.save_camera_button = QtWidgets.QPushButton("Save")
        self.save_camera_button.clicked.connect(self._on_save_camera)
        camera_button_layout.addWidget(self.save_camera_button, 1, 1)

        cameras_layout.addLayout(camera_button_layout)
        tabs.addTab(cameras_widget, "Cameras")

        # Projectors tab
        projectors_widget = QtWidgets.QWidget()
        projectors_layout = QtWidgets.QVBoxLayout(projectors_widget)

        self.projector_list = QtWidgets.QListWidget()
        self.projector_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.projector_list.itemSelectionChanged.connect(self._on_projector_selection_changed)
        self.projector_list.itemDoubleClicked.connect(self._on_projector_double_clicked)
        projectors_layout.addWidget(self.projector_list)

        # Projector buttons in 2x2 grid
        projector_button_layout = QtWidgets.QGridLayout()

        self.add_projector_button = QtWidgets.QPushButton("Add")
        self.add_projector_button.clicked.connect(self._on_add_projector)
        projector_button_layout.addWidget(self.add_projector_button, 0, 0)

        self.delete_projector_button = QtWidgets.QPushButton("Delete")
        self.delete_projector_button.clicked.connect(self._on_delete_projector)
        self.delete_projector_button.setEnabled(False)
        projector_button_layout.addWidget(self.delete_projector_button, 0, 1)

        self.load_projector_button = QtWidgets.QPushButton("Load")
        self.load_projector_button.clicked.connect(self._on_load_projector)
        projector_button_layout.addWidget(self.load_projector_button, 1, 0)

        self.save_projector_button = QtWidgets.QPushButton("Save")
        self.save_projector_button.clicked.connect(self._on_save_projector)
        projector_button_layout.addWidget(self.save_projector_button, 1, 1)

        projectors_layout.addLayout(projector_button_layout)
        tabs.addTab(projectors_widget, "Projectors")

        layout.addWidget(tabs)

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
        calibration_widget = self._create_camera_calibration_widget()
        tabs.addTab(calibration_widget, "Calibration")

        # Snapshots tab
        snapshots_widget = self._create_snapshots_widget()
        tabs.addTab(snapshots_widget, "Snapshots")

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
        self.focus_value_label = self._add_slider_to_grid(slider_grid, slider_row, "Focus:", self.focus_slider, self.focus_reset)
        slider_row += 1

        # Zoom
        self.zoom_slider, self.zoom_reset = self._create_slider_with_reset(
            "Zoom", 100, 500, 100, self._on_zoom_changed, self._on_zoom_reset
        )
        self.zoom_value_label = self._add_slider_to_grid(slider_grid, slider_row, "Zoom:", self.zoom_slider, self.zoom_reset)
        slider_row += 1

        # Brightness
        self.brightness_slider, self.brightness_reset = self._create_slider_with_reset(
            "Brightness", 0, 255, 128, self._on_brightness_changed, self._on_brightness_reset
        )
        self.brightness_value_label = self._add_slider_to_grid(slider_grid, slider_row, "Brightness:", self.brightness_slider, self.brightness_reset)
        slider_row += 1

        # Contrast
        self.contrast_slider, self.contrast_reset = self._create_slider_with_reset(
            "Contrast", 0, 255, 128, self._on_contrast_changed, self._on_contrast_reset
        )
        self.contrast_value_label = self._add_slider_to_grid(slider_grid, slider_row, "Contrast:", self.contrast_slider, self.contrast_reset)
        slider_row += 1

        # Gain
        self.gain_slider, self.gain_reset = self._create_slider_with_reset(
            "Gain", 0, 255, 128, self._on_gain_changed, self._on_gain_reset
        )
        self.gain_value_label = self._add_slider_to_grid(slider_grid, slider_row, "Gain:", self.gain_slider, self.gain_reset)
        slider_row += 1

        # Saturation
        self.saturation_slider, self.saturation_reset = self._create_slider_with_reset(
            "Saturation", 0, 255, 128, self._on_saturation_changed, self._on_saturation_reset
        )
        self.saturation_value_label = self._add_slider_to_grid(slider_grid, slider_row, "Saturation:", self.saturation_slider, self.saturation_reset)
        slider_row += 1

        # Sharpness
        self.sharpness_slider, self.sharpness_reset = self._create_slider_with_reset(
            "Sharpness", 0, 255, 128, self._on_sharpness_changed, self._on_sharpness_reset
        )
        self.sharpness_value_label = self._add_slider_to_grid(slider_grid, slider_row, "Sharpness:", self.sharpness_slider, self.sharpness_reset)
        slider_row += 1

        main_layout.addLayout(slider_grid)
        main_layout.addStretch()

        # Initially disable all controls
        self._set_camera_settings_enabled(False)

        return widget

    def _create_camera_calibration_widget(self) -> QtWidgets.QWidget:
        """Create the camera calibration widget.

        Returns:
            Widget containing calibration controls.
        """
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(widget)

        # Checkerboard settings
        checkerboard_group = QtWidgets.QGroupBox("Checkerboard Settings")
        checkerboard_layout = QtWidgets.QFormLayout(checkerboard_group)

        self.calib_squares_w_spinbox = QtWidgets.QSpinBox()
        self.calib_squares_w_spinbox.setRange(3, 50)
        self.calib_squares_w_spinbox.setValue(self.core.camera_calibration.number_of_squares_w)
        self.calib_squares_w_spinbox.valueChanged.connect(self._on_calib_squares_w_changed)
        checkerboard_layout.addRow("Squares Width:", self.calib_squares_w_spinbox)

        self.calib_squares_h_spinbox = QtWidgets.QSpinBox()
        self.calib_squares_h_spinbox.setRange(3, 50)
        self.calib_squares_h_spinbox.setValue(self.core.camera_calibration.number_of_squares_h)
        self.calib_squares_h_spinbox.valueChanged.connect(self._on_calib_squares_h_changed)
        checkerboard_layout.addRow("Squares Height:", self.calib_squares_h_spinbox)

        main_layout.addWidget(checkerboard_group)

        # Capture settings
        capture_group = QtWidgets.QGroupBox("Capture Settings")
        capture_layout = QtWidgets.QVBoxLayout(capture_group)

        # Frame capture delay
        delay_layout = QtWidgets.QHBoxLayout()
        delay_layout.addWidget(QtWidgets.QLabel("Frame Capture Delay:"))
        self.calib_delay_spinbox = QtWidgets.QSpinBox()
        self.calib_delay_spinbox.setRange(0, 10)
        self.calib_delay_spinbox.setValue(3)
        self.calib_delay_spinbox.setSuffix(" s")
        delay_layout.addWidget(self.calib_delay_spinbox)
        delay_layout.addStretch()
        capture_layout.addLayout(delay_layout)

        main_layout.addWidget(capture_group)

        # Capture buttons for each view
        views_group = QtWidgets.QGroupBox("Calibration Views")
        views_layout = QtWidgets.QVBoxLayout(views_group)

        # Store button references
        self.calib_view_buttons = {}

        # Create button for each view
        for view in [CalibrationView.TOP, CalibrationView.FRONT, CalibrationView.SIDE]:
            view_name = view.name.capitalize()

            # Create button with icon support
            button = QtWidgets.QPushButton(f"Capture {view_name}")
            button.setMinimumHeight(80)
            button.setIconSize(QtCore.QSize(64, 64))
            button.clicked.connect(lambda checked, v=view: self._on_capture_calibration_view(v))

            self.calib_view_buttons[view] = button
            views_layout.addWidget(button)

        main_layout.addWidget(views_group)

        # Calibration action buttons
        action_group = QtWidgets.QGroupBox("Calibration Actions")
        action_layout = QtWidgets.QVBoxLayout(action_group)

        # Calibrate and Uncalibrate buttons side by side
        buttons_layout = QtWidgets.QHBoxLayout()
        self.calibrate_button = QtWidgets.QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self._on_calibrate_camera)
        self.calibrate_button.setEnabled(False)
        buttons_layout.addWidget(self.calibrate_button)

        self.uncalibrate_button = QtWidgets.QPushButton("Uncalibrate")
        self.uncalibrate_button.clicked.connect(self._on_uncalibrate_camera)
        self.uncalibrate_button.setEnabled(False)
        buttons_layout.addWidget(self.uncalibrate_button)

        action_layout.addLayout(buttons_layout)

        # Mean reprojection error display
        error_layout = QtWidgets.QHBoxLayout()
        error_layout.addWidget(QtWidgets.QLabel("Mean Reprojection Error:"))
        self.calib_error_spinbox = QtWidgets.QDoubleSpinBox()
        self.calib_error_spinbox.setDecimals(5)
        self.calib_error_spinbox.setRange(-1.0, 999999.0)
        self.calib_error_spinbox.setValue(-1.0)
        self.calib_error_spinbox.setReadOnly(True)
        self.calib_error_spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        error_layout.addWidget(self.calib_error_spinbox)
        action_layout.addLayout(error_layout)

        main_layout.addWidget(action_group)
        main_layout.addStretch()

        # Calibration timer
        self.calib_capture_timer = QtCore.QTimer(self)
        self.calib_capture_timer.setSingleShot(True)
        self.calib_capture_timer.timeout.connect(self._on_calib_timer_timeout)
        self.calib_pending_view = None

        # Initially disable all controls
        self._set_calibration_enabled(False)

        return widget

    def _create_snapshots_widget(self) -> QtWidgets.QWidget:
        """Create the snapshots widget.

        Returns:
            Widget containing snapshots controls.
        """
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(widget)

        # Folder and file name settings
        naming_group = QtWidgets.QGroupBox("Snapshots Naming")
        naming_layout = QtWidgets.QFormLayout(naming_group)

        # Custom folder name
        self.snapshots_folder_edit = QtWidgets.QLineEdit()
        self.snapshots_folder_edit.setPlaceholderText("Leave empty for date (YYYY-MM-DD)")
        naming_layout.addRow("Custom folder name:", self.snapshots_folder_edit)

        # Custom file name
        self.snapshots_file_edit = QtWidgets.QLineEdit()
        self.snapshots_file_edit.setPlaceholderText("Leave empty for timestamp (YYYY-MM-DD_HH-mm-SS)")
        naming_layout.addRow("Custom file name:", self.snapshots_file_edit)

        main_layout.addWidget(naming_group)

        # Options
        options_group = QtWidgets.QGroupBox("Snapshots Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)

        # Add camera name checkbox
        self.snapshots_add_camera_checkbox = QtWidgets.QCheckBox("Add camera to file name")
        self.snapshots_add_camera_checkbox.setChecked(True)
        options_layout.addWidget(self.snapshots_add_camera_checkbox)

        # Add timestamp checkbox
        self.snapshots_add_timestamp_checkbox = QtWidgets.QCheckBox("Add timestamp to file name")
        self.snapshots_add_timestamp_checkbox.setChecked(True)
        options_layout.addWidget(self.snapshots_add_timestamp_checkbox)

        main_layout.addWidget(options_group)

        # Take snapshots button
        self.take_snapshots_button = QtWidgets.QPushButton("Take Snapshot(s)")
        self.take_snapshots_button.setMinimumHeight(40)
        self.take_snapshots_button.clicked.connect(self._on_take_snapshots)
        self.take_snapshots_button.setEnabled(False)
        main_layout.addWidget(self.take_snapshots_button)

        main_layout.addStretch()

        return widget

    def _add_slider_to_grid(self, grid: QtWidgets.QGridLayout, row: int, label_text: str,
                            slider: QtWidgets.QSlider, reset_button: QtWidgets.QPushButton) -> QtWidgets.QLabel:
        """Add a slider row to the grid layout.


        Args:
            grid: Grid layout to add to.
            row: Row number.
            label_text: Label text.
            slider: Slider widget.
            reset_button: Reset button widget.

        Returns:
            The value label widget for manual updates.
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

        return value_label

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
            List of undistorted frames from selected cameras.
        """
        frames = []
        selected_items = self.camera_list.selectedItems()

        for item in selected_items:
            camera_name = item.text()
            try:
                camera = self.core.camera_manager.get_camera(camera_name)
                frame = camera.get_undistorted_frame()
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

        # Get existing camera names
        existing_names = set(self.core.camera_manager.get_camera_names())

        # Open dialog
        dialog = AddCameraDialog(used_device_ids_by_backend, existing_names, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            camera_info = dialog.get_camera_info()
            if camera_info:
                name, backend, device_id, cam_info = camera_info

                try:
                    # Add camera through camera manager
                    camera = self.core.camera_manager.add_camera(name, backend, device_id, cam_info)

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

        # Enable snapshots button only if at least one camera is selected
        self.take_snapshots_button.setEnabled(has_selection)

        # Update camera settings based on selection
        if len(selected_items) == 1:
            # Single camera selected - enable and load settings
            camera = self._get_selected_camera()
            is_calibrated = camera is not None and camera.calibration_data is not None
            self._set_camera_settings_enabled(True, is_calibrated)
            self._load_camera_settings(selected_items[0].text())
            self._set_calibration_enabled(True)
        else:
            # Multiple or no cameras selected - disable settings
            self._set_camera_settings_enabled(False)
            self.device_id_edit.clear()
            self.device_id_edit.setPlaceholderText("No camera selected" if len(selected_items) == 0 else "Multiple cameras selected")
            self._set_calibration_enabled(False)

        # Clear calibration frames when camera selection changes
        self._clear_calibration_frames()

    @QtCore.Slot(int)
    def _on_viewport_fps_changed(self, fps: int) -> None:
        """Handle viewport FPS change.

        Args:
            fps: New frames per second value.
        """
        self.viewport.set_fps(fps)

    @QtCore.Slot()
    def _on_save_camera(self) -> None:
        """Handle save camera button click."""
        # Get all cameras data
        cameras_data = self.core.camera_manager.serialize_cameras()

        if not cameras_data:
            QtWidgets.QMessageBox.information(
                self,
                "No Cameras",
                "There are no cameras to save."
            )
            return

        # Ensure saved cameras directory exists
        os.makedirs(SAVED_CAMERAS_DIR_PATH, exist_ok=True)

        # Open file save dialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Cameras",
            SAVED_CAMERAS_DIR_PATH,
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(cameras_data, f, indent=2)

                QtWidgets.QMessageBox.information(
                    self,
                    "Cameras Saved",
                    f"Successfully saved {len(cameras_data)} camera(s) to {file_path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Saving Cameras",
                    f"Failed to save cameras: {str(e)}"
                )

    @QtCore.Slot()
    def _on_load_camera(self) -> None:
        """Handle load camera button click."""
        # Ensure saved cameras directory exists
        os.makedirs(SAVED_CAMERAS_DIR_PATH, exist_ok=True)

        # Open file open dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Cameras",
            SAVED_CAMERAS_DIR_PATH,
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            # Load JSON file
            with open(file_path, 'r') as f:
                cameras_data = json.load(f)

            if not cameras_data:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Cameras",
                    "The file contains no cameras to load."
                )
                return

            # Process each camera and find matching devices
            cameras_to_load = []
            missing_cameras = []
            used_device_ids = {}  # Track device IDs already assigned: {backend: set(device_ids)}

            for cam_data in cameras_data:
                backend = cam_data['backend']
                saved_device_id = cam_data['device_id']
                saved_camera_info = cam_data.get('camera_info')

                # Find matching device
                matched_device_id = self.core.camera_manager.find_matching_device(
                    backend, saved_camera_info, saved_device_id
                )

                if matched_device_id is None:
                    missing_cameras.append(cam_data['name'])
                    continue

                # Check if this device ID was already assigned to another camera in this load
                if backend not in used_device_ids:
                    used_device_ids[backend] = set()

                if matched_device_id in used_device_ids[backend]:
                    # Device already assigned to another camera in this load - skip as duplicate
                    missing_cameras.append(cam_data['name'])
                    continue

                # Mark this device ID as used
                used_device_ids[backend].add(matched_device_id)

                cameras_to_load.append({
                    'name': cam_data['name'],
                    'backend': backend,
                    'device_id': matched_device_id,
                    'camera_info': saved_camera_info,
                    'properties': cam_data.get('properties', {}),
                    'calibration_data': cam_data.get('calibration_data')
                })

            if not cameras_to_load:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Cameras to Load",
                    "No matching devices found for any cameras in the file."
                )
                return

            # Check for conflicts with existing cameras
            conflicts = []
            existing_cameras = self.core.camera_manager.get_camera_names()
            for cam in cameras_to_load:
                # Check name conflict
                if cam['name'] in existing_cameras:
                    conflicts.append(cam['name'])
                    continue

                # Check device conflict
                for existing_name in existing_cameras:
                    existing_cam = self.core.camera_manager.get_camera(existing_name)
                    if (
                        existing_cam.get_backend() == cam['backend'] and
                        existing_cam.get_device_id() == cam['device_id']
                    ):
                        conflicts.append(existing_name)

            # Remove duplicates from conflicts
            conflicts = list(set(conflicts))

            # Show confirmation dialog if there are conflicts or missing cameras
            if conflicts or missing_cameras:
                message_parts = []

                if conflicts:
                    conflict_list = "\n".join(f"  - {name}" for name in conflicts)
                    message_parts.append(f"The following cameras will be removed before loading:\n{conflict_list}")

                if missing_cameras:
                    missing_list = "\n".join(f"  - {name}" for name in missing_cameras)
                    message_parts.append(f"The following cameras could not be found and will be skipped:\n{missing_list}")

                message = "\n\n".join(message_parts) + "\n\nDo you want to continue?"

                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Load Cameras Confirmation",
                    message,
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )

                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    return

                # Remove conflicting cameras
                for conflict_name in conflicts:
                    try:
                        self.core.camera_manager.remove_camera(conflict_name)
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Error Removing Camera",
                            f"Failed to remove camera '{conflict_name}': {str(e)}"
                        )

            # Load cameras
            loaded_count = 0
            for cam in cameras_to_load:
                try:
                    # Add camera
                    camera = self.core.camera_manager.add_camera(
                        cam['name'],
                        cam['backend'],
                        cam['device_id'],
                        cam['camera_info']
                    )

                    # Apply properties
                    for prop_id, value in cam['properties'].items():
                        prop_id_int = int(prop_id)
                        # FOURCC, width, and height must be set as integers
                        if prop_id_int in [cv.CAP_PROP_FOURCC, cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT]:
                            camera.set_property(prop_id_int, float(int(value)))
                        else:
                            camera.set_property(prop_id_int, float(value))

                    # Deserialize calibration data if present (undistort_rectification is auto-created)
                    if cam.get('calibration_data') is not None:
                        from .camera_calibration import CameraCalibrationData
                        calib_dict = cam['calibration_data']
                        camera.calibration_data = CameraCalibrationData(
                            mtx=np.array(calib_dict['mtx']),
                            dist=np.array(calib_dict['dist']),
                            rvecs_list=[np.array(rvec) for rvec in calib_dict['rvecs_list']],
                            tvecs_list=[np.array(tvec) for tvec in calib_dict['tvecs_list']],
                            mean_reprojection_error=calib_dict['mean_reprojection_error'],
                            resolution=tuple(calib_dict['resolution'])
                        )

                    # Start camera
                    camera.start()
                    loaded_count += 1

                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Error Loading Camera",
                        f"Failed to load camera '{cam['name']}': {str(e)}"
                    )

            if loaded_count > 0:
                # Trigger selection change handler to refresh all UI controls
                self._on_camera_selection_changed()

                QtWidgets.QMessageBox.information(
                    self,
                    "Cameras Loaded",
                    f"Successfully loaded {loaded_count} camera(s)."
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading Cameras",
                f"Failed to load cameras: {str(e)}"
            )

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

    @QtCore.Slot(str)
    def _on_projector_added(self, projector_name: str) -> None:
        """Handle projector added signal.

        Args:
            projector_name: Name of the added projector.
        """
        from .projector_dialog import ProjectorDialog

        # Add to list widget
        projector = self.core.projector_manager.get_projector(projector_name)
        item = QtWidgets.QListWidgetItem(projector_name)
        # Set tooltip with resolution
        item.setToolTip(f"{projector.resolution[0]}x{projector.resolution[1]}")
        self.projector_list.addItem(item)

        # Create and show projector dialog automatically
        dialog = ProjectorDialog(projector.name, projector.resolution, self)
        projector.dialog = dialog
        dialog.show()

    @QtCore.Slot(str)
    def _on_projector_removed(self, projector_name: str) -> None:
        """Handle projector removed signal.

        Args:
            projector_name: Name of the removed projector.
        """
        # Remove from list widget
        items = self.projector_list.findItems(projector_name, QtCore.Qt.MatchFlag.MatchExactly)
        for item in items:
            row = self.projector_list.row(item)
            self.projector_list.takeItem(row)

    @QtCore.Slot()
    def _on_projector_selection_changed(self) -> None:
        """Handle projector list selection change."""
        selected_items = self.projector_list.selectedItems()
        has_selection = len(selected_items) > 0

        # Enable delete button only if at least one projector is selected
        self.delete_projector_button.setEnabled(has_selection)

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def _on_projector_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Handle projector double click to open dialog.

        Args:
            item: The clicked list item.
        """
        from .projector_dialog import ProjectorDialog

        projector_name = item.text()
        try:
            projector = self.core.projector_manager.get_projector(projector_name)

            # If dialog doesn't exist or was closed, create and show it
            if projector.dialog is None or not projector.dialog.isVisible():
                dialog = ProjectorDialog(projector.name, projector.resolution, self)
                projector.dialog = dialog
                dialog.show()

        except KeyError:
            QtWidgets.QMessageBox.warning(
                self,
                "Projector Not Found",
                f"Projector '{projector_name}' not found."
            )

    @QtCore.Slot()
    def _on_add_projector(self) -> None:
        """Handle add projector button click."""
        from .add_projector_dialog import AddProjectorDialog

        # Get existing projector names
        existing_names = [p.name for p in self.core.projector_manager.get_all_projectors()]

        # Show dialog
        dialog = AddProjectorDialog(existing_names, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            try:
                self.core.projector_manager.add_projector(
                    dialog.projector_name,
                    dialog.projector_resolution
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Adding Projector",
                    f"Failed to add projector: {str(e)}"
                )

    @QtCore.Slot()
    def _on_delete_projector(self) -> None:
        """Handle delete projector button click."""
        selected_items = self.projector_list.selectedItems()
        if not selected_items:
            return

        projector_names = [item.text() for item in selected_items]

        # Confirm deletion
        if len(projector_names) == 1:
            message = f"Are you sure you want to delete projector '{projector_names[0]}'?"
        else:
            message = f"Are you sure you want to delete {len(projector_names)} projectors?"

        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Projector(s)",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # Remove projectors
            for projector_name in projector_names:
                try:
                    self.core.projector_manager.remove_projector(projector_name)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Error Deleting Projector",
                        f"Failed to delete projector '{projector_name}': {str(e)}"
                    )

    @QtCore.Slot()
    def _on_save_projector(self) -> None:
        """Handle save projector button click."""
        import json
        import os
        from . import constants

        # Get all projectors data
        projectors_data = self.core.projector_manager.serialize_projectors()

        if not projectors_data:
            QtWidgets.QMessageBox.information(
                self,
                "No Projectors",
                "No projectors to save."
            )
            return

        # Ensure saved projectors directory exists
        os.makedirs(constants.SAVED_PROJECTORS_DIR_PATH, exist_ok=True)

        # Show file save dialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Projectors",
            constants.SAVED_PROJECTORS_DIR_PATH,
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with open(file_path, 'w') as f:
                    json.dump(projectors_data, f, indent=2)

                QtWidgets.QMessageBox.information(
                    self,
                    "Projectors Saved",
                    f"Successfully saved {len(projectors_data)} projector(s)."
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Saving Projectors",
                    f"Failed to save projectors: {str(e)}"
                )

    @QtCore.Slot()
    def _on_load_projector(self) -> None:
        """Handle load projector button click."""
        import json
        import os
        from . import constants
        from .projector import Projector

        # Ensure saved projectors directory exists
        os.makedirs(constants.SAVED_PROJECTORS_DIR_PATH, exist_ok=True)

        # Show file open dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Projectors",
            constants.SAVED_PROJECTORS_DIR_PATH,
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                projectors_data = json.load(f)

            if not projectors_data:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Projectors",
                    "No projectors found in file."
                )
                return

            # Check for conflicts
            conflicts = []
            for proj_data in projectors_data:
                if self.core.projector_manager.projector_exists(proj_data['name']):
                    conflicts.append(proj_data['name'])

            # Ask user if they want to replace conflicting projectors
            if conflicts:
                message = "The following projectors already exist:\n\n"
                message += "\n".join(conflicts)
                message += "\n\nDo you want to replace them?"

                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Replace Projectors?",
                    message,
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )

                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    return

                # Remove conflicting projectors
                for conflict_name in conflicts:
                    try:
                        self.core.projector_manager.remove_projector(conflict_name)
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Error Removing Projector",
                            f"Failed to remove projector '{conflict_name}': {str(e)}"
                        )

            # Load projectors
            loaded_count = 0
            for proj_data in projectors_data:
                try:
                    projector = Projector.from_dict(proj_data)
                    self.core.projector_manager.add_projector(
                        projector.name,
                        projector.resolution
                    )
                    loaded_count += 1
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Error Loading Projector",
                        f"Failed to load projector '{proj_data.get('name', 'Unknown')}': {str(e)}"
                    )

            if loaded_count > 0:
                QtWidgets.QMessageBox.information(
                    self,
                    "Projectors Loaded",
                    f"Successfully loaded {loaded_count} projector(s)."
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading Projectors",
                f"Failed to load projectors: {str(e)}"
            )

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

    def _set_camera_settings_enabled(self, enabled: bool, is_calibrated: bool = False) -> None:
        """Enable or disable camera settings controls.

        Args:
            enabled: True to enable, False to disable.
            is_calibrated: True if camera is calibrated (disables resolution/focus/zoom).
        """
        self.fourcc_combo.setEnabled(enabled)
        # Resolution, focus, and zoom are disabled when camera is calibrated
        self.resolution_combo.setEnabled(enabled and not is_calibrated)
        self.exposure_spinbox.setEnabled(enabled)
        self.focus_slider.setEnabled(enabled and not is_calibrated)
        self.focus_reset.setEnabled(enabled and not is_calibrated)
        self.zoom_slider.setEnabled(enabled and not is_calibrated)
        self.zoom_reset.setEnabled(enabled and not is_calibrated)
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
            focus_val = int(camera.get_property(cv.CAP_PROP_FOCUS))
            self.focus_slider.setValue(focus_val)
            self.focus_value_label.setText(str(focus_val))

            zoom_val = int(camera.get_property(cv.CAP_PROP_ZOOM))
            self.zoom_slider.setValue(zoom_val)
            self.zoom_value_label.setText(str(zoom_val))

            brightness_val = int(camera.get_property(cv.CAP_PROP_BRIGHTNESS))
            self.brightness_slider.setValue(brightness_val)
            self.brightness_value_label.setText(str(brightness_val))

            contrast_val = int(camera.get_property(cv.CAP_PROP_CONTRAST))
            self.contrast_slider.setValue(contrast_val)
            self.contrast_value_label.setText(str(contrast_val))

            gain_val = int(camera.get_property(cv.CAP_PROP_GAIN))
            self.gain_slider.setValue(gain_val)
            self.gain_value_label.setText(str(gain_val))

            saturation_val = int(camera.get_property(cv.CAP_PROP_SATURATION))
            self.saturation_slider.setValue(saturation_val)
            self.saturation_value_label.setText(str(saturation_val))

            sharpness_val = int(camera.get_property(cv.CAP_PROP_SHARPNESS))
            self.sharpness_slider.setValue(sharpness_val)
            self.sharpness_value_label.setText(str(sharpness_val))

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

    # Calibration handlers
    def _set_calibration_enabled(self, enabled: bool) -> None:
        """Enable or disable calibration controls based on camera state.

        Args:
            enabled: True if camera is selected, False otherwise.
        """
        camera = self._get_selected_camera()
        is_calibrated = camera is not None and camera.calibration_data is not None

        # Checkerboard settings and capture buttons disabled if calibrated
        self.calib_squares_w_spinbox.setEnabled(enabled and not is_calibrated)
        self.calib_squares_h_spinbox.setEnabled(enabled and not is_calibrated)
        self.calib_delay_spinbox.setEnabled(enabled and not is_calibrated)
        for button in self.calib_view_buttons.values():
            button.setEnabled(enabled and not is_calibrated)

        # Update calibration action buttons
        self._update_calibration_buttons()

    def _clear_calibration_frames(self) -> None:
        """Clear all calibration frames and button images."""
        self.core.camera_calibration.clear_frames()

        # Clear button icons
        for view, button in self.calib_view_buttons.items():
            button.setIcon(QtGui.QIcon())
            button.setText(f"Capture {view.name.capitalize()}")

    def _update_calibration_buttons(self) -> None:
        """Update calibration button states based on current state."""
        camera = self._get_selected_camera()

        if camera is None:
            self.calibrate_button.setEnabled(False)
            self.uncalibrate_button.setEnabled(False)
            self.calib_error_spinbox.setValue(-1.0)
            return

        is_calibrated = camera.calibration_data is not None

        # Calibrate button enabled when all 3 views have frames and not calibrated
        all_frames_captured = all(
            self.core.camera_calibration.get_calibration_frame(view) is not None
            for view in [CalibrationView.TOP, CalibrationView.FRONT, CalibrationView.SIDE]
        )
        self.calibrate_button.setEnabled(all_frames_captured and not is_calibrated)

        # Uncalibrate button enabled when camera is calibrated
        self.uncalibrate_button.setEnabled(is_calibrated)

        # Update error display
        if is_calibrated:
            self.calib_error_spinbox.setValue(camera.calibration_data.mean_reprojection_error)
        else:
            self.calib_error_spinbox.setValue(-1.0)

    @QtCore.Slot(int)
    def _on_calib_squares_w_changed(self, value: int) -> None:
        """Handle checkerboard width change."""
        self.core.camera_calibration.number_of_squares_w = value
        self._clear_calibration_frames()
        self._update_calibration_buttons()

    @QtCore.Slot(int)
    def _on_calib_squares_h_changed(self, value: int) -> None:
        """Handle checkerboard height change."""
        self.core.camera_calibration.number_of_squares_h = value
        self._clear_calibration_frames()
        self._update_calibration_buttons()

    @QtCore.Slot(CalibrationView)
    def _on_capture_calibration_view(self, view: CalibrationView) -> None:
        """Handle capture button click for a specific view.

        Args:
            view: The calibration view to capture.
        """
        # Get selected camera
        camera = self._get_selected_camera()
        if not camera:
            QtWidgets.QMessageBox.warning(
                self,
                "No Camera Selected",
                "Please select a camera to capture calibration frames."
            )
            return

        # Get delay
        delay_seconds = self.calib_delay_spinbox.value()

        # Store pending view
        self.calib_pending_view = view

        # Disable buttons during capture
        self._set_calibration_enabled(False)

        if delay_seconds > 0:
            # Update button text to show countdown
            button = self.calib_view_buttons[view]
            button.setText(f"Capturing in {delay_seconds}s...")

            # Start timer
            self.calib_capture_timer.start(delay_seconds * 1000)
        else:
            # Capture immediately
            self._capture_calibration_frame(view)

    @QtCore.Slot()
    def _on_calib_timer_timeout(self) -> None:
        """Handle calibration capture timer timeout."""
        if self.calib_pending_view is not None:
            self._capture_calibration_frame(self.calib_pending_view)
            self.calib_pending_view = None

    def _capture_calibration_frame(self, view: CalibrationView) -> None:
        """Capture a calibration frame for the specified view.

        Args:
            view: The calibration view to capture.
        """
        # Get selected camera
        camera = self._get_selected_camera()
        if not camera:
            self._set_calibration_enabled(True)
            return

        # Get current frame
        frame = camera.get_frame()
        if frame is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Frame Available",
                "No frame available from camera. Please try again."
            )
            self._set_calibration_enabled(True)
            return

        # Try to create calibration frame
        calib_frame = self.core.camera_calibration.make_calibration_frame(frame)

        if calib_frame is None:
            # Clear stored frame for this view
            self.core.camera_calibration.set_calibration_frame(view, None)

            # Clear button icon and reset text
            button = self.calib_view_buttons[view]
            button.setIcon(QtGui.QIcon())
            button.setText(f"Capture {view.name.capitalize()}")

            QtWidgets.QMessageBox.warning(
                self,
                "Checkerboard Not Found",
                f"Could not detect checkerboard pattern in captured frame.\n\n"
                f"Please ensure:\n"
                f"- The checkerboard is fully visible\n"
                f"- The checkerboard has {self.core.camera_calibration.number_of_squares_w}x"
                f"{self.core.camera_calibration.number_of_squares_h} squares\n"
                f"- The image is well-lit and in focus"
            )

            self._set_calibration_enabled(True)
            return

        # Store calibration frame
        self.core.camera_calibration.set_calibration_frame(view, calib_frame)

        # Draw corners on the image for visualization
        gray_with_corners = calib_frame.image.copy()
        checkerboard_dim = (
            self.core.camera_calibration.number_of_squares_w - 1,
            self.core.camera_calibration.number_of_squares_h - 1
        )
        cv.drawChessboardCorners(gray_with_corners, checkerboard_dim, calib_frame.corners, True)

        # Convert to RGB for Qt
        rgb_image = cv.cvtColor(gray_with_corners, cv.COLOR_GRAY2RGB)

        # Create QPixmap from image
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(
            rgb_image.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888
        )
        pixmap = QtGui.QPixmap.fromImage(q_image)

        # Scale pixmap to button size (64x64)
        scaled_pixmap = pixmap.scaled(
            64, 64,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )

        # Set button icon and update text
        button = self.calib_view_buttons[view]
        button.setIcon(QtGui.QIcon(scaled_pixmap))
        button.setText(f"{view.name.capitalize()} ")

        # Re-enable controls
        self._set_calibration_enabled(True)

        # Update calibration button states
        self._update_calibration_buttons()

    @QtCore.Slot()
    def _on_calibrate_camera(self) -> None:
        """Handle calibrate button click."""
        camera = self._get_selected_camera()
        if not camera:
            return

        # Get current frame resolution
        frame = camera.get_frame()
        if frame is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Frame Available",
                "Cannot determine frame resolution. Please ensure camera is active."
            )
            return

        resolution = (frame.shape[1], frame.shape[0])  # (width, height)

        # Call calibrate_camera
        calib_data = self.core.camera_calibration.calibrate_camera()

        if calib_data is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration Failed",
                "Camera calibration failed. Please ensure all calibration frames are valid."
            )
            return

        # Add resolution to calibration data
        calib_data.resolution = resolution

        # Store calibration data in camera (undistort_rectification is auto-created)
        camera.calibration_data = calib_data

        # Update UI - disable focus/zoom since camera is now calibrated
        self._set_camera_settings_enabled(True, is_calibrated=True)
        self._set_calibration_enabled(True)
        self._update_calibration_buttons()

        QtWidgets.QMessageBox.information(
            self,
            "Calibration Successful",
            f"Camera calibrated successfully!\n\n"
            f"Mean Reprojection Error: {calib_data.mean_reprojection_error:.5f}\n"
            f"Resolution: {resolution[0]}x{resolution[1]}"
        )

    @QtCore.Slot()
    def _on_uncalibrate_camera(self) -> None:
        """Handle uncalibrate button click."""
        camera = self._get_selected_camera()
        if not camera:
            return

        # Remove calibration data (undistort_rectification is inside it)
        camera.calibration_data = None

        # Clear calibration frames
        self._clear_calibration_frames()

        # Update UI - re-enable focus/zoom since camera is no longer calibrated
        self._set_camera_settings_enabled(True, is_calibrated=False)
        self._set_calibration_enabled(True)
        self._update_calibration_buttons()

    @QtCore.Slot()
    def _on_take_snapshots(self) -> None:
        """Handle take snapshots button click."""
        from datetime import datetime
        import cv2 as cv
        from . import constants

        # Get selected cameras
        selected_items = self.camera_list.selectedItems()
        if not selected_items:
            return

        # Get current timestamp
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Determine folder name
        folder_name = self.snapshots_folder_edit.text().strip()
        if not folder_name:
            folder_name = date_str

        # Determine base file name
        custom_file_name = self.snapshots_file_edit.text().strip()
        if not custom_file_name:
            base_file_name = timestamp_str
        else:
            base_file_name = custom_file_name

        # Process each selected camera
        saved_count = 0
        for item in selected_items:
            camera_name = item.text()
            try:
                camera = self.core.camera_manager.get_camera(camera_name)
                frame = camera.get_undistorted_frame()

                if frame is None:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "No Frame Available",
                        f"No frame available from camera '{camera_name}'. Skipping."
                    )
                    continue

                # Build file name with suffixes
                file_name = base_file_name

                # Add camera name suffix if checked
                if self.snapshots_add_camera_checkbox.isChecked():
                    file_name += f"__{camera_name}"

                # Add timestamp suffix if custom file name was provided and checkbox is checked
                if custom_file_name and self.snapshots_add_timestamp_checkbox.isChecked():
                    file_name += f"__{timestamp_str}"

                # Build full file path
                file_path = constants.SAVED_SNAPSHOT_FILE_PATH_TEMPLATE.format(
                    folder_name=folder_name,
                    file_name=file_name
                )

                # Ensure parent directory exists
                import os
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Save the frame as PNG
                cv.imwrite(file_path, frame)
                saved_count += 1

            except KeyError:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Camera Not Found",
                    f"Camera '{camera_name}' was removed. Skipping."
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error Saving Snapshots",
                    f"Failed to save snapshot for camera '{camera_name}': {str(e)}"
                )

        # Show success message
        if saved_count > 0:
            QtWidgets.QMessageBox.information(
                self,
                "Snapshots Saved",
                f"Successfully saved {saved_count} snapshot(s) to:\n{os.path.dirname(file_path)}"
            )
