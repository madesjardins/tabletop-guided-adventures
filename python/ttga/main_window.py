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

        # Connect zone manager signals
        self.core.zone_manager.zone_added.connect(self._on_zone_added)
        self.core.zone_manager.zone_removed.connect(self._on_zone_removed)

        # Connect speech recognition signals
        self.core.speech_partial_result.connect(self._on_speech_partial_result)
        self.core.speech_final_result.connect(self._on_speech_final_result)

        # Connect game signals
        self.core.game_loaded.connect(self._on_game_loaded)
        self.core.game_unloaded.connect(self._on_game_unloaded)

        # Initialize zone settings UI state
        self._on_zone_selection_changed()

        # Set up viewport callbacks and start timer
        self.viewport.set_get_frames_callback(
            self._get_selected_camera_frames,
            self._get_selected_camera_ids,
            self._get_selected_camera_names
        )
        self.viewport.set_zone_manager(self.core.zone_manager)
        self.viewport.set_main_core(self.core)
        self.viewport.vertex_updated.connect(self._on_viewport_vertex_updated)
        self.viewport.start()

    def _setup_menu_bar(self) -> None:
        """Set up the menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        # Save master configuration action
        save_master_action = QtGui.QAction("&Save Master Configuration", self)
        save_master_action.triggered.connect(self._on_save_master_configuration)
        file_menu.addAction(save_master_action)

        # Load master configuration action
        load_master_action = QtGui.QAction("&Load Master Configuration", self)
        load_master_action.triggered.connect(self._on_load_master_configuration)
        file_menu.addAction(load_master_action)

        file_menu.addSeparator()

        # Quit action
        quit_action = QtGui.QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Game menu
        game_menu = menu_bar.addMenu("&Game")

        # Load game action
        load_game_action = QtGui.QAction("&Load Game...", self)
        load_game_action.triggered.connect(self._on_load_game)
        game_menu.addAction(load_game_action)

        # Unload game action
        self.unload_game_action = QtGui.QAction("&Unload Game", self)
        self.unload_game_action.triggered.connect(self._on_unload_game)
        self.unload_game_action.setEnabled(False)  # Disabled until a game is loaded
        game_menu.addAction(self.unload_game_action)

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
        """Create the camera/projector/zone list group with tabs.

        Returns:
            Group box containing camera, projector, and zone tabs.
        """
        group = QtWidgets.QGroupBox("Cameras, Projectors & Zones")
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
        self.save_camera_button.setEnabled(False)
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
        self.save_projector_button.setEnabled(False)
        projector_button_layout.addWidget(self.save_projector_button, 1, 1)

        projectors_layout.addLayout(projector_button_layout)
        tabs.addTab(projectors_widget, "Projectors")

        # Zones tab
        zones_widget = QtWidgets.QWidget()
        zones_layout = QtWidgets.QVBoxLayout(zones_widget)

        self.zone_list = QtWidgets.QListWidget()
        self.zone_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.zone_list.itemSelectionChanged.connect(self._on_zone_selection_changed)
        zones_layout.addWidget(self.zone_list)

        # Zone buttons in 2x2 grid
        zone_button_layout = QtWidgets.QGridLayout()

        self.add_zone_button = QtWidgets.QPushButton("Add")
        self.add_zone_button.clicked.connect(self._on_add_zone)
        zone_button_layout.addWidget(self.add_zone_button, 0, 0)

        self.delete_zone_button = QtWidgets.QPushButton("Delete")
        self.delete_zone_button.clicked.connect(self._on_delete_zone)
        self.delete_zone_button.setEnabled(False)
        zone_button_layout.addWidget(self.delete_zone_button, 0, 1)

        self.load_zone_button = QtWidgets.QPushButton("Load")
        self.load_zone_button.clicked.connect(self._on_load_zone)
        zone_button_layout.addWidget(self.load_zone_button, 1, 0)

        self.save_zone_button = QtWidgets.QPushButton("Save")
        self.save_zone_button.clicked.connect(self._on_save_zone)
        self.save_zone_button.setEnabled(False)
        zone_button_layout.addWidget(self.save_zone_button, 1, 1)

        zones_layout.addLayout(zone_button_layout)
        tabs.addTab(zones_widget, "Zones")

        layout.addWidget(tabs)

        return group

    def _create_settings_tabs(self) -> QtWidgets.QTabWidget:
        """Create the settings tab widget.

        Returns:
            Tab widget containing settings tabs.
        """
        tabs = QtWidgets.QTabWidget()

        # Zone Settings tab
        zone_settings_widget = self._create_zone_settings_widget()
        tabs.addTab(zone_settings_widget, "Zone Settings")

        # Speech Recognition tab
        speech_recognition_widget = self._create_speech_recognition_widget()
        tabs.addTab(speech_recognition_widget, "Speech Recognition")

        # Narrator tab
        narrator_widget = self._create_narrator_widget()
        tabs.addTab(narrator_widget, "Narrator")

        # Advanced tab
        advanced_widget = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(advanced_widget)

        # Refresh rates
        refresh_form = QtWidgets.QFormLayout()

        self.viewports_fps_spinbox = QtWidgets.QSpinBox()
        self.viewports_fps_spinbox.setRange(5, 60)
        self.viewports_fps_spinbox.setValue(30)
        self.viewports_fps_spinbox.setSuffix(" fps")
        self.viewports_fps_spinbox.valueChanged.connect(self._on_viewports_fps_changed)
        refresh_form.addRow("Viewports Refresh Rate:", self.viewports_fps_spinbox)

        self.projectors_fps_spinbox = QtWidgets.QSpinBox()
        self.projectors_fps_spinbox.setRange(5, 60)
        self.projectors_fps_spinbox.setValue(15)
        self.projectors_fps_spinbox.setSuffix(" fps")
        self.projectors_fps_spinbox.valueChanged.connect(self._on_projectors_fps_changed)
        refresh_form.addRow("Projectors Refresh Rate:", self.projectors_fps_spinbox)

        self.qr_code_fps_spinbox = QtWidgets.QSpinBox()
        self.qr_code_fps_spinbox.setRange(1, 30)
        self.qr_code_fps_spinbox.setValue(5)
        self.qr_code_fps_spinbox.setSuffix(" fps")
        self.qr_code_fps_spinbox.valueChanged.connect(self._on_qr_code_fps_changed)
        refresh_form.addRow("QR Code Refresh Rate:", self.qr_code_fps_spinbox)

        advanced_layout.addLayout(refresh_form)

        advanced_layout.addStretch()
        tabs.addTab(advanced_widget, "Advanced")

        return tabs

    def _create_zone_settings_widget(self) -> QtWidgets.QWidget:
        """Create the zone settings widget.

        Returns:
            Widget containing zone settings controls.
        """
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)

        # Left section: Zone properties
        left_section = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_section)

        # Zone properties group
        properties_group = QtWidgets.QGroupBox("Zone Properties")
        properties_layout = QtWidgets.QFormLayout(properties_group)

        # Zone name (read-only display)
        self.zone_name_label = QtWidgets.QLabel("No zone selected")
        properties_layout.addRow("Name:", self.zone_name_label)

        # Dimensions
        dimensions_layout = QtWidgets.QHBoxLayout()
        self.zone_width_spinbox = QtWidgets.QDoubleSpinBox()
        self.zone_width_spinbox.setRange(0.0001, 99999.9999)
        self.zone_width_spinbox.setDecimals(4)
        self.zone_width_spinbox.setValue(34.0)
        self.zone_width_spinbox.valueChanged.connect(self._on_zone_width_changed)
        dimensions_layout.addWidget(self.zone_width_spinbox)

        dimensions_layout.addWidget(QtWidgets.QLabel("×"))

        self.zone_height_spinbox = QtWidgets.QDoubleSpinBox()
        self.zone_height_spinbox.setRange(0.0001, 99999.9999)
        self.zone_height_spinbox.setDecimals(4)
        self.zone_height_spinbox.setValue(22.0)
        self.zone_height_spinbox.valueChanged.connect(self._on_zone_height_changed)
        dimensions_layout.addWidget(self.zone_height_spinbox)

        properties_layout.addRow("Dimensions:", dimensions_layout)

        # Unit
        self.zone_unit_combo = QtWidgets.QComboBox()
        self.zone_unit_combo.addItems(["mm", "cm", "in", "px"])
        self.zone_unit_combo.setCurrentText("in")
        self.zone_unit_combo.currentTextChanged.connect(self._on_zone_unit_changed)
        properties_layout.addRow("Unit:", self.zone_unit_combo)

        # Resolution (pixels per unit)
        self.zone_resolution_spinbox = QtWidgets.QSpinBox()
        self.zone_resolution_spinbox.setRange(1, 99999)
        self.zone_resolution_spinbox.setValue(50)
        self.zone_resolution_spinbox.valueChanged.connect(self._on_zone_resolution_changed)
        properties_layout.addRow("Pixels per unit:", self.zone_resolution_spinbox)

        # Enable camera mapping checkbox
        self.zone_camera_enabled_checkbox = QtWidgets.QCheckBox()
        self.zone_camera_enabled_checkbox.stateChanged.connect(self._on_zone_camera_enabled_changed)
        properties_layout.addRow("Enable camera mapping:", self.zone_camera_enabled_checkbox)

        # Enable projector mapping checkbox
        self.zone_projector_enabled_checkbox = QtWidgets.QCheckBox()
        self.zone_projector_enabled_checkbox.stateChanged.connect(self._on_zone_projector_enabled_changed)
        properties_layout.addRow("Enable projector mapping:", self.zone_projector_enabled_checkbox)

        # Draw locked borders checkbox (applies to both camera and projector)
        self.zone_draw_locked_borders_checkbox = QtWidgets.QCheckBox()
        self.zone_draw_locked_borders_checkbox.setChecked(True)
        self.zone_draw_locked_borders_checkbox.stateChanged.connect(self._on_zone_draw_locked_borders_changed)
        properties_layout.addRow("Draw locked borders:", self.zone_draw_locked_borders_checkbox)

        left_layout.addWidget(properties_group)
        left_layout.addStretch()
        main_layout.addWidget(left_section)

        # Middle section: Camera mapping
        self.camera_section = QtWidgets.QWidget()
        camera_layout = QtWidgets.QVBoxLayout(self.camera_section)

        # Camera mapping group
        self.zone_camera_mapping_group = QtWidgets.QGroupBox("Camera Mapping")
        camera_mapping_layout = QtWidgets.QVBoxLayout(self.zone_camera_mapping_group)

        # Camera selection
        camera_selection_layout = QtWidgets.QHBoxLayout()
        camera_selection_layout.addWidget(QtWidgets.QLabel("Camera:"))
        self.zone_camera_combo = QtWidgets.QComboBox()
        self.zone_camera_combo.currentTextChanged.connect(self._on_zone_camera_changed)
        camera_selection_layout.addWidget(self.zone_camera_combo)
        camera_mapping_layout.addLayout(camera_selection_layout)

        # Vertices in a form layout
        vertices_form = QtWidgets.QFormLayout()

        # P0: Cyan
        p0_layout = QtWidgets.QHBoxLayout()
        self.zone_camera_p0_x = QtWidgets.QSpinBox()
        self.zone_camera_p0_x.setRange(0, 99999)
        self.zone_camera_p0_x.setValue(128)
        self.zone_camera_p0_x.valueChanged.connect(self._on_zone_camera_vertex_changed)
        p0_layout.addWidget(self.zone_camera_p0_x)
        self.zone_camera_p0_y = QtWidgets.QSpinBox()
        self.zone_camera_p0_y.setRange(0, 99999)
        self.zone_camera_p0_y.setValue(128)
        self.zone_camera_p0_y.valueChanged.connect(self._on_zone_camera_vertex_changed)
        p0_layout.addWidget(self.zone_camera_p0_y)
        vertices_form.addRow("P0 (Cyan):", p0_layout)

        # P1: Magenta
        p1_layout = QtWidgets.QHBoxLayout()
        self.zone_camera_p1_x = QtWidgets.QSpinBox()
        self.zone_camera_p1_x.setRange(0, 99999)
        self.zone_camera_p1_x.setValue(384)
        self.zone_camera_p1_x.valueChanged.connect(self._on_zone_camera_vertex_changed)
        p1_layout.addWidget(self.zone_camera_p1_x)
        self.zone_camera_p1_y = QtWidgets.QSpinBox()
        self.zone_camera_p1_y.setRange(0, 99999)
        self.zone_camera_p1_y.setValue(128)
        self.zone_camera_p1_y.valueChanged.connect(self._on_zone_camera_vertex_changed)
        p1_layout.addWidget(self.zone_camera_p1_y)
        vertices_form.addRow("P1 (Magenta):", p1_layout)

        # P2: Yellow
        p2_layout = QtWidgets.QHBoxLayout()
        self.zone_camera_p2_x = QtWidgets.QSpinBox()
        self.zone_camera_p2_x.setRange(0, 99999)
        self.zone_camera_p2_x.setValue(384)
        self.zone_camera_p2_x.valueChanged.connect(self._on_zone_camera_vertex_changed)
        p2_layout.addWidget(self.zone_camera_p2_x)
        self.zone_camera_p2_y = QtWidgets.QSpinBox()
        self.zone_camera_p2_y.setRange(0, 99999)
        self.zone_camera_p2_y.setValue(256)
        self.zone_camera_p2_y.valueChanged.connect(self._on_zone_camera_vertex_changed)
        p2_layout.addWidget(self.zone_camera_p2_y)
        vertices_form.addRow("P2 (Yellow):", p2_layout)

        # P3: White
        p3_layout = QtWidgets.QHBoxLayout()
        self.zone_camera_p3_x = QtWidgets.QSpinBox()
        self.zone_camera_p3_x.setRange(0, 99999)
        self.zone_camera_p3_x.setValue(128)
        self.zone_camera_p3_x.valueChanged.connect(self._on_zone_camera_vertex_changed)
        p3_layout.addWidget(self.zone_camera_p3_x)
        self.zone_camera_p3_y = QtWidgets.QSpinBox()
        self.zone_camera_p3_y.setRange(0, 99999)
        self.zone_camera_p3_y.setValue(256)
        self.zone_camera_p3_y.valueChanged.connect(self._on_zone_camera_vertex_changed)
        p3_layout.addWidget(self.zone_camera_p3_y)
        vertices_form.addRow("P3 (White):", p3_layout)

        camera_mapping_layout.addLayout(vertices_form)

        # Lock vertices checkbox
        self.zone_camera_lock_vertices_checkbox = QtWidgets.QCheckBox("Lock vertices")
        self.zone_camera_lock_vertices_checkbox.stateChanged.connect(self._on_zone_camera_lock_vertices_changed)
        camera_mapping_layout.addWidget(self.zone_camera_lock_vertices_checkbox)

        camera_layout.addWidget(self.zone_camera_mapping_group)
        camera_layout.addStretch()
        # Store reference to the camera layout for spacer management
        self.camera_section_layout = camera_layout
        main_layout.addWidget(self.camera_section)

        # Right section: Projector mapping
        self.projector_section = QtWidgets.QWidget()
        projector_layout = QtWidgets.QVBoxLayout(self.projector_section)

        # Projector mapping group
        self.zone_projector_mapping_group = QtWidgets.QGroupBox("Projector Mapping")
        projector_mapping_layout = QtWidgets.QVBoxLayout(self.zone_projector_mapping_group)

        # Projector selection
        projector_selection_layout = QtWidgets.QHBoxLayout()
        projector_selection_layout.addWidget(QtWidgets.QLabel("Projector:"))
        self.zone_projector_combo = QtWidgets.QComboBox()
        self.zone_projector_combo.currentTextChanged.connect(self._on_zone_projector_changed)
        projector_selection_layout.addWidget(self.zone_projector_combo)
        projector_mapping_layout.addLayout(projector_selection_layout)

        # Vertices
        projector_vertices_form = QtWidgets.QFormLayout()

        # P0: Cyan
        proj_p0_layout = QtWidgets.QHBoxLayout()
        self.zone_projector_p0_x = QtWidgets.QSpinBox()
        self.zone_projector_p0_x.setRange(0, 99999)
        self.zone_projector_p0_x.setValue(128)
        self.zone_projector_p0_x.valueChanged.connect(self._on_zone_projector_vertex_changed)
        proj_p0_layout.addWidget(self.zone_projector_p0_x)
        self.zone_projector_p0_y = QtWidgets.QSpinBox()
        self.zone_projector_p0_y.setRange(0, 99999)
        self.zone_projector_p0_y.setValue(128)
        self.zone_projector_p0_y.valueChanged.connect(self._on_zone_projector_vertex_changed)
        proj_p0_layout.addWidget(self.zone_projector_p0_y)
        projector_vertices_form.addRow("P0 (Cyan):", proj_p0_layout)

        # P1: Magenta
        proj_p1_layout = QtWidgets.QHBoxLayout()
        self.zone_projector_p1_x = QtWidgets.QSpinBox()
        self.zone_projector_p1_x.setRange(0, 99999)
        self.zone_projector_p1_x.setValue(384)
        self.zone_projector_p1_x.valueChanged.connect(self._on_zone_projector_vertex_changed)
        proj_p1_layout.addWidget(self.zone_projector_p1_x)
        self.zone_projector_p1_y = QtWidgets.QSpinBox()
        self.zone_projector_p1_y.setRange(0, 99999)
        self.zone_projector_p1_y.setValue(128)
        self.zone_projector_p1_y.valueChanged.connect(self._on_zone_projector_vertex_changed)
        proj_p1_layout.addWidget(self.zone_projector_p1_y)
        projector_vertices_form.addRow("P1 (Magenta):", proj_p1_layout)

        # P2: Yellow
        proj_p2_layout = QtWidgets.QHBoxLayout()
        self.zone_projector_p2_x = QtWidgets.QSpinBox()
        self.zone_projector_p2_x.setRange(0, 99999)
        self.zone_projector_p2_x.setValue(384)
        self.zone_projector_p2_x.valueChanged.connect(self._on_zone_projector_vertex_changed)
        proj_p2_layout.addWidget(self.zone_projector_p2_x)
        self.zone_projector_p2_y = QtWidgets.QSpinBox()
        self.zone_projector_p2_y.setRange(0, 99999)
        self.zone_projector_p2_y.setValue(256)
        self.zone_projector_p2_y.valueChanged.connect(self._on_zone_projector_vertex_changed)
        proj_p2_layout.addWidget(self.zone_projector_p2_y)
        projector_vertices_form.addRow("P2 (Yellow):", proj_p2_layout)

        # P3: White
        proj_p3_layout = QtWidgets.QHBoxLayout()
        self.zone_projector_p3_x = QtWidgets.QSpinBox()
        self.zone_projector_p3_x.setRange(0, 99999)
        self.zone_projector_p3_x.setValue(128)
        self.zone_projector_p3_x.valueChanged.connect(self._on_zone_projector_vertex_changed)
        proj_p3_layout.addWidget(self.zone_projector_p3_x)
        self.zone_projector_p3_y = QtWidgets.QSpinBox()
        self.zone_projector_p3_y.setRange(0, 99999)
        self.zone_projector_p3_y.setValue(256)
        self.zone_projector_p3_y.valueChanged.connect(self._on_zone_projector_vertex_changed)
        proj_p3_layout.addWidget(self.zone_projector_p3_y)
        projector_vertices_form.addRow("P3 (White):", proj_p3_layout)

        projector_mapping_layout.addLayout(projector_vertices_form)

        # Lock vertices checkbox
        self.zone_projector_lock_vertices_checkbox = QtWidgets.QCheckBox("Lock vertices")
        self.zone_projector_lock_vertices_checkbox.stateChanged.connect(self._on_zone_projector_lock_vertices_changed)
        projector_mapping_layout.addWidget(self.zone_projector_lock_vertices_checkbox)

        projector_layout.addWidget(self.zone_projector_mapping_group)
        projector_layout.addStretch()
        # Store reference to the projector layout for spacer management
        self.projector_section_layout = projector_layout
        main_layout.addWidget(self.projector_section)

        # Right section: Calibration buttons
        right_section = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_section)

        # Calibrate button
        self.zone_calibrate_button = QtWidgets.QPushButton("Calibrate")
        self.zone_calibrate_button.clicked.connect(self._on_zone_calibrate)
        right_layout.addWidget(self.zone_calibrate_button)

        # Uncalibrate button
        self.zone_uncalibrate_button = QtWidgets.QPushButton("Uncalibrate")
        self.zone_uncalibrate_button.clicked.connect(self._on_zone_uncalibrate)
        right_layout.addWidget(self.zone_uncalibrate_button)

        right_layout.addStretch()
        main_layout.addWidget(right_section)

        # Initially hide mapping groups
        self.zone_camera_mapping_group.setVisible(False)
        self.zone_projector_mapping_group.setVisible(False)

        # Create message label for when no zone or multiple zones selected
        self.zone_settings_message_label = QtWidgets.QLabel(
            "Please select one and only one Zone to enable modifications."
        )
        self.zone_settings_message_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.zone_settings_message_label.setStyleSheet("font-size: 14px; color: gray;")

        # Add message label to main layout with stretch to keep it left-aligned
        message_container = QtWidgets.QHBoxLayout()
        message_container.addWidget(self.zone_settings_message_label)
        message_container.addStretch()
        main_layout.addLayout(message_container)
        self.zone_settings_message_label.setVisible(False)

        return widget

    def _create_speech_recognition_widget(self) -> QtWidgets.QWidget:
        """Create the speech recognition widget.

        Returns:
            Widget containing speech recognition controls.
        """
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(widget)

        # Configuration group (fixed width on left)
        config_group = QtWidgets.QGroupBox("Configuration")
        config_group.setMaximumWidth(350)
        config_layout = QtWidgets.QFormLayout(config_group)

        # Vosk model combo box
        self.speech_model_combo = QtWidgets.QComboBox()
        config_layout.addRow("Vosk Model:", self.speech_model_combo)

        # Audio device combo box
        self.speech_device_combo = QtWidgets.QComboBox()
        config_layout.addRow("Audio Device:", self.speech_device_combo)

        # Similarity threshold spinbox
        self.speech_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.speech_threshold_spinbox.setRange(0.0, 1.0)
        self.speech_threshold_spinbox.setSingleStep(0.05)
        self.speech_threshold_spinbox.setValue(0.7)
        self.speech_threshold_spinbox.setDecimals(2)
        config_layout.addRow("Similarity Threshold:", self.speech_threshold_spinbox)

        main_layout.addWidget(config_group)

        # Results group (takes remaining space on right)
        results_group = QtWidgets.QGroupBox("Recognition Results")
        results_layout = QtWidgets.QFormLayout(results_group)

        # Partial result line edit
        self.speech_partial_result_edit = QtWidgets.QLineEdit()
        self.speech_partial_result_edit.setReadOnly(True)
        self.speech_partial_result_edit.setPlaceholderText("Partial recognition appears here...")
        results_layout.addRow("Partial:", self.speech_partial_result_edit)

        # Final result line edit
        self.speech_final_result_edit = QtWidgets.QLineEdit()
        self.speech_final_result_edit.setReadOnly(True)
        self.speech_final_result_edit.setPlaceholderText("Final recognition appears here...")
        results_layout.addRow("Final:", self.speech_final_result_edit)

        main_layout.addWidget(results_group)

        # Populate vosk models
        self._populate_vosk_models()

        # Populate audio devices
        self._populate_audio_devices()

        # Connect signals
        self.speech_model_combo.currentIndexChanged.connect(self._on_speech_model_changed)
        self.speech_device_combo.currentIndexChanged.connect(self._on_speech_device_changed)
        self.speech_threshold_spinbox.valueChanged.connect(self._on_speech_threshold_changed)

        # Initialize MainCore with default values if combo boxes have items
        if self.speech_model_combo.count() > 0:
            model_path = self.speech_model_combo.itemData(0)
            if model_path:
                self.core.speech_model_path = model_path

        if self.speech_device_combo.count() > 0:
            device_index = self.speech_device_combo.itemData(0)
            if device_index is not None:
                self.core.speech_device_index = device_index

        return widget

    def _populate_vosk_models(self) -> None:
        """Populate the vosk model combo box with available models."""
        import os

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        vosk_models_path = os.path.join(root_dir, "vosk_models")

        self.speech_model_combo.clear()

        if os.path.exists(vosk_models_path):
            models = [d for d in os.listdir(vosk_models_path)
                      if os.path.isdir(os.path.join(vosk_models_path, d))]
            for model in sorted(models):
                model_path = os.path.join(vosk_models_path, model)
                self.speech_model_combo.addItem(model, model_path)

    def _populate_audio_devices(self) -> None:
        """Populate the audio device combo box with available devices."""
        from .speech_recognition import get_audio_input_devices

        self.speech_device_combo.clear()

        devices = get_audio_input_devices()
        for device in devices:
            device_label = f"{device['index']}: {device['name']}"
            self.speech_device_combo.addItem(device_label, device['index'])

    def _create_narrator_widget(self) -> QtWidgets.QWidget:
        """Create the narrator widget for TTS and audio playback.

        Returns:
            Widget containing narrator controls.
        """
        widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(widget)

        # Voice Configuration Group
        voice_config_group = QtWidgets.QGroupBox("Voice Configuration")
        voice_config_layout = QtWidgets.QFormLayout(voice_config_group)

        # Voice model combo box
        self.narrator_voice_combo = QtWidgets.QComboBox()
        voice_config_layout.addRow("Voice Model:", self.narrator_voice_combo)

        # Output device combo box
        self.narrator_output_device_combo = QtWidgets.QComboBox()
        voice_config_layout.addRow("Output Device:", self.narrator_output_device_combo)

        main_layout.addWidget(voice_config_group)

        # Channel Volume Controls Group
        channels_group = QtWidgets.QGroupBox("Channel Volume Controls")
        channels_layout = QtWidgets.QGridLayout(channels_group)

        # Voice channel
        channels_layout.addWidget(QtWidgets.QLabel("Voice:"), 0, 0)
        self.narrator_voice_volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.narrator_voice_volume_slider.setRange(0, 100)
        self.narrator_voice_volume_slider.setValue(100)
        self.narrator_voice_volume_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.narrator_voice_volume_slider.setTickInterval(10)
        channels_layout.addWidget(self.narrator_voice_volume_slider, 0, 1)
        self.narrator_voice_volume_label = QtWidgets.QLabel("100%")
        self.narrator_voice_volume_label.setMinimumWidth(50)
        channels_layout.addWidget(self.narrator_voice_volume_label, 0, 2)
        self.narrator_voice_test_button = QtWidgets.QPushButton("Test")
        self.narrator_voice_test_button.setMaximumWidth(80)
        channels_layout.addWidget(self.narrator_voice_test_button, 0, 3)

        # Effect channel
        channels_layout.addWidget(QtWidgets.QLabel("Effect:"), 1, 0)
        self.narrator_effect_volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.narrator_effect_volume_slider.setRange(0, 100)
        self.narrator_effect_volume_slider.setValue(100)
        self.narrator_effect_volume_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.narrator_effect_volume_slider.setTickInterval(10)
        channels_layout.addWidget(self.narrator_effect_volume_slider, 1, 1)
        self.narrator_effect_volume_label = QtWidgets.QLabel("100%")
        self.narrator_effect_volume_label.setMinimumWidth(50)
        channels_layout.addWidget(self.narrator_effect_volume_label, 1, 2)
        self.narrator_effect_test_button = QtWidgets.QPushButton("Test")
        self.narrator_effect_test_button.setMaximumWidth(80)
        channels_layout.addWidget(self.narrator_effect_test_button, 1, 3)

        # Music channel
        channels_layout.addWidget(QtWidgets.QLabel("Music:"), 2, 0)
        self.narrator_music_volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.narrator_music_volume_slider.setRange(0, 100)
        self.narrator_music_volume_slider.setValue(100)
        self.narrator_music_volume_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.narrator_music_volume_slider.setTickInterval(10)
        channels_layout.addWidget(self.narrator_music_volume_slider, 2, 1)
        self.narrator_music_volume_label = QtWidgets.QLabel("100%")
        self.narrator_music_volume_label.setMinimumWidth(50)
        channels_layout.addWidget(self.narrator_music_volume_label, 2, 2)
        self.narrator_music_test_button = QtWidgets.QPushButton("Test")
        self.narrator_music_test_button.setMaximumWidth(80)
        channels_layout.addWidget(self.narrator_music_test_button, 2, 3)

        main_layout.addWidget(channels_group)
        main_layout.addStretch()

        # Populate voice models and output devices
        self._populate_narrator_voices()
        self._populate_narrator_output_devices()

        # Connect signals
        self.narrator_voice_combo.currentIndexChanged.connect(self._on_narrator_voice_changed)
        self.narrator_output_device_combo.currentIndexChanged.connect(self._on_narrator_output_device_changed)

        self.narrator_voice_volume_slider.valueChanged.connect(self._on_narrator_voice_volume_changed)
        self.narrator_effect_volume_slider.valueChanged.connect(self._on_narrator_effect_volume_changed)
        self.narrator_music_volume_slider.valueChanged.connect(self._on_narrator_music_volume_changed)

        self.narrator_voice_test_button.clicked.connect(self._on_narrator_voice_test)
        self.narrator_effect_test_button.clicked.connect(self._on_narrator_effect_test)
        self.narrator_music_test_button.clicked.connect(self._on_narrator_music_test)

        # Initialize narrator with default voice if available
        if self.narrator_voice_combo.count() > 0:
            voice_path = self.narrator_voice_combo.itemData(0)
            if voice_path:
                try:
                    self.core.narrator.load_voice_model(voice_path)
                except Exception as e:
                    print(f"Failed to load default voice model: {e}")

        # Test text rotation counters
        self.narrator_voice_test_index = 0
        self.narrator_effect_test_index = 0
        self.narrator_music_test_index = 0

        return widget

    def _populate_narrator_voices(self) -> None:
        """Populate the narrator voice combo box with available Piper models."""
        import os
        from .narrator import find_available_voices

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        piper_voices_path = os.path.join(root_dir, "piper_voices")

        self.narrator_voice_combo.clear()

        voices = find_available_voices(piper_voices_path)
        for voice_path in sorted(voices):
            voice_name = os.path.basename(voice_path)
            self.narrator_voice_combo.addItem(voice_name, voice_path)

    def _populate_narrator_output_devices(self) -> None:
        """Populate the narrator output device combo box."""
        self.narrator_output_device_combo.clear()

        # pygame.mixer only supports the system default output device
        self.narrator_output_device_combo.addItem("System Default Output Device", None)

    def _set_section_spacer_visible(self, layout: QtWidgets.QVBoxLayout, visible: bool) -> None:
        """Show or hide the stretch spacer in a section layout.

        Args:
            layout: The layout containing the spacer.
            visible: True to show spacer (expand), False to hide spacer (fixed size).
        """
        if not layout or not hasattr(self, 'camera_section_layout'):
            return

        # Get the last item in the layout (which should be the spacer)
        item_count = layout.count()
        if item_count > 0:
            spacer_item = layout.itemAt(item_count - 1)
            if spacer_item and spacer_item.spacerItem():
                if visible:
                    spacer_item.changeSize(0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
                else:
                    spacer_item.changeSize(0, 0, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
                layout.invalidate()

    def _set_zone_settings_visible(self, visible: bool) -> None:
        """Show or hide zone settings controls.

        Args:
            visible: True to show zone settings, False to show message.
        """
        # Show/hide the message label
        self.zone_settings_message_label.setVisible(not visible)

        # Show/hide all zone settings controls
        self.zone_name_label.parent().setVisible(visible)
        self.zone_camera_enabled_checkbox.parent().setVisible(visible)
        self.zone_projector_enabled_checkbox.parent().setVisible(visible)

        # Show/hide calibration buttons
        self.zone_calibrate_button.setVisible(visible)
        self.zone_uncalibrate_button.setVisible(visible)

        # Hide mapping sections when no zone is selected
        if not visible:
            self.camera_section.setVisible(False)
            self.projector_section.setVisible(False)

    def _load_zone_into_ui(self, zone) -> None:
        """Load zone data into the UI controls.

        Args:
            zone: Zone object to load.
        """
        # Block signals to prevent triggering change handlers
        self.zone_name_label.setText(zone.name)

        self.zone_width_spinbox.blockSignals(True)
        self.zone_width_spinbox.setValue(zone.width)
        self.zone_width_spinbox.blockSignals(False)

        self.zone_height_spinbox.blockSignals(True)
        self.zone_height_spinbox.setValue(zone.height)
        self.zone_height_spinbox.blockSignals(False)

        self.zone_unit_combo.blockSignals(True)
        self.zone_unit_combo.setCurrentText(zone.unit)
        self.zone_unit_combo.blockSignals(False)

        self.zone_resolution_spinbox.blockSignals(True)
        self.zone_resolution_spinbox.setValue(zone.resolution)
        self.zone_resolution_spinbox.setEnabled(zone.unit != 'px')
        self.zone_resolution_spinbox.blockSignals(False)

        # Load camera mapping
        self.zone_camera_enabled_checkbox.blockSignals(True)
        if zone.camera_mapping:
            # Set checkbox based on enabled flag (default to True if None)
            enabled = zone.camera_mapping.enabled if zone.camera_mapping.enabled is not None else True
            self.zone_camera_enabled_checkbox.setChecked(enabled)
            self.camera_section.setVisible(enabled)
            self.zone_camera_mapping_group.setVisible(True)

            self._update_camera_combo()
            self.zone_camera_combo.blockSignals(True)
            self.zone_camera_combo.setCurrentText(zone.camera_mapping.camera_name)
            self.zone_camera_combo.blockSignals(False)

            vertices = zone.camera_mapping.vertices
            self.zone_camera_p0_x.blockSignals(True)
            self.zone_camera_p0_y.blockSignals(True)
            self.zone_camera_p1_x.blockSignals(True)
            self.zone_camera_p1_y.blockSignals(True)
            self.zone_camera_p2_x.blockSignals(True)
            self.zone_camera_p2_y.blockSignals(True)
            self.zone_camera_p3_x.blockSignals(True)
            self.zone_camera_p3_y.blockSignals(True)

            self.zone_camera_p0_x.setValue(vertices[0][0])
            self.zone_camera_p0_y.setValue(vertices[0][1])
            self.zone_camera_p1_x.setValue(vertices[1][0])
            self.zone_camera_p1_y.setValue(vertices[1][1])
            self.zone_camera_p2_x.setValue(vertices[2][0])
            self.zone_camera_p2_y.setValue(vertices[2][1])
            self.zone_camera_p3_x.setValue(vertices[3][0])
            self.zone_camera_p3_y.setValue(vertices[3][1])

            self.zone_camera_p0_x.blockSignals(False)
            self.zone_camera_p0_y.blockSignals(False)
            self.zone_camera_p1_x.blockSignals(False)
            self.zone_camera_p1_y.blockSignals(False)
            self.zone_camera_p2_x.blockSignals(False)
            self.zone_camera_p2_y.blockSignals(False)
            self.zone_camera_p3_x.blockSignals(False)
            self.zone_camera_p3_y.blockSignals(False)

            # Load camera lock vertices from mapping
            self.zone_camera_lock_vertices_checkbox.blockSignals(True)
            self.zone_camera_lock_vertices_checkbox.setChecked(zone.camera_mapping.lock_vertices)
            self.zone_camera_lock_vertices_checkbox.blockSignals(False)
            self._update_camera_vertices_enabled()
        else:
            self.zone_camera_enabled_checkbox.setChecked(False)
            self.camera_section.setVisible(False)

        self.zone_camera_enabled_checkbox.blockSignals(False)

        # Load projector mapping
        self.zone_projector_enabled_checkbox.blockSignals(True)
        if zone.projector_mapping:
            # Set checkbox based on enabled flag (default to True if None)
            enabled = zone.projector_mapping.enabled if zone.projector_mapping.enabled is not None else True
            self.zone_projector_enabled_checkbox.setChecked(enabled)
            self.projector_section.setVisible(enabled)
            self.zone_projector_mapping_group.setVisible(True)

            self._update_projector_combo()
            self.zone_projector_combo.blockSignals(True)
            self.zone_projector_combo.setCurrentText(zone.projector_mapping.projector_name)
            self.zone_projector_combo.blockSignals(False)

            vertices = zone.projector_mapping.vertices
            self.zone_projector_p0_x.blockSignals(True)
            self.zone_projector_p0_y.blockSignals(True)
            self.zone_projector_p1_x.blockSignals(True)
            self.zone_projector_p1_y.blockSignals(True)
            self.zone_projector_p2_x.blockSignals(True)
            self.zone_projector_p2_y.blockSignals(True)
            self.zone_projector_p3_x.blockSignals(True)
            self.zone_projector_p3_y.blockSignals(True)

            self.zone_projector_p0_x.setValue(vertices[0][0])
            self.zone_projector_p0_y.setValue(vertices[0][1])
            self.zone_projector_p1_x.setValue(vertices[1][0])
            self.zone_projector_p1_y.setValue(vertices[1][1])
            self.zone_projector_p2_x.setValue(vertices[2][0])
            self.zone_projector_p2_y.setValue(vertices[2][1])
            self.zone_projector_p3_x.setValue(vertices[3][0])
            self.zone_projector_p3_y.setValue(vertices[3][1])

            self.zone_projector_p0_x.blockSignals(False)
            self.zone_projector_p0_y.blockSignals(False)
            self.zone_projector_p1_x.blockSignals(False)
            self.zone_projector_p1_y.blockSignals(False)
            self.zone_projector_p2_x.blockSignals(False)
            self.zone_projector_p2_y.blockSignals(False)
            self.zone_projector_p3_x.blockSignals(False)
            self.zone_projector_p3_y.blockSignals(False)

            # Load projector lock vertices from mapping
            self.zone_projector_lock_vertices_checkbox.blockSignals(True)
            self.zone_projector_lock_vertices_checkbox.setChecked(zone.projector_mapping.lock_vertices)
            self.zone_projector_lock_vertices_checkbox.blockSignals(False)
            self._update_projector_vertices_enabled()
        else:
            self.zone_projector_enabled_checkbox.setChecked(False)
            self.projector_section.setVisible(False)

        self.zone_projector_enabled_checkbox.blockSignals(False)

        # Load draw locked borders from zone (applies to both mappings)
        self.zone_draw_locked_borders_checkbox.blockSignals(True)
        self.zone_draw_locked_borders_checkbox.setChecked(zone.draw_locked_borders)
        self.zone_draw_locked_borders_checkbox.blockSignals(False)

        # Update UI state based on calibration status
        self._update_zone_ui_state()

    def _update_camera_combo(self) -> None:
        """Update camera combo box with available cameras."""
        current = self.zone_camera_combo.currentText()
        self.zone_camera_combo.clear()

        camera_names = self.core.camera_manager.get_camera_names()
        for camera_name in camera_names:
            self.zone_camera_combo.addItem(camera_name)

        # Restore selection if still valid
        if current:
            index = self.zone_camera_combo.findText(current)
            if index >= 0:
                self.zone_camera_combo.setCurrentIndex(index)

        # If no camera was selected before and combo has items, update zone with first camera
        zone = self._get_selected_zone()
        if zone and zone.camera_mapping:
            if not zone.camera_mapping.camera_name and self.zone_camera_combo.count() > 0:
                zone.camera_mapping.camera_name = self.zone_camera_combo.currentText()

    def _update_projector_combo(self) -> None:
        """Update projector combo box with available projectors."""
        current = self.zone_projector_combo.currentText()
        self.zone_projector_combo.clear()

        projectors = self.core.projector_manager.get_all_projectors()
        for projector in projectors:
            self.zone_projector_combo.addItem(projector.name)

        # Restore selection if still valid
        if current:
            index = self.zone_projector_combo.findText(current)
            if index >= 0:
                self.zone_projector_combo.setCurrentIndex(index)

        # If no projector was selected before and combo has items, update zone with first projector
        zone = self._get_selected_zone()
        if zone and zone.projector_mapping:
            if not zone.projector_mapping.projector_name and self.zone_projector_combo.count() > 0:
                zone.projector_mapping.projector_name = self.zone_projector_combo.currentText()

    def _update_camera_vertices_enabled(self) -> None:
        """Enable/disable camera vertex spinboxes based on lock state."""
        locked = self.zone_camera_lock_vertices_checkbox.isChecked()
        self.zone_camera_p0_x.setEnabled(not locked)
        self.zone_camera_p0_y.setEnabled(not locked)
        self.zone_camera_p1_x.setEnabled(not locked)
        self.zone_camera_p1_y.setEnabled(not locked)
        self.zone_camera_p2_x.setEnabled(not locked)
        self.zone_camera_p2_y.setEnabled(not locked)
        self.zone_camera_p3_x.setEnabled(not locked)
        self.zone_camera_p3_y.setEnabled(not locked)

    def _update_projector_vertices_enabled(self) -> None:
        """Enable/disable projector vertex spinboxes based on lock state."""
        locked = self.zone_projector_lock_vertices_checkbox.isChecked()
        self.zone_projector_p0_x.setEnabled(not locked)
        self.zone_projector_p0_y.setEnabled(not locked)
        self.zone_projector_p1_x.setEnabled(not locked)
        self.zone_projector_p1_y.setEnabled(not locked)
        self.zone_projector_p2_x.setEnabled(not locked)
        self.zone_projector_p2_y.setEnabled(not locked)
        self.zone_projector_p3_x.setEnabled(not locked)
        self.zone_projector_p3_y.setEnabled(not locked)

    def _update_zone_ui_state(self) -> None:
        """Update UI state based on zone calibration status."""
        zone = self._get_selected_zone()
        if not zone:
            return

        is_calibrated = zone.is_calibrated()

        # Check if any mapping is enabled (handle None as False)
        has_enabled_mapping = bool(
            (zone.camera_mapping and zone.camera_mapping.enabled is True) or
            (zone.projector_mapping and zone.projector_mapping.enabled is True)
        )

        # Hide calibration buttons if no mappings are enabled
        self.zone_calibrate_button.setVisible(has_enabled_mapping)
        self.zone_uncalibrate_button.setVisible(has_enabled_mapping)

        # Update button states (only matters if visible)
        self.zone_calibrate_button.setEnabled(not is_calibrated)
        self.zone_uncalibrate_button.setEnabled(is_calibrated)

        # When calibrated, disable all controls except draw_locked_borders
        self.zone_width_spinbox.setEnabled(not is_calibrated)
        self.zone_height_spinbox.setEnabled(not is_calibrated)
        self.zone_unit_combo.setEnabled(not is_calibrated)
        self.zone_resolution_spinbox.setEnabled(not is_calibrated)
        self.zone_camera_enabled_checkbox.setEnabled(not is_calibrated)
        self.zone_projector_enabled_checkbox.setEnabled(not is_calibrated)

        # Camera mapping controls
        if zone.camera_mapping and zone.camera_mapping.enabled:
            self.zone_camera_combo.setEnabled(not is_calibrated)
            self.zone_camera_lock_vertices_checkbox.setEnabled(not is_calibrated)

            # Update lock state if calibrated
            if is_calibrated and zone.camera_mapping.is_calibrated:
                self.zone_camera_lock_vertices_checkbox.setChecked(True)

            # Vertex spinboxes disabled when calibrated or locked
            # UNLESS game allows locked corner adjustment
            allows_corner_adjustment = self.core.allows_locked_corner_adjustment()
            if allows_corner_adjustment and is_calibrated:
                # Allow adjustment even when locked/calibrated
                locked_or_calibrated = zone.camera_mapping.lock_vertices and not is_calibrated
            else:
                locked_or_calibrated = is_calibrated or zone.camera_mapping.lock_vertices

            self.zone_camera_p0_x.setEnabled(not locked_or_calibrated)
            self.zone_camera_p0_y.setEnabled(not locked_or_calibrated)
            self.zone_camera_p1_x.setEnabled(not locked_or_calibrated)
            self.zone_camera_p1_y.setEnabled(not locked_or_calibrated)
            self.zone_camera_p2_x.setEnabled(not locked_or_calibrated)
            self.zone_camera_p2_y.setEnabled(not locked_or_calibrated)
            self.zone_camera_p3_x.setEnabled(not locked_or_calibrated)
            self.zone_camera_p3_y.setEnabled(not locked_or_calibrated)

        # Projector mapping controls
        if zone.projector_mapping and zone.projector_mapping.enabled:
            self.zone_projector_combo.setEnabled(not is_calibrated)
            self.zone_projector_lock_vertices_checkbox.setEnabled(not is_calibrated)

            # Update lock state if calibrated
            if is_calibrated and zone.projector_mapping.is_calibrated:
                self.zone_projector_lock_vertices_checkbox.setChecked(True)

            # Vertex spinboxes disabled when calibrated or locked
            # UNLESS game allows locked corner adjustment
            allows_corner_adjustment = self.core.allows_locked_corner_adjustment()
            if allows_corner_adjustment and is_calibrated:
                # Allow adjustment even when locked/calibrated
                locked_or_calibrated = zone.projector_mapping.lock_vertices and not is_calibrated
            else:
                locked_or_calibrated = is_calibrated or zone.projector_mapping.lock_vertices

            self.zone_projector_p0_x.setEnabled(not locked_or_calibrated)
            self.zone_projector_p0_y.setEnabled(not locked_or_calibrated)
            self.zone_projector_p1_x.setEnabled(not locked_or_calibrated)
            self.zone_projector_p1_y.setEnabled(not locked_or_calibrated)
            self.zone_projector_p2_x.setEnabled(not locked_or_calibrated)
            self.zone_projector_p2_y.setEnabled(not locked_or_calibrated)
            self.zone_projector_p3_x.setEnabled(not locked_or_calibrated)
            self.zone_projector_p3_y.setEnabled(not locked_or_calibrated)

        # Draw locked borders is always enabled
        self.zone_draw_locked_borders_checkbox.setEnabled(True)

    def _get_selected_zone(self):
        """Get the currently selected zone if exactly one is selected."""
        selected_items = self.zone_list.selectedItems()
        if len(selected_items) == 1:
            zone_name = selected_items[0].text()
            try:
                return self.core.zone_manager.get_zone(zone_name)
            except KeyError:
                return None
        return None

    def _validate_zone_references(self, zone) -> tuple[bool, str]:
        """Validate that a zone's camera and projector references exist.

        Args:
            zone: Zone object to validate.

        Returns:
            Tuple of (is_valid, error_message). error_message is empty if valid.
        """
        missing_items = []

        # Check camera mapping
        if zone.camera_mapping and zone.camera_mapping.camera_name:
            try:
                self.core.camera_manager.get_camera(zone.camera_mapping.camera_name)
            except KeyError:
                missing_items.append(f"Camera '{zone.camera_mapping.camera_name}'")

        # Check projector mapping
        if zone.projector_mapping and zone.projector_mapping.projector_name:
            try:
                self.core.projector_manager.get_projector(zone.projector_mapping.projector_name)
            except KeyError:
                missing_items.append(f"Projector '{zone.projector_mapping.projector_name}'")

        if missing_items:
            error_msg = f"Zone '{zone.name}' references missing device(s):\n" + "\n".join(f"  - {item}" for item in missing_items)
            return (False, error_msg)

        return (True, "")

    @QtCore.Slot(float)
    def _on_zone_width_changed(self, value: float) -> None:
        """Handle zone width change."""
        zone = self._get_selected_zone()
        if zone:
            zone.width = value

    @QtCore.Slot(float)
    def _on_zone_height_changed(self, value: float) -> None:
        """Handle zone height change."""
        zone = self._get_selected_zone()
        if zone:
            zone.height = value

    @QtCore.Slot(str)
    def _on_zone_unit_changed(self, unit: str) -> None:
        """Handle zone unit change."""
        zone = self._get_selected_zone()
        if zone:
            zone.unit = unit
            # Disable resolution spinbox if unit is pixels
            self.zone_resolution_spinbox.setEnabled(unit != 'px')
            if unit == 'px':
                zone.resolution = 1
                self.zone_resolution_spinbox.blockSignals(True)
                self.zone_resolution_spinbox.setValue(1)
                self.zone_resolution_spinbox.blockSignals(False)

    @QtCore.Slot(int)
    def _on_zone_resolution_changed(self, value: int) -> None:
        """Handle zone resolution change."""
        zone = self._get_selected_zone()
        if zone:
            zone.resolution = value

    @QtCore.Slot(int)
    def _on_zone_camera_enabled_changed(self, state: int) -> None:
        """Handle camera mapping enable/disable."""
        from .zone import CameraMapping

        zone = self._get_selected_zone()
        if not zone:
            return

        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.camera_section.setVisible(enabled)
        if enabled:
            self.zone_camera_mapping_group.setVisible(True)

        if enabled:
            # Refresh combo box with available cameras
            self._update_camera_combo()

            if not zone.camera_mapping:
                # Create new camera mapping with default values
                camera_name = self.zone_camera_combo.currentText() if self.zone_camera_combo.count() > 0 else ""
                zone.camera_mapping = CameraMapping(camera_name=camera_name, enabled=True)
                # Load the new mapping into UI
                self._load_zone_into_ui(zone)
            else:
                # Enable existing mapping
                zone.camera_mapping.enabled = True
        else:
            # Disable mapping but keep the data
            if zone.camera_mapping:
                zone.camera_mapping.enabled = False

    @QtCore.Slot(str)
    def _on_zone_camera_changed(self, camera_name: str) -> None:
        """Handle camera selection change."""
        zone = self._get_selected_zone()
        if zone and zone.camera_mapping and camera_name:
            zone.camera_mapping.camera_name = camera_name

    @QtCore.Slot()
    def _on_zone_camera_vertex_changed(self) -> None:
        """Handle camera vertex coordinate change."""
        zone = self._get_selected_zone()
        if zone and zone.camera_mapping:
            zone.camera_mapping.vertices = [
                (self.zone_camera_p0_x.value(), self.zone_camera_p0_y.value()),
                (self.zone_camera_p1_x.value(), self.zone_camera_p1_y.value()),
                (self.zone_camera_p2_x.value(), self.zone_camera_p2_y.value()),
                (self.zone_camera_p3_x.value(), self.zone_camera_p3_y.value())
            ]
            zone.camera_mapping.invalidate_overlay()

            # If game allows locked corner adjustment and zone is calibrated,
            # automatically uncalibrate and recalibrate
            if self.core.allows_locked_corner_adjustment() and zone.is_calibrated():
                zone.uncalibrate()
                try:
                    zone.calibrate()
                except Exception as e:
                    print(f"Error recalibrating zone {zone.name}: {e}")

    @QtCore.Slot(str)
    def _on_viewport_vertex_updated(self, zone_name: str) -> None:
        """Handle vertex update from viewport dragging.

        Args:
            zone_name: Name of the zone whose vertex was updated.
        """
        # Only update UI if this is the currently selected zone
        selected_zone = self._get_selected_zone()
        if selected_zone and selected_zone.name == zone_name:
            # Block signals to prevent triggering _on_zone_camera_vertex_changed
            self.zone_camera_p0_x.blockSignals(True)
            self.zone_camera_p0_y.blockSignals(True)
            self.zone_camera_p1_x.blockSignals(True)
            self.zone_camera_p1_y.blockSignals(True)
            self.zone_camera_p2_x.blockSignals(True)
            self.zone_camera_p2_y.blockSignals(True)
            self.zone_camera_p3_x.blockSignals(True)
            self.zone_camera_p3_y.blockSignals(True)

            # Update spinbox values from zone
            vertices = selected_zone.camera_mapping.vertices
            self.zone_camera_p0_x.setValue(vertices[0][0])
            self.zone_camera_p0_y.setValue(vertices[0][1])
            self.zone_camera_p1_x.setValue(vertices[1][0])
            self.zone_camera_p1_y.setValue(vertices[1][1])
            self.zone_camera_p2_x.setValue(vertices[2][0])
            self.zone_camera_p2_y.setValue(vertices[2][1])
            self.zone_camera_p3_x.setValue(vertices[3][0])
            self.zone_camera_p3_y.setValue(vertices[3][1])

            # Unblock signals
            self.zone_camera_p0_x.blockSignals(False)
            self.zone_camera_p0_y.blockSignals(False)
            self.zone_camera_p1_x.blockSignals(False)
            self.zone_camera_p1_y.blockSignals(False)
            self.zone_camera_p2_x.blockSignals(False)
            self.zone_camera_p2_y.blockSignals(False)
            self.zone_camera_p3_x.blockSignals(False)
            self.zone_camera_p3_y.blockSignals(False)

    @QtCore.Slot(str)
    def _on_projector_viewport_vertex_updated(self, zone_name: str) -> None:
        """Handle vertex update from projector viewport dragging.

        Args:
            zone_name: Name of the zone whose vertex was updated.
        """
        # Only update UI if this is the currently selected zone
        selected_zone = self._get_selected_zone()
        if selected_zone and selected_zone.name == zone_name:
            # Block signals to prevent triggering _on_zone_projector_vertex_changed
            self.zone_projector_p0_x.blockSignals(True)
            self.zone_projector_p0_y.blockSignals(True)
            self.zone_projector_p1_x.blockSignals(True)
            self.zone_projector_p1_y.blockSignals(True)
            self.zone_projector_p2_x.blockSignals(True)
            self.zone_projector_p2_y.blockSignals(True)
            self.zone_projector_p3_x.blockSignals(True)
            self.zone_projector_p3_y.blockSignals(True)

            # Update spinbox values from zone
            vertices = selected_zone.projector_mapping.vertices
            self.zone_projector_p0_x.setValue(vertices[0][0])
            self.zone_projector_p0_y.setValue(vertices[0][1])
            self.zone_projector_p1_x.setValue(vertices[1][0])
            self.zone_projector_p1_y.setValue(vertices[1][1])
            self.zone_projector_p2_x.setValue(vertices[2][0])
            self.zone_projector_p2_y.setValue(vertices[2][1])
            self.zone_projector_p3_x.setValue(vertices[3][0])
            self.zone_projector_p3_y.setValue(vertices[3][1])

            # Unblock signals
            self.zone_projector_p0_x.blockSignals(False)
            self.zone_projector_p0_y.blockSignals(False)
            self.zone_projector_p1_x.blockSignals(False)
            self.zone_projector_p1_y.blockSignals(False)
            self.zone_projector_p2_x.blockSignals(False)
            self.zone_projector_p2_y.blockSignals(False)
            self.zone_projector_p3_x.blockSignals(False)
            self.zone_projector_p3_y.blockSignals(False)

    @QtCore.Slot(int)
    def _on_zone_draw_locked_borders_changed(self, state: int) -> None:
        """Handle draw locked borders checkbox change (applies to both camera and projector)."""
        zone = self._get_selected_zone()
        if zone:
            zone.draw_locked_borders = state == QtCore.Qt.CheckState.Checked.value
            # Invalidate overlays since border visibility changed
            if zone.camera_mapping:
                zone.camera_mapping.invalidate_overlay()
            if zone.projector_mapping:
                zone.projector_mapping.invalidate_overlay()

    @QtCore.Slot(int)
    def _on_zone_camera_lock_vertices_changed(self, state: int) -> None:
        """Handle camera lock vertices checkbox change."""
        zone = self._get_selected_zone()
        if zone and zone.camera_mapping:
            zone.camera_mapping.lock_vertices = state == QtCore.Qt.CheckState.Checked.value
            self._update_camera_vertices_enabled()
            # Invalidate overlay since edge visibility may have changed
            zone.camera_mapping.invalidate_overlay()

    @QtCore.Slot(int)
    def _on_zone_projector_enabled_changed(self, state: int) -> None:
        """Handle projector mapping enable/disable."""
        from .zone import ProjectorMapping

        zone = self._get_selected_zone()
        if not zone:
            return

        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.projector_section.setVisible(enabled)
        if enabled:
            self.zone_projector_mapping_group.setVisible(True)

        if enabled:
            # Refresh combo box with available projectors
            self._update_projector_combo()

            if not zone.projector_mapping:
                # Create new projector mapping with default values
                projector_name = self.zone_projector_combo.currentText() if self.zone_projector_combo.count() > 0 else ""
                zone.projector_mapping = ProjectorMapping(projector_name=projector_name, enabled=True)
                # Load the new mapping into UI
                self._load_zone_into_ui(zone)
            else:
                # Enable existing mapping
                zone.projector_mapping.enabled = True
        else:
            # Disable mapping but keep the data
            if zone.projector_mapping:
                zone.projector_mapping.enabled = False

    @QtCore.Slot(str)
    def _on_zone_projector_changed(self, projector_name: str) -> None:
        """Handle projector selection change."""
        zone = self._get_selected_zone()
        if zone and zone.projector_mapping and projector_name:
            zone.projector_mapping.projector_name = projector_name

    @QtCore.Slot()
    def _on_zone_projector_vertex_changed(self) -> None:
        """Handle projector vertex coordinate change."""
        zone = self._get_selected_zone()
        if zone and zone.projector_mapping:
            zone.projector_mapping.vertices = [
                (self.zone_projector_p0_x.value(), self.zone_projector_p0_y.value()),
                (self.zone_projector_p1_x.value(), self.zone_projector_p1_y.value()),
                (self.zone_projector_p2_x.value(), self.zone_projector_p2_y.value()),
                (self.zone_projector_p3_x.value(), self.zone_projector_p3_y.value())
            ]
            zone.projector_mapping.invalidate_overlay()

            # If game allows locked corner adjustment and zone is calibrated,
            # automatically uncalibrate and recalibrate
            if self.core.allows_locked_corner_adjustment() and zone.is_calibrated():
                zone.uncalibrate()
                try:
                    zone.calibrate()
                except Exception as e:
                    print(f"Error recalibrating zone {zone.name}: {e}")

    @QtCore.Slot(int)
    def _on_zone_projector_lock_vertices_changed(self, state: int) -> None:
        """Handle projector lock vertices checkbox change."""
        zone = self._get_selected_zone()
        if zone and zone.projector_mapping:
            zone.projector_mapping.lock_vertices = state == QtCore.Qt.CheckState.Checked.value
            zone.projector_mapping.invalidate_overlay()
            self._update_projector_vertices_enabled()

    @QtCore.Slot()
    def _on_zone_calibrate(self) -> None:
        """Handle calibrate button click."""
        zone = self._get_selected_zone()
        if not zone:
            return

        try:
            zone.calibrate()
            self._update_zone_ui_state()
            QtWidgets.QMessageBox.information(
                self,
                "Calibration Complete",
                f"Zone '{zone.name}' has been calibrated successfully."
            )
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration Failed",
                str(e)
            )

    @QtCore.Slot()
    def _on_zone_uncalibrate(self) -> None:
        """Handle uncalibrate button click."""
        zone = self._get_selected_zone()
        if not zone:
            return

        zone.uncalibrate()
        self._update_zone_ui_state()
        QtWidgets.QMessageBox.information(
            self,
            "Uncalibration Complete",
            f"Zone '{zone.name}' has been uncalibrated."
        )

    @QtCore.Slot(int)
    def _on_speech_model_changed(self, index: int) -> None:
        """Handle speech model selection change."""
        if index < 0:
            return

        model_path = self.speech_model_combo.itemData(index)
        if model_path:
            self.core.update_speech_recognizer(model_path=model_path)

    @QtCore.Slot(int)
    def _on_speech_device_changed(self, index: int) -> None:
        """Handle speech device selection change."""
        if index < 0:
            return

        device_index = self.speech_device_combo.itemData(index)
        if device_index is not None:
            self.core.update_speech_recognizer(device_index=device_index)

    @QtCore.Slot(float)
    def _on_speech_threshold_changed(self, value: float) -> None:
        """Handle speech threshold change."""
        self.core.speech_threshold = value

    @QtCore.Slot(str)
    def _on_speech_partial_result(self, text: str) -> None:
        """Handle partial speech recognition result."""
        self.speech_partial_result_edit.setText(text)

    @QtCore.Slot(str)
    def _on_speech_final_result(self, text: str) -> None:
        """Handle final speech recognition result."""
        self.speech_final_result_edit.setText(text)

    @QtCore.Slot(int)
    def _on_narrator_voice_changed(self, index: int) -> None:
        """Handle narrator voice model selection change."""
        if index < 0:
            return

        voice_path = self.narrator_voice_combo.itemData(index)
        if voice_path:
            try:
                self.core.narrator.set_voice_model(voice_path)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Voice Model Error",
                    f"Failed to load voice model: {e}"
                )

    @QtCore.Slot(int)
    def _on_narrator_output_device_changed(self, index: int) -> None:
        """Handle narrator output device selection change."""
        if index < 0:
            return

        device_index = self.narrator_output_device_combo.itemData(index)
        self.core.narrator.mixer.set_output_device(device_index)

    @QtCore.Slot(int)
    def _on_narrator_voice_volume_changed(self, value: int) -> None:
        """Handle voice channel volume change."""
        from .sound_mixer import Channel
        volume = value / 100.0
        self.core.narrator.set_channel_volume(Channel.VOICE, volume)
        self.narrator_voice_volume_label.setText(f"{value}%")

    @QtCore.Slot(int)
    def _on_narrator_effect_volume_changed(self, value: int) -> None:
        """Handle effect channel volume change."""
        from .sound_mixer import Channel
        volume = value / 100.0
        self.core.narrator.set_channel_volume(Channel.EFFECT, volume)
        self.narrator_effect_volume_label.setText(f"{value}%")

    @QtCore.Slot(int)
    def _on_narrator_music_volume_changed(self, value: int) -> None:
        """Handle music channel volume change."""
        from .sound_mixer import Channel
        volume = value / 100.0
        self.core.narrator.set_channel_volume(Channel.MUSIC, volume)
        self.narrator_music_volume_label.setText(f"{value}%")

    @QtCore.Slot()
    def _on_narrator_voice_test(self) -> None:
        """Test voice channel with rotating text."""
        from .sound_mixer import Channel

        test_texts = [
            "I must not fear. Fear is the mind-killer.",
            "Violence is the last refuge of the incompetent.",
            "In a hole in the ground there lived a hobbit."
        ]

        text = test_texts[self.narrator_voice_test_index]
        self.narrator_voice_test_index = (self.narrator_voice_test_index + 1) % len(test_texts)

        try:
            self.core.narrator.synthesize_and_play(
                text,
                channel=Channel.VOICE,
                do_play_immediately=True,
                do_wait_until_played=False
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Voice Test Error",
                f"Failed to play voice test: {e}"
            )

    @QtCore.Slot()
    def _on_narrator_effect_test(self) -> None:
        """Test effect channel with rotating text."""
        from .sound_mixer import Channel

        test_texts = [
            "Boom.",
            "Swish.",
            "Pow."
        ]

        text = test_texts[self.narrator_effect_test_index]
        self.narrator_effect_test_index = (self.narrator_effect_test_index + 1) % len(test_texts)

        try:
            self.core.narrator.synthesize_and_play(
                text,
                channel=Channel.EFFECT,
                do_play_immediately=True,
                do_wait_until_played=False
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Effect Test Error",
                f"Failed to play effect test: {e}"
            )

    @QtCore.Slot()
    def _on_narrator_music_test(self) -> None:
        """Test music channel with rotating text."""
        from .sound_mixer import Channel

        test_texts = [
            "Pum, tchick pum, pum tchick pum pum.",
            "Na na naa, na. Na na naa, na. Hey hey, hey. Goodbye.",
            "Do re mi fa sol la ti do."
        ]

        text = test_texts[self.narrator_music_test_index]
        self.narrator_music_test_index = (self.narrator_music_test_index + 1) % len(test_texts)

        try:
            self.core.narrator.synthesize_and_play(
                text,
                channel=Channel.MUSIC,
                do_play_immediately=True,
                do_wait_until_played=False
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Music Test Error",
                f"Failed to play music test: {e}"
            )

    def _find_matching_vosk_model(self, saved_model_path: str) -> int:
        """Find matching vosk model in combo box.

        Args:
            saved_model_path: The saved model path to match.

        Returns:
            Index of matching model, or index with 'small' in name, or 0 if no match.
        """
        import os

        # Try exact match first
        for i in range(self.speech_model_combo.count()):
            if self.speech_model_combo.itemData(i) == saved_model_path:
                return i

        # Try matching just the model name (last part of path)
        saved_model_name = os.path.basename(saved_model_path)
        for i in range(self.speech_model_combo.count()):
            model_path = self.speech_model_combo.itemData(i)
            if model_path and os.path.basename(model_path) == saved_model_name:
                return i

        # Try finding one with 'small' in the name
        for i in range(self.speech_model_combo.count()):
            if 'small' in self.speech_model_combo.itemText(i).lower():
                return i

        # Default to first item
        return 0 if self.speech_model_combo.count() > 0 else -1

    def _find_matching_audio_device(self, saved_device_index: int, saved_device_name: str = None) -> int:
        """Find matching audio device in combo box.

        Prioritizes matching by device name, then by device index.

        Args:
            saved_device_index: The saved device index to match.
            saved_device_name: The saved device name to match (optional but preferred).

        Returns:
            Combo box index of matching device, or 0 if no match.
        """
        from .speech_recognition import get_audio_input_devices

        # Get current devices
        devices = get_audio_input_devices()

        # 1. Try matching by device name first (most reliable)
        if saved_device_name:
            for i in range(self.speech_device_combo.count()):
                device_index = self.speech_device_combo.itemData(i)
                if device_index is not None:
                    for device in devices:
                        if device['index'] == device_index and device['name'] == saved_device_name:
                            return i

            # Try partial match on device name (in case name changed slightly)
            for i in range(self.speech_device_combo.count()):
                item_text = self.speech_device_combo.itemText(i)
                if saved_device_name in item_text or item_text in saved_device_name:
                    return i

        # 2. Try matching by device index
        if saved_device_index is not None:
            for i in range(self.speech_device_combo.count()):
                device_index = self.speech_device_combo.itemData(i)
                if device_index == saved_device_index:
                    return i

        # 3. Default to first item
        return 0 if self.speech_device_combo.count() > 0 else -1

    def _find_matching_narrator_voice(self, saved_voice_path: str) -> int:
        """Find matching narrator voice in combo box.

        Args:
            saved_voice_path: The saved voice model path to match.

        Returns:
            Index of matching voice, or 0 if no match.
        """
        import os

        # Try exact match first
        for i in range(self.narrator_voice_combo.count()):
            if self.narrator_voice_combo.itemData(i) == saved_voice_path:
                return i

        # Try matching just the voice name (last part of path)
        saved_voice_name = os.path.basename(saved_voice_path)
        for i in range(self.narrator_voice_combo.count()):
            voice_path = self.narrator_voice_combo.itemData(i)
            if voice_path and os.path.basename(voice_path) == saved_voice_name:
                return i

        # Default to first item
        return 0 if self.narrator_voice_combo.count() > 0 else -1

    def _find_matching_narrator_output_device(self, saved_device_index: int, saved_device_name: str = None) -> int:
        """Find matching narrator output device in combo box.

        Prioritizes matching by device name, then by device index.

        Args:
            saved_device_index: The saved device index to match.
            saved_device_name: The saved device name to match (optional but preferred).

        Returns:
            Combo box index of matching device, or 0 (default device) if no match.
        """
        # Try matching by device name first (most reliable)
        if saved_device_name:
            for i in range(self.narrator_output_device_combo.count()):
                item_text = self.narrator_output_device_combo.itemText(i)
                if saved_device_name in item_text or item_text in saved_device_name:
                    return i

        # Try matching by device index
        if saved_device_index is not None:
            for i in range(self.narrator_output_device_combo.count()):
                device_index = self.narrator_output_device_combo.itemData(i)
                if device_index == saved_device_index:
                    return i

        # Default to first item (Default Output Device)
        return 0

    def _create_camera_tabs(self) -> QtWidgets.QTabWidget:
        """Create the camera tab widget.

        Returns:
            Tab widget containing camera-specific tabs.
        """
        tabs = QtWidgets.QTabWidget()

        # Settings tab
        settings_widget = self._create_camera_settings_widget()
        tabs.addTab(settings_widget, "Camera Settings")

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

    def _get_selected_camera_names(self) -> list[str]:
        """Get names of selected cameras for zone overlay compositing.

        Returns:
            List of camera names.
        """
        selected_items = self.camera_list.selectedItems()
        return [item.text() for item in selected_items]

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

        # Check if any cameras are used by zones
        cameras_in_use = {}
        for camera_name in camera_names:
            zones_using_camera = []
            for zone in self.core.zone_manager.get_all_zones():
                if zone.camera_mapping and zone.camera_mapping.camera_name == camera_name:
                    zones_using_camera.append(zone.name)
            if zones_using_camera:
                cameras_in_use[camera_name] = zones_using_camera

        # If any cameras are in use, show error and abort
        if cameras_in_use:
            message = "Cannot delete the following camera(s) because they are in use by zones:\n\n"
            for camera_name, zone_names in cameras_in_use.items():
                message += f"  • {camera_name} is used by: {', '.join(zone_names)}\n"
            message += "\nPlease remove or modify the zone mappings first."

            QtWidgets.QMessageBox.warning(
                self,
                "Cameras In Use",
                message
            )
            return

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
    def _on_viewports_fps_changed(self, fps: int) -> None:
        """Handle viewports FPS change.

        Args:
            fps: New frames per second value.
        """
        self.core.viewports_refresh_rate = fps
        self.viewport.set_fps(fps)

    @QtCore.Slot(int)
    def _on_projectors_fps_changed(self, fps: int) -> None:
        """Handle projectors FPS change.

        Args:
            fps: New frames per second value.
        """
        self.core.projectors_refresh_rate = fps
        # Update all existing projectors
        for projector in self.core.projector_manager.get_all_projectors():
            projector.set_fps(fps)

    @QtCore.Slot(int)
    def _on_qr_code_fps_changed(self, fps: int) -> None:
        """Handle QR code scanning FPS change.

        Args:
            fps: New frames per second value.
        """
        self.core.set_qr_code_refresh_rate(fps)

    @QtCore.Slot()
    def _on_load_game(self) -> None:
        """Handle load game menu action."""
        from .game_loader import GameInfo

        # Discover available games
        games = self.core.game_loader.discover_games()

        if not games:
            QtWidgets.QMessageBox.information(
                self,
                "No Games Found",
                "No games were found in the games/ or test_games/ directories.\n\n"
                "To add a game, create a folder in games/ or test_games/ with a game.py file."
            )
            return

        # Create selection dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Load Game")
        dialog.setMinimumWidth(500)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Instructions
        label = QtWidgets.QLabel("Select a game to load:")
        layout.addWidget(label)

        # Game list
        game_list = QtWidgets.QListWidget()
        for game in games:
            item = QtWidgets.QListWidgetItem(str(game))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, game)

            # Add tooltip with description
            if game.description:
                item.setToolTip(game.description)

            game_list.addItem(item)

        game_list.setCurrentRow(0)
        layout.addWidget(game_list)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Show dialog
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            selected_item = game_list.currentItem()
            if selected_item:
                game_info: GameInfo = selected_item.data(QtCore.Qt.ItemDataRole.UserRole)

                # Load the game
                success = self.core.load_game(game_info)

                if not success:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Game Load Failed",
                        f"Failed to load game: {game_info.name}\n\n"
                        "Check the console for error details."
                    )

    @QtCore.Slot()
    def _on_unload_game(self) -> None:
        """Handle unload game menu action."""
        if self.core.current_game_info:
            game_name = self.core.current_game_info.name

            reply = QtWidgets.QMessageBox.question(
                self,
                "Unload Game",
                f"Unload '{game_name}'?",
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.No
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self.core.unload_game()

    @QtCore.Slot(str)
    def _on_game_loaded(self, game_name: str) -> None:
        """Handle game loaded signal.

        Args:
            game_name: Name of the loaded game.
        """
        self.unload_game_action.setEnabled(True)
        self.setWindowTitle(f"Tabletop Guided Adventures - {game_name}")

        # Refresh zone UI state to apply game-specific overrides
        self._update_zone_ui_state()

        # Auto-open game dialog
        if self.core.current_game:
            self.core.current_game.show_dialog(parent=self)

    @QtCore.Slot()
    def _on_game_unloaded(self) -> None:
        """Handle game unloaded signal."""
        self.unload_game_action.setEnabled(False)
        self.setWindowTitle("Tabletop Guided Adventures")

        # Refresh zone UI state to remove game-specific overrides
        self._update_zone_ui_state()

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
                # Wrap data in standard structure
                save_data = {
                    "type": "cameras",
                    "version": "1.0",
                    "data": cameras_data
                }

                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2)

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
                file_data = json.load(f)

            # Validate file structure
            if not isinstance(file_data, dict):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid File Format",
                    "The file format is invalid. Expected a JSON object with 'type', 'version', and 'data' fields."
                )
                return

            # Validate type field
            if 'type' not in file_data:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing Type Field",
                    "The file is missing the required 'type' field."
                )
                return

            # Accept both 'cameras' and 'master' file types
            if file_data['type'] not in ['cameras', 'master']:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid File Type",
                    f"Cannot load cameras from a '{file_data['type']}' file.\n"
                    f"Please select a cameras or master configuration file."
                )
                return

            # Validate version field
            if 'version' not in file_data:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing Version Field",
                    "The file is missing the required 'version' field."
                )
                return

            if file_data['version'] != '1.0':
                QtWidgets.QMessageBox.critical(
                    self,
                    "Unknown Version",
                    f"Unknown file version '{file_data['version']}'. This application only supports version '1.0'."
                )
                return

            # Extract cameras data based on file type
            if file_data['type'] == 'master':
                master_data = file_data.get('data', {})
                cameras_data = master_data.get('cameras', [])
            else:
                cameras_data = file_data.get('data', [])

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

        # Enable save button since we have at least one camera
        self.save_camera_button.setEnabled(True)

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

        # Disable save button if no cameras left
        self.save_camera_button.setEnabled(self.camera_list.count() > 0)

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
        dialog.viewport.set_zone_manager(self.core.zone_manager)
        dialog.viewport.set_main_core(self.core)
        dialog.viewport.vertex_updated.connect(self._on_projector_viewport_vertex_updated)
        projector.dialog = dialog
        dialog.show()

        # Enable save button since we have at least one projector
        self.save_projector_button.setEnabled(True)

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

        # Disable save button if no projectors left
        self.save_projector_button.setEnabled(self.projector_list.count() > 0)

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
                dialog.viewport.set_zone_manager(self.core.zone_manager)
                dialog.viewport.set_main_core(self.core)
                dialog.viewport.vertex_updated.connect(self._on_projector_viewport_vertex_updated)
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
                    dialog.projector_resolution,
                    self.core.projectors_refresh_rate
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

        # Check if any projectors are used by zones
        projectors_in_use = {}
        for projector_name in projector_names:
            zones_using_projector = []
            for zone in self.core.zone_manager.get_all_zones():
                if zone.projector_mapping and zone.projector_mapping.projector_name == projector_name:
                    zones_using_projector.append(zone.name)
            if zones_using_projector:
                projectors_in_use[projector_name] = zones_using_projector

        # If any projectors are in use, show error and abort
        if projectors_in_use:
            message = "Cannot delete the following projector(s) because they are in use by zones:\n\n"
            for projector_name, zone_names in projectors_in_use.items():
                message += f"  • {projector_name} is used by: {', '.join(zone_names)}\n"
            message += "\nPlease remove or modify the zone mappings first."

            QtWidgets.QMessageBox.warning(
                self,
                "Projectors In Use",
                message
            )
            return

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

                # Wrap data in standard structure
                save_data = {
                    "type": "projectors",
                    "version": "1.0",
                    "data": projectors_data
                }

                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2)

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
                file_data = json.load(f)

            # Validate file structure
            if not isinstance(file_data, dict):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid File Format",
                    "The file format is invalid. Expected a JSON object with 'type', 'version', and 'data' fields."
                )
                return

            # Validate type field
            if 'type' not in file_data:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing Type Field",
                    "The file is missing the required 'type' field."
                )
                return

            # Accept both 'projectors' and 'master' file types
            if file_data['type'] not in ['projectors', 'master']:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid File Type",
                    f"Cannot load projectors from a '{file_data['type']}' file.\n"
                    f"Please select a projectors or master configuration file."
                )
                return

            # Validate version field
            if 'version' not in file_data:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing Version Field",
                    "The file is missing the required 'version' field."
                )
                return

            if file_data['version'] != '1.0':
                QtWidgets.QMessageBox.critical(
                    self,
                    "Unknown Version",
                    f"Unknown file version '{file_data['version']}'. This application only supports version '1.0'."
                )
                return

            # Extract projectors data based on file type
            if file_data['type'] == 'master':
                master_data = file_data.get('data', {})
                projectors_data = master_data.get('projectors', [])
            else:
                projectors_data = file_data.get('data', [])

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

    @QtCore.Slot(str)
    def _on_zone_added(self, zone_name: str) -> None:
        """Handle zone added signal.

        Args:
            zone_name: Name of the added zone.
        """
        # Add to list widget
        item = QtWidgets.QListWidgetItem(zone_name)
        self.zone_list.addItem(item)

        # Enable save button since we have at least one zone
        self.save_zone_button.setEnabled(True)

    @QtCore.Slot(str)
    def _on_zone_removed(self, zone_name: str) -> None:
        """Handle zone removed signal.

        Args:
            zone_name: Name of the removed zone.
        """
        # Remove from list widget
        items = self.zone_list.findItems(zone_name, QtCore.Qt.MatchFlag.MatchExactly)
        for item in items:
            row = self.zone_list.row(item)
            self.zone_list.takeItem(row)

        # Disable save button if no zones left
        self.save_zone_button.setEnabled(self.zone_list.count() > 0)

    @QtCore.Slot()
    def _on_zone_selection_changed(self) -> None:
        """Handle zone list selection change."""
        selected_items = self.zone_list.selectedItems()
        has_selection = len(selected_items) > 0

        # Enable delete button only if at least one zone is selected
        self.delete_zone_button.setEnabled(has_selection)

        # Show zone settings only if exactly one zone is selected
        if len(selected_items) == 1:
            zone_name = selected_items[0].text()
            try:
                zone = self.core.zone_manager.get_zone(zone_name)
                self._load_zone_into_ui(zone)
                self._set_zone_settings_visible(True)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error Loading Zone",
                    f"Failed to load zone '{zone_name}': {str(e)}"
                )
                self._set_zone_settings_visible(False)
        else:
            self._set_zone_settings_visible(False)

    @QtCore.Slot()
    def _on_add_zone(self) -> None:
        """Handle add zone button click."""
        from .zone import Zone
        from .add_zone_dialog import AddZoneDialog

        # Show dialog to get zone name
        dialog = AddZoneDialog(self.core.zone_manager, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            zone_name = dialog.get_zone_name()

            try:
                # Create and add zone with default values
                zone = Zone(zone_name)
                self.core.zone_manager.add_zone(zone)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Adding Zone",
                    f"Failed to add zone: {str(e)}"
                )

    @QtCore.Slot()
    def _on_delete_zone(self) -> None:
        """Handle delete zone button click."""
        selected_items = self.zone_list.selectedItems()
        if not selected_items:
            return

        zone_names = [item.text() for item in selected_items]

        # Confirm deletion
        if len(zone_names) == 1:
            message = f"Are you sure you want to delete zone '{zone_names[0]}'?"
        else:
            message = f"Are you sure you want to delete {len(zone_names)} zones?"

        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Zone(s)",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # Remove zones
            for zone_name in zone_names:
                try:
                    self.core.zone_manager.remove_zone(zone_name)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Error Deleting Zone",
                        f"Failed to delete zone '{zone_name}': {str(e)}"
                    )

    @QtCore.Slot()
    def _on_save_zone(self) -> None:
        """Handle save zone button click."""
        import json
        import os
        from . import constants

        # Get all zones data
        zones_data = self.core.zone_manager.serialize_zones()

        if not zones_data:
            QtWidgets.QMessageBox.information(
                self,
                "No Zones",
                "No zones to save."
            )
            return

        # Ensure saved zones directory exists
        os.makedirs(constants.SAVED_ZONES_DIR_PATH, exist_ok=True)

        # Show file save dialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Zones",
            constants.SAVED_ZONES_DIR_PATH,
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Wrap data in standard structure
                save_data = {
                    "type": "zones",
                    "version": "1.0",
                    "data": zones_data
                }

                with open(file_path, 'w') as f:
                    json.dump(save_data, f, indent=2)

                QtWidgets.QMessageBox.information(
                    self,
                    "Zones Saved",
                    f"Successfully saved {len(zones_data)} zone(s)."
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Saving Zones",
                    f"Failed to save zones: {str(e)}"
                )

    @QtCore.Slot()
    def _on_load_zone(self) -> None:
        """Handle load zone button click."""
        import json
        import os
        from . import constants
        from .zone import Zone

        # Ensure saved zones directory exists
        os.makedirs(constants.SAVED_ZONES_DIR_PATH, exist_ok=True)

        # Show file open dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Zones",
            constants.SAVED_ZONES_DIR_PATH,
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                file_data = json.load(f)

            # Validate file structure
            if not isinstance(file_data, dict):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid File Format",
                    "The file format is invalid. Expected a JSON object with 'type', 'version', and 'data' fields."
                )
                return

            # Validate type field
            if 'type' not in file_data:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing Type Field",
                    "The file is missing the required 'type' field."
                )
                return

            # Accept both 'zones' and 'master' file types
            if file_data['type'] not in ['zones', 'master']:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid File Type",
                    f"Cannot load zones from a '{file_data['type']}' file.\n"
                    f"Please select a zones or master configuration file."
                )
                return

            # Validate version field
            if 'version' not in file_data:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing Version Field",
                    "The file is missing the required 'version' field."
                )
                return

            if file_data['version'] != '1.0':
                QtWidgets.QMessageBox.critical(
                    self,
                    "Unknown Version",
                    f"Unknown file version '{file_data['version']}'. This application only supports version '1.0'."
                )
                return

            # Extract zones data based on file type
            if file_data['type'] == 'master':
                master_data = file_data.get('data', {})
                zones_data = master_data.get('zones', [])
            else:
                zones_data = file_data.get('data', [])

            if not zones_data:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Zones",
                    "No zones found in file."
                )
                return

            # Check for conflicts
            conflicts = []
            for zone_data in zones_data:
                if self.core.zone_manager.zone_exists(zone_data['name']):
                    conflicts.append(zone_data['name'])

            # Ask user if they want to replace conflicting zones
            if conflicts:
                message = "The following zones already exist:\n\n"
                message += "\n".join(conflicts)
                message += "\n\nDo you want to replace them?"

                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Replace Zones?",
                    message,
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )

                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    return

                # Remove conflicting zones
                for conflict_name in conflicts:
                    try:
                        self.core.zone_manager.remove_zone(conflict_name)
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Error Removing Zone",
                            f"Failed to remove zone '{conflict_name}': {str(e)}"
                        )

            # Load zones
            loaded_count = 0
            for zone_data in zones_data:
                try:
                    zone = Zone.from_dict(zone_data)

                    # Validate that referenced cameras/projectors exist
                    is_valid, error_msg = self._validate_zone_references(zone)
                    if not is_valid:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Missing Device Reference",
                            error_msg
                        )
                        continue

                    self.core.zone_manager.add_zone(zone)
                    loaded_count += 1
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Error Loading Zone",
                        f"Failed to load zone '{zone_data.get('name', 'Unknown')}': {str(e)}"
                    )

            if loaded_count > 0:
                QtWidgets.QMessageBox.information(
                    self,
                    "Zones Loaded",
                    f"Successfully loaded {loaded_count} zone(s)."
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading Zones",
                f"Failed to load zones: {str(e)}"
            )

    @QtCore.Slot()
    def _on_save_master_configuration(self) -> None:
        """Handle save master configuration menu action."""
        import json
        import os
        from . import constants

        # Collect all data
        cameras_data = []
        projectors_data = []
        zones_data = []

        try:
            # Get cameras data
            for camera_name in self.core.camera_manager.get_camera_names():
                camera = self.core.camera_manager.get_camera(camera_name)
                camera_dict = {
                    'name': camera.name,
                    'backend': camera.get_backend(),
                    'device_id': camera.get_device_id(),
                    'camera_info': camera.get_camera_info(),
                    'properties': {}
                }

                # Get all properties
                for prop_id in [cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT,
                                cv.CAP_PROP_FPS, cv.CAP_PROP_FOURCC, cv.CAP_PROP_BRIGHTNESS,
                                cv.CAP_PROP_CONTRAST, cv.CAP_PROP_SATURATION, cv.CAP_PROP_HUE,
                                cv.CAP_PROP_GAIN, cv.CAP_PROP_EXPOSURE, cv.CAP_PROP_AUTO_EXPOSURE,
                                cv.CAP_PROP_AUTOFOCUS, cv.CAP_PROP_FOCUS]:
                    value = camera.get_property(prop_id)
                    if value is not None:
                        camera_dict['properties'][str(prop_id)] = value

                # Add calibration data if available
                if camera.calibration_data is not None:
                    camera_dict['calibration_data'] = {
                        'mtx': camera.calibration_data.mtx.tolist(),
                        'dist': camera.calibration_data.dist.tolist(),
                        'rvecs_list': [rvec.tolist() for rvec in camera.calibration_data.rvecs_list],
                        'tvecs_list': [tvec.tolist() for tvec in camera.calibration_data.tvecs_list],
                        'mean_reprojection_error': camera.calibration_data.mean_reprojection_error,
                        'resolution': camera.calibration_data.resolution
                    }

                cameras_data.append(camera_dict)

            # Get projectors data
            projectors_data = self.core.projector_manager.serialize_projectors()

            # Get zones data
            zones_data = self.core.zone_manager.serialize_zones()

            # Get speech recognition configuration
            speech_config = {
                'model_path': self.core.speech_model_path,
                'device_index': self.core.speech_device_index,
                'device_name': None,
                'threshold': self.core.speech_threshold
            }

            # Get device name if device index is set
            if self.core.speech_device_index is not None:
                from .speech_recognition import get_audio_input_devices
                devices = get_audio_input_devices()
                for device in devices:
                    if device['index'] == self.core.speech_device_index:
                        speech_config['device_name'] = device['name']
                        break

            # Get narrator configuration
            from .sound_mixer import Channel
            narrator_config = {
                'voice_model_path': self.core.narrator.voice_model_path,
                'output_device_index': None,
                'output_device_name': None,
                'voice_volume': self.core.narrator.get_channel_volume(Channel.VOICE),
                'effect_volume': self.core.narrator.get_channel_volume(Channel.EFFECT),
                'music_volume': self.core.narrator.get_channel_volume(Channel.MUSIC)
            }

            # Get output device info if selected
            current_device_index = self.narrator_output_device_combo.currentIndex()
            if current_device_index > 0:  # Skip "Default Output Device" at index 0
                device_index = self.narrator_output_device_combo.itemData(current_device_index)
                device_name = self.narrator_output_device_combo.itemText(current_device_index)
                narrator_config['output_device_index'] = device_index
                narrator_config['output_device_name'] = device_name

            # Get refresh rates
            refresh_rates = {
                'viewports_fps': self.core.viewports_refresh_rate,
                'projectors_fps': self.core.projectors_refresh_rate,
                'qr_code_fps': self.core.qr_code_refresh_rate
            }

            # Create master configuration dictionary with standard structure
            master_config = {
                'type': 'master',
                'version': '1.0',
                'data': {
                    'cameras': cameras_data,
                    'projectors': projectors_data,
                    'zones': zones_data,
                    'speech_recognition': speech_config,
                    'narrator': narrator_config,
                    'refresh_rates': refresh_rates
                }
            }

            # Ensure directory exists
            os.makedirs(constants.SAVED_MASTER_CONFIGURATIONS_DIR_PATH, exist_ok=True)

            # Show file save dialog
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Master Configuration",
                constants.SAVED_MASTER_CONFIGURATIONS_DIR_PATH,
                "JSON Files (*.json)"
            )

            if file_path:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with open(file_path, 'w') as f:
                    json.dump(master_config, f, indent=2)

                QtWidgets.QMessageBox.information(
                    self,
                    "Master Configuration Saved",
                    f"Successfully saved configuration with {len(cameras_data)} camera(s), "
                    f"{len(projectors_data)} projector(s), and {len(zones_data)} zone(s)."
                )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Saving Master Configuration",
                f"Failed to save master configuration: {str(e)}"
            )

    @QtCore.Slot()
    def _on_load_master_configuration(self) -> None:
        """Handle load master configuration menu action."""
        import json
        import os
        from . import constants
        from .zone import Zone

        # Ensure directory exists
        os.makedirs(constants.SAVED_MASTER_CONFIGURATIONS_DIR_PATH, exist_ok=True)

        # Show file open dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Master Configuration",
            constants.SAVED_MASTER_CONFIGURATIONS_DIR_PATH,
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                file_data = json.load(f)

            # Validate file structure
            if not isinstance(file_data, dict):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid File Format",
                    "The file format is invalid. Expected a JSON object with 'type', 'version', and 'data' fields."
                )
                return

            # Validate type field
            if 'type' not in file_data:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing Type Field",
                    "The file is missing the required 'type' field."
                )
                return

            if file_data['type'] != 'master':
                QtWidgets.QMessageBox.critical(
                    self,
                    "Invalid File Type",
                    f"Cannot load master configuration from a '{file_data['type']}' file.\n"
                    f"Please select a master configuration file."
                )
                return

            # Validate version field
            if 'version' not in file_data:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing Version Field",
                    "The file is missing the required 'version' field."
                )
                return

            if file_data['version'] != '1.0':
                QtWidgets.QMessageBox.critical(
                    self,
                    "Unknown Version",
                    f"Unknown file version '{file_data['version']}'. This application only supports version '1.0'."
                )
                return

            master_data = file_data.get('data', {})

            cameras_data = master_data.get('cameras', [])
            projectors_data = master_data.get('projectors', [])
            zones_data = master_data.get('zones', [])

            if not cameras_data and not projectors_data and not zones_data:
                QtWidgets.QMessageBox.information(
                    self,
                    "Empty Configuration",
                    "The master configuration file is empty."
                )
                return

            # Show confirmation dialog
            message = "This will clear all existing cameras, projectors, and zones, then load:\n"
            message += f"  - {len(cameras_data)} camera(s)\n"
            message += f"  - {len(projectors_data)} projector(s)\n"
            message += f"  - {len(zones_data)} zone(s)\n\n"
            message += "Do you want to continue?"

            reply = QtWidgets.QMessageBox.question(
                self,
                "Load Master Configuration",
                message,
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )

            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

            # Clear all existing items first
            # Remove all zones
            for zone in list(self.core.zone_manager.get_all_zones()):
                try:
                    self.core.zone_manager.remove_zone(zone.name)
                except Exception:
                    pass

            # Remove all projectors
            for projector in list(self.core.projector_manager.get_all_projectors()):
                try:
                    self.core.projector_manager.remove_projector(projector.name)
                except Exception:
                    pass

            # Remove all cameras
            for camera_name in list(self.core.camera_manager.get_camera_names()):
                try:
                    self.core.camera_manager.remove_camera(camera_name)
                except Exception:
                    pass

            loaded_cameras = 0
            loaded_projectors = 0
            loaded_zones = 0

            # Load cameras first
            if cameras_data:
                cameras_to_load = []
                missing_cameras = []
                used_device_ids = {}

                for cam_data in cameras_data:
                    backend = cam_data['backend']
                    saved_device_id = cam_data['device_id']
                    saved_camera_info = cam_data.get('camera_info')

                    matched_device_id = self.core.camera_manager.find_matching_device(
                        backend, saved_camera_info, saved_device_id
                    )

                    if matched_device_id is None:
                        missing_cameras.append(cam_data['name'])
                        continue

                    if backend not in used_device_ids:
                        used_device_ids[backend] = set()

                    if matched_device_id in used_device_ids[backend]:
                        missing_cameras.append(cam_data['name'])
                        continue

                    used_device_ids[backend].add(matched_device_id)

                    cameras_to_load.append({
                        'name': cam_data['name'],
                        'backend': backend,
                        'device_id': matched_device_id,
                        'camera_info': saved_camera_info,
                        'properties': cam_data.get('properties', {}),
                        'calibration_data': cam_data.get('calibration_data')
                    })

                # Remove conflicting cameras
                existing_cameras = self.core.camera_manager.get_camera_names()
                for cam in cameras_to_load:
                    if cam['name'] in existing_cameras:
                        try:
                            self.core.camera_manager.remove_camera(cam['name'])
                        except Exception:
                            pass

                # Load cameras
                for cam in cameras_to_load:
                    try:
                        camera = self.core.camera_manager.add_camera(
                            cam['name'],
                            cam['backend'],
                            cam['device_id'],
                            cam['camera_info']
                        )

                        for prop_id, value in cam['properties'].items():
                            prop_id_int = int(prop_id)
                            if prop_id_int in [cv.CAP_PROP_FOURCC, cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT]:
                                camera.set_property(prop_id_int, float(int(value)))
                            else:
                                camera.set_property(prop_id_int, float(value))

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

                        camera.start()
                        loaded_cameras += 1

                    except Exception as e:
                        print(f"Failed to load camera '{cam['name']}': {str(e)}")

            # Load projectors second
            if projectors_data:
                for proj_data in projectors_data:
                    if self.core.projector_manager.projector_exists(proj_data['name']):
                        try:
                            self.core.projector_manager.remove_projector(proj_data['name'])
                        except Exception:
                            pass

                    try:
                        self.core.projector_manager.add_projector(
                            proj_data['name'],
                            tuple(proj_data['resolution'])
                        )
                        loaded_projectors += 1
                    except Exception as e:
                        print(f"Failed to load projector '{proj_data['name']}': {str(e)}")

            # Load zones last
            if zones_data:
                for zone_data in zones_data:
                    if self.core.zone_manager.zone_exists(zone_data['name']):
                        try:
                            self.core.zone_manager.remove_zone(zone_data['name'])
                        except Exception:
                            pass

                    try:
                        zone = Zone.from_dict(zone_data)

                        # Validate that referenced cameras/projectors exist
                        is_valid, error_msg = self._validate_zone_references(zone)
                        if not is_valid:
                            print(f"Skipping zone due to missing references: {error_msg}")
                            continue

                        self.core.zone_manager.add_zone(zone)
                        loaded_zones += 1
                    except Exception as e:
                        import traceback
                        error_msg = f"Failed to load zone '{zone_data.get('name', 'unknown')}': {str(e)}\n{traceback.format_exc()}"
                        print(error_msg)

            # Load speech recognition configuration
            speech_config = master_data.get('speech_recognition', {})
            if speech_config:
                # Load threshold
                threshold = speech_config.get('threshold', 0.7)
                self.core.speech_threshold = threshold
                self.speech_threshold_spinbox.blockSignals(True)
                self.speech_threshold_spinbox.setValue(threshold)
                self.speech_threshold_spinbox.blockSignals(False)

                # Load and match vosk model
                saved_model_path = speech_config.get('model_path')
                if saved_model_path:
                    model_index = self._find_matching_vosk_model(saved_model_path)
                    if model_index >= 0:
                        self.speech_model_combo.blockSignals(True)
                        self.speech_model_combo.setCurrentIndex(model_index)
                        self.speech_model_combo.blockSignals(False)
                        model_path = self.speech_model_combo.itemData(model_index)
                        self.core.update_speech_recognizer(model_path=model_path)

                # Load and match audio device
                saved_device_index = speech_config.get('device_index')
                saved_device_name = speech_config.get('device_name')
                if saved_device_index is not None or saved_device_name:
                    device_combo_index = self._find_matching_audio_device(saved_device_index, saved_device_name)
                    if device_combo_index >= 0:
                        self.speech_device_combo.blockSignals(True)
                        self.speech_device_combo.setCurrentIndex(device_combo_index)
                        self.speech_device_combo.blockSignals(False)
                        device_index = self.speech_device_combo.itemData(device_combo_index)
                        self.core.update_speech_recognizer(device_index=device_index)

            # Load narrator configuration
            narrator_config = master_data.get('narrator', {})
            if narrator_config:
                from .sound_mixer import Channel

                # Load voice model
                saved_voice_path = narrator_config.get('voice_model_path')
                if saved_voice_path:
                    voice_index = self._find_matching_narrator_voice(saved_voice_path)
                    if voice_index >= 0:
                        self.narrator_voice_combo.blockSignals(True)
                        self.narrator_voice_combo.setCurrentIndex(voice_index)
                        self.narrator_voice_combo.blockSignals(False)
                        voice_path = self.narrator_voice_combo.itemData(voice_index)
                        if voice_path:
                            try:
                                self.core.narrator.set_voice_model(voice_path)
                            except Exception as e:
                                print(f"Failed to load narrator voice model: {e}")

                # Load output device
                saved_device_index = narrator_config.get('output_device_index')
                saved_device_name = narrator_config.get('output_device_name')
                if saved_device_index is not None or saved_device_name:
                    device_combo_index = self._find_matching_narrator_output_device(saved_device_index, saved_device_name)
                    if device_combo_index >= 0:
                        self.narrator_output_device_combo.blockSignals(True)
                        self.narrator_output_device_combo.setCurrentIndex(device_combo_index)
                        self.narrator_output_device_combo.blockSignals(False)
                        # Apply the device to the mixer
                        device_index = self.narrator_output_device_combo.itemData(device_combo_index)
                        self.core.narrator.mixer.set_output_device(device_index)

                # Load channel volumes
                voice_volume = narrator_config.get('voice_volume', 1.0)
                effect_volume = narrator_config.get('effect_volume', 1.0)
                music_volume = narrator_config.get('music_volume', 1.0)

                self.narrator_voice_volume_slider.blockSignals(True)
                self.narrator_voice_volume_slider.setValue(int(voice_volume * 100))
                self.narrator_voice_volume_slider.blockSignals(False)
                self.narrator_voice_volume_label.setText(f"{int(voice_volume * 100)}%")
                self.core.narrator.set_channel_volume(Channel.VOICE, voice_volume)

                self.narrator_effect_volume_slider.blockSignals(True)
                self.narrator_effect_volume_slider.setValue(int(effect_volume * 100))
                self.narrator_effect_volume_slider.blockSignals(False)
                self.narrator_effect_volume_label.setText(f"{int(effect_volume * 100)}%")
                self.core.narrator.set_channel_volume(Channel.EFFECT, effect_volume)

                self.narrator_music_volume_slider.blockSignals(True)
                self.narrator_music_volume_slider.setValue(int(music_volume * 100))
                self.narrator_music_volume_slider.blockSignals(False)
                self.narrator_music_volume_label.setText(f"{int(music_volume * 100)}%")
                self.core.narrator.set_channel_volume(Channel.MUSIC, music_volume)

            # Load refresh rates
            refresh_rates = master_data.get('refresh_rates', {})
            if refresh_rates:
                viewports_fps = refresh_rates.get('viewports_fps', 30)
                projectors_fps = refresh_rates.get('projectors_fps', 15)
                qr_code_fps = refresh_rates.get('qr_code_fps', 5)

                # Update UI spinboxes
                self.viewports_fps_spinbox.blockSignals(True)
                self.viewports_fps_spinbox.setValue(viewports_fps)
                self.viewports_fps_spinbox.blockSignals(False)

                self.projectors_fps_spinbox.blockSignals(True)
                self.projectors_fps_spinbox.setValue(projectors_fps)
                self.projectors_fps_spinbox.blockSignals(False)

                self.qr_code_fps_spinbox.blockSignals(True)
                self.qr_code_fps_spinbox.setValue(qr_code_fps)
                self.qr_code_fps_spinbox.blockSignals(False)

                # Apply to core and all objects
                self.core.viewports_refresh_rate = viewports_fps
                self.viewport.set_fps(viewports_fps)

                self.core.projectors_refresh_rate = projectors_fps
                for projector in self.core.projector_manager.get_all_projectors():
                    projector.set_fps(projectors_fps)

                self.core.set_qr_code_refresh_rate(qr_code_fps)

            # Trigger selection change callbacks to update UI
            self._on_camera_selection_changed()
            self._on_projector_selection_changed()
            self._on_zone_selection_changed()

            # Show summary
            QtWidgets.QMessageBox.information(
                self,
                "Master Configuration Loaded",
                f"Successfully loaded:\n"
                f"  - {loaded_cameras} camera(s)\n"
                f"  - {loaded_projectors} projector(s)\n"
                f"  - {loaded_zones} zone(s)"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading Master Configuration",
                f"Failed to load master configuration: {str(e)}"
            )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window close event.

        Args:
            event: Close event.
        """
        # Unload game if one is loaded
        if self.core.current_game:
            self.core.unload_game()

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
            self._clear_calibration_frames()

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
