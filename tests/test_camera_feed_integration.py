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

"""Integration test for CameraFeed with live preview and controls.

This script provides a GUI application to test the CameraFeed class with
real-time preview and adjustable camera parameters.
"""

import os
import sys
import time

import cv2 as cv
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QComboBox,
    QLineEdit,
    QGroupBox,
    QFormLayout
)

root_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_path = os.path.join(root_dir_path, "python")
if python_path not in sys.path:
    sys.path.append(python_path)

from ttga.camera_feed import CameraFeed, DEFAULT_CAPTURE_PROPERTIES  # noqa: E402


class CameraFeedTestWindow(QMainWindow):
    """Main window for testing CameraFeed with live preview and controls."""

    def __init__(self) -> None:
        """Initialize the test window."""
        super().__init__()

        self.camera_feed: CameraFeed | None = None
        self.last_frame_time: float = 0.0
        self.frame_count: int = 0
        self.fps_update_interval: float = 0.5
        self.current_capture_api: int = cv.CAP_DSHOW

        self.setWindowTitle("CameraFeed Integration Test")
        self.setMinimumSize(1280, 800)

        self._setup_ui()
        self._setup_menu()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left side: Video preview
        preview_layout = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        preview_layout.addWidget(self.video_label)

        # FPS display
        self.fps_input = QLineEdit()
        self.fps_input.setReadOnly(True)
        self.fps_input.setPlaceholderText("FPS: --")
        preview_layout.addWidget(self.fps_input)

        main_layout.addLayout(preview_layout, stretch=3)

        # Right side: Controls
        controls_layout = QVBoxLayout()

        # Device and Capture API control
        device_group = QGroupBox("Device")
        device_layout = QFormLayout()

        # Capture API
        self.capture_api_combo = QComboBox()
        self.capture_api_combo.addItem("DirectShow (CAP_DSHOW)", cv.CAP_DSHOW)
        self.capture_api_combo.addItem("Media Foundation (CAP_MSMF)", cv.CAP_MSMF)
        self.capture_api_combo.currentIndexChanged.connect(self._on_capture_api_changed)
        device_layout.addRow("Capture API:", self.capture_api_combo)

        # Device ID
        self.device_id_combo = QComboBox()
        device_layout.addRow("Device ID:", self.device_id_combo)

        device_group.setLayout(device_layout)
        controls_layout.addWidget(device_group)

        # Populate device list for initial API
        self._update_device_list()

        # Capture properties group
        properties_group = QGroupBox("Capture Properties")
        properties_layout = QFormLayout()

        # Focus
        self.focus_spinbox = QSpinBox()
        self.focus_spinbox.setRange(0, 255)
        self.focus_spinbox.setValue(int(DEFAULT_CAPTURE_PROPERTIES.get(cv.CAP_PROP_FOCUS, 0)))
        self.focus_spinbox.valueChanged.connect(self._on_focus_changed)
        properties_layout.addRow("Focus:", self.focus_spinbox)

        # Brightness
        self.brightness_spinbox = QSpinBox()
        self.brightness_spinbox.setRange(0, 255)
        self.brightness_spinbox.setValue(int(DEFAULT_CAPTURE_PROPERTIES.get(cv.CAP_PROP_BRIGHTNESS, 128)))
        self.brightness_spinbox.valueChanged.connect(self._on_brightness_changed)
        properties_layout.addRow("Brightness:", self.brightness_spinbox)

        # Gain
        self.gain_spinbox = QSpinBox()
        self.gain_spinbox.setRange(0, 255)
        self.gain_spinbox.setValue(int(DEFAULT_CAPTURE_PROPERTIES.get(cv.CAP_PROP_GAIN, 128)))
        self.gain_spinbox.valueChanged.connect(self._on_gain_changed)
        properties_layout.addRow("Gain:", self.gain_spinbox)

        # Saturation
        self.saturation_spinbox = QSpinBox()
        self.saturation_spinbox.setRange(0, 255)
        self.saturation_spinbox.setValue(int(DEFAULT_CAPTURE_PROPERTIES.get(cv.CAP_PROP_SATURATION, 128)))
        self.saturation_spinbox.valueChanged.connect(self._on_saturation_changed)
        properties_layout.addRow("Saturation:", self.saturation_spinbox)

        # Sharpness
        self.sharpness_spinbox = QSpinBox()
        self.sharpness_spinbox.setRange(0, 255)
        self.sharpness_spinbox.setValue(int(DEFAULT_CAPTURE_PROPERTIES.get(cv.CAP_PROP_SHARPNESS, 128)))
        self.sharpness_spinbox.valueChanged.connect(self._on_sharpness_changed)
        properties_layout.addRow("Sharpness:", self.sharpness_spinbox)

        # Zoom
        self.zoom_spinbox = QSpinBox()
        self.zoom_spinbox.setRange(100, 255)
        self.zoom_spinbox.setValue(int(DEFAULT_CAPTURE_PROPERTIES.get(cv.CAP_PROP_ZOOM, 100)))
        self.zoom_spinbox.valueChanged.connect(self._on_zoom_changed)
        properties_layout.addRow("Zoom:", self.zoom_spinbox)

        # FOURCC
        self.fourcc_combo = QComboBox()
        self.fourcc_combo.addItems(["YUY2", "MJPG"])
        self.fourcc_combo.setCurrentText("YUY2")
        self.fourcc_combo.currentTextChanged.connect(self._on_fourcc_changed)
        properties_layout.addRow("FOURCC:", self.fourcc_combo)

        # Exposure
        self.exposure_spinbox = QSpinBox()
        self.exposure_spinbox.setRange(-10, 10)
        self.exposure_spinbox.setValue(int(DEFAULT_CAPTURE_PROPERTIES.get(cv.CAP_PROP_EXPOSURE, -5)))
        self.exposure_spinbox.valueChanged.connect(self._on_exposure_changed)
        properties_layout.addRow("Exposure:", self.exposure_spinbox)

        properties_group.setLayout(properties_layout)
        controls_layout.addWidget(properties_group)

        controls_layout.addStretch()

        main_layout.addLayout(controls_layout, stretch=1)

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        self.start_action = QAction("&Start/Restart Feed", self)
        self.start_action.setShortcut("Ctrl+S")
        self.start_action.triggered.connect(self._start_camera)
        file_menu.addAction(self.start_action)

        self.stop_action = QAction("S&top Feed", self)
        self.stop_action.setShortcut("Ctrl+T")
        self.stop_action.triggered.connect(self._stop_camera)
        self.stop_action.setEnabled(False)
        file_menu.addAction(self.stop_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _update_device_list(self) -> None:
        """Update the device list based on current capture API."""
        self.device_id_combo.clear()

        try:
            cameras = enumerate_cameras(self.current_capture_api)

            if cameras:
                for camera_info in cameras:
                    device_id = camera_info.index
                    device_name = camera_info.name if camera_info.name else f"Camera {device_id}"
                    self.device_id_combo.addItem(f"{device_id}: {device_name}", device_id)
            else:
                # Fallback if no cameras detected
                for i in range(9):
                    self.device_id_combo.addItem(f"Device {i}", i)
        except Exception as e:
            print(f"Error enumerating cameras: {e}")
            # Fallback if enumeration fails
            for i in range(9):
                self.device_id_combo.addItem(f"Device {i}", i)

    def _start_camera(self) -> None:
        """Start the camera feed."""
        if self.camera_feed is not None:
            self.camera_feed.release()

        device_id = self.device_id_combo.currentData()
        if device_id is None:
            device_id = 0

        # Collect current property values from UI
        capture_properties = {
            cv.CAP_PROP_FOCUS: self.focus_spinbox.value(),
            cv.CAP_PROP_BRIGHTNESS: self.brightness_spinbox.value(),
            cv.CAP_PROP_GAIN: self.gain_spinbox.value(),
            cv.CAP_PROP_SATURATION: self.saturation_spinbox.value(),
            cv.CAP_PROP_SHARPNESS: self.sharpness_spinbox.value(),
            cv.CAP_PROP_ZOOM: self.zoom_spinbox.value(),
            cv.CAP_PROP_FOURCC: cv.VideoWriter_fourcc(*self.fourcc_combo.currentText()),
            cv.CAP_PROP_EXPOSURE: self.exposure_spinbox.value()
        }

        self.camera_feed = CameraFeed(
            device_id=device_id,
            capture_api=self.current_capture_api,
            capture_properties=capture_properties
        )
        self.camera_feed.frame_captured.connect(self._on_frame_captured)
        self.camera_feed.error_occurred.connect(self._on_error_occurred)

        self.camera_feed.start()

        self.last_frame_time = time.perf_counter()
        self.frame_count = 0

        # Update UI state
        self.capture_api_combo.setEnabled(False)
        self.device_id_combo.setEnabled(False)
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(True)

    def _stop_camera(self) -> None:
        """Stop the camera feed and release resources."""
        if self.camera_feed is not None:
            self.camera_feed.release()
            self.camera_feed = None

        # Clear video display
        self.video_label.clear()
        self.fps_input.clear()
        self.fps_input.setPlaceholderText("FPS: --")

        # Update UI state
        self.capture_api_combo.setEnabled(True)
        self.device_id_combo.setEnabled(True)
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)

    @Slot(np.ndarray, bool)
    def _on_frame_captured(self, frame: np.ndarray, success: bool) -> None:
        """Handle captured frame and update display.

        Args:
            frame: The captured frame as numpy array.
            success: True if actual camera frame, False if error frame.
        """
        # Update FPS counter
        current_time = time.perf_counter()
        self.frame_count += 1

        if current_time - self.last_frame_time >= self.fps_update_interval:
            fps = self.frame_count / (current_time - self.last_frame_time)
            self.fps_input.setText(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_frame_time = current_time

        # Convert frame to QPixmap and display
        height, width, channels = frame.shape
        bytes_per_line = channels * width

        # Convert BGR to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        q_image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(q_image)

        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    @Slot(str)
    def _on_error_occurred(self, error_msg: str) -> None:
        """Handle error messages.

        Args:
            error_msg: The error message string.
        """
        print(f"Camera Error: {error_msg}")

    @Slot(int)
    def _on_capture_api_changed(self, index: int) -> None:
        """Handle capture API change.

        Args:
            index: New combo box index.
        """
        self.current_capture_api = self.capture_api_combo.currentData()
        self._update_device_list()

    @Slot(int)
    def _on_focus_changed(self, value: int) -> None:
        """Handle focus change.

        Args:
            value: New focus value.
        """
        if self.camera_feed is not None:
            self.camera_feed.update_capture_property(cv.CAP_PROP_FOCUS, value)

    @Slot(int)
    def _on_brightness_changed(self, value: int) -> None:
        """Handle brightness change.

        Args:
            value: New brightness value.
        """
        if self.camera_feed is not None:
            self.camera_feed.update_capture_property(cv.CAP_PROP_BRIGHTNESS, value)

    @Slot(int)
    def _on_gain_changed(self, value: int) -> None:
        """Handle gain change.

        Args:
            value: New gain value.
        """
        if self.camera_feed is not None:
            self.camera_feed.update_capture_property(cv.CAP_PROP_GAIN, value)

    @Slot(int)
    def _on_saturation_changed(self, value: int) -> None:
        """Handle saturation change.

        Args:
            value: New saturation value.
        """
        if self.camera_feed is not None:
            self.camera_feed.update_capture_property(cv.CAP_PROP_SATURATION, value)

    @Slot(int)
    def _on_sharpness_changed(self, value: int) -> None:
        """Handle sharpness change.

        Args:
            value: New sharpness value.
        """
        if self.camera_feed is not None:
            self.camera_feed.update_capture_property(cv.CAP_PROP_SHARPNESS, value)

    @Slot(int)
    def _on_zoom_changed(self, value: int) -> None:
        """Handle zoom change.

        Args:
            value: New zoom value.
        """
        if self.camera_feed is not None:
            self.camera_feed.update_capture_property(cv.CAP_PROP_ZOOM, value)

    @Slot(str)
    def _on_fourcc_changed(self, value: str) -> None:
        """Handle FOURCC codec change.

        Args:
            value: New FOURCC codec string.
        """
        if self.camera_feed is not None:
            fourcc = cv.VideoWriter_fourcc(*value)
            self.camera_feed.update_capture_property(cv.CAP_PROP_FOURCC, fourcc)

    @Slot(int)
    def _on_exposure_changed(self, value: int) -> None:
        """Handle exposure change.

        Args:
            value: New exposure value.
        """
        if self.camera_feed is not None:
            self.camera_feed.update_capture_property(cv.CAP_PROP_EXPOSURE, value)

    def closeEvent(self, event) -> None:
        """Handle window close event.

        Args:
            event: The close event.
        """
        if self.camera_feed is not None:
            self.camera_feed.release()

        event.accept()


def main() -> None:
    """Main entry point for the test application."""
    app = QApplication(sys.argv)

    window = CameraFeedTestWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
