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

"""Main core module for the Tabletop Guided Adventures application.

This module contains the MainCore class which manages the application's
core functionality and state.
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore

from .camera_manager import CameraManager
from .camera_calibration import CameraCalibration
from .projector_manager import ProjectorManager
from .zone_manager import ZoneManager
from .speech_recognition import SpeechRecognizer
from .narrator import Narrator


class MainCore(QtCore.QObject):
    """Main core class for managing application state and functionality.

    This class serves as the central coordinator for the application,
    managing cameras, game state, and other core functionality.

    Attributes:
        camera_manager: Manager for all active cameras.
        camera_calibration: Camera calibration manager.
        projector_manager: Manager for all projectors.
        zone_manager: Manager for all zones.
        speech_recognizer: Speech recognition instance.
        speech_model_path: Path to Vosk model for speech recognition.
        speech_device_index: Audio input device index for speech recognition.
        speech_threshold: Similarity threshold for speech recognition matching.
        narrator: Narrator instance for TTS and audio playback.
        viewports_refresh_rate: Refresh rate for camera viewports in FPS.
        projectors_refresh_rate: Refresh rate for projector displays in FPS.
        qr_code_refresh_rate: Refresh rate for QR code scanning in FPS.

    Signals:
        speech_partial_result: Emitted when partial speech recognition result is received.
        speech_final_result: Emitted when final speech recognition result is received.
    """

    speech_partial_result = QtCore.Signal(str)
    speech_final_result = QtCore.Signal(str)

    def __init__(self) -> None:
        """Initialize the main core."""
        super().__init__()
        self.camera_manager = CameraManager()
        self.camera_calibration = CameraCalibration()
        self.projector_manager = ProjectorManager()
        self.zone_manager = ZoneManager()

        # Speech recognition
        self.speech_recognizer: Optional[SpeechRecognizer] = None
        self.speech_model_path: Optional[str] = None
        self.speech_device_index: Optional[int] = None
        self.speech_threshold: float = 0.7

        # Narrator (TTS and audio)
        self.narrator = Narrator()

        # Refresh rates
        self.viewports_refresh_rate: int = 30
        self.projectors_refresh_rate: int = 15
        self.qr_code_refresh_rate: int = 5

    def update_speech_recognizer(
        self, model_path: Optional[str] = None, device_index: Optional[int] = None
    ) -> None:
        """Update speech recognizer with new configuration.

        Args:
            model_path: Path to Vosk model (None to keep current).
            device_index: Audio device index (None to keep current).
        """
        # Update configuration
        if model_path is not None:
            self.speech_model_path = model_path
        if device_index is not None:
            self.speech_device_index = device_index

        # Stop existing recognizer
        if self.speech_recognizer is not None:
            self.speech_recognizer.stop()
            self.speech_recognizer = None

        # Start new recognizer if we have both model and device
        if self.speech_model_path and self.speech_device_index is not None:
            try:
                self.speech_recognizer = SpeechRecognizer(
                    model_path=self.speech_model_path,
                    device_index=self.speech_device_index
                )
                self.speech_recognizer.partial_result.connect(self.speech_partial_result.emit)
                self.speech_recognizer.final_result.connect(self._on_speech_final_result)
                self.speech_recognizer.start()
            except Exception as e:
                print(f"Failed to start speech recognizer: {e}")
                self.speech_recognizer = None

    @QtCore.Slot(str)
    def _on_speech_final_result(self, text: str) -> None:
        """Handle final speech recognition result.

        Args:
            text: Recognized text.
        """
        # Emit signal for UI and other listeners
        self.speech_final_result.emit(text)

        # TODO: When a game is loaded, call game method with final result
        # if self.current_game:
        #     self.current_game.on_speech_command(text)

    @QtCore.Slot(int)
    def set_qr_code_refresh_rate(self, fps: int) -> None:
        """Set the QR code scanning refresh rate.

        Args:
            fps: Refresh rate in frames per second.
        """
        self.qr_code_refresh_rate = fps
        # TODO: Apply to QR code scanner when implemented

    def release_all(self) -> None:
        """Release all resources."""
        if self.speech_recognizer is not None:
            self.speech_recognizer.stop()
            self.speech_recognizer = None
        self.narrator.shutdown()
        self.camera_manager.release_all()
        self.projector_manager.clear_all()
        self.zone_manager.clear_all()
