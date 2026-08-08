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

import os
import threading
from typing import Optional

from PySide6 import QtCore

from .camera_manager import CameraManager
from .camera_calibration import CameraCalibration
from .projector_manager import ProjectorManager
from .zone_manager import ZoneManager
from .speech_recognition import SpeechRecognizer
from .narrator import Narrator
from .llm_client import LLMClient, LLMConfig
from .game_loader import GameLoader, GameInfo
from .game_base import GameBase
from .help_agent import HelpAgent


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
        game_loader: Game loader for discovering and loading game plugins.
        current_game: Currently loaded game instance (None if no game loaded).
        current_game_info: Info about the currently loaded game.

    Signals:
        speech_partial_result: Emitted when partial speech recognition result is received.
        speech_final_result: Emitted when final speech recognition result is received.
        game_loaded: Emitted when a game is loaded.
        game_unloaded: Emitted when a game is unloaded.
    """

    speech_partial_result = QtCore.Signal(str)
    speech_final_result = QtCore.Signal(str)
    game_loaded = QtCore.Signal(str)  # game name
    game_unloaded = QtCore.Signal()
    # Emitted after an asynchronous LLM model load attempt: (success, message).
    # On success, message is the loaded model name (empty when unloaded).
    llm_model_loaded = QtCore.Signal(bool, str)
    # Emitted when the user changes the narrator persona text.
    persona_changed = QtCore.Signal(str)
    # Emitted when the user changes temperature or max_tokens.
    narration_params_changed = QtCore.Signal(float, int)

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

        # Local LLM narrator brains (optional). Shared by all games. No model
        # is loaded initially, so is_available() is False and games fall back
        # to scripted text until the user selects a model.
        self.llm_client = LLMClient(LLMConfig(enabled=True))
        self.llm_model_path: Optional[str] = None
        self.llm_n_gpu_layers: int = -1

        # Help / Q&A agent (core-level, voice-triggered, read-only).
        self.help_agent = HelpAgent(self.llm_client, self.narrator)

        # Narrator persona (user-editable, overrides game defaults when non-empty).
        self._persona: str = ""
        # LLM generation parameters (user-editable, override engine defaults).
        self._temperature: float = 0.7
        self._max_tokens: int = 120

        # Refresh rates
        self.viewports_refresh_rate: int = 30
        self.projectors_refresh_rate: int = 15
        self.qr_code_refresh_rate: int = 5

        # Game management
        self.game_loader = GameLoader()
        self.current_game: Optional[GameBase] = None
        self.current_game_info: Optional[GameInfo] = None

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

        The help agent is checked *before* normal game routing so that the
        wake phrase ("Help me...") works in both the main menu and in-game.

        Args:
            text: Recognized text.
        """
        # Emit signal for UI and other listeners
        self.speech_final_result.emit(text)

        # Help agent pre-filter (consumes wake-phrase utterances).
        if self.help_agent.handle_utterance(text, self.current_game):
            return

        # Pass to current game if loaded
        if self.current_game:
            self.current_game.on_speech_command(text)

    def set_llm_model(
        self, model_path: Optional[str], n_gpu_layers: int = -1
    ) -> None:
        """Load (or unload) the LLM model on a background thread.

        Loading a multi-gigabyte GGUF model is slow, so it runs off the Qt
        thread to keep the UI responsive. The outcome is reported via the
        :attr:`llm_model_loaded` signal.

        Args:
            model_path: Path to a ``.gguf`` model, or a falsy value to unload
                (disabling the LLM so games use scripted text).
            n_gpu_layers: GPU layers to offload (-1 = all, 0 = CPU only).
        """
        self.llm_n_gpu_layers = n_gpu_layers
        self.llm_client.config.n_gpu_layers = n_gpu_layers

        def _load() -> None:
            try:
                if not model_path:
                    self.llm_client.unload()
                    self.llm_model_path = None
                    self.llm_model_loaded.emit(True, "")
                    return
                self.llm_client.load_model(model_path)
                self.llm_model_path = model_path
                self.llm_model_loaded.emit(True, os.path.basename(model_path))
            except Exception as e:
                self.llm_model_loaded.emit(False, str(e))

        threading.Thread(target=_load, daemon=True).start()

    @property
    def persona(self) -> str:
        """The user-configured narrator persona (empty = use game default)."""
        return self._persona

    def set_persona(self, persona: str) -> None:
        """Set the narrator persona and propagate to any active game.

        Args:
            persona: The new persona text. Empty string means "use the
                game's built-in default persona".
        """
        self._persona = persona
        self.persona_changed.emit(persona)

    @property
    def temperature(self) -> float:
        """The user-configured LLM sampling temperature."""
        return self._temperature

    def set_temperature(self, temperature: float) -> None:
        """Set the LLM sampling temperature and propagate to active games.

        Args:
            temperature: Sampling temperature (0.0–2.0).
        """
        self._temperature = temperature
        self.narration_params_changed.emit(temperature, self._max_tokens)

    @property
    def max_tokens(self) -> int:
        """The user-configured max generation length."""
        return self._max_tokens

    def set_max_tokens(self, max_tokens: int) -> None:
        """Set the max generation length and propagate to active games.

        Args:
            max_tokens: Maximum tokens to generate per phrase.
        """
        self._max_tokens = max_tokens
        self.narration_params_changed.emit(self._temperature, max_tokens)

    @QtCore.Slot(int)
    def set_qr_code_refresh_rate(self, fps: int) -> None:
        """Set the QR code scanning refresh rate.

        Args:
            fps: Refresh rate in frames per second.
        """
        self.qr_code_refresh_rate = fps

        # Update QR detectors in current game if loaded
        if self.current_game:
            self.current_game.set_qr_detectors_refresh_rate(fps)

    def load_game(self, game_info: GameInfo) -> bool:
        """Load a game plugin.

        Unloads the current game if one is loaded, then loads the new game.

        Args:
            game_info: GameInfo object for the game to load.

        Returns:
            True if game loaded successfully, False otherwise.
        """
        # Unload current game if any
        if self.current_game:
            self.unload_game()

        # Load the new game
        game_instance = self.game_loader.load_game(game_info, self)
        if game_instance is None:
            return False

        self.current_game = game_instance
        self.current_game_info = game_info

        # Call game's on_load
        try:
            self.current_game.on_load()
            self.game_loaded.emit(game_info.name)
            return True
        except Exception as e:
            print(f"Error calling on_load for game {game_info.name}: {e}")
            import traceback
            traceback.print_exc()
            self.current_game = None
            self.current_game_info = None
            return False

    def unload_game(self) -> None:
        """Unload the current game if one is loaded."""
        if self.current_game is None:
            return

        # Call game's on_unload
        try:
            self.current_game.on_unload()
        except Exception as e:
            print(f"Error calling on_unload for game: {e}")

        # Clean up
        if self.current_game_info:
            self.game_loader.unload_game(self.current_game_info)

        self.current_game = None
        self.current_game_info = None
        self.game_unloaded.emit()

    def get_game_camera_overlay(self, zone_name: str):
        """Get the camera overlay image from the current game for a specific zone.

        Args:
            zone_name: Name of the zone to get overlay for.

        Returns:
            numpy.ndarray with shape (height, width, 4) in BGRA format with game coordinates,
            or None if no game loaded or no overlay for this zone.
        """
        if self.current_game is None:
            return None
        return self.current_game.get_camera_overlay(zone_name)

    def get_game_projector_overlay(self, zone_name: str):
        """Get the projector overlay image from the current game for a specific zone.

        Args:
            zone_name: Name of the zone to get overlay for.

        Returns:
            numpy.ndarray with shape (height, width, 4) in BGRA format with game coordinates,
            or None if no game loaded or no overlay for this zone.
        """
        if self.current_game is None:
            return None
        return self.current_game.get_projector_overlay(zone_name)

    def allows_locked_corner_adjustment(self) -> bool:
        """Check if the current game allows corner adjustments when calibrated.

        Returns:
            True if current game allows corner adjustments with locked vertices, False otherwise.
        """
        if self.current_game_info is None:
            return False
        return self.current_game_info.allow_locked_corner_adjustment

    def release_all(self) -> None:
        """Release all resources."""
        # Unload current game
        if self.current_game:
            self.unload_game()

        if self.speech_recognizer is not None:
            self.speech_recognizer.stop()
            self.speech_recognizer = None
        self.help_agent.shutdown()
        self.llm_client.unload()
        self.narrator.shutdown()
        self.camera_manager.release_all()
        self.projector_manager.clear_all()
        self.zone_manager.clear_all()
