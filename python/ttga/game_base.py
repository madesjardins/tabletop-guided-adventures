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

"""Base class for game plugins.

This module provides the GameBase abstract class that all game plugins must inherit from.
Games have access to all application resources through the MainCore instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_core import MainCore


class GameBase(ABC):
    """Abstract base class for all game plugins.

    Game plugins inherit from this class and implement the required methods.
    Games have full access to the application's resources through self.core:
    - Cameras (core.camera_manager)
    - Projectors (core.projector_manager)
    - Zones (core.zone_manager)
    - Narrator/TTS (core.narrator)
    - Speech recognition (core.speech_recognizer)

    Example:
        >>> class MyGame(GameBase):
        ...     def get_metadata(self):
        ...         return {
        ...             'name': 'My Game',
        ...             'version': '1.0.0',
        ...             'author': 'Your Name',
        ...             'description': 'A simple game'
        ...         }
        ...
        ...     def on_load(self):
        ...         self.core.narrator.synthesize_and_play("Welcome to my game!")
        ...
        ...     def on_speech_command(self, text):
        ...         if "roll dice" in text.lower():
        ...             self.core.narrator.synthesize_and_play("Rolling dice!")
    """

    def __init__(self, core: MainCore) -> None:
        """Initialize the game with access to the application core.

        Args:
            core: MainCore instance providing access to all application resources.
        """
        self.core = core

    @abstractmethod
    def get_metadata(self) -> dict[str, str]:
        """Get game metadata.

        Returns:
            Dictionary with keys: 'name', 'version', 'author', 'description'.

        Example:
            >>> return {
            ...     'name': 'Example Game',
            ...     'version': '1.0.0',
            ...     'author': 'John Doe',
            ...     'description': 'A simple example game'
            ... }
        """
        pass

    @abstractmethod
    def on_load(self) -> None:
        """Called when the game is loaded.

        Use this to initialize game state, set up UI elements, configure
        projectors/cameras, etc.

        Example:
            >>> def on_load(self):
            ...     self.score = 0
            ...     self.core.narrator.synthesize_and_play("Game loaded!")
        """
        pass

    @abstractmethod
    def on_unload(self) -> None:
        """Called when the game is unloaded.

        Use this to clean up resources, save state, etc.

        Example:
            >>> def on_unload(self):
            ...     self.core.narrator.synthesize_and_play("Thanks for playing!")
        """
        pass

    def on_speech_command(self, text: str) -> None:
        """Handle speech recognition results.

        Override this method to respond to voice commands.

        Args:
            text: Recognized speech text.

        Example:
            >>> def on_speech_command(self, text):
            ...     if "help" in text.lower():
            ...         self.core.narrator.synthesize_and_play("Say roll dice to play")
        """
        pass

    def on_camera_frame(self, camera_name: str, frame) -> None:
        """Handle camera frame updates.

        Override this method to process camera frames (e.g., for object detection,
        QR code scanning, etc.).

        Args:
            camera_name: Name of the camera that captured the frame.
            frame: OpenCV frame (numpy array).

        Example:
            >>> def on_camera_frame(self, camera_name, frame):
            ...     # Detect objects in frame
            ...     pass
        """
        pass

    def on_zone_calibrated(self, zone_name: str) -> None:
        """Handle zone calibration events.

        Override this method to respond when a zone is calibrated.

        Args:
            zone_name: Name of the zone that was calibrated.

        Example:
            >>> def on_zone_calibrated(self, zone_name):
            ...     self.core.narrator.synthesize_and_play(f"Zone {zone_name} ready!")
        """
        pass

    def show_dialog(self, parent=None) -> None:
        """Show the game's configuration dialog.

        Override this method to display a custom game dialog for configuration,
        zone mapping, and match management.

        Args:
            parent: Parent widget for the dialog.

        Example:
            >>> def show_dialog(self, parent=None):
            ...     dialog = MyGameDialog(self.core, parent)
            ...     dialog.exec()
        """
        pass

    def set_qr_detectors_refresh_rate(self, fps: int) -> None:
        """Set the refresh rate for all QR detectors used by this game.

        Override this method if your game uses QR detectors and needs to
        update their refresh rates dynamically.

        Args:
            fps: New refresh rate in frames per second.

        Example:
            >>> def set_qr_detectors_refresh_rate(self, fps):
            ...     for detector in self.qr_detectors.values():
            ...         detector.set_refresh_rate(fps)
        """
        pass

    def get_camera_overlay(self, zone_name: str):
        """Get the camera overlay image for a specific zone in game coordinates.

        Override this method to provide camera overlay images that will be warped
        and composited onto the camera view in the viewport.

        Args:
            zone_name: Name of the zone to get overlay for.

        Returns:
            numpy.ndarray with shape (height, width, 4) in BGRA format with game coordinates,
            or None if no overlay for this zone. The image dimensions should match
            zone.height * zone.resolution by zone.width * zone.resolution.

        Example:
            >>> def get_camera_overlay(self, zone_name):
            ...     return self.camera_overlays.get(zone_name)
        """
        return None

    def get_projector_overlay(self, zone_name: str):
        """Get the projector overlay image for a specific zone in game coordinates.

        Override this method to provide projector overlay images that will be warped
        and displayed on the projector.

        Args:
            zone_name: Name of the zone to get overlay for.

        Returns:
            numpy.ndarray with shape (height, width, 4) in BGRA format with game coordinates,
            or None if no overlay for this zone. The image dimensions should match
            zone.height * zone.resolution by zone.width * zone.resolution.

        Example:
            >>> def get_projector_overlay(self, zone_name):
            ...     return self.projector_overlays.get(zone_name)
        """
        return None
