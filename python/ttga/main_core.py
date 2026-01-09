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

from .camera_manager import CameraManager
from .camera_calibration import CameraCalibration
from .projector_manager import ProjectorManager
from .zone_manager import ZoneManager


class MainCore:
    """Main core class for managing application state and functionality.

    This class serves as the central coordinator for the application,
    managing cameras, game state, and other core functionality.

    Attributes:
        camera_manager: Manager for all active cameras.
        camera_calibration: Camera calibration manager.
        projector_manager: Manager for all projectors.
        zone_manager: Manager for all zones.
    """

    def __init__(self) -> None:
        """Initialize the main core."""
        self.camera_manager = CameraManager()
        self.camera_calibration = CameraCalibration()
        self.projector_manager = ProjectorManager()
        self.zone_manager = ZoneManager()

    def release_all(self) -> None:
        """Release all resources."""
        self.camera_manager.release_all()
        self.projector_manager.clear_all()
        self.zone_manager.clear_all()
