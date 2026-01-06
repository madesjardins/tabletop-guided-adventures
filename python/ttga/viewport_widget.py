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

"""Viewport widget module for displaying camera feeds.

This module contains the ViewportWidget class which displays the selected
camera feed(s) in the main application window.
"""

from __future__ import annotations

from PySide6 import QtWidgets, QtCore


class ViewportWidget(QtWidgets.QLabel):
    """Widget for displaying camera feed viewport.

    This widget displays the selected camera feed(s) and handles
    rendering of the video stream.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the viewport widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        # Set default appearance
        self.setStyleSheet("background-color: rgb(32, 32, 32); border: 1px solid rgb(128, 128, 128);")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setText("No camera selected")
        self.setMinimumSize(640, 480)
