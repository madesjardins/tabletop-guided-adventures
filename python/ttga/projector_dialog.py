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

"""Dialog window for projector display."""

from PySide6 import QtWidgets, QtCore, QtGui

from .projector_viewport import ProjectorViewport


class ProjectorDialog(QtWidgets.QDialog):
    """Dialog window for displaying projector output.

    Press F to toggle fullscreen mode.
    """

    def __init__(self, projector_name: str, resolution: tuple[int, int],
                 parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the projector dialog.

        Args:
            projector_name: Name of the projector.
            resolution: Tuple of (width, height) for the projector resolution.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.projector_name = projector_name
        self.resolution = resolution

        self.setWindowTitle(f"Projector - {projector_name} - {resolution[0]}x{resolution[1]}")

        # Remove close button and prevent closing with Escape
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window |
            QtCore.Qt.WindowType.CustomizeWindowHint |
            QtCore.Qt.WindowType.WindowTitleHint |
            QtCore.Qt.WindowType.WindowMinMaxButtonsHint
        )

        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create viewport
        self.viewport = ProjectorViewport(resolution, projector_name)
        layout.addWidget(self.viewport)

        # Allow dialog to be resized smaller
        self.setMinimumSize(100, 100)

        # Flag to allow closing when explicitly requested
        self._allow_close = False

        # Set initial size to half resolution after layout is set up
        # Use QTimer to ensure this happens after the window is shown
        QtCore.QTimer.singleShot(0, lambda: self.resize(resolution[0] // 2, resolution[1] // 2))

    def close(self) -> bool:
        """Override close to allow explicit closing.

        Returns:
            True if dialog was closed.
        """
        self._allow_close = True
        return super().close()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key press events.

        Args:
            event: Key event.
        """
        if event.key() == QtCore.Qt.Key.Key_F:
            # Toggle fullscreen
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == QtCore.Qt.Key.Key_Escape:
            # Prevent Escape from closing the dialog
            event.ignore()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle close event to prevent closing.

        Args:
            event: Close event.
        """
        # Only allow closing if explicitly requested (e.g., via Delete button)
        if self._allow_close:
            event.accept()
        else:
            event.ignore()
