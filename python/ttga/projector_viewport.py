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

"""Projector viewport widget for displaying projector output."""

import numpy as np
import cv2 as cv
from PySide6 import QtWidgets, QtGui, QtCore


class ProjectorViewport(QtWidgets.QLabel):
    """Viewport widget for displaying projector output.

    This widget maintains the aspect ratio of the projector resolution
    when resized.
    """

    def __init__(self, resolution: tuple[int, int], projector_name: str,
                 parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the projector viewport.

        Args:
            resolution: Tuple of (width, height) for the projector resolution.
            projector_name: Name of the projector.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.resolution = resolution
        self.projector_name = projector_name

        # Set size policy to maintain aspect ratio
        self.setScaledContents(False)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Generate test image
        self._generate_test_image()

    def _generate_test_image(self) -> None:
        """Generate a test image with black background, white border, and projector name."""
        width, height = self.resolution

        # Create black image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add white border (10 pixels)
        cv.rectangle(image, (0, 0), (width - 1, height - 1), (255, 255, 255), 10)

        # Add projector name in the center
        font = cv.FONT_HERSHEY_SIMPLEX
        text = self.projector_name

        # Calculate font scale to make text about 1/4 of width
        target_width = width // 4
        font_scale = 1.0
        text_size = cv.getTextSize(text, font, font_scale, 2)[0]

        # Adjust font scale
        if text_size[0] > 0:
            font_scale = target_width / text_size[0]
            text_size = cv.getTextSize(text, font, font_scale, 2)[0]

        # Center the text
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        # Draw text in white
        cv.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv.LINE_AA)

        # Convert to QPixmap
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(
            rgb_image.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888
        )
        self.original_pixmap = QtGui.QPixmap.fromImage(q_image)
        self.setPixmap(self.original_pixmap)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Handle resize event to maintain aspect ratio.

        Args:
            event: Resize event.
        """
        super().resizeEvent(event)

        if hasattr(self, 'original_pixmap'):
            # Scale pixmap to fit while maintaining aspect ratio
            scaled_pixmap = self.original_pixmap.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
