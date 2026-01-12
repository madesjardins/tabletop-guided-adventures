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

    Signals:
        vertex_updated: Emitted when a vertex is updated via dragging (zone_name: str).
    """

    vertex_updated = QtCore.Signal(str)

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

        # Zone manager reference for overlay compositing
        self._zone_manager = None

        # MainCore reference for game overlay queries
        self._main_core = None

        # Vertex dragging state
        self._dragging_vertex: tuple | None = None  # (zone, vertex_idx)
        self._drag_start_pos: tuple[int, int] | None = None

        # Current refresh rate (fps)
        self._fps = 30

        # Frame update timer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update_display)

        # Enable mouse tracking for interactions
        self.setMouseTracking(True)

        # Generate initial test image
        self._generate_test_image()

        # Start timer
        self._timer.start(int(1000 / self._fps))

    def set_zone_manager(self, zone_manager) -> None:
        """Set the zone manager for overlay compositing.

        Args:
            zone_manager: ZoneManager instance.
        """
        self._zone_manager = zone_manager

    def set_main_core(self, main_core) -> None:
        """Set the main core for game overlay queries.

        Args:
            main_core: MainCore instance.
        """
        self._main_core = main_core

    def _update_display(self) -> None:
        """Update the displayed image with zone overlays."""
        width, height = self.resolution

        # Check if we have zones with projector mapping
        if self._zone_manager is not None:
            zones = self._zone_manager.get_zones_with_projector_mapping(self.projector_name)

            if zones:
                # Start with black base image
                image = np.zeros((height, width, 3), dtype=np.uint8)

                # Composite zone overlays (if draw_locked_borders is enabled)
                overlays_to_composite = []
                for zone in zones:
                    overlay_data = zone.get_projector_overlay((height, width, 3))
                    if overlay_data is not None:
                        overlays_to_composite.append(overlay_data)

                for overlay_data in overlays_to_composite:
                    overlay, x, y, w, h = overlay_data

                    # Extract ROI
                    roi = image[y:y + h, x:x + w]

                    # Composite BGRA overlay onto BGR ROI
                    alpha = overlay[:, :, 3:4] / 255.0
                    blended_roi = (roi * (1 - alpha) + overlay[:, :, :3] * alpha).astype(np.uint8)

                    # Put blended ROI back
                    image[y:y + h, x:x + w] = blended_roi

                # Always composite game overlays (independent of zone overlay settings)
                if self._main_core is not None:
                    for zone in zones:
                        # Only process zones with calibrated projector mapping
                        if not zone.projector_mapping or not zone.projector_mapping.is_calibrated:
                            continue

                        # Get game overlay for this zone
                        game_overlay = self._main_core.get_game_projector_overlay(zone.name)
                        if game_overlay is None:
                            continue

                        # Check if overlay has any non-zero alpha values
                        if np.max(game_overlay[:, :, 3]) == 0:
                            continue  # Skip fully transparent overlays

                        # Get transformation matrix and ROI
                        matrix = zone.projector_mapping.game_to_projector_matrix
                        roi = zone.projector_mapping.roi
                        if matrix is None or roi is None:
                            continue

                        # Calculate ROI dimensions
                        roi_width = roi['max_x'] - roi['min_x']
                        roi_height = roi['max_y'] - roi['min_y']

                        # Warp game overlay to projector coordinates
                        warped_overlay = cv.warpPerspective(
                            game_overlay,
                            matrix,
                            (roi_width, roi_height)
                        )

                        # Ensure we don't go out of frame bounds
                        x_start = max(0, roi['min_x'])
                        y_start = max(0, roi['min_y'])
                        x_end = min(image.shape[1], roi['max_x'])
                        y_end = min(image.shape[0], roi['max_y'])

                        # Calculate actual dimensions after bounds checking
                        actual_width = x_end - x_start
                        actual_height = y_end - y_start

                        if actual_width <= 0 or actual_height <= 0:
                            continue

                        # Crop warped overlay if needed
                        warped_crop = warped_overlay[:actual_height, :actual_width]

                        # Extract ROI from frame
                        frame_roi = image[y_start:y_end, x_start:x_end]

                        # Composite BGRA overlay onto BGR ROI
                        alpha = warped_crop[:, :, 3:4] / 255.0
                        blended = (frame_roi * (1 - alpha) + warped_crop[:, :, :3] * alpha).astype(np.uint8)

                        # Put blended ROI back into frame
                        image[y_start:y_end, x_start:x_end] = blended

                self._display_image(image)
                return

        # No zones or no zone manager - show test image
        self._generate_test_image()

    def _display_image(self, image: np.ndarray) -> None:
        """Display a BGR image in the viewport.

        Args:
            image: BGR image to display.
        """
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

        # Scale to fit viewport
        scaled_pixmap = self.original_pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)

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

        # Display the image
        self._display_image(image)

    def _viewport_to_frame_coords(self, viewport_x: int, viewport_y: int) -> tuple[int, int] | None:
        """Convert viewport coordinates to projector frame coordinates.

        Args:
            viewport_x: X coordinate in viewport.
            viewport_y: Y coordinate in viewport.

        Returns:
            Tuple of (x, y) in projector frame coordinates, or None if invalid.
        """
        if not hasattr(self, 'original_pixmap'):
            return None

        # Get viewport and pixmap dimensions
        viewport_width = self.width()
        viewport_height = self.height()

        frame_width, frame_height = self.resolution

        # Calculate scaled pixmap dimensions (maintaining aspect ratio)
        frame_aspect = frame_width / frame_height
        viewport_aspect = viewport_width / viewport_height

        if frame_aspect > viewport_aspect:
            # Width-limited
            scaled_width = viewport_width
            scaled_height = int(viewport_width / frame_aspect)
        else:
            # Height-limited
            scaled_height = viewport_height
            scaled_width = int(viewport_height * frame_aspect)

        # Calculate offset (pixmap is centered)
        x_offset = (viewport_width - scaled_width) // 2
        y_offset = (viewport_height - scaled_height) // 2

        # Check if click is within the scaled pixmap area
        if viewport_x < x_offset or viewport_x >= x_offset + scaled_width:
            return None
        if viewport_y < y_offset or viewport_y >= y_offset + scaled_height:
            return None

        # Normalize to scaled pixmap coordinates
        rel_x = viewport_x - x_offset
        rel_y = viewport_y - y_offset

        # Map to frame coordinates
        frame_x = int((rel_x / scaled_width) * frame_width)
        frame_y = int((rel_y / scaled_height) * frame_height)

        # Clamp to frame bounds
        frame_x = max(0, min(frame_x, frame_width - 1))
        frame_y = max(0, min(frame_y, frame_height - 1))

        return (frame_x, frame_y)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press events for vertex dragging.

        Args:
            event: Mouse event.
        """
        pos = event.position()
        mouse_x = int(pos.x())
        mouse_y = int(pos.y())

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Check for vertex dragging
            if self._zone_manager is not None:
                # Convert viewport coords to frame coords
                frame_coords = self._viewport_to_frame_coords(mouse_x, mouse_y)

                if frame_coords is not None:
                    from .constants import VERTEX_RADIUS
                    frame_x, frame_y = frame_coords

                    # Find vertex at this position
                    result = self._zone_manager.find_projector_vertex_at_position(
                        self.projector_name, frame_x, frame_y, VERTEX_RADIUS
                    )

                    if result is not None:
                        zone, vertex_idx = result
                        self._dragging_vertex = (zone, vertex_idx)
                        self._drag_start_pos = (mouse_x, mouse_y)
                        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                        event.accept()
                        return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move events for vertex dragging.

        Args:
            event: Mouse event.
        """
        pos = event.position()
        mouse_x = int(pos.x())
        mouse_y = int(pos.y())

        # Handle vertex dragging
        if self._dragging_vertex is not None:
            zone, vertex_idx = self._dragging_vertex

            # Convert viewport coords to frame coords
            frame_coords = self._viewport_to_frame_coords(mouse_x, mouse_y)

            if frame_coords is not None:
                frame_x, frame_y = frame_coords

                # Update vertex position
                vertices = list(zone.projector_mapping.vertices)
                vertices[vertex_idx] = (frame_x, frame_y)
                zone.projector_mapping.vertices = vertices

                # Invalidate overlay to force regeneration
                zone.projector_mapping.invalidate_overlay()

                # Emit signal to update UI
                self.vertex_updated.emit(zone.name)

                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release events for vertex dragging.

        Args:
            event: Mouse event.
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._dragging_vertex is not None:
            # End vertex dragging
            self._dragging_vertex = None
            self._drag_start_pos = None
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

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
