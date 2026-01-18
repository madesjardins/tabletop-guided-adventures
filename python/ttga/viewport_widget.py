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

from typing import Callable

import cv2 as cv
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui


class ViewportWidget(QtWidgets.QLabel):
    """Widget for displaying camera feed viewport.

    This widget displays the selected camera feed(s) and handles
    rendering of the video stream with configurable refresh rate.

    Signals:
        vertex_updated: Emitted when a vertex is updated via dragging (zone_name: str).
    """

    vertex_updated = QtCore.Signal(str)

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
        self.setScaledContents(False)

        # Frame update timer
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update_frame)

        # Callback to get frames from selected cameras
        self._get_frames_callback: Callable[[], list[np.ndarray]] | None = None
        self._get_camera_ids_callback: Callable[[], list[int]] | None = None
        self._get_camera_names_callback: Callable[[], list[str]] | None = None

        # Zone manager reference for overlay compositing
        self._zone_manager = None

        # MainCore reference for game overlay queries
        self._main_core = None

        # Vertex dragging state
        self._dragging_vertex: tuple | None = None  # (zone, vertex_idx, camera_name)
        self._drag_start_pos: tuple[int, int] | None = None

        # Current refresh rate (fps)
        self._fps = 30

        # Zoom state per cell: {cell_index: {'zoom': float, 'center_x': float, 'center_y': float, 'pan_x': float, 'pan_y': float}}
        self._zoom_states: dict[int, dict[str, float]] = {}

        # Pan state for middle mouse button
        self._is_panning = False
        self._pan_start_pos: tuple[int, int] | None = None
        self._pan_cell_idx = -1

        # Current grid layout info
        self._current_rows = 1
        self._current_cols = 1
        self._cell_width = 0
        self._cell_height = 0

        # Track selected camera identities to detect selection changes
        self._last_camera_ids: list[int] = []

        # Enable mouse tracking for wheel events
        self.setMouseTracking(True)

    def set_get_frames_callback(self, callback: Callable[[], list[np.ndarray]], get_camera_ids_callback: Callable[[], list[int]] | None = None, get_camera_names_callback: Callable[[], list[str]] | None = None) -> None:
        """Set the callback function to get frames from selected cameras.

        Args:
            callback: Function that returns list of frames from selected cameras.
            get_camera_ids_callback: Optional function that returns list of camera IDs for selection tracking.
            get_camera_names_callback: Optional function that returns list of camera names for zone overlay.
        """
        self._get_frames_callback = callback
        self._get_camera_ids_callback = get_camera_ids_callback
        self._get_camera_names_callback = get_camera_names_callback

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

    def set_fps(self, fps: int) -> None:
        """Set the viewport refresh rate.

        Args:
            fps: Frames per second (5-60).
        """
        self._fps = max(5, min(60, fps))
        if self._timer.isActive():
            self._timer.setInterval(int(1000 / self._fps))

    def start(self) -> None:
        """Start the viewport frame updates."""
        if not self._timer.isActive():
            self._timer.start(int(1000 / self._fps))

    def stop(self) -> None:
        """Stop the viewport frame updates."""
        self._timer.stop()

    def reset_zoom(self) -> None:
        """Reset all zoom states."""
        self._zoom_states.clear()

    @QtCore.Slot()
    def _update_frame(self) -> None:
        """Update the displayed frame(s)."""
        if self._get_frames_callback is None:
            return

        # Get frames from selected cameras
        frames = self._get_frames_callback()

        if not frames:
            # No cameras selected
            self.clear()
            self.setText("No camera selected")
            return

        # Filter out None frames
        valid_frames = [f for f in frames if f is not None]

        if not valid_frames:
            # No valid frames available yet
            self.clear()
            self.setText("Waiting for camera feed...")
            return

        # Check if camera selection changed (by IDs, not just count)
        current_camera_ids = []
        if self._get_camera_ids_callback is not None:
            current_camera_ids = self._get_camera_ids_callback()

        if current_camera_ids != self._last_camera_ids:
            self.reset_zoom()
            self._last_camera_ids = current_camera_ids.copy()

        # Composite zone overlays on frames (returns QImages)
        qimages_with_overlays = self._composite_zone_overlays(valid_frames)

        # Compose QImages into grid (returns QImage)
        composed_qimage = self._compose_frames(qimages_with_overlays)

        # Display QImage directly (no conversion!)
        self._display_qimage(composed_qimage)

    def _composite_zone_overlays(self, frames: list[np.ndarray]) -> list[QtGui.QImage]:
        """Composite zone overlays on camera frames.

        Args:
            frames: List of camera frames (numpy BGR).

        Returns:
            List of QImages with zone overlays composited.
        """
        if self._zone_manager is None or self._get_camera_names_callback is None:
            # No zone manager - just convert frames to QImages
            result = []
            for frame in frames:
                height, width = frame.shape[:2]
                qimage = QtGui.QImage(frame.data, width, height, width * 3, QtGui.QImage.Format.Format_BGR888).copy()
                result.append(qimage)
            return result

        # Get camera names
        camera_names = self._get_camera_names_callback()
        if len(camera_names) != len(frames):
            # Mismatch - just convert frames to QImages
            result = []
            for frame in frames:
                height, width = frame.shape[:2]
                qimage = QtGui.QImage(frame.data, width, height, width * 3, QtGui.QImage.Format.Format_BGR888).copy()
                result.append(qimage)
            return result

        result_frames = []
        for frame, camera_name in zip(frames, camera_names):
            # Get zones with camera mapping for this camera
            zones = self._zone_manager.get_zones_with_camera_mapping(camera_name)

            if not zones:
                # No zones - convert frame to QImage
                height, width = frame.shape[:2]
                qimage = QtGui.QImage(frame.data, width, height, width * 3, QtGui.QImage.Format.Format_BGR888).copy()
                result_frames.append(qimage)
                continue

            # Collect overlays first to avoid unnecessary frame copy
            overlays_to_composite = []
            for zone in zones:
                overlay_data = zone.get_camera_overlay(frame.shape)
                if overlay_data is not None:
                    overlays_to_composite.append(overlay_data)

            # Only copy frame if we have overlays to composite
            if overlays_to_composite:
                # Use Qt QPainter for fast compositing (22x faster than NumPy)
                height, width = frame.shape[:2]

                # Convert frame to QImage
                qimage = QtGui.QImage(frame.data, width, height, width * 3, QtGui.QImage.Format.Format_BGR888).copy()

                # Create painter
                painter = QtGui.QPainter(qimage)
                painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)

                for overlay_data in overlays_to_composite:
                    # Unpack ROI overlay data
                    overlay, x, y, ov_width, ov_height = overlay_data

                    # Convert BGRA overlay to RGBA for Qt
                    overlay_rgba = cv.cvtColor(overlay, cv.COLOR_BGRA2RGBA)
                    overlay_qimage = QtGui.QImage(
                        overlay_rgba.data, ov_width, ov_height, ov_width * 4,
                        QtGui.QImage.Format.Format_RGBA8888
                    )

                    # Draw overlay at position
                    painter.drawImage(x, y, overlay_qimage)

                painter.end()

                # Keep as QImage (no numpy conversion!)
                result_frames.append(qimage)
            else:
                # Convert frame to QImage
                height, width = frame.shape[:2]
                qimage = QtGui.QImage(frame.data, width, height, width * 3, QtGui.QImage.Format.Format_BGR888).copy()
                result_frames.append(qimage)

            # Composite game overlays after zone overlays
            if self._main_core is not None:
                qimage_with_game = result_frames[-1]
                game_overlays_to_composite = []

                for zone in zones:
                    # Only process zones with calibrated camera mapping
                    if not zone.camera_mapping or not zone.camera_mapping.is_calibrated:
                        continue

                    # Get game overlay for this zone
                    game_overlay = self._main_core.get_game_camera_overlay(zone.name)
                    if game_overlay is None:
                        continue

                    # Check if overlay has any non-zero alpha values
                    if np.max(game_overlay[:, :, 3]) == 0:
                        continue  # Skip fully transparent overlays

                    # Get transformation matrix and ROI
                    matrix = zone.camera_mapping.game_to_camera_matrix
                    roi = zone.camera_mapping.roi
                    if matrix is None or roi is None:
                        continue

                    # Calculate ROI dimensions
                    roi_width = roi['max_x'] - roi['min_x'] + 1
                    roi_height = roi['max_y'] - roi['min_y'] + 1

                    # Warp game overlay to camera coordinates
                    warped_overlay = cv.warpPerspective(
                        game_overlay,
                        matrix,
                        (roi_width, roi_height)
                    )

                    # Ensure we don't go out of frame bounds
                    x_start = max(0, roi['min_x'])
                    y_start = max(0, roi['min_y'])
                    x_end = min(qimage_with_game.width(), roi['max_x'])
                    y_end = min(qimage_with_game.height(), roi['max_y'])

                    # Calculate actual dimensions after bounds checking
                    actual_width = x_end - x_start
                    actual_height = y_end - y_start

                    if actual_width <= 0 or actual_height <= 0:
                        continue

                    # Crop warped overlay if needed
                    warped_crop = warped_overlay[:actual_height, :actual_width]

                    game_overlays_to_composite.append((warped_crop, x_start, y_start))

                # Use Qt QPainter for fast compositing if we have game overlays
                if game_overlays_to_composite:
                    # Create painter on existing QImage
                    painter = QtGui.QPainter(qimage_with_game)
                    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)

                    for warped_crop, x_pos, y_pos in game_overlays_to_composite:
                        ov_height, ov_width = warped_crop.shape[:2]

                        # Convert BGRA overlay to RGBA for Qt
                        overlay_rgba = cv.cvtColor(warped_crop, cv.COLOR_BGRA2RGBA)
                        overlay_qimage = QtGui.QImage(
                            overlay_rgba.data, ov_width, ov_height, ov_width * 4,
                            QtGui.QImage.Format.Format_RGBA8888
                        )

                        # Draw overlay at position
                        painter.drawImage(x_pos, y_pos, overlay_qimage)

                    painter.end()

                    # Keep as QImage (no numpy conversion!)

        return result_frames

    def _compose_frames(self, frames: list[QtGui.QImage]) -> QtGui.QImage:
        """Compose multiple QImages into a single grid layout.

        Args:
            frames: List of QImages to compose.

        Returns:
            Composed frame as QImage.
        """
        num_frames = len(frames)

        if num_frames == 1:
            # Single camera - full viewport, but still apply zoom/pan
            viewport_width = self.width()
            viewport_height = self.height()
            cell_width = viewport_width
            cell_height = viewport_height
            rows = 1
            cols = 1

            # Process single frame with zoom/pan on FULL RESOLUTION first
            qimage = frames[0]

            # Apply zoom/pan on full resolution if set
            if 0 in self._zoom_states:
                qimage = self._apply_zoom_pan_qimage(qimage, self._zoom_states[0])

            # Now scale to fit viewport
            w = qimage.width()
            h = qimage.height()
            aspect = w / h
            cell_aspect = cell_width / cell_height

            if aspect > cell_aspect:
                new_w = cell_width
                new_h = int(cell_width / aspect)
            else:
                new_h = cell_height
                new_w = int(cell_height * aspect)

            # Scale QImage
            resized = qimage.scaled(new_w, new_h, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)

            # Create cell with black padding using QPainter
            cell = QtGui.QImage(cell_width, cell_height, QtGui.QImage.Format.Format_BGR888)
            cell.fill(QtCore.Qt.GlobalColor.black)

            painter = QtGui.QPainter(cell)
            y_offset = (cell_height - new_h) // 2
            x_offset = (cell_width - new_w) // 2
            painter.drawImage(x_offset, y_offset, resized)
            painter.end()

            # Store layout info
            self._current_rows = rows
            self._current_cols = cols
            self._cell_width = cell_width
            self._cell_height = cell_height

            return cell

        # Determine grid layout based on number of frames
        # 2 cameras: side by side (1x2)
        # 3-4 cameras: 2x2 grid

        # Get viewport dimensions
        viewport_width = self.width()
        viewport_height = self.height()

        if num_frames == 2:
            # Side by side
            cell_width = viewport_width // 2
            cell_height = viewport_height
            rows = 1
            cols = 2
        else:
            # 2x2 grid for 3-4 cameras
            cell_width = viewport_width // 2
            cell_height = viewport_height // 2
            rows = 2
            cols = 2

        # Resize frames to fit cells
        resized_frames = []
        for cell_idx, qimage in enumerate(frames):
            # Apply zoom/pan on FULL RESOLUTION first if set for this cell
            if cell_idx in self._zoom_states:
                qimage = self._apply_zoom_pan_qimage(qimage, self._zoom_states[cell_idx])

            # Calculate aspect-preserving resize
            w = qimage.width()
            h = qimage.height()
            aspect = w / h
            cell_aspect = cell_width / cell_height

            if aspect > cell_aspect:
                # Width-limited
                new_w = cell_width
                new_h = int(cell_width / aspect)
            else:
                # Height-limited
                new_h = cell_height
                new_w = int(cell_height * aspect)

            # Scale QImage
            resized = qimage.scaled(new_w, new_h, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)

            # Create cell with black padding using QPainter
            cell = QtGui.QImage(cell_width, cell_height, QtGui.QImage.Format.Format_BGR888)
            cell.fill(QtCore.Qt.GlobalColor.black)

            painter = QtGui.QPainter(cell)
            y_offset = (cell_height - new_h) // 2
            x_offset = (cell_width - new_w) // 2
            painter.drawImage(x_offset, y_offset, resized)
            painter.end()

            resized_frames.append(cell)

        # Pad with black frames if needed
        total_cells = rows * cols
        while len(resized_frames) < total_cells:
            black_cell = QtGui.QImage(cell_width, cell_height, QtGui.QImage.Format.Format_BGR888)
            black_cell.fill(QtCore.Qt.GlobalColor.black)
            resized_frames.append(black_cell)

        # Compose grid using QPainter
        composed = QtGui.QImage(viewport_width, viewport_height, QtGui.QImage.Format.Format_BGR888)
        composed.fill(QtCore.Qt.GlobalColor.black)

        painter = QtGui.QPainter(composed)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx < len(resized_frames):
                    x_pos = c * cell_width
                    y_pos = r * cell_height
                    painter.drawImage(x_pos, y_pos, resized_frames[idx])
        painter.end()

        # Store current grid layout for mouse event handling
        self._current_rows = rows
        self._current_cols = cols
        self._cell_width = cell_width
        self._cell_height = cell_height

        return composed

    def _apply_zoom_pan_qimage(self, qimage: QtGui.QImage, zoom_state: dict[str, float]) -> QtGui.QImage:
        """Apply zoom and pan to a QImage by extracting ROI.

        Args:
            qimage: Input QImage.
            zoom_state: Zoom state dict with zoom, center_x, center_y, pan_x, pan_y.

        Returns:
            Cropped ROI QImage.
        """
        zoom_factor = zoom_state['zoom']
        center_x = zoom_state['center_x']
        center_y = zoom_state['center_y']
        pan_x = zoom_state.get('pan_x', 0.0)
        pan_y = zoom_state.get('pan_y', 0.0)

        w = qimage.width()
        h = qimage.height()

        # Calculate crop region centered on zoom point with pan offset
        crop_w = int(w / zoom_factor)
        crop_h = int(h / zoom_factor)

        # Apply pan offset (normalized to image coordinates)
        center_x_px = center_x * w + pan_x * w
        center_y_px = center_y * h + pan_y * h

        # Calculate crop coordinates
        x1 = int(center_x_px - crop_w / 2)
        y1 = int(center_y_px - crop_h / 2)

        # Clamp to image bounds
        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))

        # Return cropped QImage
        return qimage.copy(x1, y1, crop_w, crop_h)

    def _apply_zoom_pan_full_res(self, image: np.ndarray, zoom_state: dict[str, float]) -> np.ndarray:
        """Apply zoom and pan to a full resolution image by extracting ROI.

        This method works on the full resolution image and returns the cropped ROI,
        which should then be scaled to viewport size. This preserves maximum quality.

        Args:
            image: Full resolution input image.
            zoom_state: Zoom state dict with zoom, center_x, center_y, pan_x, pan_y.

        Returns:
            Cropped ROI from full resolution image.
        """
        zoom_factor = zoom_state['zoom']
        center_x = zoom_state['center_x']
        center_y = zoom_state['center_y']
        pan_x = zoom_state.get('pan_x', 0.0)
        pan_y = zoom_state.get('pan_y', 0.0)

        h, w = image.shape[:2]

        # Calculate crop region centered on zoom point with pan offset
        crop_w = int(w / zoom_factor)
        crop_h = int(h / zoom_factor)

        # Apply pan offset (normalized to image coordinates)
        center_x_px = center_x * w + pan_x * w
        center_y_px = center_y * h + pan_y * h

        # Calculate crop coordinates
        x1 = int(center_x_px - crop_w / 2)
        y1 = int(center_y_px - crop_h / 2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Clamp to image bounds
        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Return cropped ROI at full resolution
        cropped = image[y1:y2, x1:x2]
        return cropped

    def _apply_zoom_pan(self, image: np.ndarray, zoom_state: dict[str, float]) -> np.ndarray:
        """Apply zoom and pan to an image (legacy method - resizes back to original size).

        Args:
            image: Input image.
            zoom_state: Zoom state dict with zoom, center_x, center_y, pan_x, pan_y.

        Returns:
            Transformed image.
        """
        zoom_factor = zoom_state['zoom']
        center_x = zoom_state['center_x']
        center_y = zoom_state['center_y']
        pan_x = zoom_state.get('pan_x', 0.0)
        pan_y = zoom_state.get('pan_y', 0.0)

        h, w = image.shape[:2]

        # Calculate crop region centered on zoom point with pan offset
        crop_w = int(w / zoom_factor)
        crop_h = int(h / zoom_factor)

        # Apply pan offset (normalized to image coordinates)
        center_x_px = center_x * w + pan_x * w
        center_y_px = center_y * h + pan_y * h

        # Calculate crop coordinates
        x1 = int(center_x_px - crop_w / 2)
        y1 = int(center_y_px - crop_h / 2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Clamp to image bounds
        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Crop and resize back to original size
        cropped = image[y1:y2, x1:x2]
        resized = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)

        return resized

    def _display_qimage(self, qimage: QtGui.QImage) -> None:
        """Display a QImage directly in the viewport (optimized - no conversion).

        Args:
            qimage: QImage to display (BGR format).
        """
        # Convert BGR QImage to RGB for display
        rgb_qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(rgb_qimage)

        # Scale to fit viewport while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

    def _display_frame(self, frame: np.ndarray) -> None:
        """Display a numpy frame in the viewport (legacy method).

        Args:
            frame: Frame to display (BGR format).
        """
        # Convert BGR to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # Create QImage
        q_image = QtGui.QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888
        )

        # Convert to QPixmap and display
        pixmap = QtGui.QPixmap.fromImage(q_image)

        # Scale to fit viewport while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Handle mouse wheel events for zooming.

        Args:
            event: Wheel event.
        """
        if self._get_frames_callback is None:
            return

        # Get frames to check if we have any
        frames = self._get_frames_callback()
        if not frames or all(f is None for f in frames):
            return

        # Get mouse position relative to the label
        pos = event.position()
        mouse_x = int(pos.x())
        mouse_y = int(pos.y())

        # Determine which cell the mouse is over
        cell_idx = self._get_cell_at_position(mouse_x, mouse_y)
        if cell_idx < 0 or cell_idx >= len(frames):
            return

        # Calculate zoom delta
        delta = event.angleDelta().y()
        zoom_delta = 1.1 if delta > 0 else 0.9

        # Get or create zoom state for this cell
        if cell_idx not in self._zoom_states:
            self._zoom_states[cell_idx] = {
                'zoom': 1.0,
                'center_x': 0.5,
                'center_y': 0.5,
                'pan_x': 0.0,
                'pan_y': 0.0
            }

        zoom_state = self._zoom_states[cell_idx]
        old_zoom = zoom_state['zoom']
        new_zoom = old_zoom * zoom_delta

        # Clamp zoom to minimum 1.0 (100%)
        if new_zoom < 1.0:
            new_zoom = 1.0

        # Calculate maximum zoom to allow 9x9 pixel magnification
        # At max zoom, 1 source pixel should occupy 9x9 viewport pixels
        # This means: (viewport_size / source_size) * zoom = 9
        # So: max_zoom = 9 * (source_size / viewport_size)
        frame = frames[cell_idx] if cell_idx < len(frames) else None
        if frame is not None:
            src_h, src_w = frame.shape[:2]
            # Use the smaller dimension to calculate max zoom (more restrictive)
            src_min = min(src_w, src_h)
            viewport_min = min(self._cell_width, self._cell_height)
            if viewport_min > 0:
                # Max zoom where 1 source pixel = 9x9 viewport pixels
                max_zoom = 9.0 * (src_min / viewport_min)
                max_zoom = max(10.0, max_zoom)  # At least 10x for backward compatibility
            else:
                max_zoom = 100.0
        else:
            max_zoom = 100.0

        new_zoom = min(new_zoom, max_zoom)

        # Calculate normalized position within the cell for zoom center
        cell_row = cell_idx // self._current_cols
        cell_col = cell_idx % self._current_cols

        # Get cell position in viewport
        cell_x = cell_col * self._cell_width
        cell_y = cell_row * self._cell_height

        # Calculate position within cell (normalized 0-1)
        rel_x = (mouse_x - cell_x) / self._cell_width if self._cell_width > 0 else 0.5
        rel_y = (mouse_y - cell_y) / self._cell_height if self._cell_height > 0 else 0.5

        # Clamp to valid range
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))

        # Smooth zoom: adjust center to keep pixel under mouse fixed
        # The pixel under the mouse in the current view should stay under the mouse after zoom
        if old_zoom != new_zoom and old_zoom > 0:
            # Current center with pan
            old_center_x = zoom_state['center_x'] + zoom_state['pan_x']
            old_center_y = zoom_state['center_y'] + zoom_state['pan_y']

            # Calculate the image coordinate of the pixel under the mouse
            # In the current zoomed view, the mouse is at rel_x, rel_y in the cell
            # This corresponds to a position in the cropped region
            # The cropped region spans from (center - 0.5/zoom) to (center + 0.5/zoom)
            old_crop_half_w = 0.5 / old_zoom
            old_crop_half_h = 0.5 / old_zoom

            # Image coordinate under mouse in normalized space
            img_x = old_center_x + (rel_x - 0.5) * 2 * old_crop_half_w
            img_y = old_center_y + (rel_y - 0.5) * 2 * old_crop_half_h

            # New crop half sizes
            new_crop_half_w = 0.5 / new_zoom
            new_crop_half_h = 0.5 / new_zoom

            # Calculate new center so that img_x, img_y appears at rel_x, rel_y
            new_center_x = img_x - (rel_x - 0.5) * 2 * new_crop_half_w
            new_center_y = img_y - (rel_y - 0.5) * 2 * new_crop_half_h

            # Split into base center and pan offset
            zoom_state['center_x'] = 0.5
            zoom_state['center_y'] = 0.5
            zoom_state['pan_x'] = new_center_x - 0.5
            zoom_state['pan_y'] = new_center_y - 0.5

        # Update zoom
        zoom_state['zoom'] = new_zoom

        # If zoom is back to 1.0, remove the zoom state
        if new_zoom == 1.0:
            del self._zoom_states[cell_idx]

        event.accept()

    def _get_cell_at_position(self, x: int, y: int) -> int:
        """Get the cell index at the given position.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            Cell index, or -1 if out of bounds.
        """
        if self._cell_width == 0 or self._cell_height == 0:
            return 0 if self._current_rows == 1 and self._current_cols == 1 else -1

        # Calculate cell row and column
        cell_col = x // self._cell_width
        cell_row = y // self._cell_height

        # Check bounds
        if cell_col < 0 or cell_col >= self._current_cols:
            return -1
        if cell_row < 0 or cell_row >= self._current_rows:
            return -1

        # Calculate cell index
        cell_idx = cell_row * self._current_cols + cell_col

        return cell_idx

    def _viewport_to_frame_coords(self, viewport_x: int, viewport_y: int, cell_idx: int, frame_shape: tuple) -> tuple[int, int] | None:
        """Convert viewport coordinates to original frame coordinates.

        Args:
            viewport_x: X coordinate in viewport.
            viewport_y: Y coordinate in viewport.
            cell_idx: Cell index.
            frame_shape: Shape of the original frame (height, width, channels).

        Returns:
            Tuple of (x, y) in original frame coordinates, or None if invalid.
        """
        if self._cell_width == 0 or self._cell_height == 0:
            return None

        # Get cell position
        cell_row = cell_idx // self._current_cols
        cell_col = cell_idx % self._current_cols
        cell_x = cell_col * self._cell_width
        cell_y = cell_row * self._cell_height

        # Position within cell
        rel_x = viewport_x - cell_x
        rel_y = viewport_y - cell_y

        # Get frame dimensions
        frame_h, frame_w = frame_shape[:2]

        # Account for zoom/pan if active
        if cell_idx in self._zoom_states:
            zoom_state = self._zoom_states[cell_idx]
            zoom_factor = zoom_state['zoom']
            center_x = zoom_state['center_x']
            center_y = zoom_state['center_y']
            pan_x = zoom_state.get('pan_x', 0.0)
            pan_y = zoom_state.get('pan_y', 0.0)

            # Calculate the visible region in the original frame
            crop_w = int(frame_w / zoom_factor)
            crop_h = int(frame_h / zoom_factor)

            # Center with pan offset
            center_x_px = center_x * frame_w + pan_x * frame_w
            center_y_px = center_y * frame_h + pan_y * frame_h

            # Crop coordinates
            x1 = int(center_x_px - crop_w / 2)
            y1 = int(center_y_px - crop_h / 2)
            x1 = max(0, min(x1, frame_w - crop_w))
            y1 = max(0, min(y1, frame_h - crop_h))

            # Calculate aspect-preserving resize dimensions
            aspect = frame_w / frame_h
            cell_aspect = self._cell_width / self._cell_height

            if aspect > cell_aspect:
                new_w = self._cell_width
                new_h = int(self._cell_width / aspect)
            else:
                new_h = self._cell_height
                new_w = int(self._cell_height * aspect)

            # Account for padding
            x_offset = (self._cell_width - new_w) // 2
            y_offset = (self._cell_height - new_h) // 2

            # Check if click is within the actual image area
            if rel_x < x_offset or rel_x >= x_offset + new_w:
                return None
            if rel_y < y_offset or rel_y >= y_offset + new_h:
                return None

            # Normalize to cropped region
            norm_x = (rel_x - x_offset) / new_w
            norm_y = (rel_y - y_offset) / new_h

            # Map to original frame coordinates
            frame_x = int(x1 + norm_x * crop_w)
            frame_y = int(y1 + norm_y * crop_h)

        else:
            # No zoom - calculate aspect-preserving resize
            aspect = frame_w / frame_h
            cell_aspect = self._cell_width / self._cell_height

            if aspect > cell_aspect:
                new_w = self._cell_width
                new_h = int(self._cell_width / aspect)
            else:
                new_h = self._cell_height
                new_w = int(self._cell_height * aspect)

            # Account for padding
            x_offset = (self._cell_width - new_w) // 2
            y_offset = (self._cell_height - new_h) // 2

            # Check if click is within the actual image area
            if rel_x < x_offset or rel_x >= x_offset + new_w:
                return None
            if rel_y < y_offset or rel_y >= y_offset + new_h:
                return None

            # Normalize and map to original frame
            norm_x = (rel_x - x_offset) / new_w
            norm_y = (rel_y - y_offset) / new_h
            frame_x = int(norm_x * frame_w)
            frame_y = int(norm_y * frame_h)

        # Clamp to frame bounds
        frame_x = max(0, min(frame_x, frame_w - 1))
        frame_y = max(0, min(frame_y, frame_h - 1))

        return (frame_x, frame_y)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press events for panning and vertex dragging.

        Args:
            event: Mouse event.
        """
        pos = event.position()
        mouse_x = int(pos.x())
        mouse_y = int(pos.y())

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Check for vertex dragging
            if (self._zone_manager is not None and
                    self._get_frames_callback is not None and
                    self._get_camera_names_callback is not None):

                # Get frames and camera names
                frames = self._get_frames_callback()
                camera_names = self._get_camera_names_callback()

                if frames and camera_names and len(frames) == len(camera_names):
                    # Determine which cell was clicked
                    cell_idx = self._get_cell_at_position(mouse_x, mouse_y)

                    if 0 <= cell_idx < len(frames):
                        frame = frames[cell_idx]
                        camera_name = camera_names[cell_idx]

                        if frame is not None:
                            # Convert viewport coords to frame coords
                            frame_coords = self._viewport_to_frame_coords(mouse_x, mouse_y, cell_idx, frame.shape)

                            if frame_coords is not None:
                                from .constants import VERTEX_RADIUS
                                frame_x, frame_y = frame_coords

                                # Find vertex at this position
                                result = self._zone_manager.find_vertex_at_position(
                                    camera_name, frame_x, frame_y, VERTEX_RADIUS
                                )

                                if result is not None:
                                    zone, vertex_idx = result
                                    self._dragging_vertex = (zone, vertex_idx, camera_name, cell_idx)
                                    self._drag_start_pos = (mouse_x, mouse_y)
                                    self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                                    event.accept()
                                    return

        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            # Start panning
            cell_idx = self._get_cell_at_position(mouse_x, mouse_y)
            if cell_idx >= 0:
                # Only pan if this cell has zoom
                if cell_idx in self._zoom_states and self._zoom_states[cell_idx]['zoom'] > 1.0:
                    self._is_panning = True
                    self._pan_start_pos = (mouse_x, mouse_y)
                    self._pan_cell_idx = cell_idx
                    self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    event.accept()
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move events for panning and vertex dragging.

        Args:
            event: Mouse event.
        """
        pos = event.position()
        mouse_x = int(pos.x())
        mouse_y = int(pos.y())

        # Handle vertex dragging
        if self._dragging_vertex is not None and self._get_frames_callback is not None:
            zone, vertex_idx, camera_name, cell_idx = self._dragging_vertex

            # Get frames to access frame shape
            frames = self._get_frames_callback()
            if frames and 0 <= cell_idx < len(frames):
                frame = frames[cell_idx]
                if frame is not None:
                    # Convert viewport coords to frame coords
                    frame_coords = self._viewport_to_frame_coords(mouse_x, mouse_y, cell_idx, frame.shape)

                    if frame_coords is not None:
                        frame_x, frame_y = frame_coords

                        # Update vertex position
                        vertices = list(zone.camera_mapping.vertices)
                        vertices[vertex_idx] = (frame_x, frame_y)
                        zone.camera_mapping.vertices = vertices

                        # Invalidate overlay to force regeneration
                        zone.camera_mapping.invalidate_overlay()

                        # Emit signal to update UI
                        self.vertex_updated.emit(zone.name)

                        event.accept()
                        return

        # Handle panning
        if self._is_panning and self._pan_start_pos is not None:
            # Calculate delta
            dx = mouse_x - self._pan_start_pos[0]
            dy = mouse_y - self._pan_start_pos[1]

            # Update pan offset (normalized to cell size)
            if self._pan_cell_idx in self._zoom_states:
                zoom_state = self._zoom_states[self._pan_cell_idx]
                zoom_factor = zoom_state['zoom']

                # Convert pixel delta to normalized delta in the zoomed view
                # The cell shows a region of size 1/zoom, so pixel movement needs scaling
                if self._cell_width > 0 and self._cell_height > 0:
                    norm_dx = -(dx / self._cell_width) / zoom_factor
                    norm_dy = -(dy / self._cell_height) / zoom_factor

                    zoom_state['pan_x'] += norm_dx
                    zoom_state['pan_y'] += norm_dy

                    # Update start position for next delta
                    self._pan_start_pos = (mouse_x, mouse_y)

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release events for panning and vertex dragging.

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

        if event.button() == QtCore.Qt.MouseButton.MiddleButton and self._is_panning:
            self._is_panning = False
            self._pan_start_pos = None
            self._pan_cell_idx = -1
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)
