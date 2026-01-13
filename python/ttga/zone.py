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

"""Zone class for managing virtual world zones with camera/projector mappings."""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CameraMapping:
    """Camera mapping for a zone.

    Attributes:
        camera_name: Name of the camera.
        vertices: List of 4 (x, y) tuples representing the quadrilateral vertices.
        lock_vertices: Whether vertices are locked from editing.
        enabled: Whether the camera mapping is enabled.
        is_calibrated: Whether the mapping has been calibrated.
        camera_to_game_matrix: Transform matrix from camera to game coordinates.
        game_to_camera_matrix: Transform matrix from game to camera coordinates.
        roi: ROI bounding box dict with min_x, min_y, max_x, max_y.
        overlay_needs_update: Flag indicating overlay image needs regeneration.
        camera_overlay: Cached overlay image (not serialized).
    """
    camera_name: str
    vertices: list[tuple[int, int]] = field(default_factory=lambda: [
        (128, 128),   # P0: Cyan (0, 0)
        (384, 128),   # P1: Magenta (wpx, 0)
        (384, 256),   # P2: Yellow (wpx, hpx)
        (128, 256)    # P3: White (0, hpx)
    ])
    lock_vertices: bool = False
    enabled: bool = True
    is_calibrated: bool = False
    camera_to_game_matrix: np.ndarray | None = None
    game_to_camera_matrix: np.ndarray | None = None
    roi: dict[str, int] | None = None
    overlay_needs_update: bool = field(default=True, init=False, repr=False)
    camera_overlay: any = field(default=None, init=False, repr=False)

    def to_dict(self) -> dict:
        """Serialize camera mapping to dictionary.

        Returns:
            Dictionary containing camera mapping data.
        """
        # Note: overlay_needs_update and camera_overlay are not serialized
        result = {
            'camera_name': self.camera_name,
            'vertices': self.vertices,
            'lock_vertices': self.lock_vertices,
            'enabled': self.enabled,
            'is_calibrated': self.is_calibrated,
            'roi': self.roi
        }
        # Serialize matrices as lists if they exist
        if self.camera_to_game_matrix is not None:
            result['camera_to_game_matrix'] = self.camera_to_game_matrix.tolist()
        if self.game_to_camera_matrix is not None:
            result['game_to_camera_matrix'] = self.game_to_camera_matrix.tolist()
        return result

    def invalidate_overlay(self) -> None:
        """Mark overlay as needing update and clear cached overlay."""
        self.overlay_needs_update = True
        self.camera_overlay = None

    @staticmethod
    def from_dict(data: dict) -> 'CameraMapping':
        """Deserialize camera mapping from dictionary.

        Args:
            data: Dictionary containing camera mapping data.

        Returns:
            CameraMapping instance.
        """
        mapping = CameraMapping(
            camera_name=data['camera_name'],
            vertices=[tuple(v) for v in data['vertices']],
            lock_vertices=data.get('lock_vertices', False),
            enabled=data.get('enabled', True),
            is_calibrated=data.get('is_calibrated', False),
            roi=data.get('roi')
        )
        # Deserialize matrices from lists if they exist
        if 'camera_to_game_matrix' in data:
            mapping.camera_to_game_matrix = np.array(data['camera_to_game_matrix'])
        if 'game_to_camera_matrix' in data:
            mapping.game_to_camera_matrix = np.array(data['game_to_camera_matrix'])
        return mapping


@dataclass
class ProjectorMapping:
    """Projector mapping for a zone.

    Attributes:
        projector_name: Name of the projector.
        vertices: List of 4 (x, y) tuples representing the quadrilateral vertices.
        lock_vertices: Whether vertices are locked from editing.
        enabled: Whether the projector mapping is enabled.
        is_calibrated: Whether the mapping has been calibrated.
        projector_to_game_matrix: Transform matrix from projector to game coordinates.
        game_to_projector_matrix: Transform matrix from game to projector coordinates.
        roi: ROI bounding box dict with min_x, min_y, max_x, max_y.
        overlay_needs_update: Flag indicating overlay image needs regeneration.
        projector_overlay: Cached overlay image (not serialized).
    """
    projector_name: str
    vertices: list[tuple[int, int]] = field(default_factory=lambda: [
        (128, 128),   # P0: Cyan (0, 0)
        (384, 128),   # P1: Magenta (wpx, 0)
        (384, 256),   # P2: Yellow (wpx, hpx)
        (128, 256)    # P3: White (0, hpx)
    ])
    lock_vertices: bool = False
    enabled: bool = True
    is_calibrated: bool = False
    projector_to_game_matrix: np.ndarray | None = None
    game_to_projector_matrix: np.ndarray | None = None
    roi: dict[str, int] | None = None
    overlay_needs_update: bool = field(default=True, init=False, repr=False)
    projector_overlay: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize overlay fields after dataclass initialization."""
        self.overlay_needs_update = True
        self.projector_overlay = None

    def invalidate_overlay(self) -> None:
        """Mark overlay as needing update and clear cached overlay."""
        self.overlay_needs_update = True
        self.projector_overlay = None

    def to_dict(self) -> dict:
        """Serialize projector mapping to dictionary.

        Returns:
            Dictionary containing projector mapping data.
        """
        result = {
            'projector_name': self.projector_name,
            'vertices': self.vertices,
            'lock_vertices': self.lock_vertices,
            'enabled': self.enabled,
            'is_calibrated': self.is_calibrated,
            'roi': self.roi
        }
        # Serialize matrices as lists if they exist
        if self.projector_to_game_matrix is not None:
            result['projector_to_game_matrix'] = self.projector_to_game_matrix.tolist()
        if self.game_to_projector_matrix is not None:
            result['game_to_projector_matrix'] = self.game_to_projector_matrix.tolist()
        return result

    @staticmethod
    def from_dict(data: dict) -> 'ProjectorMapping':
        """Deserialize projector mapping from dictionary.

        Args:
            data: Dictionary containing projector mapping data.

        Returns:
            ProjectorMapping instance.
        """
        mapping = ProjectorMapping(
            projector_name=data['projector_name'],
            vertices=[tuple(v) for v in data['vertices']],
            lock_vertices=data.get('lock_vertices', False),
            enabled=data.get('enabled', True),
            is_calibrated=data.get('is_calibrated', False),
            roi=data.get('roi')
        )
        # Deserialize matrices from lists if they exist
        if 'projector_to_game_matrix' in data:
            mapping.projector_to_game_matrix = np.array(data['projector_to_game_matrix'])
        if 'game_to_projector_matrix' in data:
            mapping.game_to_projector_matrix = np.array(data['game_to_projector_matrix'])
        return mapping


class Zone:
    """Represents a rectangular zone in the virtual world.

    A zone can be mapped to a camera and/or projector quadrilateral.

    Attributes:
        name: Unique name for the zone.
        width: Width of the zone.
        height: Height of the zone.
        unit: Unit of measurement (mm, cm, in, px).
        resolution: Pixels per unit (disabled when unit is px).
        camera_mapping: Optional camera mapping.
        projector_mapping: Optional projector mapping.
        draw_locked_borders: Whether to draw locked borders for mappings.
    """

    VALID_UNITS = ['mm', 'cm', 'in', 'px']

    def __init__(self, name: str, width: float = 34.0, height: float = 22.0,
                 unit: str = 'in', resolution: int = 50) -> None:
        """Initialize a zone.

        Args:
            name: Unique name for the zone.
            width: Width of the zone.
            height: Height of the zone.
            unit: Unit of measurement (mm, cm, in, px).
            resolution: Pixels per unit (set to 1 when unit is px).
        """
        self.name = name
        self.width = width
        self.height = height
        self.unit = unit if unit in self.VALID_UNITS else 'in'
        self.resolution = 1 if self.unit == 'px' else int(resolution)
        self.camera_mapping: CameraMapping | None = None
        self.projector_mapping: ProjectorMapping | None = None
        self.draw_locked_borders: bool = True

    def to_dict(self) -> dict:
        """Serialize zone to dictionary.

        Returns:
            Dictionary containing zone data.
        """
        return {
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'unit': self.unit,
            'resolution': self.resolution,
            'camera_mapping': self.camera_mapping.to_dict() if self.camera_mapping else None,
            'projector_mapping': self.projector_mapping.to_dict() if self.projector_mapping else None,
            'draw_locked_borders': self.draw_locked_borders
        }

    @staticmethod
    def from_dict(data: dict) -> 'Zone':
        """Deserialize zone from dictionary.

        Args:
            data: Dictionary containing zone data.

        Returns:
            Zone instance.
        """
        zone = Zone(
            name=data['name'],
            width=data['width'],
            height=data['height'],
            unit=data['unit'],
            resolution=data['resolution']
        )

        if data.get('camera_mapping'):
            zone.camera_mapping = CameraMapping.from_dict(data['camera_mapping'])

        if data.get('projector_mapping'):
            zone.projector_mapping = ProjectorMapping.from_dict(data['projector_mapping'])

        # Load zone-level attributes with backward compatibility
        zone.draw_locked_borders = data.get('draw_locked_borders', True)

        return zone

    def get_camera_overlay(self, frame_shape: tuple[int, int, int]):
        """Generate or retrieve cached camera overlay with vertices and edges.

        Args:
            frame_shape: Shape of the camera frame (height, width, channels).

        Returns:
            Tuple of (overlay, x, y, width, height) where overlay is a BGRA ROI image,
            or None if no camera mapping. The x, y, width, height define the ROI position.
        """
        from .constants import VERTEX_RADIUS, EDGE_THICKNESS, VERTEX_CIRCLE_THICKNESS

        if not self.camera_mapping or not self.camera_mapping.enabled:
            return None

        # Early return if nothing to draw (vertices locked and borders disabled)
        if self.camera_mapping.lock_vertices and not self.draw_locked_borders:
            return None

        vertices = self.camera_mapping.vertices

        # Calculate bounding box of vertices with padding for drawing
        padding = VERTEX_RADIUS + VERTEX_CIRCLE_THICKNESS + 2  # Extra padding for anti-aliasing
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]

        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = min(frame_shape[1] - 1, max(xs) + padding)
        y_max = min(frame_shape[0] - 1, max(ys) + padding)

        roi_width = x_max - x_min + 1
        roi_height = y_max - y_min + 1

        # Validate ROI has positive dimensions
        if roi_width <= 0 or roi_height <= 0:
            return None

        # Check if we can use cached overlay (must match current ROI dimensions)
        if (self.camera_mapping.camera_overlay is not None and
                not self.camera_mapping.overlay_needs_update):
            cached_overlay, cached_x, cached_y, cached_w, cached_h = self.camera_mapping.camera_overlay
            if (cached_x == x_min and cached_y == y_min and
                    cached_w == roi_width and cached_h == roi_height):
                return self.camera_mapping.camera_overlay

        # Create transparent BGRA image for ROI only
        overlay = np.zeros((roi_height, roi_width, 4), dtype=np.uint8)

        # Vertex colors (BGR format): P0=Cyan, P1=Magenta, P2=Yellow, P3=White
        vertex_colors = [
            (255, 255, 0),    # P0: Cyan
            (255, 0, 255),    # P1: Magenta
            (0, 255, 255),    # P2: Yellow
            (255, 255, 255)   # P3: White
        ]

        # Adjust vertices to ROI coordinates
        roi_vertices = [(x - x_min, y - y_min) for x, y in vertices]

        # Draw edges if draw_locked_borders is enabled or vertices are unlocked
        if self.draw_locked_borders or not self.camera_mapping.lock_vertices:
            edge_color = (255, 255, 255, 255)  # White BGRA
            pts = np.array(roi_vertices, dtype=np.int32)

            # Draw quadrilateral edges with anti-aliasing
            for i in range(4):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i + 1) % 4])
                cv2.line(overlay, pt1, pt2, edge_color, EDGE_THICKNESS, cv2.LINE_AA)

        # Draw vertices only if unlocked
        if not self.camera_mapping.lock_vertices:
            for i, (x, y) in enumerate(roi_vertices):
                color_bgr = vertex_colors[i]
                color_bgra = (*color_bgr, 255)  # Add alpha channel
                cv2.circle(overlay, (x, y), VERTEX_RADIUS, color_bgra, VERTEX_CIRCLE_THICKNESS, cv2.LINE_AA)

        # Cache the overlay with ROI info
        result = (overlay, x_min, y_min, roi_width, roi_height)
        self.camera_mapping.camera_overlay = result
        self.camera_mapping.overlay_needs_update = False

        return result

    def get_projector_overlay(self, frame_shape: tuple[int, int, int]):
        """Generate or retrieve cached projector overlay with vertices and edges.

        Args:
            frame_shape: Shape of the projector frame (height, width, channels).

        Returns:
            Tuple of (overlay, x, y, width, height) where overlay is a BGRA ROI image,
            or None if no projector mapping. The x, y, width, height define the ROI position.
        """
        from .constants import VERTEX_RADIUS, EDGE_THICKNESS, VERTEX_CIRCLE_THICKNESS

        if not self.projector_mapping or not self.projector_mapping.enabled:
            return None

        # Early return if nothing to draw (vertices locked and borders disabled)
        if self.projector_mapping.lock_vertices and not self.draw_locked_borders:
            return None

        vertices = self.projector_mapping.vertices

        # Calculate bounding box of vertices with padding for drawing
        padding = VERTEX_RADIUS + VERTEX_CIRCLE_THICKNESS + 2  # Extra padding for anti-aliasing
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]

        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = min(frame_shape[1] - 1, max(xs) + padding)
        y_max = min(frame_shape[0] - 1, max(ys) + padding)

        roi_width = x_max - x_min + 1
        roi_height = y_max - y_min + 1

        # Validate ROI has positive dimensions
        if roi_width <= 0 or roi_height <= 0:
            return None

        # Check if we can use cached overlay (must match current ROI dimensions)
        if (self.projector_mapping.projector_overlay is not None and
                not self.projector_mapping.overlay_needs_update):
            cached_overlay, cached_x, cached_y, cached_w, cached_h = self.projector_mapping.projector_overlay
            if (cached_x == x_min and cached_y == y_min and
                    cached_w == roi_width and cached_h == roi_height):
                return self.projector_mapping.projector_overlay

        # Create transparent BGRA image for ROI only
        overlay = np.zeros((roi_height, roi_width, 4), dtype=np.uint8)

        # Vertex colors (BGR format): P0=Cyan, P1=Magenta, P2=Yellow, P3=White
        vertex_colors = [
            (255, 255, 0),    # P0: Cyan
            (255, 0, 255),    # P1: Magenta
            (0, 255, 255),    # P2: Yellow
            (255, 255, 255)   # P3: White
        ]

        # Adjust vertices to ROI coordinates
        roi_vertices = [(x - x_min, y - y_min) for x, y in vertices]

        # Draw edges if draw_locked_borders is enabled or vertices are unlocked
        if self.draw_locked_borders or not self.projector_mapping.lock_vertices:
            edge_color = (255, 255, 255, 255)  # White BGRA
            pts = np.array(roi_vertices, dtype=np.int32)

            # Draw quadrilateral edges with anti-aliasing
            for i in range(4):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i + 1) % 4])
                cv2.line(overlay, pt1, pt2, edge_color, EDGE_THICKNESS, cv2.LINE_AA)

        # Draw vertices only if unlocked
        if not self.projector_mapping.lock_vertices:
            for i, (x, y) in enumerate(roi_vertices):
                color_bgr = vertex_colors[i]
                color_bgra = (*color_bgr, 255)  # Add alpha channel
                cv2.circle(overlay, (x, y), VERTEX_RADIUS, color_bgra, VERTEX_CIRCLE_THICKNESS, cv2.LINE_AA)

        # Cache the overlay with ROI info
        result = (overlay, x_min, y_min, roi_width, roi_height)
        self.projector_mapping.projector_overlay = result
        self.projector_mapping.overlay_needs_update = False

        return result

    def get_game_dimensions(self) -> tuple[int, int]:
        """Get game dimensions in pixels.

        Returns:
            Tuple of (width_px, height_px).
        """
        width_px = int(round(self.width * self.resolution))
        height_px = int(round(self.height * self.resolution))
        return (width_px, height_px)

    def camera_to_game(self, pos: tuple[float, float], rounded: bool = False) -> tuple[float, float]:
        """Transform a position from camera coordinates to game coordinates.

        Args:
            pos: The (x, y) position in camera ROI coordinates.
            rounded: Whether to round the result and return integers.

        Returns:
            Transformed position in game coordinates.

        Raises:
            ValueError: If camera mapping is not calibrated.
        """
        if not self.camera_mapping or not self.camera_mapping.is_calibrated:
            raise ValueError("Camera mapping is not calibrated")

        pos_homo = np.float32([pos[0], pos[1], 1.0])
        warp_pos_homo = self.camera_mapping.camera_to_game_matrix.dot(pos_homo)
        warp_pos = (warp_pos_homo / warp_pos_homo[2])[:2]

        if rounded:
            return (round(warp_pos[0]), round(warp_pos[1]))
        return tuple(warp_pos)

    def game_to_camera(self, pos: tuple[float, float], rounded: bool = False) -> tuple[float, float]:
        """Transform a position from game coordinates to camera coordinates.

        Args:
            pos: The (x, y) position in game coordinates.
            rounded: Whether to round the result and return integers.

        Returns:
            Transformed position in camera ROI coordinates.

        Raises:
            ValueError: If camera mapping is not calibrated.
        """
        if not self.camera_mapping or not self.camera_mapping.is_calibrated:
            raise ValueError("Camera mapping is not calibrated")

        pos_homo = np.float32([pos[0], pos[1], 1.0])
        warp_pos_homo = self.camera_mapping.game_to_camera_matrix.dot(pos_homo)
        warp_pos = (warp_pos_homo / warp_pos_homo[2])[:2]

        if rounded:
            return (round(warp_pos[0]), round(warp_pos[1]))
        return tuple(warp_pos)

    def projector_to_game(self, pos: tuple[float, float], rounded: bool = False) -> tuple[float, float]:
        """Transform a position from projector coordinates to game coordinates.

        Args:
            pos: The (x, y) position in projector ROI coordinates.
            rounded: Whether to round the result and return integers.

        Returns:
            Transformed position in game coordinates.

        Raises:
            ValueError: If projector mapping is not calibrated.
        """
        if not self.projector_mapping or not self.projector_mapping.is_calibrated:
            raise ValueError("Projector mapping is not calibrated")

        pos_homo = np.float32([pos[0], pos[1], 1.0])
        warp_pos_homo = self.projector_mapping.projector_to_game_matrix.dot(pos_homo)
        warp_pos = (warp_pos_homo / warp_pos_homo[2])[:2]

        if rounded:
            return (round(warp_pos[0]), round(warp_pos[1]))
        return tuple(warp_pos)

    def game_to_projector(self, pos: tuple[float, float], rounded: bool = False) -> tuple[float, float]:
        """Transform a position from game coordinates to projector coordinates.

        Args:
            pos: The (x, y) position in game coordinates.
            rounded: Whether to round the result and return integers.

        Returns:
            Transformed position in projector ROI coordinates.

        Raises:
            ValueError: If projector mapping is not calibrated.
        """
        if not self.projector_mapping or not self.projector_mapping.is_calibrated:
            raise ValueError("Projector mapping is not calibrated")

        pos_homo = np.float32([pos[0], pos[1], 1.0])
        warp_pos_homo = self.projector_mapping.game_to_projector_matrix.dot(pos_homo)
        warp_pos = (warp_pos_homo / warp_pos_homo[2])[:2]

        if rounded:
            return (round(warp_pos[0]), round(warp_pos[1]))
        return tuple(warp_pos)

    def warp_game_to_camera(self, image: np.ndarray) -> np.ndarray:
        """Warp a game image to camera ROI coordinates.

        Args:
            image: Game image (numpy array).

        Returns:
            Warped image in camera ROI dimensions.

        Raises:
            ValueError: If camera mapping is not calibrated.
        """
        if not self.camera_mapping or not self.camera_mapping.is_calibrated:
            raise ValueError("Camera mapping is not calibrated")

        roi = self.camera_mapping.roi
        width = roi['max_x'] - roi['min_x'] + 1
        height = roi['max_y'] - roi['min_y'] + 1
        return cv2.warpPerspective(image, self.camera_mapping.game_to_camera_matrix, (width, height))

    def warp_game_to_projector(self, image: np.ndarray) -> np.ndarray:
        """Warp a game image to projector ROI coordinates.

        Args:
            image: Game image (numpy array).

        Returns:
            Warped image in projector ROI dimensions.

        Raises:
            ValueError: If projector mapping is not calibrated.
        """
        if not self.projector_mapping or not self.projector_mapping.is_calibrated:
            raise ValueError("Projector mapping is not calibrated")

        roi = self.projector_mapping.roi
        width = roi['max_x'] - roi['min_x'] + 1
        height = roi['max_y'] - roi['min_y'] + 1
        return cv2.warpPerspective(image, self.projector_mapping.game_to_projector_matrix, (width, height))

    def calibrate(self) -> None:
        """Calibrate the zone by calculating transform matrices for enabled mappings.

        This calculates the perspective transform matrices between game coordinates
        and camera/projector coordinates for all enabled mappings.

        Raises:
            ValueError: If no mappings are enabled.
        """
        if not self.camera_mapping or not self.camera_mapping.enabled:
            if not self.projector_mapping or not self.projector_mapping.enabled:
                raise ValueError("Cannot calibrate: no mappings are enabled")

        # Calculate game dimensions
        width_px, height_px = self.get_game_dimensions()

        # Game corner points (P0, P1, P2, P3)
        game_points = np.float32([
            [0, 0],                                 # P0: top-left
            [width_px - 1, 0],                      # P1: top-right
            [width_px - 1, height_px - 1],          # P2: bottom-right
            [0, height_px - 1]                      # P3: bottom-left
        ])

        # Calibrate camera mapping if enabled
        if self.camera_mapping and self.camera_mapping.enabled:
            vertices = self.camera_mapping.vertices

            # Calculate ROI
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            roi = {
                'min_x': min(xs),
                'min_y': min(ys),
                'max_x': max(xs),
                'max_y': max(ys)
            }

            # Adjust vertices to ROI coordinates
            roi_vertices = np.float32([
                [v[0] - roi['min_x'], v[1] - roi['min_y']] for v in vertices
            ])

            # Calculate transform matrices
            self.camera_mapping.camera_to_game_matrix = cv2.getPerspectiveTransform(roi_vertices, game_points)
            self.camera_mapping.game_to_camera_matrix = cv2.getPerspectiveTransform(game_points, roi_vertices)
            self.camera_mapping.roi = roi
            self.camera_mapping.is_calibrated = True
            self.camera_mapping.lock_vertices = True

        # Calibrate projector mapping if enabled
        if self.projector_mapping and self.projector_mapping.enabled:
            vertices = self.projector_mapping.vertices

            # Calculate ROI
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            roi = {
                'min_x': min(xs),
                'min_y': min(ys),
                'max_x': max(xs),
                'max_y': max(ys)
            }

            # Adjust vertices to ROI coordinates
            roi_vertices = np.float32([
                [v[0] - roi['min_x'], v[1] - roi['min_y']] for v in vertices
            ])

            # Calculate transform matrices
            self.projector_mapping.projector_to_game_matrix = cv2.getPerspectiveTransform(roi_vertices, game_points)
            self.projector_mapping.game_to_projector_matrix = cv2.getPerspectiveTransform(game_points, roi_vertices)
            self.projector_mapping.roi = roi
            self.projector_mapping.is_calibrated = True
            self.projector_mapping.lock_vertices = True

    def uncalibrate(self) -> None:
        """Uncalibrate the zone by clearing transform matrices and calibration state."""
        if self.camera_mapping:
            self.camera_mapping.is_calibrated = False
            self.camera_mapping.camera_to_game_matrix = None
            self.camera_mapping.game_to_camera_matrix = None
            self.camera_mapping.roi = None

        if self.projector_mapping:
            self.projector_mapping.is_calibrated = False
            self.projector_mapping.projector_to_game_matrix = None
            self.projector_mapping.game_to_projector_matrix = None
            self.projector_mapping.roi = None

    def is_calibrated(self) -> bool:
        """Check if the zone is calibrated.

        Returns:
            True if any enabled mapping is calibrated.
        """
        if self.camera_mapping and self.camera_mapping.enabled and self.camera_mapping.is_calibrated:
            return True
        if self.projector_mapping and self.projector_mapping.enabled and self.projector_mapping.is_calibrated:
            return True
        return False

    def get_latest_camera_image_cropped(self, camera_manager) -> np.ndarray | None:
        """Get the latest camera image cropped to the ROI.

        Args:
            camera_manager: CameraManager instance to get camera from.

        Returns:
            Cropped camera image (ROI only) or None if no camera mapping or no frame available.
        """
        if not self.camera_mapping or not self.camera_mapping.enabled:
            return None

        if not self.camera_mapping.roi:
            return None

        # Get camera
        try:
            camera = camera_manager.get_camera(self.camera_mapping.camera_name)
        except KeyError:
            return None

        # Get latest frame
        frame = camera.get_undistorted_frame()
        if frame is None:
            return None

        # Crop to ROI
        roi = self.camera_mapping.roi
        min_x = max(0, roi['min_x'])
        min_y = max(0, roi['min_y'])
        max_x = min(frame.shape[1], roi['max_x'])
        max_y = min(frame.shape[0], roi['max_y'])

        cropped = frame[min_y:max_y, min_x:max_x]
        return cropped
