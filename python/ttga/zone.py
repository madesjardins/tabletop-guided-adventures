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


@dataclass
class CameraMapping:
    """Camera mapping for a zone.

    Attributes:
        camera_name: Name of the camera.
        vertices: List of 4 (x, y) tuples representing the quadrilateral vertices.
        lock_vertices: Whether vertices are locked from editing.
    """
    camera_name: str
    vertices: list[tuple[int, int]] = field(default_factory=lambda: [
        (128, 128),   # P0: Cyan (0, 0)
        (384, 128),   # P1: Magenta (wpx, 0)
        (384, 256),   # P2: Yellow (wpx, hpx)
        (128, 256)    # P3: White (0, hpx)
    ])
    lock_vertices: bool = False

    def to_dict(self) -> dict:
        """Serialize camera mapping to dictionary.

        Returns:
            Dictionary containing camera mapping data.
        """
        return {
            'camera_name': self.camera_name,
            'vertices': self.vertices,
            'lock_vertices': self.lock_vertices
        }

    @staticmethod
    def from_dict(data: dict) -> 'CameraMapping':
        """Deserialize camera mapping from dictionary.

        Args:
            data: Dictionary containing camera mapping data.

        Returns:
            CameraMapping instance.
        """
        return CameraMapping(
            camera_name=data['camera_name'],
            vertices=[tuple(v) for v in data['vertices']],
            lock_vertices=data.get('lock_vertices', False)
        )


@dataclass
class ProjectorMapping:
    """Projector mapping for a zone.

    Attributes:
        projector_name: Name of the projector.
        vertices: List of 4 (x, y) tuples representing the quadrilateral vertices.
        lock_vertices: Whether vertices are locked from editing.
    """
    projector_name: str
    vertices: list[tuple[int, int]] = field(default_factory=lambda: [
        (128, 128),   # P0: Cyan (0, 0)
        (384, 128),   # P1: Magenta (wpx, 0)
        (384, 256),   # P2: Yellow (wpx, hpx)
        (128, 256)    # P3: White (0, hpx)
    ])
    lock_vertices: bool = False

    def to_dict(self) -> dict:
        """Serialize projector mapping to dictionary.

        Returns:
            Dictionary containing projector mapping data.
        """
        return {
            'projector_name': self.projector_name,
            'vertices': self.vertices,
            'lock_vertices': self.lock_vertices
        }

    @staticmethod
    def from_dict(data: dict) -> 'ProjectorMapping':
        """Deserialize projector mapping from dictionary.

        Args:
            data: Dictionary containing projector mapping data.

        Returns:
            ProjectorMapping instance.
        """
        return ProjectorMapping(
            projector_name=data['projector_name'],
            vertices=[tuple(v) for v in data['vertices']],
            lock_vertices=data.get('lock_vertices', False)
        )


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
