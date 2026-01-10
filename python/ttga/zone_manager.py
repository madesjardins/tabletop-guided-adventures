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

"""Zone manager for handling multiple zones."""

from PySide6 import QtCore

from .zone import Zone


class ZoneManager(QtCore.QObject):
    """Manager for handling multiple zones.

    Signals:
        zone_added: Emitted when a zone is added (zone_name).
        zone_removed: Emitted when a zone is removed (zone_name).
    """

    zone_added = QtCore.Signal(str)
    zone_removed = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        """Initialize the zone manager.

        Args:
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._zones: dict[str, Zone] = {}

    def add_zone(self, zone: Zone) -> None:
        """Add a new zone.

        Args:
            zone: Zone instance to add.

        Raises:
            ValueError: If a zone with this name already exists.
        """
        if zone.name in self._zones:
            raise ValueError(f"Zone '{zone.name}' already exists")

        self._zones[zone.name] = zone
        self.zone_added.emit(zone.name)

    def remove_zone(self, name: str) -> None:
        """Remove a zone.

        Args:
            name: Name of the zone to remove.

        Raises:
            KeyError: If zone doesn't exist.
        """
        if name not in self._zones:
            raise KeyError(f"Zone '{name}' not found")

        del self._zones[name]
        self.zone_removed.emit(name)

    def get_zone(self, name: str) -> Zone:
        """Get a zone by name.

        Args:
            name: Name of the zone.

        Returns:
            The Zone instance.

        Raises:
            KeyError: If zone doesn't exist.
        """
        return self._zones[name]

    def get_all_zones(self) -> list[Zone]:
        """Get all zones.

        Returns:
            List of all Zone instances.
        """
        return list(self._zones.values())

    def zone_exists(self, name: str) -> bool:
        """Check if a zone exists.

        Args:
            name: Name to check.

        Returns:
            True if zone exists, False otherwise.
        """
        return name in self._zones

    def serialize_zones(self) -> list[dict]:
        """Serialize all zones to a list of dictionaries.

        Returns:
            List of zone data dictionaries.
        """
        return [zone.to_dict() for zone in self._zones.values()]

    def clear_all(self) -> None:
        """Remove all zones."""
        zone_names = list(self._zones.keys())
        for name in zone_names:
            self.remove_zone(name)

    def get_zones_with_camera_mapping(self, camera_name: str) -> list[Zone]:
        """Get all zones that have camera mapping enabled for the specified camera.

        Args:
            camera_name: Name of the camera to filter by.

        Returns:
            List of zones with camera mapping for the specified camera.
        """
        zones = []
        for zone in self._zones.values():
            if (zone.camera_mapping and
                    zone.camera_mapping.enabled and
                    zone.camera_mapping.camera_name == camera_name):
                zones.append(zone)
        return zones

    def get_zones_with_projector_mapping(self, projector_name: str) -> list[Zone]:
        """Get all zones that have projector mapping enabled for the specified projector.

        Args:
            projector_name: Name of the projector to filter by.

        Returns:
            List of zones with projector mapping for the specified projector.
        """
        zones = []
        for zone in self._zones.values():
            if (zone.projector_mapping and
                    zone.projector_mapping.enabled and
                    zone.projector_mapping.projector_name == projector_name):
                zones.append(zone)
        return zones

    def find_vertex_at_position(self, camera_name: str, x: int, y: int, max_distance: int = 7) -> tuple[Zone, int] | None:
        """Find the closest vertex to a click position for zones with unlocked vertices.

        Args:
            camera_name: Name of the camera.
            x: X coordinate of the click in camera frame pixels.
            y: Y coordinate of the click in camera frame pixels.
            max_distance: Maximum distance in pixels to consider (default: VERTEX_RADIUS).

        Returns:
            Tuple of (Zone, vertex_index) if found, None otherwise.
            If multiple vertices are within max_distance, returns the closest one.
        """
        import math

        closest_zone = None
        closest_vertex_idx = -1
        closest_distance = float('inf')

        # Check all zones with camera mapping enabled for this camera
        for zone in self.get_zones_with_camera_mapping(camera_name):
            # Skip if vertices are locked
            if zone.camera_mapping.lock_vertices:
                continue

            # Check each vertex
            for idx, (vx, vy) in enumerate(zone.camera_mapping.vertices):
                distance = math.sqrt((x - vx) ** 2 + (y - vy) ** 2)

                # Check if within max_distance and closer than previous best
                if distance <= max_distance and distance < closest_distance:
                    closest_distance = distance
                    closest_zone = zone
                    closest_vertex_idx = idx

        if closest_zone is not None:
            return (closest_zone, closest_vertex_idx)
        return None

    def find_projector_vertex_at_position(self, projector_name: str, x: int, y: int, max_distance: int = 7) -> tuple[Zone, int] | None:
        """Find the closest vertex to a click position for zones with unlocked projector vertices.

        Args:
            projector_name: Name of the projector.
            x: X coordinate of the click in projector frame pixels.
            y: Y coordinate of the click in projector frame pixels.
            max_distance: Maximum distance in pixels to consider (default: VERTEX_RADIUS).

        Returns:
            Tuple of (Zone, vertex_index) if found, None otherwise.
            If multiple vertices are within max_distance, returns the closest one.
        """
        import math

        closest_zone = None
        closest_vertex_idx = -1
        closest_distance = float('inf')

        # Check all zones with projector mapping enabled for this projector
        for zone in self.get_zones_with_projector_mapping(projector_name):
            # Skip if vertices are locked
            if zone.projector_mapping.lock_vertices:
                continue

            # Check each vertex
            for idx, (vx, vy) in enumerate(zone.projector_mapping.vertices):
                distance = math.sqrt((x - vx) ** 2 + (y - vy) ** 2)

                # Check if within max_distance and closer than previous best
                if distance <= max_distance and distance < closest_distance:
                    closest_distance = distance
                    closest_zone = zone
                    closest_vertex_idx = idx

        if closest_zone is not None:
            return (closest_zone, closest_vertex_idx)
        return None
