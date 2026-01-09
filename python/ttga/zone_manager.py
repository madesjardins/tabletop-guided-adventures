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
