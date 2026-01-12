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

"""Game loader module for discovering and loading game plugins.

This module provides functionality to discover game plugins from the games/
and test_games/ directories and load them dynamically.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .game_base import GameBase

if TYPE_CHECKING:
    from .main_core import MainCore


class GameInfo:
    """Information about a discovered game.

    Attributes:
        name: Display name of the game.
        version: Game version string.
        author: Game author name.
        description: Game description.
        module_path: Path to the game's module file.
        folder_path: Path to the game's folder.
        is_test_game: True if from test_games/, False if from games/.
    """

    def __init__(
        self,
        name: str,
        version: str,
        author: str,
        description: str,
        module_path: str,
        folder_path: str,
        is_test_game: bool
    ) -> None:
        """Initialize game info.

        Args:
            name: Display name of the game.
            version: Game version string.
            author: Game author name.
            description: Game description.
            module_path: Path to the game's module file.
            folder_path: Path to the game's folder.
            is_test_game: True if from test_games/, False if from games/.
        """
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.module_path = module_path
        self.folder_path = folder_path
        self.is_test_game = is_test_game

    def __str__(self) -> str:
        """String representation of game info."""
        source = "test_games" if self.is_test_game else "games"
        return f"{self.name} v{self.version} by {self.author} ({source})"


class GameLoader:
    """Discovers and loads game plugins from games/ and test_games/ directories."""

    def __init__(self) -> None:
        """Initialize the game loader."""
        # Get root directory (3 levels up from this file)
        self.root_dir = Path(__file__).parent.parent.parent.resolve()
        self.games_dir = self.root_dir / "games"
        self.test_games_dir = self.root_dir / "test_games"

    def discover_games(self) -> list[GameInfo]:
        """Discover all available games.

        Searches both games/ and test_games/ directories for valid game plugins.
        Each game must be in its own folder with a game.py file containing a
        Game class that inherits from GameBase.

        Returns:
            List of GameInfo objects for all discovered games.

        Example:
            >>> loader = GameLoader()
            >>> games = loader.discover_games()
            >>> for game in games:
            ...     print(game.name, game.version)
        """
        discovered_games = []

        # Discover from test_games/ (tracked)
        if self.test_games_dir.exists():
            discovered_games.extend(self._discover_from_directory(self.test_games_dir, is_test_game=True))

        # Discover from games/ (untracked)
        if self.games_dir.exists():
            discovered_games.extend(self._discover_from_directory(self.games_dir, is_test_game=False))

        return discovered_games

    def _discover_from_directory(self, directory: Path, is_test_game: bool) -> list[GameInfo]:
        """Discover games from a specific directory.

        Args:
            directory: Directory to search for games.
            is_test_game: True if this is the test_games directory.

        Returns:
            List of GameInfo objects found in this directory.
        """
        games = []

        # Iterate through subdirectories
        for item in directory.iterdir():
            if not item.is_dir():
                continue

            # Skip __pycache__ and hidden directories
            if item.name.startswith('__') or item.name.startswith('.'):
                continue

            # Look for game.py in the subdirectory
            game_file = item / "game.py"
            if not game_file.exists():
                continue

            # Try to load metadata without fully loading the game
            try:
                metadata = self._load_game_metadata(str(game_file))
                if metadata:
                    games.append(GameInfo(
                        name=metadata.get('name', item.name),
                        version=metadata.get('version', '0.0.0'),
                        author=metadata.get('author', 'Unknown'),
                        description=metadata.get('description', ''),
                        module_path=str(game_file),
                        folder_path=str(item),
                        is_test_game=is_test_game
                    ))
            except Exception as e:
                print(f"Error loading game metadata from {game_file}: {e}")

        return games

    def _load_game_metadata(self, module_path: str) -> Optional[dict]:
        """Load game metadata without instantiating the game.

        Args:
            module_path: Path to the game.py file.

        Returns:
            Metadata dictionary or None if loading failed.
        """
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("temp_game_module", module_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for Game class
            if not hasattr(module, 'Game'):
                return None

            game_class = getattr(module, 'Game')

            # Check if it's a subclass of GameBase
            if not issubclass(game_class, GameBase):
                return None

            # Try to get metadata (we need a dummy core for this)
            # We'll create a minimal metadata extraction
            # For now, we'll just return a basic dict and let the actual load get full metadata
            return {
                'name': getattr(game_class, 'GAME_NAME', None),
                'version': getattr(game_class, 'GAME_VERSION', None),
                'author': getattr(game_class, 'GAME_AUTHOR', None),
                'description': getattr(game_class, 'GAME_DESCRIPTION', None)
            }

        except Exception as e:
            print(f"Error loading metadata from {module_path}: {e}")
            return None

    def load_game(self, game_info: GameInfo, core: MainCore) -> Optional[GameBase]:
        """Load and instantiate a game.

        Args:
            game_info: GameInfo object for the game to load.
            core: MainCore instance to pass to the game.

        Returns:
            Instantiated game object or None if loading failed.

        Example:
            >>> loader = GameLoader()
            >>> games = loader.discover_games()
            >>> game = loader.load_game(games[0], core)
            >>> game.on_load()
        """
        try:
            # Add the game's folder to sys.path so it can import local modules
            game_folder = Path(game_info.folder_path)
            if str(game_folder) not in sys.path:
                sys.path.insert(0, str(game_folder))

            # Load the module
            spec = importlib.util.spec_from_file_location(
                f"game_{game_folder.name}",
                game_info.module_path
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the Game class
            if not hasattr(module, 'Game'):
                print(f"No 'Game' class found in {game_info.module_path}")
                return None

            game_class = getattr(module, 'Game')

            # Verify it's a GameBase subclass
            if not issubclass(game_class, GameBase):
                print(f"Game class in {game_info.module_path} does not inherit from GameBase")
                return None

            # Instantiate the game
            game_instance = game_class(core)

            return game_instance

        except Exception as e:
            print(f"Error loading game from {game_info.module_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def unload_game(self, game_info: GameInfo) -> None:
        """Clean up after unloading a game.

        Removes the game's folder from sys.path.

        Args:
            game_info: GameInfo object for the game being unloaded.
        """
        game_folder = str(Path(game_info.folder_path))
        if game_folder in sys.path:
            sys.path.remove(game_folder)
