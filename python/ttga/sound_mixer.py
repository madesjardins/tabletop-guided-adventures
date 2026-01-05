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

"""Sound mixer module for managing multi-channel audio playback.

This module provides a thread-safe sound mixer that supports independent audio
channels for music, effects, and voice with queuing capabilities.
"""

from __future__ import annotations

import threading
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Optional

import pygame


class Channel(Enum):
    """Audio channel types for the sound mixer.

    Attributes:
        MUSIC: Channel for background music playback.
        EFFECT: Channel for sound effects playback.
        VOICE: Channel for voice/dialogue playback.
    """
    MUSIC = "music"
    EFFECT = "effect"
    VOICE = "voice"


class SoundMixer:
    """Multi-channel audio mixer with independent queuing per channel.

    The SoundMixer manages three independent audio channels (MUSIC, EFFECT, VOICE),
    each with its own queue and playback thread. Audio files are played sequentially
    from each channel's queue, with support for immediate playback that clears the
    queue and interrupts current playback.

    Attributes:
        _queues: Dictionary mapping channels to their audio file queues.
        _pygame_channels: Dictionary mapping channels to pygame mixer channels.
        _locks: Dictionary of threading locks for thread-safe channel operations.
        _running: Flag indicating if worker threads should continue running.
        _worker_threads: Dictionary of worker threads for each channel.

    Example:
        >>> mixer = SoundMixer()
        >>> mixer.play("background.wav", Channel.MUSIC)
        >>> mixer.play("explosion.wav", Channel.EFFECT, do_play_immediately=True)
        >>> mixer.shutdown()
    """

    def __init__(
        self,
        frequency: int = 44100,
        size: int = -16,
        channels: int = 2,
        buffer: int = 512
    ) -> None:
        """Initialize the sound mixer with pygame audio settings.

        Args:
            frequency: Audio sampling frequency in Hz. Default is 44100.
            size: Sample size in bits. Negative values indicate signed samples.
                Default is -16 (16-bit signed).
            channels: Number of audio channels (1=mono, 2=stereo). Default is 2.
            buffer: Audio buffer size. Smaller values reduce latency but may cause
                audio artifacts. Default is 512.
        """
        pygame.mixer.init(frequency=frequency, size=size, channels=channels, buffer=buffer)

        self._queues: dict[Channel, Queue[str]] = {
            Channel.MUSIC: Queue(),
            Channel.EFFECT: Queue(),
            Channel.VOICE: Queue()
        }

        self._pygame_channels: dict[Channel, pygame.mixer.Channel] = {
            Channel.MUSIC: pygame.mixer.Channel(0),
            Channel.EFFECT: pygame.mixer.Channel(1),
            Channel.VOICE: pygame.mixer.Channel(2)
        }

        self._locks: dict[Channel, threading.Lock] = {
            Channel.MUSIC: threading.Lock(),
            Channel.EFFECT: threading.Lock(),
            Channel.VOICE: threading.Lock()
        }

        self._is_busy: dict[Channel, bool] = {
            Channel.MUSIC: False,
            Channel.EFFECT: False,
            Channel.VOICE: False
        }

        self._running: bool = True
        self._worker_threads: dict[Channel, threading.Thread] = {}

        for channel in Channel:
            thread = threading.Thread(target=self._channel_worker, args=(channel,), daemon=True)
            thread.start()
            self._worker_threads[channel] = thread

    def _channel_worker(self, channel: Channel) -> None:
        """Worker thread that processes audio files from a channel's queue.

        This method runs in a separate thread for each channel, continuously
        checking the queue for audio files to play. When a file is found, it
        loads and plays it, blocking until playback completes before processing
        the next file.

        Args:
            channel: The audio channel this worker manages.
        """
        while self._running:
            if not self._queues[channel].empty():
                sound_file_path = self._queues[channel].get()

                try:
                    sound = pygame.mixer.Sound(sound_file_path)
                    pygame_channel = self._pygame_channels[channel]
                    pygame_channel.play(sound)

                    while pygame_channel.get_busy():
                        pygame.time.wait(100)

                except Exception as e:
                    print(f"Error playing sound on {channel.value} channel: {e}")
                finally:
                    self._queues[channel].task_done()
                    if self._queues[channel].empty():
                        with self._locks[channel]:
                            self._is_busy[channel] = False
            else:
                pygame.time.wait(100)

    def play(
        self,
        sound_file_path: str,
        channel: Channel,
        do_play_immediately: bool = False
    ) -> None:
        """Queue or immediately play an audio file on the specified channel.

        By default, the audio file is added to the channel's queue and will play
        when all previously queued files have finished. If do_play_immediately is
        True, the queue is cleared, current playback is stopped, and the new file
        plays immediately.

        Args:
            sound_file_path: Path to the WAV audio file to play.
            channel: The audio channel to play on (MUSIC, EFFECT, or VOICE).
            do_play_immediately: If True, clear queue and interrupt current playback.
                Default is False.

        Raises:
            ValueError: If channel is not a valid Channel enum value.
            FileNotFoundError: If the audio file does not exist.

        Example:
            >>> mixer.play("background.wav", Channel.MUSIC)
            >>> mixer.play("urgent.wav", Channel.VOICE, do_play_immediately=True)
        """
        if not isinstance(channel, Channel):
            raise ValueError(f"Invalid channel. Must be one of {list(Channel)}")

        sound_path = Path(sound_file_path)
        if not sound_path.exists():
            raise FileNotFoundError(f"Sound file not found: {sound_file_path}")

        with self._locks[channel]:
            if do_play_immediately:
                while not self._queues[channel].empty():
                    try:
                        self._queues[channel].get_nowait()
                        self._queues[channel].task_done()
                    except Exception:
                        break

                self._pygame_channels[channel].stop()

            self._is_busy[channel] = True
            self._queues[channel].put(str(sound_path))

    def stop_channel(self, channel: Channel) -> None:
        """Stop playback and clear the queue for a specific channel.

        Args:
            channel: The audio channel to stop.

        Example:
            >>> mixer.stop_channel(Channel.MUSIC)
        """
        with self._locks[channel]:
            while not self._queues[channel].empty():
                try:
                    self._queues[channel].get_nowait()
                    self._queues[channel].task_done()
                except Exception:
                    break

            self._pygame_channels[channel].stop()
            self._is_busy[channel] = False

    def stop_all(self) -> None:
        """Stop playback and clear queues for all channels.

        Example:
            >>> mixer.stop_all()
        """
        for channel in Channel:
            self.stop_channel(channel)

    def is_channel_busy(self, channel: Channel) -> bool:
        """Check if a channel is currently playing or has queued audio.

        Args:
            channel: The audio channel to check.

        Returns:
            True if the channel is playing audio or has files in queue, False otherwise.

        Example:
            >>> if mixer.is_channel_busy(Channel.MUSIC):
            ...     print("Music is playing")
        """
        with self._locks[channel]:
            return self._is_busy[channel]

    def get_queue_size(self, channel: Channel) -> int:
        """Get the number of audio files queued for a channel.

        Args:
            channel: The audio channel to query.

        Returns:
            Number of audio files waiting in the channel's queue.

        Example:
            >>> queue_size = mixer.get_queue_size(Channel.EFFECT)
        """
        return self._queues[channel].qsize()

    def shutdown(self) -> None:
        """Shutdown the mixer and clean up resources.

        Stops all worker threads and quits the pygame mixer. This should be
        called before the application exits to ensure proper cleanup.

        Example:
            >>> mixer.shutdown()
        """
        self._running = False

        for thread in self._worker_threads.values():
            thread.join(timeout=2.0)

        pygame.mixer.quit()

    def __enter__(self) -> SoundMixer:
        """Context manager entry.

        Returns:
            The SoundMixer instance.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object]
    ) -> None:
        """Context manager exit with automatic cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self.shutdown()
