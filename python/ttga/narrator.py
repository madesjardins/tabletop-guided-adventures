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

"""Narrator module for text-to-speech and audio playback management.

This module provides the Narrator class which manages Piper TTS voice synthesis
and SoundMixer audio playback across multiple channels (Voice, Effect, Music).
"""

import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

from piper import PiperVoice

from .sound_mixer import SoundMixer, Channel


def find_available_voices(voices_dir: str) -> list[str]:
    """Find all available Piper voice models in the voices directory.

    A valid voice model consists of both a .onnx file and a .onnx.json file
    with the same base name.

    Args:
        voices_dir: Path to the directory containing voice models.

    Returns:
        List of paths to valid .onnx voice model files.
    """
    voices_path = Path(voices_dir)
    if not voices_path.exists():
        return []

    available_voices = []

    for onnx_file in voices_path.glob("*.onnx"):
        json_file = voices_path / f"{onnx_file.name}.json"

        if json_file.exists():
            available_voices.append(str(onnx_file))

    return available_voices


class Narrator:
    """Manages text-to-speech synthesis and audio playback.

    This class handles Piper TTS voice model management and SoundMixer
    integration for playing synthesized speech and audio files across
    multiple channels.

    Attributes:
        voice_model_path: Path to the currently loaded Piper voice model.
        mixer: SoundMixer instance for audio playback.
        piper_voice: Currently loaded PiperVoice instance.
    """

    def __init__(self, voice_model_path: Optional[str] = None) -> None:
        """Initialize the narrator.

        Args:
            voice_model_path: Optional path to Piper voice model to load initially.
        """
        self.voice_model_path: Optional[str] = voice_model_path
        self.mixer = SoundMixer()
        self.piper_voice: Optional[PiperVoice] = None
        self._temp_dir = tempfile.mkdtemp(prefix="ttga_narrator_")

        if voice_model_path:
            self.load_voice_model(voice_model_path)

    def load_voice_model(self, model_path: str) -> None:
        """Load a Piper voice model.

        Args:
            model_path: Path to the .onnx voice model file.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
            ValueError: If the model file is invalid.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Voice model not found: {model_path}")

        json_path = f"{model_path}.json"
        if not os.path.exists(json_path):
            raise ValueError(f"Voice model config not found: {json_path}")

        self.piper_voice = PiperVoice.load(model_path)
        self.voice_model_path = model_path

    def set_voice_model(self, model_path: str) -> None:
        """Set and load a new voice model, replacing the current one.

        Args:
            model_path: Path to the .onnx voice model file.
        """
        self.load_voice_model(model_path)

    def set_channel_volume(self, channel: Channel, volume: float) -> None:
        """Set the volume for a specific channel.

        Args:
            channel: The channel to set volume for.
            volume: Volume level (0.0 to 1.0).
        """
        self.mixer.set_volume(channel, volume)

    def get_channel_volume(self, channel: Channel) -> float:
        """Get the volume for a specific channel.

        Args:
            channel: The channel to get volume for.

        Returns:
            Volume level (0.0 to 1.0).
        """
        return self.mixer.get_volume(channel)

    def synthesize_and_play(
        self,
        text: str,
        channel: Channel = Channel.VOICE,
        do_play_immediately: bool = False,
        do_wait_until_played: bool = False
    ) -> None:
        """Synthesize text to speech and play it on the specified channel.

        Args:
            text: The text to synthesize.
            channel: The channel to play on (default: VOICE).
            do_play_immediately: If True, clear channel queue and play immediately.
            do_wait_until_played: If True, block until playback is complete.

        Raises:
            RuntimeError: If no voice model is loaded.
        """
        if not self.piper_voice:
            raise RuntimeError("No voice model loaded. Call load_voice_model() first.")

        # Create temporary WAV file
        temp_wav = os.path.join(self._temp_dir, f"tts_{id(text)}.wav")

        try:
            # Synthesize speech to WAV file
            with wave.open(temp_wav, 'wb') as wav_file:
                self.piper_voice.synthesize_wav(text, wav_file)

            # Play the synthesized audio
            self.mixer.play(
                sound_file_path=temp_wav,
                channel=channel,
                do_play_immediately=do_play_immediately
            )

            # Wait for playback to complete if requested
            if do_wait_until_played:
                while self.mixer.is_channel_busy(channel):
                    time.sleep(0.1)

        finally:
            # Clean up temporary file after playback
            if do_wait_until_played and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    def play_audio_file(
        self,
        file_path: str,
        channel: Channel = Channel.EFFECT,
        do_play_immediately: bool = False,
        do_wait_until_played: bool = False
    ) -> None:
        """Play an audio file (MP3 or WAV) on the specified channel.

        Args:
            file_path: Path to the audio file.
            channel: The channel to play on (default: EFFECT).
            do_play_immediately: If True, clear channel queue and play immediately.
            do_wait_until_played: If True, block until playback is complete.

        Raises:
            FileNotFoundError: If the audio file doesn't exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self.mixer.play(
            sound_file_path=file_path,
            channel=channel,
            do_play_immediately=do_play_immediately
        )

        # Wait for playback to complete if requested
        if do_wait_until_played:
            while self.mixer.is_channel_busy(channel):
                time.sleep(0.1)

    def is_channel_busy(self, channel: Channel) -> bool:
        """Check if a channel is currently playing audio.

        Args:
            channel: The channel to check.

        Returns:
            True if the channel is busy, False otherwise.
        """
        return self.mixer.is_channel_busy(channel)

    def shutdown(self) -> None:
        """Shutdown the narrator and clean up resources."""
        self.mixer.shutdown()

        # Clean up temporary directory
        if os.path.exists(self._temp_dir):
            try:
                import shutil
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
