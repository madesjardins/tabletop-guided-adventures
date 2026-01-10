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

"""Speech recognition module using Vosk for speech-to-text.

This module provides speech recognition capabilities using the Vosk library,
along with string similarity functions for comparing recognized text."""

from __future__ import annotations

import json
import queue
import threading
from typing import Optional

import sounddevice as sd
import vosk
from Levenshtein import distance as levenshtein_distance
from PySide6.QtCore import QObject, Signal


def get_audio_input_devices() -> list[dict[str, any]]:
    """Get list of available audio input devices.

    Returns:
        List of dictionaries containing device information with keys:
        - 'index': Device index
        - 'name': Device name
        - 'channels': Number of input channels
        - 'sample_rate': Default sample rate

    Example:
        >>> devices = get_audio_input_devices()
        >>> for device in devices:
        ...     print(f"{device['index']}: {device['name']}")
    """
    devices = []
    device_list = sd.query_devices()

    for idx, device in enumerate(device_list):
        # Only include devices with input channels
        if device['max_input_channels'] > 0:
            devices.append({
                'index': idx,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate']
            })

    return devices


def levenshtein_similarity(str1: str, str2: str) -> float:
    """Calculate Levenshtein similarity between two strings.

    Returns a normalized similarity score between 0.0 and 1.0, where 1.0
    means the strings are identical.

    Args:
        str1: First string to compare.
        str2: Second string to compare.

    Returns:
        Similarity score between 0.0 and 1.0.

    Example:
        >>> levenshtein_similarity("hello", "hallo")
        0.8
        >>> levenshtein_similarity("hello", "hello")
        1.0
    """
    if not str1 and not str2:
        return 1.0

    if not str1 or not str2:
        return 0.0

    max_len = max(len(str1), len(str2))
    distance = levenshtein_distance(str1, str2)

    return 1.0 - (distance / max_len)


def jaccard_similarity(str1: str, str2: str) -> float:
    """Calculate Jaccard similarity between two strings.

    Computes similarity based on character set overlap. Returns a score
    between 0.0 and 1.0, where 1.0 means the strings have identical character sets.

    Args:
        str1: First string to compare.
        str2: Second string to compare.

    Returns:
        Similarity score between 0.0 and 1.0.

    Example:
        >>> jaccard_similarity("hello", "hallo")
        0.8
        >>> jaccard_similarity("abc", "xyz")
        0.0
    """
    if not str1 and not str2:
        return 1.0

    if not str1 or not str2:
        return 0.0

    set1 = set(str1)
    set2 = set(str2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def string_similarity(str1: str, str2: str) -> float:
    """Calculate combined string similarity using Levenshtein and Jaccard.

    Returns the average of Levenshtein and Jaccard similarity scores.

    Args:
        str1: First string to compare.
        str2: Second string to compare.

    Returns:
        Combined similarity score between 0.0 and 1.0.

    Example:
        >>> string_similarity("hello world", "hallo world")
        0.9
    """
    lev_sim = levenshtein_similarity(str1, str2)
    jac_sim = jaccard_similarity(str1, str2)

    return (lev_sim + jac_sim) / 2.0


class SpeechRecognizer(QObject):
    """Speech recognition using Vosk speech-to-text.

    This class captures audio from an input device and performs real-time
    speech recognition using Vosk. It emits signals for both partial results
    (during speech) and final results (after silence detection).

    Signals:
        partial_result: Emitted during speech with partial recognition text.
        final_result: Emitted after silence with final recognition text.
        error_occurred: Emitted when an error occurs with error message.

    Example:
        >>> recognizer = SpeechRecognizer(model_path="/path/to/vosk-model")
        >>> recognizer.partial_result.connect(lambda text: print(f"Partial: {text}"))
        >>> recognizer.final_result.connect(lambda text: print(f"Final: {text}"))
        >>> recognizer.start()
    """

    partial_result = Signal(str)
    final_result = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        model_path: str,
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        parent: Optional[QObject] = None
    ) -> None:
        """Initialize the speech recognizer.

        Args:
            model_path: Path to the Vosk model directory.
            device_index: Audio input device index (None for default device).
            sample_rate: Audio sample rate in Hz (default: 16000).
            parent: Parent QObject for Qt ownership.

        Raises:
            Exception: If the Vosk model cannot be loaded.
        """
        super().__init__(parent)

        self._model_path: str = model_path
        self._device_index: Optional[int] = device_index
        self._sample_rate: int = sample_rate

        self._model: Optional[vosk.Model] = None
        self._recognizer: Optional[vosk.KaldiRecognizer] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._is_running: bool = False
        self._recognition_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

        # Load Vosk model
        try:
            self._model = vosk.Model(model_path)
        except Exception as e:
            raise Exception(f"Failed to load Vosk model from {model_path}: {e}")

    def start(self) -> None:
        """Start speech recognition.

        Opens the audio input stream and begins processing audio data.

        Raises:
            RuntimeError: If recognition is already running.
        """
        if self._is_running:
            raise RuntimeError("Speech recognition is already running")

        self._is_running = True

        # Create recognizer
        self._recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)

        # Start recognition thread
        self._recognition_thread = threading.Thread(
            target=self._recognition_loop,
            daemon=True
        )
        self._recognition_thread.start()

        # Start audio stream
        try:
            self._stream = sd.InputStream(
                device=self._device_index,
                samplerate=self._sample_rate,
                channels=1,
                dtype='int16',
                callback=self._audio_callback
            )
            self._stream.start()
        except Exception as e:
            self._is_running = False
            self.error_occurred.emit(f"Failed to start audio stream: {e}")

    def stop(self) -> None:
        """Stop speech recognition and release resources."""
        if not self._is_running:
            return

        self._is_running = False

        # Stop audio stream
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Wait for recognition thread to finish
        if self._recognition_thread is not None:
            self._recognition_thread.join(timeout=2.0)
            self._recognition_thread = None

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def is_running(self) -> bool:
        """Check if speech recognition is currently running.

        Returns:
            True if running, False otherwise.
        """
        return self._is_running

    def _audio_callback(
        self,
        indata: any,
        frames: int,
        time_info: any,
        status: sd.CallbackFlags
    ) -> None:
        """Callback for audio input stream.

        Args:
            indata: Input audio data.
            frames: Number of frames.
            time_info: Time information.
            status: Stream status flags.
        """
        if status:
            self.error_occurred.emit(f"Audio stream status: {status}")

        # Add audio data to queue
        self._audio_queue.put(bytes(indata))

    def _recognition_loop(self) -> None:
        """Main recognition loop running in separate thread."""
        while self._is_running:
            try:
                # Get audio data from queue with timeout
                data = self._audio_queue.get(timeout=0.1)

                if self._recognizer.AcceptWaveform(data):
                    # Final result (after silence)
                    result = json.loads(self._recognizer.Result())
                    text = result.get('text', '').strip()
                    if text in ["huh", "hum", "ah", "ha"]:  # Ignore these common words
                        continue
                    if text:
                        self.final_result.emit(text)
                else:
                    # Partial result (during speech)
                    partial = json.loads(self._recognizer.PartialResult())
                    text = partial.get('partial', '').strip()

                    if text:
                        self.partial_result.emit(text)

            except queue.Empty:
                continue
            except Exception as e:
                if self._is_running:
                    self.error_occurred.emit(f"Recognition error: {e}")

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.stop()
