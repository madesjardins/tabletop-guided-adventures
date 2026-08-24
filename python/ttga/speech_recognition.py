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

"""Speech recognition module for the TTGA application.

Provides two speech recognition backends:

* :class:`SpeechRecognizer` — Vosk-based, lightweight, real-time streaming.
* :class:`WhisperSpeechRecognizer` — faster-whisper-based, higher accuracy,
  especially for proper nouns, with ``initial_prompt`` support for custom
  vocabulary.

Also provides string similarity utilities and post-processing helpers for
cleaning up recognized text and fuzzy-matching it against a candidate list.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from typing import Optional

import numpy as np
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


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

_LEADING_ARTICLES = re.compile(r"^\s*(the|a|an)\s+", re.IGNORECASE)
_TRAILING_ARTICLES = re.compile(r"\s+(the|a|an)\s*$", re.IGNORECASE)
_FILLER_WORDS = {"uh", "um", "huh", "hum", "ah", "ha", "er", "mm", "hmm"}


def clean_transcript(text: str) -> str:
    """Clean up a recognized transcript for matching.

    Strips leading/trailing articles ("the", "a", "an"), filler words,
    and extra whitespace.  Does not change internal words.

    Args:
        text: Raw recognized text.

    Returns:
        Cleaned text suitable for name matching.
    """
    text = text.strip()
    text = _LEADING_ARTICLES.sub("", text)
    text = _TRAILING_ARTICLES.sub("", text)
    text = text.strip()
    return text


def fuzzy_match_name(
    spoken: str,
    candidates: list[str],
    threshold: float = 0.7,
) -> Optional[str]:
    """Find the best fuzzy match for *spoken* from *candidates*.

    Uses :func:`string_similarity` (Levenshtein + Jaccard average) to compare
    the cleaned spoken text against each candidate.  Returns the best
    candidate if its similarity score meets *threshold*, otherwise ``None``.

    Args:
        spoken: The raw recognized text.
        candidates: List of candidate names to match against.
        threshold: Minimum similarity score (0.0–1.0) to accept a match.

    Returns:
        The best-matching candidate string, or ``None`` if no candidate
        meets the threshold.
    """
    cleaned = clean_transcript(spoken).lower()
    if not cleaned:
        return None

    best_score = 0.0
    best_match: Optional[str] = None
    for candidate in candidates:
        cand_clean = clean_transcript(candidate).lower()
        if not cand_clean:
            continue
        if cleaned == cand_clean:
            return candidate
        score = string_similarity(cleaned, cand_clean)
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return best_match
    return None


# ---------------------------------------------------------------------------
# Whisper-based speech recognizer (faster-whisper backend)
# ---------------------------------------------------------------------------


class WhisperSpeechRecognizer(QObject):
    """Speech recognition using faster-whisper for high-accuracy STT.

    This class captures audio from an input device and performs speech
    recognition using a Whisper model via the ``faster-whisper`` library
    (CTranslate2 backend).  It uses energy-based voice activity detection
    to segment speech, then transcribes each segment.

    The ``initial_prompt`` parameter biases recognition toward domain-specific
    vocabulary (e.g. model names from the database), dramatically improving
    proper-noun accuracy compared to Vosk.

    Signals:
        partial_result: Emitted during speech with partial recognition text.
        final_result: Emitted after silence with final recognition text.
        error_occurred: Emitted when an error occurs with error message.
    """

    partial_result = Signal(str)
    final_result = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        model_size: str = "small.en",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        initial_prompt: str = "",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
        energy_threshold: float = 0.01,
        silence_duration: float = 0.8,
        max_recording_duration: float = 15.0,
        parent: Optional[QObject] = None,
    ) -> None:
        """Initialize the Whisper speech recognizer.

        Args:
            model_size: Whisper model name (e.g. "tiny.en", "base.en",
                "small.en", "medium.en").  Downloaded automatically on first
                use.
            device_index: Audio input device index (None for default).
            sample_rate: Audio sample rate in Hz (default: 16000).
            initial_prompt: Domain vocabulary text to bias recognition
                (e.g. "Winter Guard, Khador, Cygnar, Stormblade").
            device: "cuda", "cpu", or "auto".
            compute_type: "float16", "int8", "int8_float16", or "auto".
            language: Language code (default: "en").
            energy_threshold: RMS energy threshold for voice activity
                detection.  Audio below this is considered silence.
            silence_duration: Seconds of silence to end an utterance.
            max_recording_duration: Maximum seconds for a single utterance
                before forcing transcription.
            parent: Parent QObject for Qt ownership.

        Raises:
            ImportError: If faster-whisper is not installed.
            Exception: If the model cannot be loaded.
        """
        super().__init__(parent)

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            ) from e

        self._device_index: Optional[int] = device_index
        self._sample_rate: int = sample_rate
        self._initial_prompt: str = initial_prompt
        self._language: str = language
        self._energy_threshold: float = energy_threshold
        self._silence_duration: float = silence_duration
        self._max_recording_duration: float = max_recording_duration

        self._model: Optional[WhisperModel] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._is_running: bool = False
        self._recognition_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

        # Partial transcription state — runs in a separate thread so it
        # never blocks the main recognition loop's silence detection.
        self._partial_thread: Optional[threading.Thread] = None
        self._partial_busy: bool = False
        self._partial_interval: float = 1.5  # seconds between partials

        # Load the Whisper model
        try:
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        except Exception as e:
            raise Exception(f"Failed to load Whisper model '{model_size}': {e}")

    def start(self) -> None:
        """Start speech recognition.

        Raises:
            RuntimeError: If recognition is already running.
        """
        if self._is_running:
            raise RuntimeError("Speech recognition is already running")

        self._is_running = True

        # Start recognition thread
        self._recognition_thread = threading.Thread(
            target=self._recognition_loop,
            daemon=True,
        )
        self._recognition_thread.start()

        # Start audio stream
        try:
            self._stream = sd.InputStream(
                device=self._device_index,
                samplerate=self._sample_rate,
                channels=1,
                dtype='int16',
                callback=self._audio_callback,
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

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._recognition_thread is not None:
            self._recognition_thread.join(timeout=3.0)
            self._recognition_thread = None

        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def is_running(self) -> bool:
        """Check if speech recognition is currently running."""
        return self._is_running

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: any,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for audio input stream."""
        if status:
            self.error_occurred.emit(f"Audio stream status: {status}")
        self._audio_queue.put(bytes(indata))

    def _recognition_loop(self) -> None:
        """Main recognition loop: accumulate audio, detect silence, transcribe.

        This loop only handles audio accumulation and silence detection.
        Partial transcriptions are dispatched to a separate thread so they
        never block silence detection (which would delay final results).
        """
        silence_frames_needed = int(self._sample_rate * self._silence_duration)
        max_frames = int(self._sample_rate * self._max_recording_duration)
        # Minimum bytes before we attempt a partial (1 second of audio)
        min_partial_bytes = self._sample_rate * 2  # 16-bit mono
        # Sliding window for partials: only transcribe the last N seconds
        partial_window_bytes = int(self._sample_rate * 5) * 2  # last 5s

        audio_buffer = bytearray()
        silence_frames = 0
        is_recording = False
        total_frames = 0
        last_partial_time = 0.0

        while self._is_running:
            try:
                data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._is_running:
                break

            audio_buffer.extend(data)
            chunk_samples = len(data) // 2  # 16-bit = 2 bytes per sample
            total_frames += chunk_samples

            # Compute RMS energy of this chunk
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples ** 2))) / 32768.0 if len(samples) > 0 else 0.0

            if rms >= self._energy_threshold:
                if not is_recording:
                    is_recording = True
                    # Immediate feedback that we're listening
                    self.partial_result.emit("...")
                silence_frames = 0
            elif is_recording:
                silence_frames += chunk_samples

            # Dispatch partial transcription asynchronously
            if is_recording and len(audio_buffer) >= min_partial_bytes:
                now = time.monotonic()
                if now - last_partial_time >= self._partial_interval and not self._partial_busy:
                    last_partial_time = now
                    # Only send the last N seconds to keep latency bounded
                    partial_data = bytes(audio_buffer[-partial_window_bytes:])
                    self._partial_thread = threading.Thread(
                        target=self._emit_partial,
                        args=(partial_data,),
                        daemon=True,
                    )
                    self._partial_thread.start()

            # End of utterance: silence detected or max duration reached
            if is_recording and (
                silence_frames >= silence_frames_needed
                or total_frames >= max_frames
            ):
                self._emit_final(audio_buffer)
                audio_buffer = bytearray()
                silence_frames = 0
                is_recording = False
                total_frames = 0

    def _emit_partial(self, audio_bytes: bytes) -> None:
        """Transcribe audio bytes and emit a partial result.

        Runs in a separate thread so it does not block silence detection.
        Uses a sliding window (last N seconds) to keep latency bounded.
        """
        self._partial_busy = True
        try:
            if self._model is None or len(audio_bytes) < self._sample_rate * 2:
                return  # Need at least 1 second of audio
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            kwargs = {"language": self._language, "beam_size": 1, "best_of": 1}
            if self._initial_prompt:
                kwargs["initial_prompt"] = self._initial_prompt
            segments, _ = self._model.transcribe(audio, **kwargs)
            text = " ".join(s.text.strip() for s in segments).strip()
            if text:
                self.partial_result.emit(text)
        except Exception:
            pass  # Partial failures are non-critical
        finally:
            self._partial_busy = False

    def _emit_final(self, buffer: bytearray) -> None:
        """Transcribe the full buffer and emit the final result."""
        if self._model is None or len(buffer) < self._sample_rate * 2:
            return  # Need at least 1 second of audio
        try:
            audio = np.frombuffer(bytes(buffer), dtype=np.int16).astype(np.float32) / 32768.0
            kwargs = {"language": self._language, "beam_size": 1}
            if self._initial_prompt:
                kwargs["initial_prompt"] = self._initial_prompt
            segments, _ = self._model.transcribe(audio, **kwargs)
            text = " ".join(s.text.strip() for s in segments).strip()
            if text:
                # Strip common filler-only results
                words = set(text.lower().split())
                if words and words.issubset(_FILLER_WORDS):
                    return
                # Strip trailing punctuation that Whisper often appends.
                text = text.rstrip(".!?\"'")
                if text:
                    self.final_result.emit(text)
        except Exception as e:
            if self._is_running:
                self.error_occurred.emit(f"Transcription error: {e}")

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.stop()
