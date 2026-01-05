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

"""Integration test for Piper TTS with SoundMixer.

This script tests the Piper TTS engine by synthesizing speech and playing
it through the SoundMixer audio system.
"""

import os
import random
import sys
import time
import wave
from pathlib import Path

root_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_path = os.path.join(root_dir_path, "python")
if python_path not in sys.path:
    sys.path.append(python_path)

from piper import PiperVoice  # noqa: E402
from ttga.sound_mixer import SoundMixer, Channel  # noqa: E402

PIPER_VOICES_DIR = os.path.join(root_dir_path, "piper_voices")
OUTPUT_DIR = os.path.join(root_dir_path, "temp")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test_piper.wav")
TEST_TEXT = "If you hear me, it means piper text to speech module works."


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


def synthesize_speech(text: str, model_path: str, output_path: str) -> None:
    """Synthesize speech using Piper TTS.

    Args:
        text: The text to synthesize.
        model_path: Path to the Piper voice model (.onnx file).
        output_path: Path where the output WAV file will be saved.
    """
    print(f"Loading Piper voice model from: {model_path}")
    voice = PiperVoice.load(model_path)

    print(f"Synthesizing text: '{text}'")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with wave.open(output_path, 'wb') as wav_file:
        voice.synthesize_wav(text, wav_file)

    print(f"Audio saved to: {output_path}")


def main() -> None:
    """Main test function."""
    print("=" * 60)
    print("Piper TTS Integration Test")
    print("=" * 60)

    try:
        print(f"\nScanning for available voices in: {PIPER_VOICES_DIR}")
        available_voices = find_available_voices(PIPER_VOICES_DIR)

        if not available_voices:
            print(f"\nERROR: No voice files are available in the folder: {PIPER_VOICES_DIR}")
            print("Please ensure you have both .onnx and .onnx.json files for at least one voice model.")
            return

        print(f"Found {len(available_voices)} available voice(s):")
        for voice in available_voices:
            print(f"  - {Path(voice).name}")

        selected_voice = random.choice(available_voices)
        print(f"\nRandomly selected voice: {Path(selected_voice).name}")

        synthesize_speech(TEST_TEXT, selected_voice, OUTPUT_FILE)

        print("\nInitializing SoundMixer...")
        mixer = SoundMixer()

        print("Playing synthesized speech...")
        mixer.play(
            sound_file_path=OUTPUT_FILE,
            channel=Channel.VOICE,
            do_play_immediately=True
        )

        while mixer.is_channel_busy(Channel.VOICE):
            time.sleep(0.1)

        print("\nPlayback complete!")
        print("Shutting down mixer...")
        mixer.shutdown()

        print("=" * 60)
        print("Test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during test: {e}")
        raise


if __name__ == "__main__":
    main()
