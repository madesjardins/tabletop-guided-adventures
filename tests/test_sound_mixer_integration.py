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

"""Integration test for SoundMixer audio system.

This script tests the SoundMixer audio system by playing various sound files
through different audio channels to ensure proper functionality.
"""

import time
import sys
import os

root_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_path = os.path.join(root_dir_path, "python")
if python_path not in sys.path:
    sys.path.append(python_path)

from ttga.sound_mixer import SoundMixer, Channel  # noqa: E402

MUSIC_SOUND_FILE_PATH = r"C:\Windows\Media\Ring05.wav"
VOICE_SOUND_FILE_PATH_1 = r"C:\Windows\Media\Ring07.wav"
VOICE_SOUND_FILE_PATH_2 = r"C:\Windows\Media\tada.wav"
EFFECT_SOUND_FILE_PATH = r"C:\Windows\Media\Windows Critical Stop.wav"


def main():
    mixer = SoundMixer()

    print("Testing SoundMixer initialization...")

    print("Playing music ...")
    mixer.play(
        sound_file_path=MUSIC_SOUND_FILE_PATH,
        channel=Channel.MUSIC,
        do_play_immediately=False
    )

    time.sleep(4)

    print("Playing voice 1...")
    mixer.play(
        sound_file_path=VOICE_SOUND_FILE_PATH_1,
        channel=Channel.VOICE,
        do_play_immediately=True
    )
    time.sleep(1)

    print("Playing effect...")
    mixer.play(
        sound_file_path=EFFECT_SOUND_FILE_PATH,
        channel=Channel.EFFECT,
        do_play_immediately=True
    )
    time.sleep(0.25)

    print("Playing voice 2 immediately...")
    mixer.play(
        sound_file_path=VOICE_SOUND_FILE_PATH_2,
        channel=Channel.VOICE,
        do_play_immediately=True
    )

    print("Waiting for music to finish...")
    while mixer.is_channel_busy(Channel.MUSIC):
        time.sleep(0.1)

    print("Done - shutting down...")
    mixer.shutdown()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()
