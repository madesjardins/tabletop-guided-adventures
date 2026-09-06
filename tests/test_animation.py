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

"""Unit tests for the FrameSequence animation utility."""

import sys
from pathlib import Path

import pytest

# Ensure the warmachine python package is importable.
_warmachine_py = Path(__file__).resolve().parent.parent / "games" / "ttga-warmachine" / "python"
if str(_warmachine_py) not in sys.path:
    sys.path.insert(0, str(_warmachine_py))

from animation import FrameSequence, seconds_to_frames, DEFAULT_FPS


class TestSecondsToFrames:
    def test_basic_conversion(self):
        assert seconds_to_frames(1.0, fps=30) == 30

    def test_half_second(self):
        assert seconds_to_frames(0.5, fps=30) == 15

    def test_minimum_one_frame(self):
        assert seconds_to_frames(0.0, fps=30) == 1
        assert seconds_to_frames(-1.0, fps=30) == 1

    def test_custom_fps(self):
        assert seconds_to_frames(1.0, fps=60) == 60
        assert seconds_to_frames(0.5, fps=60) == 30


class TestFrameSequenceConstruction:
    def test_simple_two_layer(self):
        seq = FrameSequence([10, 10])
        assert seq.total_frames == 20
        assert seq.num_layers == 2

    def test_single_layer(self):
        seq = FrameSequence([5])
        assert seq.total_frames == 5
        assert seq.num_layers == 1

    def test_uneven_layers(self):
        seq = FrameSequence([3, 7, 2])
        assert seq.total_frames == 12
        assert seq.num_layers == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            FrameSequence([])

    def test_zero_frame_count_raises(self):
        with pytest.raises(ValueError):
            FrameSequence([5, 0, 3])


class TestFrameSequenceLayerAt:
    def test_two_layer_boundaries(self):
        seq = FrameSequence([10, 10])
        # Layer 0: frames 0-9
        for f in range(10):
            assert seq.layer_at(f) == 0
        # Layer 1: frames 10-19
        for f in range(10, 20):
            assert seq.layer_at(f) == 1

    def test_modulo_wraparound(self):
        seq = FrameSequence([10, 10])
        assert seq.layer_at(0) == 0
        assert seq.layer_at(20) == 0  # wraps
        assert seq.layer_at(30) == 1  # wraps to layer 1
        assert seq.layer_at(40) == 0  # wraps again

    def test_three_layer_sequence(self):
        seq = FrameSequence([3, 7, 2])
        assert seq.layer_at(0) == 0
        assert seq.layer_at(2) == 0
        assert seq.layer_at(3) == 1
        assert seq.layer_at(9) == 1
        assert seq.layer_at(10) == 2
        assert seq.layer_at(11) == 2
        # Wrap
        assert seq.layer_at(12) == 0
        assert seq.layer_at(15) == 1

    def test_negative_frame_wraps(self):
        seq = FrameSequence([10, 10])
        # -1 % 20 == 19 -> layer 1
        assert seq.layer_at(-1) == 1
        # -10 % 20 == 10 -> layer 1
        assert seq.layer_at(-10) == 1
        # -20 % 20 == 0 -> layer 0
        assert seq.layer_at(-20) == 0

    def test_single_layer_always_zero(self):
        seq = FrameSequence([5])
        for f in range(20):
            assert seq.layer_at(f) == 0


class TestBlinkConstructor:
    def test_default_blink(self):
        seq = FrameSequence.blink()
        assert seq.num_layers == 2
        on_frames = seconds_to_frames(0.5, DEFAULT_FPS)
        off_frames = seconds_to_frames(0.5, DEFAULT_FPS)
        assert seq.total_frames == on_frames + off_frames

    def test_blink_layer_pattern(self):
        seq = FrameSequence.blink(0.5, 0.5, fps=30)
        # 15 on + 15 off = 30 total
        assert seq.total_frames == 30
        # On phase
        for f in range(15):
            assert seq.layer_at(f) == 0
        # Off phase
        for f in range(15, 30):
            assert seq.layer_at(f) == 1
        # Wraps
        assert seq.layer_at(30) == 0
        assert seq.layer_at(45) == 1

    def test_custom_blink_timing(self):
        seq = FrameSequence.blink(1.0, 0.25, fps=30)
        # 30 on + 8 off (0.25*30=7.5 -> 8) = 38 total
        assert seq.total_frames == 30 + 8
        assert seq.layer_at(0) == 0
        assert seq.layer_at(29) == 0
        assert seq.layer_at(30) == 1
        assert seq.layer_at(37) == 1
        assert seq.layer_at(38) == 0  # wraps
