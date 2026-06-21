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

"""Tests for the Warmachine NarrationEngine (NLG, Step 2).

Verifies the LLM-optional contract:
  - No client / disabled / unavailable / erroring => scripted fallback verbatim.
  - Active client => in-character rephrasing returned.

Run with:
    uv run python tests/test_narration_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "games" / "ttga-warmachine" / "python"))

from ttga.narration_engine import NarrationEngine, DEFAULT_PERSONA  # noqa: E402
from persona import WARMACHINE_PERSONA  # noqa: E402


class FakeClient:
    """Minimal LLMClient-compatible stub for testing."""

    def __init__(self, available: bool = True, reply: str = "A grim reply.",
                 raise_on_generate: bool = False) -> None:
        self._available = available
        self._reply = reply
        self._raise = raise_on_generate
        self.last_system = None
        self.last_prompt = None
        self.last_kwargs = None

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt, *, system=None, temperature=None, max_tokens=None, **_):
        if self._raise:
            raise RuntimeError("boom")
        self.last_prompt = prompt
        self.last_system = system
        self.last_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        return self._reply


SCRIPTED = "Player 1, speak the name of the next model or unit."


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return condition


def run() -> int:
    print("=" * 60)
    print("NarrationEngine tests")
    print("=" * 60)
    ok = True

    # No client => fallback verbatim, never active.
    engine = NarrationEngine(llm_client=None)
    ok &= _check("no client: not active", not engine.is_active())
    ok &= _check("no client: returns scripted", engine.phrase(SCRIPTED) == SCRIPTED)

    # Disabled => fallback even with a working client.
    disabled = NarrationEngine(llm_client=FakeClient(), enabled=False)
    ok &= _check("disabled: not active", not disabled.is_active())
    ok &= _check("disabled: returns scripted", disabled.phrase(SCRIPTED) == SCRIPTED)

    # Client present but unavailable => fallback.
    unavail = NarrationEngine(llm_client=FakeClient(available=False))
    ok &= _check("unavailable: not active", not unavail.is_active())
    ok &= _check("unavailable: returns scripted", unavail.phrase(SCRIPTED) == SCRIPTED)

    # Active client => rephrased text returned.
    client = FakeClient(reply="Player One, name the next warrior to the muster.")
    active = NarrationEngine(llm_client=client)
    ok &= _check("active: is_active True", active.is_active())
    result = active.phrase(SCRIPTED)
    ok &= _check("active: returns LLM reply", result == "Player One, name the next warrior to the muster.")
    ok &= _check("active: persona used as system", client.last_system == DEFAULT_PERSONA)
    ok &= _check("active: scripted included in prompt", SCRIPTED in client.last_prompt)

    # Empty LLM reply => fallback to scripted.
    empty = NarrationEngine(llm_client=FakeClient(reply="   "))
    ok &= _check("empty reply: returns scripted", empty.phrase(SCRIPTED) == SCRIPTED)

    # Generation error => fallback to scripted (never raises).
    erroring = NarrationEngine(llm_client=FakeClient(raise_on_generate=True))
    ok &= _check("error: returns scripted", erroring.phrase(SCRIPTED) == SCRIPTED)

    # Explicit situation overrides the prompt body.
    sit_client = FakeClient(reply="Drama.")
    sit_engine = NarrationEngine(llm_client=sit_client)
    sit_engine.phrase(SCRIPTED, situation="ask for next model")
    ok &= _check("situation used in prompt", "ask for next model" in sit_client.last_prompt)

    # Game-supplied persona is used as the system prompt (core engine is generic).
    persona_client = FakeClient(reply="By blade and steam!")
    persona_engine = NarrationEngine(llm_client=persona_client, persona=WARMACHINE_PERSONA)
    persona_engine.phrase(SCRIPTED)
    ok &= _check("game persona used as system", persona_client.last_system == WARMACHINE_PERSONA)
    ok &= _check("game persona differs from default", WARMACHINE_PERSONA != DEFAULT_PERSONA)

    print("=" * 60)
    if ok:
        print("All tests passed.")
        return 0
    print("Some tests FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(run())
