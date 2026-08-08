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

from ttga.narration_engine import (  # noqa: E402
    NarrationEngine,
    DEFAULT_PERSONA,
    Intent,
    UNKNOWN_INTENT,
    _extract_json_object,
    split_sentences,
)
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

    def generate(self, prompt, *, system=None, temperature=None, max_tokens=None,
                 stream=False, **_):
        if self._raise:
            raise RuntimeError("boom")
        self.last_prompt = prompt
        self.last_system = system
        self.last_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if stream:
            # Emit the reply in small chunks to exercise incremental assembly.
            return iter([tok + " " for tok in self._reply.split(" ")])
        return self._reply


SCRIPTED = "Player 1, speak the name of the next model or unit."


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return condition


ALLOWED = {
    "add_model": "the player named a model to add (value = model name)",
    "army_completed": "the player is finished building their army",
}


def run_nlu_tests() -> bool:
    """parse_intent (NLU) tests using fake clients returning JSON."""
    print("\nNLU parse_intent tests:")
    ok = True

    # Inactive (no client) => unknown, never raises.
    inactive = NarrationEngine(llm_client=None)
    res = inactive.parse_intent("add the juggernaut", ALLOWED)
    ok &= _check("inactive: unknown", res.is_unknown and res.intent == UNKNOWN_INTENT)

    # Valid JSON with an allowed intent and value.
    c = FakeClient(reply='{"intent": "add_model", "value": "Juggernaut", "confidence": 0.95}')
    eng = NarrationEngine(llm_client=c)
    res = eng.parse_intent("I'd like to add the Juggernaut", ALLOWED)
    ok &= _check("valid: intent add_model", res.intent == "add_model")
    ok &= _check("valid: value extracted", res.value == "Juggernaut")
    ok &= _check("valid: confidence parsed", abs(res.confidence - 0.95) < 1e-9)
    ok &= _check("valid: allowed listed in prompt", "add_model" in c.last_prompt)
    ok &= _check("valid: utterance in prompt", "Juggernaut" in c.last_prompt)

    # army_completed with null value.
    c = FakeClient(reply='{"intent": "army_completed", "value": null, "confidence": 0.9}')
    res = NarrationEngine(llm_client=c).parse_intent("that's everything", ALLOWED)
    ok &= _check("army_completed parsed", res.intent == "army_completed" and res.value is None)

    # Markdown-fenced JSON is still extracted.
    c = FakeClient(reply='```json\n{"intent": "add_model", "value": "Kreoss", "confidence": 0.8}\n```')
    res = NarrationEngine(llm_client=c).parse_intent("Kreoss", ALLOWED)
    ok &= _check("fenced JSON extracted", res.intent == "add_model" and res.value == "Kreoss")

    # Disallowed intent name => unknown.
    c = FakeClient(reply='{"intent": "delete_everything", "value": null, "confidence": 1.0}')
    res = NarrationEngine(llm_client=c).parse_intent("nuke it", ALLOWED)
    ok &= _check("disallowed intent => unknown", res.is_unknown)

    # Below confidence threshold => unknown.
    c = FakeClient(reply='{"intent": "add_model", "value": "Maybe", "confidence": 0.2}')
    res = NarrationEngine(llm_client=c).parse_intent("uhh", ALLOWED, confidence_threshold=0.5)
    ok &= _check("low confidence => unknown", res.is_unknown)

    # Malformed output => unknown (never raises).
    c = FakeClient(reply="I think you want to add a model, friend.")
    res = NarrationEngine(llm_client=c).parse_intent("add something", ALLOWED)
    ok &= _check("malformed => unknown", res.is_unknown)

    # Generation error => unknown.
    c = FakeClient(raise_on_generate=True)
    res = NarrationEngine(llm_client=c).parse_intent("add", ALLOWED)
    ok &= _check("error => unknown", res.is_unknown)

    # Non-string value is coerced to str.
    c = FakeClient(reply='{"intent": "add_model", "value": 42, "confidence": 0.7}')
    res = NarrationEngine(llm_client=c).parse_intent("number", ALLOWED)
    ok &= _check("value coerced to str", res.value == "42")

    # Context is included in the prompt when provided.
    c = FakeClient(reply='{"intent": "add_model", "value": "X", "confidence": 0.6}')
    eng = NarrationEngine(llm_client=c)
    eng.parse_intent("x", ALLOWED, context={"player": "Player 1"})
    ok &= _check("context in prompt", "Player 1" in c.last_prompt)

    # _extract_json_object direct checks.
    ok &= _check("extract: plain", _extract_json_object('{"a": 1}') == {"a": 1})
    ok &= _check("extract: embedded", _extract_json_object('noise {"a": 1} tail') == {"a": 1})
    ok &= _check("extract: junk => None", _extract_json_object("no json here") is None)
    ok &= _check("extract: empty => None", _extract_json_object("") is None)

    return ok


def run_stream_tests() -> bool:
    """phrase_stream + split_sentences tests."""
    print("\nStreaming (phrase_stream / split_sentences):")
    ok = True

    # split_sentences: complete sentences + remainder.
    sents, rem = split_sentences("Hello there. How are you? I am")
    ok &= _check("split: two sentences", sents == ["Hello there.", "How are you?"])
    ok &= _check("split: remainder kept", rem.strip() == "I am")
    sents, rem = split_sentences("no terminator yet")
    ok &= _check("split: no sentence", sents == [] and rem == "no terminator yet")

    # Inactive engine streams the scripted line as a single item.
    inactive = NarrationEngine(llm_client=None)
    chunks = list(inactive.phrase_stream(SCRIPTED))
    ok &= _check("stream inactive: scripted only", chunks == [SCRIPTED])

    # Active engine streams sentence by sentence.
    c = FakeClient(reply="The siege begins. Steel meets steel!")
    eng = NarrationEngine(llm_client=c)
    chunks = list(eng.phrase_stream(SCRIPTED))
    ok &= _check("stream: sentence split", chunks == ["The siege begins.", "Steel meets steel!"])

    # Error before any output falls back to scripted.
    c = FakeClient(raise_on_generate=True)
    chunks = list(NarrationEngine(llm_client=c).phrase_stream(SCRIPTED))
    ok &= _check("stream error: scripted fallback", chunks == [SCRIPTED])

    return ok


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

    ok &= run_nlu_tests()
    ok &= run_stream_tests()

    print("=" * 60)
    if ok:
        print("All tests passed.")
        return 0
    print("Some tests FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(run())
