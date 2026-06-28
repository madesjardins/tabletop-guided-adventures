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

"""Tests for ttga.narration_service.NarrationService (Step 5).

Two suites:
  1. Direct worker-logic tests (call the worker methods synchronously and
     capture signals on the same thread).
  2. A threaded end-to-end test that submits real background work and pumps the
     Qt event loop to confirm results marshal back to the main thread.

Run with:
    uv run python tests/test_narration_service.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6 import QtCore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ttga.narration_service import NarrationService  # noqa: E402
from ttga.narration_engine import Intent  # noqa: E402

_app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication(sys.argv)


class FakeEngine:
    """NarrationEngine-compatible stub."""

    def __init__(self, sentences=None, intent=None) -> None:
        # sentences=None => "inactive": stream yields the scripted line.
        self._sentences = sentences
        self._intent = intent or Intent(intent="add_model", value="X", confidence=0.9)

    def phrase_stream(self, scripted, *, situation=None):
        if self._sentences is None:
            yield scripted
        else:
            for s in self._sentences:
                yield s

    def parse_intent(self, utterance, allowed, *, context=None, confidence_threshold=0.0):
        return self._intent


class FakeNarrator:
    """Records synthesize_and_play calls."""

    def __init__(self) -> None:
        self.spoken = []

    def synthesize_and_play(self, text, channel=None, *a, **k) -> None:
        self.spoken.append(text)


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return condition


def _pump_until(predicate, timeout_s: float = 5.0) -> bool:
    """Process Qt events until predicate() is true or timeout elapses."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        _app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_speak_logic() -> bool:
    print("Speak (worker logic):")
    ok = True
    narrator = FakeNarrator()
    engine = FakeEngine(sentences=["The siege begins.", "Steel meets steel!"])
    svc = NarrationService(engine, narrator)

    sentences, narrated = [], []
    svc.sentence_spoken.connect(sentences.append)
    svc.narrated.connect(narrated.append)

    svc._do_speak("scripted fallback", None, use_persona=True)

    ok &= _check("each sentence emitted", sentences == ["The siege begins.", "Steel meets steel!"])
    ok &= _check("each sentence synthesized", narrator.spoken == sentences)
    ok &= _check("narrated is full text", narrated == ["The siege begins. Steel meets steel!"])
    return ok


def test_speak_fallback_no_narrator() -> bool:
    print("\nSpeak (inactive engine, no narrator):")
    ok = True
    engine = FakeEngine(sentences=None)  # inactive => yields scripted
    svc = NarrationService(engine, narrator=None)
    narrated = []
    svc.narrated.connect(narrated.append)
    svc._do_speak("just the script", None)
    ok &= _check("narrated equals scripted", narrated == ["just the script"])
    return ok


def test_parse_logic() -> bool:
    print("\nparse_intent (worker logic):")
    ok = True
    engine = FakeEngine(intent=Intent(intent="army_completed", confidence=0.8))
    svc = NarrationService(engine)
    results = []
    svc.intent_parsed.connect(lambda rid, intent: results.append((rid, intent)))
    svc._do_parse(7, "that's all", {"army_completed": "done"}, None, 0.0)
    ok &= _check("request id preserved", results and results[0][0] == 7)
    ok &= _check("intent delivered", results and results[0][1].intent == "army_completed")
    return ok


def test_threaded_end_to_end() -> bool:
    print("\nThreaded end-to-end (event-loop marshaling):")
    ok = True
    narrator = FakeNarrator()
    engine = FakeEngine(sentences=["Alpha.", "Beta."],
                        intent=Intent(intent="add_model", value="Juggernaut", confidence=0.9))
    svc = NarrationService(engine, narrator)

    narrated = []
    intents = []
    svc.narrated.connect(narrated.append)
    svc.intent_parsed.connect(lambda rid, intent: intents.append((rid, intent)))

    svc.speak("scripted", use_persona=True)
    req_id = svc.parse_intent_async("add the juggernaut", {"add_model": "add"})

    got = _pump_until(lambda: narrated and intents)
    ok &= _check("speak result marshaled", bool(narrated) and narrated[0] == "Alpha. Beta.")
    ok &= _check("sentences synthesized in bg", narrator.spoken == ["Alpha.", "Beta."])
    ok &= _check("intent result marshaled", bool(intents) and intents[0][0] == req_id)
    ok &= _check("intent value correct", bool(intents) and intents[0][1].value == "Juggernaut")
    ok &= _check("pump completed", got)

    svc.shutdown()
    return ok


def main() -> int:
    print("=" * 60)
    print("NarrationService tests")
    print("=" * 60)
    ok = True
    ok &= test_speak_logic()
    ok &= test_speak_fallback_no_narrator()
    ok &= test_parse_logic()
    ok &= test_threaded_end_to_end()
    print("=" * 60)
    if ok:
        print("All tests passed.")
        return 0
    print("Some tests FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
