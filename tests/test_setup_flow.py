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

"""Tests for the Warmachine SetupFlow state machine (Step 4).

Covers the deterministic (LLM-off) conversational flow, edge cases (invalid
points, restart, cancel), the number-word parser, and the LLM-on path where
parse_intent extracts a value deterministic parsing could not.

Run with:
    uv run python tests/test_setup_flow.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6 import QtCore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "games" / "ttga-warmachine" / "python"))

from setup_flow import SetupFlow, SetupState, words_to_int  # noqa: E402
from ttga.narration_engine import NarrationEngine, Intent  # noqa: E402
from ttga.narration_service import NarrationService  # noqa: E402

# Ensure a QCoreApplication exists for QObject signal machinery.
_app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication(sys.argv)


class FakeEventManager:
    """Minimal speech-handler stack for driving the flow."""

    def __init__(self) -> None:
        self._handlers = []

    def push_speech_handler(self, handler) -> None:
        if handler not in self._handlers:
            self._handlers.append(handler)

    def pop_speech_handler(self, handler) -> None:
        if handler in self._handlers:
            self._handlers.remove(handler)

    def route(self, text: str) -> None:
        if self._handlers:
            self._handlers[-1](text)

    @property
    def has_handler(self) -> bool:
        return bool(self._handlers)


class FakeLog:
    """Captures log calls without touching disk."""

    def __init__(self) -> None:
        self.narrations = []
        self.player_lines = []
        self.system_lines = []

    def narrate(self, text: str) -> None:
        self.narrations.append(text)

    def player_said(self, who: str, text: str) -> None:
        self.player_lines.append((who, text))

    def system(self, text: str) -> None:
        self.system_lines.append(text)


class FixedClient:
    """LLMClient stub returning a fixed JSON reply for every generate call."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, *, system=None, temperature=None, max_tokens=None, **_):
        return self._reply


def _make_flow(narration_engine=None):
    em = FakeEventManager()
    log = FakeLog()
    flow = SetupFlow(event_manager=em, game_log=log, narration_engine=narration_engine)
    completed = {}
    cancelled = {"called": False}
    flow.setup_complete.connect(lambda cfg: completed.update(cfg))
    flow.setup_cancelled.connect(lambda: cancelled.update(called=True))
    return flow, em, log, completed, cancelled


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return condition


def test_words_to_int() -> bool:
    print("words_to_int:")
    ok = True
    ok &= _check("digits", words_to_int("50 points") == 50)
    ok &= _check("simple word", words_to_int("fifty") == 50)
    ok &= _check("tens+units", words_to_int("seventy five") == 75)
    ok &= _check("hyphenated", words_to_int("seventy-five") == 75)
    ok &= _check("teen", words_to_int("fifteen") == 15)
    ok &= _check("none", words_to_int("no number here") is None)
    ok &= _check("digits win", words_to_int("about 35 or forty") == 35)
    return ok


def test_deterministic_flow() -> bool:
    print("\nDeterministic flow (LLM off):")
    ok = True
    flow, em, log, completed, cancelled = _make_flow()

    flow.start()
    ok &= _check("starts in GAME_MODE", flow.state == SetupState.GAME_MODE)
    ok &= _check("handler pushed", em.has_handler)
    ok &= _check("asked a question", len(log.narrations) == 1)

    em.route("let's play a single match")
    ok &= _check("advanced to POINTS", flow.state == SetupState.POINTS)
    ok &= _check("game_mode set", flow.config["game_mode"] == "single_match")

    em.route("seventy five points")
    ok &= _check("advanced to CONFIRM", flow.state == SetupState.CONFIRM)
    ok &= _check("points set", flow.config["points"] == 75)

    em.route("yes")
    ok &= _check("reached DONE", flow.state == SetupState.DONE)
    ok &= _check("setup_complete emitted", completed == {"game_mode": "single_match", "points": 75})
    ok &= _check("handler popped", not em.has_handler)
    return ok


def test_invalid_points_reprompt() -> bool:
    print("\nInvalid points then valid:")
    ok = True
    flow, em, log, completed, _ = _make_flow()
    flow.start()
    em.route("single match")
    em.route("uhh I dunno")  # no number
    ok &= _check("stays in POINTS", flow.state == SetupState.POINTS)
    ok &= _check("points still unset", flow.config["points"] is None)
    em.route("fifty")
    ok &= _check("accepts valid points", flow.config["points"] == 50)
    return ok


def test_restart_and_cancel() -> bool:
    print("\nRestart and cancel:")
    ok = True
    # Restart at confirm goes back to GAME_MODE and clears state.
    flow, em, log, completed, _ = _make_flow()
    flow.start()
    em.route("single match")
    em.route("fifty")
    em.route("restart")
    ok &= _check("restart -> GAME_MODE", flow.state == SetupState.GAME_MODE)
    ok &= _check("restart clears points", flow.config["points"] is None)

    # Cancel ends the flow with the cancelled signal.
    flow2, em2, log2, completed2, cancelled2 = _make_flow()
    flow2.start()
    em2.route("cancel")
    ok &= _check("cancel -> DONE", flow2.state == SetupState.DONE)
    ok &= _check("cancelled emitted", cancelled2["called"])
    ok &= _check("no completion", completed2 == {})
    ok &= _check("handler popped on cancel", not em2.has_handler)
    return ok


def test_llm_path() -> bool:
    print("\nLLM-on path (parse_intent extracts where deterministic fails):")
    ok = True
    # Client always returns set_points=35. For the mode/confirm steps that
    # intent is not allowed, so it is ignored and deterministic parsing runs.
    client = FixedClient('{"intent": "set_points", "value": "35", "confidence": 0.95}')
    engine = NarrationEngine(llm_client=client)
    flow, em, log, completed, _ = _make_flow(narration_engine=engine)

    flow.start()
    em.route("single match")  # mode resolved deterministically
    ok &= _check("mode set (LLM intent ignored)", flow.config["game_mode"] == "single_match")

    # Deterministic parser finds no number here; the LLM supplies 35.
    em.route("make it a small skirmish")
    ok &= _check("LLM-extracted points", flow.config["points"] == 35)
    ok &= _check("advanced to CONFIRM", flow.state == SetupState.CONFIRM)

    em.route("yes")  # confirm resolved deterministically
    ok &= _check("completed via LLM points", completed == {"game_mode": "single_match", "points": 35})
    return ok


class _UnknownEngine:
    """Engine stub for the service: always returns unknown so the flow's
    deterministic parsing drives it, exercising the async plumbing."""

    def phrase_stream(self, scripted, *, situation=None):
        yield scripted

    def parse_intent(self, utterance, allowed, *, context=None, confidence_threshold=0.0):
        return Intent()


def _pump_until(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        _app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_service_async_flow() -> bool:
    print("\nAsync flow driven through NarrationService:")
    ok = True
    em = FakeEventManager()
    log = FakeLog()
    service = NarrationService(_UnknownEngine(), narrator=None)
    flow = SetupFlow(event_manager=em, game_log=log, narration_service=service)
    completed = {}
    flow.setup_complete.connect(lambda cfg: completed.update(cfg))

    flow.start()
    ok &= _check("starts in GAME_MODE", flow.state == SetupState.GAME_MODE)

    # Each route triggers an async parse; pump until the state advances.
    em.route("single match")
    ok &= _check("async advanced to POINTS", _pump_until(lambda: flow.state == SetupState.POINTS))

    em.route("seventy five")
    ok &= _check("async advanced to CONFIRM", _pump_until(lambda: flow.state == SetupState.CONFIRM))

    em.route("yes")
    ok &= _check("async completed", _pump_until(lambda: completed))
    ok &= _check("async config correct", completed == {"game_mode": "single_match", "points": 75})

    service.shutdown()
    return ok


def main() -> int:
    print("=" * 60)
    print("SetupFlow tests")
    print("=" * 60)
    ok = True
    ok &= test_words_to_int()
    ok &= test_deterministic_flow()
    ok &= test_invalid_points_reprompt()
    ok &= test_restart_and_cancel()
    ok &= test_llm_path()
    ok &= test_service_async_flow()
    print("=" * 60)
    if ok:
        print("All tests passed.")
        return 0
    print("Some tests FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
