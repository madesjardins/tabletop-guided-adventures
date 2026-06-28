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

"""Tests for ttga.help_agent.HelpAgent (Step 8).

Covers:
  - Wake-phrase detection (inline + conversational modes)
  - Non-help utterances pass through (returns False)
  - Context assembly from game.get_help_context() and menu fallback
  - LLM streaming answer path
  - Fallback (no LLM) scripted answer path
  - Listening state lifecycle

Run with:
    uv run python tests/test_help_agent.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6 import QtCore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ttga.help_agent import HelpAgent  # noqa: E402
from ttga.llm_client import LLMClient, LLMConfig  # noqa: E402

_app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication(sys.argv)


class FakeNarrator:
    """Records synthesize_and_play calls."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synthesize_and_play(self, text, channel=None, *a, **k) -> None:
        self.spoken.append(text)


class FakeBackend:
    """LLMBackend stub that returns a canned streaming response."""

    def __init__(self, chunks=None, loaded=True) -> None:
        self._chunks = chunks or ["This is ", "a helpful answer. ", "You're welcome."]
        self._loaded = loaded

    def runtime_available(self) -> bool:
        return True

    def list_models(self, models_dir):
        return []

    def load(self, model_path, *, n_gpu_layers, n_ctx) -> None:
        pass

    def is_loaded(self) -> bool:
        return self._loaded

    def chat(self, messages, *, stream, temperature, max_tokens):
        if not stream:
            return "".join(self._chunks)
        for chunk in self._chunks:
            yield chunk

    def unload(self) -> None:
        self._loaded = False


class FakeGame:
    """Minimal game stub with get_help_context()."""

    def __init__(self, context=None) -> None:
        self._context = context or {
            "state": "setup_game_mode",
            "summary": "The player is choosing a game mode.",
            "topics": {
                "game_mode": "Say 'single match' to choose a single-game format.",
                "cancel": "Say 'cancel' to abort the setup.",
            },
        }

    def get_help_context(self) -> dict:
        return self._context


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_wake_phrase_detection() -> bool:
    print("Wake-phrase detection:")
    ok = True
    client = LLMClient(LLMConfig(enabled=False))
    agent = HelpAgent(client, narrator=None)

    # Non-help utterance → returns False (pass through)
    ok &= _check(
        "non-help returns False",
        agent.handle_utterance("single match", None) is False,
    )

    # "Help me" alone → returns True, enters listening state
    ok &= _check(
        "'Help me' returns True",
        agent.handle_utterance("Help me", None) is True,
    )
    ok &= _check("agent is listening", agent.is_listening is True)

    # Next utterance is treated as the question
    agent.cancel()  # don't actually submit to LLM in this test
    ok &= _check("cancel exits listening", agent.is_listening is False)

    agent.shutdown()
    return ok


def test_inline_question() -> bool:
    print("\nInline question ('Help me, how do I...'):")
    ok = True
    narrator = FakeNarrator()
    backend = FakeBackend()
    client = LLMClient(LLMConfig(enabled=True), backend=backend)
    agent = HelpAgent(client, narrator)

    answers: list[str] = []
    agent.answered.connect(answers.append)

    game = FakeGame()
    consumed = agent.handle_utterance("Help me, how do I choose a game mode?", game)

    ok &= _check("inline returns True", consumed is True)
    ok &= _check("not listening (inline mode)", agent.is_listening is False)

    got = _pump_until(lambda: len(answers) > 0)
    ok &= _check("answer emitted", got and len(answers) == 1)
    ok &= _check("answer is full text", answers and "helpful" in answers[0].lower())
    ok &= _check("narrator spoke sentences", len(narrator.spoken) > 0)

    agent.shutdown()
    return ok


def test_conversational_mode() -> bool:
    print("\nConversational mode ('Help me' -> 'What can I help you with?'):")
    ok = True
    narrator = FakeNarrator()
    backend = FakeBackend()
    client = LLMClient(LLMConfig(enabled=True), backend=backend)
    agent = HelpAgent(client, narrator)

    started: list[bool] = []
    agent.help_started.connect(lambda: started.append(True))

    # Step 1: "Help me" → prompt
    consumed = agent.handle_utterance("Help me", None)
    ok &= _check("wake returns True", consumed is True)
    ok &= _check("agent is listening", agent.is_listening is True)
    ok &= _check("help_started emitted", len(started) == 1)
    ok &= _check(
        "narrator spoke prompt",
        len(narrator.spoken) == 1 and "what can i help" in narrator.spoken[0].lower(),
    )

    # Step 2: next utterance is the question
    answers: list[str] = []
    agent.answered.connect(answers.append)
    consumed = agent.handle_utterance("How do I add a model?", FakeGame())
    ok &= _check("question returns True", consumed is True)
    ok &= _check("no longer listening", agent.is_listening is False)

    got = _pump_until(lambda: len(answers) > 0)
    ok &= _check("answer emitted", got and len(answers) == 1)

    agent.shutdown()
    return ok


def test_fallback_no_llm() -> bool:
    print("\nFallback (LLM unavailable):")
    ok = True
    narrator = FakeNarrator()
    client = LLMClient(LLMConfig(enabled=False))
    agent = HelpAgent(client, narrator)

    answers: list[str] = []
    agent.answered.connect(answers.append)

    game = FakeGame()
    consumed = agent.handle_utterance("Help me, what can I say?", game)

    ok &= _check("returns True", consumed is True)
    got = _pump_until(lambda: len(answers) > 0)
    ok &= _check("answer emitted", got)
    ok &= _check(
        "answer contains summary",
        answers and "game mode" in answers[0].lower(),
    )
    ok &= _check(
        "answer contains topics",
        answers and "single match" in answers[0].lower(),
    )
    ok &= _check("narrator spoke", len(narrator.spoken) > 0)

    agent.shutdown()
    return ok


def test_context_assembly() -> bool:
    print("\nContext assembly:")
    ok = True
    client = LLMClient(LLMConfig(enabled=False))
    agent = HelpAgent(client, narrator=None)

    # With game → uses game context
    game = FakeGame(context={"state": "army_creation", "summary": "Building army."})
    ctx = agent._assemble_context(game)
    ok &= _check("game context has state", '"army_creation"' in ctx)
    ok &= _check("game context has summary", '"Building army."' in ctx)

    # Without game → menu fallback
    ctx_menu = agent._assemble_context(None)
    ok &= _check("menu context has state", '"menu"' in ctx_menu)
    ok &= _check("menu context has topics", '"load_game"' in ctx_menu)

    agent.shutdown()
    return ok


def test_help_prefix_variations() -> bool:
    print("\nWake-phrase prefix variations:")
    ok = True
    client = LLMClient(LLMConfig(enabled=False))
    agent = HelpAgent(client, narrator=None)

    # "Help, ..." (comma prefix)
    q = HelpAgent._extract_question("help, how do i play", "Help, how do I play")
    ok &= _check("'help,' prefix extracts question", q == "how do I play")

    # "Help me, ..." (me + comma)
    q = HelpAgent._extract_question("help me, what now", "Help me, what now")
    ok &= _check("'help me,' extracts question", q == "what now")

    # "Help me" alone → empty question (conversational)
    q = HelpAgent._extract_question("help me", "Help me")
    ok &= _check("'help me' alone -> empty", q == "")

    # Non-help → None
    q = HelpAgent._extract_question("single match", "single match")
    ok &= _check("non-help -> None", q is None)

    agent.shutdown()
    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("HelpAgent tests")
    print("=" * 60)
    ok = True
    ok &= test_wake_phrase_detection()
    ok &= test_inline_question()
    ok &= test_conversational_mode()
    ok &= test_fallback_no_llm()
    ok &= test_context_assembly()
    ok &= test_help_prefix_variations()
    print("=" * 60)
    if ok:
        print("All tests passed.")
        return 0
    print("Some tests FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
