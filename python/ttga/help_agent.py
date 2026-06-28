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

"""Voice-triggered help / Q&A agent (Step 8).

:class:`HelpAgent` is a core-level, read-only assistant that answers player
questions about the game and the application.  It works in **both** the main
menu (before a game starts) and in-game, because it is intercepted in
``MainCore._on_speech_final_result`` *before* normal game routing.

Trigger
-------
The player says ``"Help me"`` (or ``"Help me, <question>"``).  The agent
detects the wake phrase, enters a listening state, and either:

1. **Inline question** — ``"Help me, how do I add a model?"`` — the question
   after the wake phrase is answered immediately.
2. **Conversational** — ``"Help me"`` alone — the agent says *"What can I help
   you with?"* and treats the *next* utterance as the question.  After
   answering, it returns to idle.

Context
-------
The agent assembles a context string from:

- **Game state**: ``GameBase.get_help_context()`` returns a dict with the
  current phase, available options, and topic-specific guidance.  When no
  game is loaded, a minimal menu context is used.
- **Rules knowledge**: v1 uses only the game-supplied context (no RAG).  The
  system prompt instructs the LLM to answer *only* from the provided context
  and to suggest consulting the rulebook when it doesn't know.

LLM-optional
------------
When ``llm_client.is_available()`` is ``False``, the agent still triggers but
reads the context's ``summary`` and ``topics`` as scripted text via the
Narrator, providing a functional but less conversational fallback.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

from PySide6 import QtCore

from .llm_client import LLMClient
from .narration_engine import split_sentences
from .sound_mixer import Channel

if TYPE_CHECKING:
    from .game_base import GameBase
    from .narrator import Narrator

# Wake phrase prefixes (case-insensitive).  STT may produce slight variations.
_WAKE_PREFIXES = ("help me", "help,")

_HELP_SYSTEM_PROMPT = """\
You are a helpful tabletop gaming assistant. A player asked you a question.

Answer using ONLY the information in the CONTEXT block below. Be concise (1-3 \
sentences). Speak naturally — your answer will be read aloud by a TTS engine.

If the question is about game rules not covered in the context, say you're not \
sure and suggest checking the rulebook. Never invent rules, stats, or mechanics.

CONTEXT:
{context}"""

_MENU_CONTEXT = {
    "state": "menu",
    "summary": "The user is in the main menu. No game is running.",
    "topics": {
        "load_game": "Open the game's dialog and click Start Game after configuring zones.",
        "configure_speech": "Go to Settings > Speech Recognition to configure microphone and Vosk model.",
        "configure_narrator": "Go to Settings > Narrator to select a Piper TTS voice.",
        "load_llm": "Go to Settings > LLM Narrator to load a local model for dynamic narration.",
    },
}


class HelpAgent(QtCore.QObject):
    """Voice-triggered help / Q&A agent.

    Signals:
        help_started(): Emitted when the agent enters help mode (wake phrase
            detected or listening for a question).
        help_finished(): Emitted when the agent has finished answering and
            returned to idle.
        answered(str): Emitted with the full answer text once the LLM response
            is complete (suitable for logging / UI display).
    """

    help_started = QtCore.Signal()
    help_finished = QtCore.Signal()
    answered = QtCore.Signal(str)

    def __init__(
        self,
        llm_client: LLMClient,
        narrator: Optional[Narrator] = None,
        *,
        channel: Channel = Channel.VOICE,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        """Initialize the help agent.

        Args:
            llm_client: The shared :class:`LLMClient` (may be unavailable).
            narrator: Optional :class:`Narrator` for TTS playback.
            channel: Audio channel for playback.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._llm_client = llm_client
        self._narrator = narrator
        self._channel = channel
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="help-agent"
        )
        self._listening = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_listening(self) -> bool:
        """True when the agent is waiting for the player's question."""
        return self._listening

    def handle_utterance(self, text: str, game: Optional[GameBase] = None) -> bool:
        """Check if *text* is a help request and handle it.

        Called from ``MainCore._on_speech_final_result`` *before* normal game
        routing.  Returns ``True`` if the utterance was consumed (wake phrase
        detected or already in listening state), ``False`` to let it pass
        through to the game.

        Args:
            text: The recognised speech text.
            game: The currently loaded game (``None`` in main menu).

        Returns:
            ``True`` if consumed, ``False`` if not a help request.
        """
        lower = text.strip().lower()

        # Already listening → treat this utterance as the question.
        if self._listening:
            self._listening = False
            question = text.strip()
            self._submit_question(question, game)
            return True

        # Check for wake phrase.
        question = self._extract_question(lower, text.strip())
        if question is not None:
            if question:
                # Inline: "Help me, how do I..." → answer immediately.
                self._submit_question(question, game)
            else:
                # Conversational: "Help me" → prompt for question.
                self._listening = True
                self.help_started.emit()
                self._speak("What can I help you with?")
            return True

        return False

    def cancel(self) -> None:
        """Cancel any pending listening state."""
        self._listening = False

    def shutdown(self) -> None:
        """Stop the background worker. Does not wait for pending work."""
        self._listening = False
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_question(lowered: str, original: str) -> Optional[str]:
        """Extract the question from a wake-phrase utterance.

        Returns:
            The question text (may be empty for conversational mode), or
            ``None`` if *lowered* does not start with a wake phrase.
        """
        for prefix in _WAKE_PREFIXES:
            if lowered.startswith(prefix):
                remainder = original[len(prefix):].strip()
                # Strip leading punctuation like ", how do I..." → "how do I..."
                while remainder and remainder[0] in ",;:- ":
                    remainder = remainder[1:].strip()
                return remainder
        return None

    def _submit_question(self, question: str, game: Optional[GameBase]) -> None:
        """Submit the question to the background worker."""
        self.help_started.emit()
        context = self._assemble_context(game)
        self._executor.submit(self._do_answer, question, context)

    def _assemble_context(self, game: Optional[GameBase]) -> str:
        """Build the context string from the current game state.

        Args:
            game: The currently loaded game, or ``None`` for menu.

        Returns:
            A JSON-formatted context string for the LLM prompt.
        """
        if game is not None:
            try:
                ctx = game.get_help_context()
                if ctx:
                    return json.dumps(ctx, indent=2, default=str)
            except Exception:
                pass
        return json.dumps(_MENU_CONTEXT, indent=2)

    def _do_answer(self, question: str, context: str) -> None:
        """Worker-thread: generate the answer and stream it to TTS."""
        try:
            if self._llm_client.is_available():
                self._answer_with_llm(question, context)
            else:
                self._answer_fallback(context)
        except Exception:
            self._answer_fallback(context)

        self.help_finished.emit()

    def _answer_with_llm(self, question: str, context: str) -> None:
        """Stream an LLM-generated answer to the narrator."""
        system = _HELP_SYSTEM_PROMPT.format(context=context)
        stream = self._llm_client.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": question}],
            stream=True,
            temperature=0.5,
            max_tokens=256,
        )

        parts: list[str] = []
        buffer = ""
        for chunk in stream:
            buffer += chunk
            sentences, buffer = split_sentences(buffer)
            for sentence in sentences:
                if not sentence.strip():
                    continue
                parts.append(sentence)
                self._speak(sentence)

        # Flush any remaining text.
        if buffer.strip():
            parts.append(buffer.strip())
            self._speak(buffer.strip())

        full = " ".join(parts).strip()
        if full:
            self.answered.emit(full)

    def _answer_fallback(self, context: str) -> None:
        """Read the context summary + topics as scripted text (no LLM)."""
        try:
            ctx = json.loads(context)
        except (json.JSONDecodeError, TypeError):
            ctx = {}

        summary = ctx.get("summary", "I'm not sure how to help with that.")
        topics = ctx.get("topics", {})

        text = summary
        if topics:
            topic_lines = [f"{k}: {v}" for k, v in topics.items()]
            text += " " + ". ".join(topic_lines) + "."

        self._speak(text)
        self.answered.emit(text)

    def _speak(self, text: str) -> None:
        """Synthesize and play *text* via the narrator."""
        if self._narrator is None:
            return
        try:
            self._narrator.synthesize_and_play(text, self._channel)
        except Exception:
            pass
