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

"""Core narration engine (NLG role), shared by all games.

This module implements the generic **phrasing** (NLG) mechanics of the LLM
narrator design in ``docs/llm_narrator_architecture.md``. It wraps an optional
``LLMClient`` and a persona system-prompt, turning a state-machine-supplied
scripted line into in-character narration.

The engine contains no game-specific content: each game supplies its own
persona (and, later, allowed intents) via the constructor. It is strictly
LLM-optional: when no client is given, the client is unavailable, the feature
is disabled, or generation errors out, ``phrase()`` returns the scripted
fallback text verbatim so the game runs identically without an LLM.
"""

from __future__ import annotations

from typing import Any, Optional

# Neutral default persona. Games should pass their own in-character persona; this
# generic one keeps the engine usable (and testable) on its own.
DEFAULT_PERSONA = """\
You are the Narrator for a solo tabletop game session: a dramatic but concise \
storyteller who never breaks character and never explains rules unless asked.

Hard rules:
- Speak only the narration text. No stage directions, no markdown, no quotes.
- Keep it to 1-2 short sentences unless told otherwise.
- Never invent game rules, names, numbers, or player decisions.
- Use only the facts provided in the SITUATION block. If a fact is missing, do \
not fabricate it.
- This text will be read aloud by a TTS engine; avoid symbols, emoji, and \
numbers written as digits when a word reads more naturally."""


class NarrationEngine:
    """Turns scripted prompts into in-character narration (NLG).

    The engine is game-agnostic; supply a game-specific ``persona`` to give the
    narrator its character. Intent parsing (NLU) is added in a later step.

    Example:
        >>> engine = NarrationEngine(llm_client=client, persona=MY_PERSONA)
        >>> engine.phrase("Player 1, name your next model.")
        'Player One, call forth your next warrior to the muster.'
    """

    def __init__(
        self,
        llm_client: Any = None,
        *,
        persona: str = DEFAULT_PERSONA,
        enabled: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 120,
    ) -> None:
        """Initialize the narration engine.

        Args:
            llm_client: An ``LLMClient`` (or compatible) used for phrasing. May
                be ``None`` to force scripted output.
            persona: The system prompt establishing the narrator's character.
            enabled: Master flag for in-character phrasing. When False, the
                engine always returns the scripted fallback.
            temperature: Sampling temperature for phrasing generation.
            max_tokens: Maximum tokens to generate per phrase.
        """
        self._client = llm_client
        self._persona = persona
        self._enabled = enabled
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def enabled(self) -> bool:
        """Whether in-character phrasing is enabled."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable in-character phrasing at runtime."""
        self._enabled = enabled

    def is_active(self) -> bool:
        """Return True if phrasing will actually use the LLM.

        Requires the feature enabled, a client present, and the client
        reporting availability (runtime installed and a model loaded).
        """
        if not self._enabled or self._client is None:
            return False
        try:
            return bool(self._client.is_available())
        except Exception:
            return False

    def phrase(self, scripted: str, *, situation: Optional[str] = None) -> str:
        """Rephrase a scripted line in-character, falling back to the script.

        Args:
            scripted: The deterministic, state-machine-supplied line. This is
                both the source of facts and the fallback returned when the LLM
                is unavailable.
            situation: Optional explicit SITUATION block. When omitted, the
                scripted line itself is used as the situation to convey.

        Returns:
            In-character narration when the LLM is active, otherwise the
            ``scripted`` text verbatim. Never raises.
        """
        if not self.is_active():
            return scripted

        block = situation if situation is not None else scripted
        user_message = (
            "SITUATION:\n"
            f"{block}\n\n"
            "Speak the narration that conveys this to the player, in character."
        )

        try:
            result = self._client.generate(
                user_message,
                system=self._persona,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            text = (result or "").strip() if isinstance(result, str) else ""
            return text or scripted
        except Exception:
            # Any inference failure degrades gracefully to the scripted line.
            return scripted
