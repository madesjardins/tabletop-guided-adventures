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

import json
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

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

# Intent name returned when parsing fails, the LLM is unavailable, or the
# result is below the confidence threshold. Callers fall back to their own
# deterministic logic when they receive this.
UNKNOWN_INTENT = "unknown"

# System prompt for the intent-parsing (NLU) role. Game-agnostic: the allowed
# intents and their descriptions are supplied per call in the user message.
_NLU_SYSTEM = """\
You are an intent parser for a voice-controlled tabletop game. Given a player's \
spoken utterance and a list of allowed intents, identify which intent best \
matches and extract its value.

Respond with ONLY a single-line JSON object, no markdown, no extra text, using \
exactly this schema:
{"intent": "<one of the allowed intent names, or 'unknown'>", "value": \
"<extracted value, or null>", "confidence": <number between 0 and 1>}

Rules:
- The "intent" MUST be one of the allowed names, or "unknown" if none fit.
- "value" is the relevant extracted text (e.g. a name the player said), or null \
when the intent carries no value.
- "confidence" reflects how certain you are (1.0 = certain, 0.0 = guess).
- Do not invent intents that are not in the allowed list."""


@dataclass
class Intent:
    """A parsed player intent (NLU result).

    Attributes:
        intent: The matched intent name, or :data:`UNKNOWN_INTENT`.
        value: Extracted value (e.g. a spoken model name), or ``None``.
        confidence: Model-reported confidence in ``[0.0, 1.0]``.
        raw: Raw model output, retained for logging / debugging.
    """

    intent: str = UNKNOWN_INTENT
    value: Optional[str] = None
    confidence: float = 0.0
    raw: str = ""

    @property
    def is_unknown(self) -> bool:
        """True if this intent is the unknown sentinel."""
        return self.intent == UNKNOWN_INTENT


class NarrationEngine:
    """Turns scripted prompts into in-character narration (NLG).

    The engine is game-agnostic; supply a game-specific ``persona`` to give the
    narrator its character. It provides both NLG (:meth:`phrase`) and NLU
    (:meth:`parse_intent`) roles, each LLM-optional with graceful fallback.

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

    def phrase_stream(
        self, scripted: str, *, situation: Optional[str] = None
    ) -> Iterator[str]:
        """Stream in-character narration one complete sentence at a time.

        This enables time-to-first-sentence latency: callers can start TTS on
        the first sentence while the rest is still generating. LLM-optional and
        never raises.

        Args:
            scripted: The deterministic fallback line (also the source of facts).
            situation: Optional explicit SITUATION block.

        Yields:
            Complete sentences in order. If the LLM is inactive or fails before
            producing anything, yields the ``scripted`` text as a single item.
        """
        if not self.is_active():
            yield scripted
            return

        block = situation if situation is not None else scripted
        user_message = (
            "SITUATION:\n"
            f"{block}\n\n"
            "Speak the narration that conveys this to the player, in character."
        )

        buffer = ""
        produced_any = False
        try:
            stream = self._client.generate(
                user_message,
                system=self._persona,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk:
                    continue
                buffer += chunk
                sentences, buffer = split_sentences(buffer)
                for sentence in sentences:
                    if sentence:
                        produced_any = True
                        yield sentence
            tail = buffer.strip()
            if tail:
                produced_any = True
                yield tail
        except Exception:
            # Fall back only if nothing has been emitted yet; otherwise stop.
            if not produced_any:
                yield scripted
            return

        if not produced_any:
            yield scripted

    # ------------------------------------------------------------------
    # NLU (intent parsing)
    # ------------------------------------------------------------------

    def parse_intent(
        self,
        utterance: str,
        allowed: Mapping[str, str],
        *,
        context: Optional[Mapping[str, Any]] = None,
        confidence_threshold: float = 0.0,
    ) -> Intent:
        """Parse a player utterance into a structured intent (NLU).

        The model proposes an intent and value; callers remain authoritative
        (e.g. validating an extracted name against a database). This method is
        LLM-optional and never raises: when the LLM is inactive, returns an
        unparsable result, picks a disallowed intent, or falls below
        ``confidence_threshold``, it returns an :class:`Intent` with
        ``intent == UNKNOWN_INTENT`` so the caller can fall back.

        Args:
            utterance: The raw player speech to classify.
            allowed: Mapping of allowed intent name -> short description. The
                sentinel ``unknown`` is always permitted implicitly.
            context: Optional small dict of state facts to aid disambiguation.
            confidence_threshold: Minimum confidence to accept; below this the
                result is coerced to unknown.

        Returns:
            An :class:`Intent`. ``intent`` is always either one of ``allowed``
            or :data:`UNKNOWN_INTENT`.
        """
        if not self.is_active():
            return Intent(intent=UNKNOWN_INTENT)

        allowed_block = "\n".join(
            f"- {name}: {desc}" for name, desc in allowed.items()
        )
        parts = [
            "ALLOWED INTENTS:",
            allowed_block,
        ]
        if context:
            ctx = ", ".join(f"{k}={v}" for k, v in context.items())
            parts.append(f"\nCONTEXT: {ctx}")
        parts.append(f'\nPLAYER UTTERANCE: "{utterance}"')
        parts.append("\nReturn the JSON intent object.")
        user_message = "\n".join(parts)

        try:
            result = self._client.generate(
                user_message,
                system=_NLU_SYSTEM,
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
        except Exception:
            return Intent(intent=UNKNOWN_INTENT)

        raw = result if isinstance(result, str) else ""
        data = _extract_json_object(raw)
        if data is None:
            return Intent(intent=UNKNOWN_INTENT, raw=raw)

        name = data.get("intent")
        value = data.get("value")
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        # Validate the intent name and confidence; coerce to unknown otherwise.
        if not isinstance(name, str) or name not in allowed:
            return Intent(intent=UNKNOWN_INTENT, raw=raw, confidence=confidence)
        if confidence < confidence_threshold:
            return Intent(intent=UNKNOWN_INTENT, raw=raw, confidence=confidence)

        if value is not None and not isinstance(value, str):
            value = str(value)
        return Intent(intent=name, value=value, confidence=confidence, raw=raw)


# Matches a run of text ending in sentence-final punctuation, consuming any
# trailing whitespace so the remainder starts at the next sentence.
_SENTENCE_RE = re.compile(r"(.+?[.!?])(?:\s+|$)", re.DOTALL)


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a text buffer into complete sentences plus a trailing remainder.

    Used for incremental streaming: feed accumulated tokens, emit any complete
    sentences, and keep the remainder for the next chunk.

    Args:
        buffer: Accumulated text, possibly ending mid-sentence.

    Returns:
        ``(sentences, remainder)`` where ``sentences`` are complete, stripped
        sentences and ``remainder`` is the leftover partial sentence.
    """
    sentences: list[str] = []
    pos = 0
    for match in _SENTENCE_RE.finditer(buffer):
        sentence = match.group(1).strip()
        if sentence:
            sentences.append(sentence)
        pos = match.end()
    return sentences, buffer[pos:]


def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from model output.

    Tries a direct parse first, then falls back to the first ``{...}`` span
    found in the text (handles models that wrap JSON in prose or fences).

    Args:
        text: Raw model output.

    Returns:
        The parsed dict, or ``None`` if no valid JSON object was found.
    """
    if not text:
        return None
    candidate = text.strip()
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None
