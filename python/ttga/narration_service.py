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

"""Asynchronous narration service (Step 5: threading + streaming).

:class:`NarrationService` runs the (potentially slow) LLM work off the Qt main
thread so the UI stays responsive, and streams narration to the TTS engine one
sentence at a time so audio starts at time-to-first-sentence rather than after
the full response.

It wraps a core ``NarrationEngine`` and a ``Narrator``, exposing:

- :meth:`speak` — phrase a scripted line in-character (streaming) and play each
  sentence through the narrator as it is produced.
- :meth:`parse_intent_async` — parse a player utterance into an intent without
  blocking the caller; the result is delivered via the :attr:`intent_parsed`
  signal.

Work runs on a single background worker (so calls are serialized and ordered),
and results are delivered through Qt signals which, because this object lives on
the main thread, are marshaled back to the main thread automatically.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Optional

from PySide6 import QtCore

from .narration_engine import Intent, NarrationEngine, split_sentences
from .sound_mixer import Channel


class NarrationService(QtCore.QObject):
    """Off-thread narration + intent parsing with streaming TTS.

    Signals:
        narrated(str): Emitted with the full phrased text once a ``speak``
            request finishes (suitable for logging / UI display).
        sentence_spoken(str): Emitted for each sentence as it is synthesized.
        intent_parsed(int, object): ``(request_id, Intent)`` for a completed
            ``parse_intent_async`` request.
        error(str): Emitted on a background failure (non-fatal).
    """

    narrated = QtCore.Signal(str)
    sentence_spoken = QtCore.Signal(str)
    intent_parsed = QtCore.Signal(int, object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        engine: NarrationEngine,
        narrator: Any = None,
        *,
        channel: Channel = Channel.VOICE,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        """Initialize the service.

        Args:
            engine: The core :class:`NarrationEngine` (phrasing + NLU).
            narrator: Optional ``Narrator`` for TTS playback. When ``None``,
                ``speak`` still emits text signals but produces no audio.
            channel: Audio channel for narration playback.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self._engine = engine
        self._narrator = narrator
        self._channel = channel
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="narration"
        )
        self._req_ids = itertools.count(1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(
        self,
        scripted: str,
        *,
        situation: Optional[str] = None,
        use_persona: bool = False,
    ) -> None:
        """Phrase and speak *scripted*, off the main thread.

        When *use_persona* is True, the scripted line is rephrased in-character
        by the LLM (streamed sentence by sentence). When False (default), the
        text is spoken verbatim — suitable for functional messages, error
        notices, and confirmations where dramatization is undesirable.

        Args:
            scripted: The text to speak (or source of facts for LLM phrasing).
            situation: Optional explicit SITUATION block for LLM phrasing.
            use_persona: When True, rephrase via the LLM persona. When False,
                speak the text as-is.
        """
        self._executor.submit(self._do_speak, scripted, situation, use_persona)

    def parse_intent_async(
        self,
        utterance: str,
        allowed: Mapping[str, str],
        *,
        context: Optional[Mapping[str, Any]] = None,
        confidence_threshold: float = 0.0,
    ) -> int:
        """Parse an utterance into an intent off the main thread.

        Args:
            utterance: The raw player speech.
            allowed: Allowed intent name -> description mapping.
            context: Optional state facts to aid disambiguation.
            confidence_threshold: Minimum confidence to accept.

        Returns:
            A request id; the matching result is delivered via
            :attr:`intent_parsed` as ``(request_id, Intent)``.
        """
        req_id = next(self._req_ids)
        self._executor.submit(
            self._do_parse, req_id, utterance, allowed, context, confidence_threshold
        )
        return req_id

    def shutdown(self) -> None:
        """Stop the background worker. Does not wait for pending work."""
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Worker-thread implementations
    # ------------------------------------------------------------------

    def _do_speak(
        self,
        scripted: str,
        situation: Optional[str],
        use_persona: bool = False,
    ) -> None:
        parts: list[str] = []
        try:
            if use_persona:
                iterator = self._engine.phrase_stream(scripted, situation=situation)
            else:
                iterator = split_sentences(scripted)[0] or [scripted]
            for sentence in iterator:
                if not sentence:
                    continue
                parts.append(sentence)
                self.sentence_spoken.emit(sentence)
                self._synthesize(sentence)
        except Exception as exc:  # pragma: no cover - defensive
            self.error.emit(str(exc))

        full = " ".join(parts).strip() or scripted
        self.narrated.emit(full)

    def _do_parse(
        self,
        req_id: int,
        utterance: str,
        allowed: Mapping[str, str],
        context: Optional[Mapping[str, Any]],
        confidence_threshold: float,
    ) -> None:
        try:
            intent = self._engine.parse_intent(
                utterance,
                allowed,
                context=context,
                confidence_threshold=confidence_threshold,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.error.emit(str(exc))
            intent = Intent()
        self.intent_parsed.emit(req_id, intent)

    def _synthesize(self, text: str) -> None:
        if self._narrator is None:
            return
        try:
            self._narrator.synthesize_and_play(text, self._channel)
        except Exception:
            # TTS failures must never break narration flow.
            pass
