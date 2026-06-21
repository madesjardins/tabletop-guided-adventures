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

"""Tests for ttga.llm_client.

Runs two suites:
  1. Unit tests using a fake backend (no real model / runtime needed).
  2. A live smoke test that loads a real GGUF model if one is present in
     models/ and llama-cpp-python is installed (otherwise skipped).

Run with:
    uv run python tests/test_llm_client.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from ttga.llm_client import (  # noqa: E402
    LLMClient,
    LLMConfig,
    LLMBackend,
    ModelInfo,
    discover_gguf_models,
)

MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Fake backend for unit tests
# ---------------------------------------------------------------------------
class FakeBackend(LLMBackend):
    """Deterministic in-memory backend for testing client logic."""

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self._loaded = False
        self.last_messages = None
        self.last_kwargs = None

    def runtime_available(self) -> bool:
        return self._available

    def list_models(self, models_dir: str) -> list[ModelInfo]:
        return [ModelInfo(name="fake.gguf", path="/fake/fake.gguf", size_gb=4.5)]

    def load(self, model_path: str, *, n_gpu_layers: int, n_ctx: int) -> None:
        self.last_kwargs = {"n_gpu_layers": n_gpu_layers, "n_ctx": n_ctx}
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def chat(self, messages, *, stream, temperature, max_tokens):
        self.last_messages = messages
        self.last_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if stream:
            return iter(["Hel", "lo!"])
        return "Hello!"

    def unload(self) -> None:
        self._loaded = False


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return condition


def run_unit_tests() -> bool:
    """Run fake-backend unit tests. Returns True if all pass."""
    print("Unit tests (fake backend):")
    ok = True

    # Not available before a model is loaded.
    client = LLMClient(LLMConfig(model_path="/fake/fake.gguf"), backend=FakeBackend())
    ok &= _check("runtime_available is True", client.runtime_available())
    ok &= _check("not available before load", not client.is_available())

    # list_models passes through the backend.
    ok &= _check("list_models returns one model", len(client.list_models()) == 1)

    # After load it becomes available.
    client.load_model()
    ok &= _check("loaded after load_model", client.is_loaded())
    ok &= _check("available after load", client.is_available())

    # generate() builds system+user messages and returns text.
    reply = client.generate("hi", system="be terse")
    backend: FakeBackend = client._backend  # type: ignore[assignment]
    ok &= _check("generate returns text", reply == "Hello!")
    ok &= _check("system message included", backend.last_messages[0]["role"] == "system")
    ok &= _check("user message included", backend.last_messages[1]["content"] == "hi")

    # config defaults applied to chat kwargs.
    ok &= _check(
        "default temperature applied",
        backend.last_kwargs["temperature"] == client.config.temperature,
    )

    # streaming returns an iterator that joins to the full text.
    chunks = list(client.generate("hi", stream=True))
    ok &= _check("stream yields chunks", "".join(chunks) == "Hello!")

    # disabled config => never available, chat raises.
    disabled = LLMClient(LLMConfig(enabled=False), backend=FakeBackend())
    ok &= _check("disabled is not available", not disabled.is_available())
    try:
        disabled.generate("hi")
        ok &= _check("disabled chat raises", False)
    except RuntimeError:
        ok &= _check("disabled chat raises", True)

    # runtime unavailable => not available even if "loaded".
    unavail = LLMClient(LLMConfig(model_path="/fake/fake.gguf"), backend=FakeBackend(available=False))
    ok &= _check("unavailable runtime not available", not unavail.is_available())

    # unload resets state.
    client.unload()
    ok &= _check("not loaded after unload", not client.is_loaded())

    return ok


def run_live_smoke_test() -> bool:
    """Load a real GGUF model and generate a short reply. Skips if unavailable."""
    print("\nLive smoke test (real model):")

    real = LLMClient(LLMConfig(n_ctx=2048, max_tokens=32))
    if not real.runtime_available():
        print("  [SKIP] llama-cpp-python not installed.")
        return True

    models = discover_gguf_models(MODELS_DIR)
    if not models:
        print(f"  [SKIP] no .gguf models found in {MODELS_DIR}.")
        return True

    model = models[0]
    print(f"  Loading: {model.name} ({model.vram_hint})")
    try:
        real.load_model(model.path)
        reply = real.generate(
            "Reply with exactly the word: pong",
            system="You are a terse test assistant.",
            temperature=0.0,
            max_tokens=8,
        )
        print(f"  Model replied: {reply!r}")
        ok = _check("non-empty reply", bool(reply.strip()))
        real.unload()
        return ok
    except Exception as e:
        print(f"  [FAIL] live test error: {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("ttga.llm_client tests")
    print("=" * 60)
    unit_ok = run_unit_tests()
    live_ok = run_live_smoke_test()
    print("=" * 60)
    if unit_ok and live_ok:
        print("All tests passed.")
        return 0
    print("Some tests FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
