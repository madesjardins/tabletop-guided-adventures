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

"""Local LLM client for the Tabletop Guided Adventures narrator.

This module provides a backend-agnostic ``LLMClient`` used by games for the two
bounded LLM roles described in ``docs/llm_narrator_architecture.md``:

- **NLG** (phrasing scripted prompts in-character)
- **NLU** (parsing player speech into structured intents)

The client is LLM-optional: when disabled, no model is selected, or the runtime
is unavailable, ``is_available()`` returns ``False`` and callers fall back to
scripted text. The default backend is ``llama-cpp-python`` (in-process GGUF
inference); the ``LLMBackend`` interface allows other backends (e.g. Ollama) to
be dropped in later.
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

from .constants import MODELS_DIR_PATH

# A chat message is a dict like {"role": "system"|"user"|"assistant", "content": str}.
Message = dict
# chat() returns a full string, or an iterator of text chunks when streaming.
ChatResult = Union[str, Iterator[str]]


# ---------------------------------------------------------------------------
# CUDA DLL registration (Windows)
# ---------------------------------------------------------------------------
def register_cuda_dll_dir() -> Optional[str]:
    """Add the CUDA toolkit bin directory to the DLL search path on Windows.

    The prebuilt cu124 ``llama-cpp-python`` wheel's ``llama.dll`` depends on the
    CUDA runtime DLLs (``cudart64_12.dll``, ``cublas64_12.dll``, ...) which are
    only found if the CUDA bin directory is on the DLL search path. Registering
    it explicitly makes GPU loading robust regardless of whether CUDA is on
    ``PATH`` at launch time.

    Returns:
        The CUDA bin path that was registered, or ``None`` if not found / not
        applicable (non-Windows).
    """
    if os.name != "nt":
        return None

    candidates: list[Path] = []
    # Explicit env vars set by the CUDA installer (e.g. CUDA_PATH, CUDA_PATH_V12_6).
    for key, val in os.environ.items():
        if key.startswith("CUDA_PATH") and val:
            candidates.append(Path(val) / "bin")
    # Standard install location, newest version first.
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if base.exists():
        for version_dir in sorted(base.glob("v*"), reverse=True):
            candidates.append(version_dir / "bin")

    for bin_dir in candidates:
        if bin_dir.is_dir() and (bin_dir / "cudart64_12.dll").exists():
            try:
                os.add_dll_directory(str(bin_dir))
                return str(bin_dir)
            except OSError:
                continue
    return None


# ---------------------------------------------------------------------------
# Config + model info
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Runtime configuration for the LLM client.

    Attributes:
        enabled: Master feature flag. When False, the client is never available
            and callers fall back to scripted text.
        model_path: Absolute path to the selected GGUF model file (None = none
            selected).
        models_dir: Directory scanned by ``list_models()`` for ``*.gguf`` files.
        n_gpu_layers: Layers to offload to GPU (-1 = all, 0 = CPU only).
        n_ctx: Context window size in tokens.
        temperature: Default sampling temperature for generation.
        max_tokens: Default maximum tokens to generate per response.
    """

    enabled: bool = True
    model_path: Optional[str] = None
    models_dir: str = MODELS_DIR_PATH
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    temperature: float = 0.7
    max_tokens: int = 256


@dataclass
class ModelInfo:
    """Metadata about a discovered GGUF model file.

    Attributes:
        name: File name of the model (e.g. ``Qwen2.5-7B-Instruct-Q4_K_M.gguf``).
        path: Absolute path to the model file.
        size_gb: On-disk size in gigabytes (approximate VRAM footprint at load).
        vram_hint: Human-readable VRAM hint, e.g. ``~4.6 GB VRAM``.
    """

    name: str
    path: str
    size_gb: float
    vram_hint: str = ""

    def __post_init__(self) -> None:
        if not self.vram_hint:
            self.vram_hint = f"~{self.size_gb:.1f} GB VRAM"


def discover_gguf_models(models_dir: Union[str, Path]) -> list[ModelInfo]:
    """Scan a directory for ``*.gguf`` model files.

    Args:
        models_dir: Directory to scan (non-recursive).

    Returns:
        Sorted list of ``ModelInfo`` for each ``.gguf`` file found. Empty list if
        the directory does not exist.
    """
    path = Path(models_dir)
    if not path.is_dir():
        return []

    models: list[ModelInfo] = []
    for gguf in sorted(path.glob("*.gguf")):
        size_gb = gguf.stat().st_size / (1024 ** 3)
        models.append(ModelInfo(name=gguf.name, path=str(gguf), size_gb=size_gb))
    return models


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class LLMBackend(ABC):
    """Abstract inference backend.

    Implementations wrap a concrete runtime (llama-cpp-python, Ollama, ...) and
    expose a minimal chat interface. All methods must be safe to call when the
    runtime is unavailable (return falsy / raise only from ``chat``/``load``).
    """

    @abstractmethod
    def runtime_available(self) -> bool:
        """Return True if the underlying runtime can be imported / reached."""

    @abstractmethod
    def list_models(self, models_dir: str) -> list[ModelInfo]:
        """Return models available to this backend."""

    @abstractmethod
    def load(self, model_path: str, *, n_gpu_layers: int, n_ctx: int) -> None:
        """Load a model into memory. Raises on failure."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if a model is currently loaded and ready."""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        stream: bool,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        """Run a chat completion. Returns a string, or an iterator if streaming."""

    @abstractmethod
    def unload(self) -> None:
        """Release the loaded model and free resources."""


class LlamaCppBackend(LLMBackend):
    """In-process GGUF inference backend using ``llama-cpp-python``."""

    def __init__(self) -> None:
        self._llm = None  # type: ignore[var-annotated]
        self._model_path: Optional[str] = None
        self._cuda_bin: Optional[str] = None

    def runtime_available(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("llama_cpp") is not None
        except Exception:
            return False

    def list_models(self, models_dir: str) -> list[ModelInfo]:
        return discover_gguf_models(models_dir)

    def load(self, model_path: str, *, n_gpu_layers: int, n_ctx: int) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        # Ensure CUDA runtime DLLs are discoverable before importing llama_cpp.
        self._cuda_bin = register_cuda_dll_dir()

        from llama_cpp import Llama

        # Free any previously loaded model first.
        self.unload()

        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        self._model_path = model_path

    def is_loaded(self) -> bool:
        return self._llm is not None

    def chat(
        self,
        messages: list[Message],
        *,
        stream: bool,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        if self._llm is None:
            raise RuntimeError("No model loaded. Call load() first.")

        if stream:
            return self._chat_stream(messages, temperature, max_tokens)

        resp = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"].strip()

    def _chat_stream(
        self, messages: list[Message], temperature: float, max_tokens: int
    ) -> Iterator[str]:
        stream = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield piece

    def unload(self) -> None:
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception:
                pass
            self._llm = None
            self._model_path = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class LLMClient:
    """Backend-agnostic local LLM client.

    Wraps an ``LLMBackend`` and a config. The client is LLM-optional: callers
    should check ``is_available()`` and fall back to scripted text when it is
    False.

    Example:
        >>> client = LLMClient(LLMConfig(model_path="models/qwen2.5-7b.gguf"))
        >>> if client.is_available():
        ...     client.load_model()
        ...     reply = client.generate("Say hello", system="You are terse.")
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        backend: Optional[LLMBackend] = None,
    ) -> None:
        """Initialize the client.

        Args:
            config: Runtime configuration (defaults to a fresh ``LLMConfig``).
            backend: Inference backend (defaults to ``LlamaCppBackend``).
        """
        self.config: LLMConfig = config or LLMConfig()
        self._backend: LLMBackend = backend or LlamaCppBackend()
        self._lock = threading.Lock()

    # -- discovery / availability ------------------------------------------
    def runtime_available(self) -> bool:
        """Return True if the inference runtime is installed / reachable."""
        return self._backend.runtime_available()

    def list_models(self) -> list[ModelInfo]:
        """Return models available to the backend (scans ``config.models_dir``)."""
        return self._backend.list_models(self.config.models_dir)

    def is_available(self) -> bool:
        """Return True if the client can serve requests.

        Requires: feature enabled, runtime installed, and a model loaded.
        """
        return (
            self.config.enabled
            and self._backend.runtime_available()
            and self._backend.is_loaded()
        )

    def is_loaded(self) -> bool:
        """Return True if a model is currently loaded."""
        return self._backend.is_loaded()

    # -- model lifecycle ---------------------------------------------------
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load a model into the backend.

        Args:
            model_path: Path to the GGUF model. If None, uses
                ``config.model_path``.

        Raises:
            RuntimeError: If the feature is disabled or no model path is set.
            FileNotFoundError: If the model file does not exist.
        """
        if not self.config.enabled:
            raise RuntimeError("LLM is disabled (config.enabled is False).")

        path = model_path or self.config.model_path
        if not path:
            raise RuntimeError("No model_path provided or configured.")

        with self._lock:
            self._backend.load(
                path,
                n_gpu_layers=self.config.n_gpu_layers,
                n_ctx=self.config.n_ctx,
            )
            self.config.model_path = path

    def unload(self) -> None:
        """Unload the current model and free resources."""
        with self._lock:
            self._backend.unload()

    # -- inference ---------------------------------------------------------
    def chat(
        self,
        messages: list[Message],
        *,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        """Run a chat completion.

        Args:
            messages: List of chat messages (role/content dicts).
            stream: If True, return an iterator of text chunks instead of a
                full string.
            temperature: Sampling temperature (defaults to config value).
            max_tokens: Max tokens to generate (defaults to config value).

        Returns:
            The completion text, or an iterator of text chunks if streaming.

        Raises:
            RuntimeError: If the client is not available (disabled / no model).
        """
        if not self.is_available():
            raise RuntimeError("LLM not available (disabled, missing runtime, or no model loaded).")

        temp = self.config.temperature if temperature is None else temperature
        max_tok = self.config.max_tokens if max_tokens is None else max_tokens

        # Streaming results must not hold the lock across the generator's
        # lifetime; non-streaming calls are serialized.
        if stream:
            return self._backend.chat(
                messages, stream=True, temperature=temp, max_tokens=max_tok
            )
        with self._lock:
            return self._backend.chat(
                messages, stream=False, temperature=temp, max_tokens=max_tok
            )

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        """Convenience wrapper around ``chat`` for a single user prompt.

        Args:
            prompt: The user prompt text.
            system: Optional system message prepended to the conversation.
            stream: If True, return an iterator of text chunks.
            temperature: Sampling temperature (defaults to config value).
            max_tokens: Max tokens to generate (defaults to config value).

        Returns:
            The completion text, or an iterator of text chunks if streaming.
        """
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages, stream=stream, temperature=temperature, max_tokens=max_tokens
        )
