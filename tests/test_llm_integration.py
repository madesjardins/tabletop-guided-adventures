"""LLM integration test — NLU + NLG pipeline with speech in and TTS out.

Run with:
    uv run python tests/test_llm_integration.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
VOSK_DIR = ROOT / "vosk_models"
VOICES_DIR = ROOT / "piper_voices"

sys.path.insert(0, str(ROOT / "python"))


def _register_cuda_dll_dir() -> Optional[str]:
    """Add the CUDA toolkit bin directory to the DLL search path on Windows.

    The prebuilt cu124 llama-cpp-python wheel's llama.dll depends on the CUDA
    runtime DLLs (cudart64_12.dll, cublas64_12.dll, ...) which are only found if
    the CUDA bin directory is on the DLL search path. This makes the test robust
    regardless of whether CUDA is on PATH at launch time.

    Returns:
        The CUDA bin path that was registered, or None if not found.
    """
    if os.name != "nt":
        return None

    candidates: list[Path] = []
    # Explicit env vars set by the CUDA installer (e.g. CUDA_PATH_V12_6).
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
            os.add_dll_directory(str(bin_dir))
            return str(bin_dir)
    return None


_CUDA_BIN = _register_cuda_dll_dir()

from ttga.speech_recognition import SpeechRecognizer, get_audio_input_devices
from ttga.narrator import Narrator, find_available_voices
from ttga.sound_mixer import Channel

# ---------------------------------------------------------------------------
# Known intents for the intent picker / NLU schema
# ---------------------------------------------------------------------------
KNOWN_INTENTS = [
    "funny",
    "aggressive",
    "flirting",
    "surprised",
    "sad",
    "excited",
    "sarcastic",
    "neutral",
]

NLU_SYSTEM = """\
You are an emotional tone parser. Given a person's spoken sentence, classify the underlying tone or intent and extract its core meaning.

Respond ONLY with a valid JSON object on a single line using this exact schema:
{"intent": "<one of: funny, aggressive, flirting, surprised, sad, excited, sarcastic, neutral>", "subject": "<who or what it is about, or null>", "details": "<the core message in a few words, or null>"}

Do not include any explanation, markdown, or extra text."""

NLG_SYSTEM = """\
You are an expressive voice actor. Given a structured intent describing a tone and a message, rephrase the message so it strongly conveys that tone (1-2 sentences max).
Fully embody the given tone (e.g. funny, aggressive, flirting, surprised). Do not mention JSON. Do not use the original words verbatim."""


# ---------------------------------------------------------------------------
# LLM worker — runs inference off the Qt thread
# ---------------------------------------------------------------------------
class LLMWorker(QtCore.QObject):
    nlu_done = QtCore.Signal(str)   # raw JSON string
    nlg_done = QtCore.Signal(str)   # narrator phrase
    error = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._llm = None
        self._model_path: Optional[str] = None
        self._n_gpu_layers: int = -1
        self._lock = threading.Lock()

    def load_model(self, model_path: str, n_gpu_layers: int) -> None:
        def _load():
            with self._lock:
                try:
                    from llama_cpp import Llama
                    self._llm = Llama(
                        model_path=model_path,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=2048,
                        verbose=False,
                    )
                    self._model_path = model_path
                    self._n_gpu_layers = n_gpu_layers
                except Exception as e:
                    self.error.emit(f"Model load failed: {e}")
        threading.Thread(target=_load, daemon=True).start()

    def run_pipeline(self, text: str) -> None:
        def _run():
            with self._lock:
                if self._llm is None:
                    self.error.emit("No model loaded.")
                    return
                try:
                    # --- NLU ---
                    nlu_resp = self._llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": NLU_SYSTEM},
                            {"role": "user", "content": text},
                        ],
                        max_tokens=128,
                        temperature=0.0,
                    )
                    nlu_raw = nlu_resp["choices"][0]["message"]["content"].strip()
                    self.nlu_done.emit(nlu_raw)

                    # --- NLG ---
                    nlg_resp = self._llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": NLG_SYSTEM},
                            {"role": "user", "content": nlu_raw},
                        ],
                        max_tokens=128,
                        temperature=0.7,
                    )
                    nlg_text = nlg_resp["choices"][0]["message"]["content"].strip()
                    self.nlg_done.emit(nlg_text)

                except Exception as e:
                    self.error.emit(f"Inference error: {e}")
        threading.Thread(target=_run, daemon=True).start()

    def run_nlg_only(self, intent_json: str) -> None:
        def _run():
            with self._lock:
                if self._llm is None:
                    self.error.emit("No model loaded.")
                    return
                try:
                    nlg_resp = self._llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": NLG_SYSTEM},
                            {"role": "user", "content": intent_json},
                        ],
                        max_tokens=128,
                        temperature=0.7,
                    )
                    nlg_text = nlg_resp["choices"][0]["message"]["content"].strip()
                    self.nlg_done.emit(nlg_text)
                except Exception as e:
                    self.error.emit(f"Inference error: {e}")
        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class LLMTestWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LLM Integration Test — NLU + NLG Pipeline")
        self.resize(960, 680)

        self._speech_recognizer: Optional[SpeechRecognizer] = None
        self._narrator: Optional[Narrator] = None
        self._llm_worker = LLMWorker()
        self._llm_worker.nlu_done.connect(self._on_nlu_done)
        self._llm_worker.nlg_done.connect(self._on_nlg_done)
        self._llm_worker.error.connect(self._on_llm_error)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setSpacing(8)

        root_layout.addWidget(self._build_config_panel(), stretch=0)
        root_layout.addWidget(self._build_pipeline_panel(), stretch=1)

    # ------------------------------------------------------------------
    # Config panel (left, fixed width)
    # ------------------------------------------------------------------
    def _build_config_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setAlignment(QtCore.Qt.AlignTop)

        # --- LLM ---
        llm_group = QtWidgets.QGroupBox("LLM Model")
        llm_layout = QtWidgets.QFormLayout(llm_group)

        self._llm_combo = QtWidgets.QComboBox()
        self._populate_llm_models()
        llm_layout.addRow("GGUF:", self._llm_combo)

        gpu_layout = QtWidgets.QHBoxLayout()
        self._gpu_layers_spin = QtWidgets.QSpinBox()
        self._gpu_layers_spin.setRange(-1, 200)
        self._gpu_layers_spin.setValue(-1)
        self._gpu_layers_spin.setSpecialValueText("All (GPU)")
        self._gpu_layers_spin.setToolTip("-1 = all layers on GPU, 0 = CPU only")
        gpu_layout.addWidget(self._gpu_layers_spin)
        llm_layout.addRow("GPU layers:", gpu_layout)

        self._load_llm_btn = QtWidgets.QPushButton("Load Model")
        self._load_llm_btn.clicked.connect(self._on_load_llm)
        llm_layout.addRow(self._load_llm_btn)

        self._llm_status_label = QtWidgets.QLabel("No model loaded")
        self._llm_status_label.setWordWrap(True)
        self._llm_status_label.setStyleSheet("color: gray; font-style: italic;")
        llm_layout.addRow(self._llm_status_label)

        layout.addWidget(llm_group)

        # --- Speech ---
        speech_group = QtWidgets.QGroupBox("Speech Recognition (Vosk)")
        speech_layout = QtWidgets.QFormLayout(speech_group)

        self._vosk_combo = QtWidgets.QComboBox()
        self._populate_vosk_models()
        speech_layout.addRow("Model:", self._vosk_combo)

        self._mic_combo = QtWidgets.QComboBox()
        self._populate_audio_inputs()
        speech_layout.addRow("Mic:", self._mic_combo)

        self._mic_toggle_btn = QtWidgets.QPushButton("Start Mic")
        self._mic_toggle_btn.setCheckable(True)
        self._mic_toggle_btn.clicked.connect(self._on_mic_toggle)
        speech_layout.addRow(self._mic_toggle_btn)

        layout.addWidget(speech_group)

        # --- TTS ---
        tts_group = QtWidgets.QGroupBox("TTS (Piper)")
        tts_layout = QtWidgets.QFormLayout(tts_group)

        self._voice_combo = QtWidgets.QComboBox()
        self._populate_voices()
        tts_layout.addRow("Voice:", self._voice_combo)

        self._load_voice_btn = QtWidgets.QPushButton("Load Voice")
        self._load_voice_btn.clicked.connect(self._on_load_voice)
        tts_layout.addRow(self._load_voice_btn)

        self._tts_status_label = QtWidgets.QLabel("No voice loaded")
        self._tts_status_label.setWordWrap(True)
        self._tts_status_label.setStyleSheet("color: gray; font-style: italic;")
        tts_layout.addRow(self._tts_status_label)

        layout.addWidget(tts_group)
        layout.addStretch()

        return panel

    # ------------------------------------------------------------------
    # Pipeline panel (right, expands)
    # ------------------------------------------------------------------
    def _build_pipeline_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)

        # --- Input ---
        input_group = QtWidgets.QGroupBox("1 — Input")
        input_layout = QtWidgets.QVBoxLayout(input_group)

        self._partial_label = QtWidgets.QLabel("")
        self._partial_label.setStyleSheet("color: gray; font-style: italic;")
        input_layout.addWidget(self._partial_label)

        text_row = QtWidgets.QHBoxLayout()
        self._input_edit = QtWidgets.QLineEdit()
        self._input_edit.setPlaceholderText("Type or speak a command…")
        self._input_edit.returnPressed.connect(self._on_submit)
        text_row.addWidget(self._input_edit)
        self._submit_btn = QtWidgets.QPushButton("Submit")
        self._submit_btn.clicked.connect(self._on_submit)
        text_row.addWidget(self._submit_btn)
        input_layout.addLayout(text_row)

        layout.addWidget(input_group)

        # --- NLU result ---
        nlu_group = QtWidgets.QGroupBox("2 — NLU: Extracted Intent (JSON)")
        nlu_layout = QtWidgets.QVBoxLayout(nlu_group)

        self._nlu_edit = QtWidgets.QTextEdit()
        self._nlu_edit.setReadOnly(False)
        self._nlu_edit.setFixedHeight(70)
        self._nlu_edit.setPlaceholderText('{"intent": "...", "subject": "...", "details": "..."}')
        nlu_layout.addWidget(self._nlu_edit)

        intent_row = QtWidgets.QHBoxLayout()
        intent_row.addWidget(QtWidgets.QLabel("Override intent:"))
        self._intent_combo = QtWidgets.QComboBox()
        self._intent_combo.addItems(KNOWN_INTENTS)
        self._intent_combo.currentTextChanged.connect(self._on_intent_override)
        intent_row.addWidget(self._intent_combo)
        intent_row.addStretch()
        self._rephrase_btn = QtWidgets.QPushButton("Re-narrate from JSON")
        self._rephrase_btn.clicked.connect(self._on_rephrase)
        intent_row.addWidget(self._rephrase_btn)
        nlu_layout.addLayout(intent_row)

        layout.addWidget(nlu_group)

        # --- NLG result ---
        nlg_group = QtWidgets.QGroupBox("3 — NLG: Narrator Phrase")
        nlg_layout = QtWidgets.QVBoxLayout(nlg_group)

        self._nlg_edit = QtWidgets.QTextEdit()
        self._nlg_edit.setReadOnly(True)
        self._nlg_edit.setFixedHeight(80)
        self._nlg_edit.setPlaceholderText("Narrator phrase will appear here…")
        nlg_layout.addWidget(self._nlg_edit)

        speak_row = QtWidgets.QHBoxLayout()
        speak_row.addStretch()
        self._speak_btn = QtWidgets.QPushButton("Speak")
        self._speak_btn.clicked.connect(self._on_speak)
        speak_row.addWidget(self._speak_btn)
        nlg_layout.addLayout(speak_row)

        layout.addWidget(nlg_group)

        # --- Log ---
        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(120)
        log_layout.addWidget(self._log)
        layout.addWidget(log_group)

        layout.addStretch()
        return panel

    # ------------------------------------------------------------------
    # Populate helpers
    # ------------------------------------------------------------------
    def _populate_llm_models(self) -> None:
        self._llm_combo.clear()
        gguf_files = sorted(MODELS_DIR.glob("*.gguf"))
        for p in gguf_files:
            self._llm_combo.addItem(p.name, str(p))
        if not gguf_files:
            self._llm_combo.addItem("(no .gguf files in models/)", "")

    def _populate_vosk_models(self) -> None:
        self._vosk_combo.clear()
        for p in sorted(VOSK_DIR.iterdir()):
            if p.is_dir() and not p.name.startswith("_"):
                self._vosk_combo.addItem(p.name, str(p))
        if self._vosk_combo.count() == 0:
            self._vosk_combo.addItem("(no vosk models found)", "")

    def _populate_audio_inputs(self) -> None:
        self._mic_combo.clear()
        for dev in get_audio_input_devices():
            self._mic_combo.addItem(f"{dev['index']}: {dev['name']}", dev['index'])

    def _populate_voices(self) -> None:
        self._voice_combo.clear()
        for p in find_available_voices(str(VOICES_DIR)):
            self._voice_combo.addItem(Path(p).stem, p)
        if self._voice_combo.count() == 0:
            self._voice_combo.addItem("(no voices found)", "")

    # ------------------------------------------------------------------
    # Slots — config
    # ------------------------------------------------------------------
    def _on_load_llm(self) -> None:
        model_path = self._llm_combo.currentData()
        if not model_path:
            self._log_msg("No GGUF model selected.")
            return
        n_gpu = self._gpu_layers_spin.value()
        self._llm_status_label.setText(f"Loading {Path(model_path).name}…")
        self._llm_status_label.setStyleSheet("color: orange;")
        if _CUDA_BIN:
            self._log_msg(f"CUDA DLLs: {_CUDA_BIN}")
        else:
            self._log_msg("CUDA bin not found — GPU offload may fail (CPU fallback).")
        self._log_msg(f"Loading model: {Path(model_path).name} (gpu_layers={n_gpu})")
        self._llm_worker.load_model(model_path, n_gpu)
        # Poll until loaded (simple approach via timer)
        self._load_poll_timer = QtCore.QTimer(self)
        self._load_poll_timer.setInterval(500)
        self._load_poll_timer.timeout.connect(self._poll_model_loaded)
        self._load_poll_timer.start()

    def _poll_model_loaded(self) -> None:
        if self._llm_worker._llm is not None:
            self._load_poll_timer.stop()
            name = Path(self._llm_worker._model_path).name
            self._llm_status_label.setText(f"Loaded: {name}")
            self._llm_status_label.setStyleSheet("color: green;")
            self._log_msg(f"Model ready: {name}")

    def _on_load_voice(self) -> None:
        voice_path = self._voice_combo.currentData()
        if not voice_path:
            self._log_msg("No voice selected.")
            return
        try:
            if self._narrator is None:
                self._narrator = Narrator(voice_path)
            else:
                self._narrator.set_voice_model(voice_path)
            self._tts_status_label.setText(f"Loaded: {Path(voice_path).stem}")
            self._tts_status_label.setStyleSheet("color: green;")
            self._log_msg(f"Voice loaded: {Path(voice_path).stem}")
        except Exception as e:
            self._tts_status_label.setText(f"Error: {e}")
            self._tts_status_label.setStyleSheet("color: red;")
            self._log_msg(f"Voice load error: {e}")

    def _on_mic_toggle(self, checked: bool) -> None:
        if checked:
            vosk_path = self._vosk_combo.currentData()
            device_idx = self._mic_combo.currentData()
            if not vosk_path:
                self._log_msg("No Vosk model selected.")
                self._mic_toggle_btn.setChecked(False)
                return
            try:
                self._speech_recognizer = SpeechRecognizer(
                    model_path=vosk_path,
                    device_index=device_idx,
                )
                self._speech_recognizer.partial_result.connect(self._on_partial)
                self._speech_recognizer.final_result.connect(self._on_final_speech)
                self._speech_recognizer.error_occurred.connect(self._log_msg)
                self._speech_recognizer.start()
                self._mic_toggle_btn.setText("Stop Mic")
                self._log_msg("Mic started.")
            except Exception as e:
                self._log_msg(f"Mic error: {e}")
                self._mic_toggle_btn.setChecked(False)
        else:
            if self._speech_recognizer:
                self._speech_recognizer.stop()
                self._speech_recognizer = None
            self._mic_toggle_btn.setText("Start Mic")
            self._partial_label.setText("")
            self._log_msg("Mic stopped.")

    # ------------------------------------------------------------------
    # Slots — pipeline
    # ------------------------------------------------------------------
    @QtCore.Slot(str)
    def _on_partial(self, text: str) -> None:
        self._partial_label.setText(f"…{text}")

    @QtCore.Slot(str)
    def _on_final_speech(self, text: str) -> None:
        self._partial_label.setText("")
        self._input_edit.setText(text)
        self._log_msg(f"Speech: {text}")
        self._run_pipeline(text)

    def _on_submit(self) -> None:
        text = self._input_edit.text().strip()
        if not text:
            return
        self._log_msg(f"Submit: {text}")
        self._run_pipeline(text)

    def _run_pipeline(self, text: str) -> None:
        self._nlu_edit.setPlaceholderText("Running NLU…")
        self._nlu_edit.clear()
        self._nlg_edit.clear()
        self._submit_btn.setEnabled(False)
        self._llm_worker.run_pipeline(text)

    def _on_rephrase(self) -> None:
        intent_json = self._nlu_edit.toPlainText().strip()
        if not intent_json:
            self._log_msg("No intent JSON to rephrase.")
            return
        self._nlg_edit.clear()
        self._llm_worker.run_nlg_only(intent_json)

    def _on_intent_override(self, intent: str) -> None:
        current = self._nlu_edit.toPlainText().strip()
        try:
            obj = json.loads(current) if current else {}
        except json.JSONDecodeError:
            obj = {}
        obj["intent"] = intent
        self._nlu_edit.setPlainText(json.dumps(obj))

    @QtCore.Slot(str)
    def _on_nlu_done(self, raw: str) -> None:
        self._nlu_edit.setPlainText(raw)
        self._log_msg(f"NLU: {raw}")
        # Sync the intent combo if possible
        try:
            obj = json.loads(raw)
            intent = obj.get("intent", "")
            idx = self._intent_combo.findText(intent)
            if idx >= 0:
                self._intent_combo.blockSignals(True)
                self._intent_combo.setCurrentIndex(idx)
                self._intent_combo.blockSignals(False)
        except json.JSONDecodeError:
            pass

    @QtCore.Slot(str)
    def _on_nlg_done(self, text: str) -> None:
        self._nlg_edit.setPlainText(text)
        self._log_msg(f"NLG: {text}")
        self._submit_btn.setEnabled(True)

    @QtCore.Slot(str)
    def _on_llm_error(self, msg: str) -> None:
        self._log_msg(f"ERROR: {msg}")
        self._submit_btn.setEnabled(True)

    def _on_speak(self) -> None:
        text = self._nlg_edit.toPlainText().strip()
        if not text:
            self._log_msg("Nothing to speak.")
            return
        if self._narrator is None or self._narrator.piper_voice is None:
            self._log_msg("No voice loaded. Load a Piper voice first.")
            return
        try:
            self._narrator.synthesize_and_play(text, Channel.VOICE, do_play_immediately=True)
            self._log_msg("Speaking…")
        except Exception as e:
            self._log_msg(f"TTS error: {e}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _log_msg(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    def closeEvent(self, event) -> None:
        if self._speech_recognizer:
            self._speech_recognizer.stop()
        if self._narrator:
            self._narrator.shutdown()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LLMTestWindow()
    win.show()
    sys.exit(app.exec())
