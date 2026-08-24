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

"""Integration test for speech recognition with live preview and similarity matching.

This script provides a GUI application to test the SpeechRecognizer (Vosk) and
WhisperSpeechRecognizer (faster-whisper) classes with real-time speech
recognition, string similarity comparison, and fuzzy name matching.
"""

import os
import sys

from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout
)

root_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_path = os.path.join(root_dir_path, "python")
if python_path not in sys.path:
    sys.path.append(python_path)

from ttga.speech_recognition import (  # noqa: E402
    SpeechRecognizer,
    WhisperSpeechRecognizer,
    get_audio_input_devices,
    string_similarity,
    fuzzy_match_name,
)


# Reference strings for similarity comparison
REFERENCE_STRINGS = [
    "Warmachine",
    "Winter Guard Rifle Corps",
    "Omodamos The Black Gate",
    "Combined Melee Attack",
    "Fury Manipulation"
]


class SpeechRecognitionTestWindow(QMainWindow):
    """Main window for testing SpeechRecognizer and WhisperSpeechRecognizer."""

    # Whisper model sizes offered in the UI
    _WHISPER_MODELS = [
        "tiny.en", "base.en", "small.en", "medium.en",
        "large-v3", "large-v3-turbo",
    ]

    def __init__(self, available_models: list[str], default_model_index: int = 0) -> None:
        """Initialize the test window.

        Args:
            available_models: List of available Vosk model directory names.
            default_model_index: Index of the model to select by default.
        """
        super().__init__()

        self.available_models: list[str] = available_models
        self.default_model_index: int = default_model_index
        self.vosk_models_path: str = os.path.join(root_dir_path, "vosk_models")
        self.speech_recognizer: SpeechRecognizer | WhisperSpeechRecognizer | None = None
        self.similarity_spinboxes: list[QDoubleSpinBox] = []
        self.threshold_spinbox: QDoubleSpinBox | None = None
        self.fuzzy_match_label: QLabel | None = None

        self.setWindowTitle("Speech Recognition Integration Test")
        self.setMinimumSize(800, 600)

        self._setup_ui()
        self._setup_menu()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Model and device selection group
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout()

        # STT engine selection
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Vosk", "vosk")
        self.engine_combo.addItem("Whisper (faster-whisper)", "whisper")
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        config_layout.addRow("STT Engine:", self.engine_combo)

        # Model selection (populated based on engine)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        config_layout.addRow("Model:", self.model_combo)

        # Audio device selection
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        config_layout.addRow("Audio Device:", self.device_combo)

        # Initial prompt (Whisper only)
        self.initial_prompt_edit = QLineEdit()
        self.initial_prompt_edit.setPlaceholderText(
            "Domain vocabulary, e.g. Winter Guard, Khador, Cygnar, Stormblade"
        )
        self.initial_prompt_edit.textChanged.connect(self._on_initial_prompt_changed)
        config_layout.addRow("Initial Prompt (Whisper):", self.initial_prompt_edit)

        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # Populate model list for default engine (Vosk)
        self._populate_models()

        # Populate device list
        self._update_device_list()

        # Recognition results group
        results_group = QGroupBox("Recognition Results")
        results_layout = QFormLayout()

        self.partial_result_edit = QLineEdit()
        self.partial_result_edit.setReadOnly(True)
        self.partial_result_edit.setPlaceholderText("Partial result will appear here...")
        results_layout.addRow("Partial:", self.partial_result_edit)

        self.final_result_edit = QLineEdit()
        self.final_result_edit.setReadOnly(True)
        self.final_result_edit.setPlaceholderText("Final result will appear here...")
        results_layout.addRow("Final:", self.final_result_edit)

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        # Similarity matching group
        similarity_group = QGroupBox("Similarity Matching")
        similarity_layout = QVBoxLayout()

        # Threshold control
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Similarity Threshold:"))

        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setDecimals(2)
        self.threshold_spinbox.setSingleStep(0.05)
        self.threshold_spinbox.setValue(0.75)
        self.threshold_spinbox.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_spinbox)
        threshold_layout.addStretch()

        similarity_layout.addLayout(threshold_layout)

        # Grid layout for reference strings and similarity scores
        grid_layout = QGridLayout()
        grid_layout.setColumnStretch(0, 3)  # Label column wider
        grid_layout.setColumnStretch(1, 1)  # Spinbox column narrower

        # Header
        grid_layout.addWidget(QLabel("<b>Reference String</b>"), 0, 0)
        grid_layout.addWidget(QLabel("<b>Similarity</b>"), 0, 1)

        # Create rows for each reference string
        for i, ref_string in enumerate(REFERENCE_STRINGS, start=1):
            label = QLabel(ref_string)
            grid_layout.addWidget(label, i, 0)

            spinbox = QDoubleSpinBox()
            spinbox.setRange(0.0, 1.0)
            spinbox.setDecimals(4)
            spinbox.setReadOnly(True)
            spinbox.setButtonSymbols(QDoubleSpinBox.NoButtons)
            spinbox.setValue(0.0)
            grid_layout.addWidget(spinbox, i, 1)

            self.similarity_spinboxes.append(spinbox)

        similarity_layout.addLayout(grid_layout)

        # Fuzzy match result
        fuzzy_layout = QHBoxLayout()
        fuzzy_layout.addWidget(QLabel("<b>Fuzzy Match:</b>"))
        self.fuzzy_match_label = QLabel("—")
        fuzzy_layout.addWidget(self.fuzzy_match_label)
        fuzzy_layout.addStretch()
        similarity_layout.addLayout(fuzzy_layout)

        similarity_group.setLayout(similarity_layout)
        main_layout.addWidget(similarity_group)

        main_layout.addStretch()

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _update_device_list(self) -> None:
        """Update the device list combo box."""
        self.device_combo.clear()

        try:
            devices = get_audio_input_devices()

            if devices:
                for device in devices:
                    device_name = f"{device['index']}: {device['name']}"
                    self.device_combo.addItem(device_name, device['index'])
            else:
                self.device_combo.addItem("No input devices found", None)
        except Exception as e:
            print(f"Error enumerating devices: {e}")
            self.device_combo.addItem("Error loading devices", None)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        Converts to lowercase and removes spaces and periods.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        return text.lower().replace(" ", "").replace(".", "")

    def _update_similarity_scores(self, recognized_text: str) -> None:
        """Update similarity scores and fuzzy match for all reference strings.

        Args:
            recognized_text: The recognized text to compare against.
        """
        normalized_recognized = self._normalize_text(recognized_text)
        threshold = self.threshold_spinbox.value()

        for i, ref_string in enumerate(REFERENCE_STRINGS):
            normalized_ref = self._normalize_text(ref_string)
            similarity = string_similarity(normalized_recognized, normalized_ref)

            spinbox = self.similarity_spinboxes[i]
            spinbox.setValue(similarity)

            # Update color based on threshold
            palette = spinbox.palette()
            if similarity >= threshold:
                palette.setColor(QPalette.Text, QColor(0, 128, 0))  # Green
            else:
                palette.setColor(QPalette.Text, QColor(255, 0, 0))  # Red
            spinbox.setPalette(palette)

        # Fuzzy match against reference strings
        match = fuzzy_match_name(recognized_text, REFERENCE_STRINGS, threshold=threshold)
        if match is not None:
            self.fuzzy_match_label.setText(match)
            self.fuzzy_match_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.fuzzy_match_label.setText("—")
            self.fuzzy_match_label.setStyleSheet("color: red;")

    def _populate_models(self) -> None:
        """Populate the model combo box based on the selected STT engine."""
        engine = self.engine_combo.currentData() or "vosk"
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        if engine == "whisper":
            for name in self._WHISPER_MODELS:
                self.model_combo.addItem(name)
            # Default to small.en
            idx = self.model_combo.findText("small.en")
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        else:
            for model in self.available_models:
                self.model_combo.addItem(model)
            if self.default_model_index < self.model_combo.count():
                self.model_combo.setCurrentIndex(self.default_model_index)

        self.model_combo.blockSignals(False)

    def _start_recognizer(self) -> None:
        """Start speech recognizer with current engine, model and device selection."""
        # Stop existing recognizer
        if self.speech_recognizer is not None:
            self.speech_recognizer.stop()
            self.speech_recognizer = None

        # Get selected model
        model_name = self.model_combo.currentText()
        if not model_name:
            return

        # Get selected device index
        device_index = self.device_combo.currentData()
        if device_index is None:
            return

        engine = self.engine_combo.currentData() or "vosk"

        # Start new recognizer with selected engine, model and device
        try:
            if engine == "whisper":
                initial_prompt = self.initial_prompt_edit.text().strip()
                self.speech_recognizer = WhisperSpeechRecognizer(
                    model_size=model_name,
                    device_index=device_index,
                    initial_prompt=initial_prompt,
                )
            else:
                model_path = os.path.join(self.vosk_models_path, model_name)
                self.speech_recognizer = SpeechRecognizer(
                    model_path=model_path,
                    device_index=device_index,
                )

            self.speech_recognizer.partial_result.connect(self._on_partial_result)
            self.speech_recognizer.final_result.connect(self._on_final_result)
            self.speech_recognizer.error_occurred.connect(self._on_error_occurred)

            self.speech_recognizer.start()
            print(f"Started {engine} speech recognition with model '{model_name}' on device {device_index}")
        except Exception as e:
            print(f"Error starting speech recognizer: {e}")
            self.final_result_edit.setText(f"ERROR: {e}")

    @Slot(int)
    def _on_engine_changed(self, index: int) -> None:
        """Handle STT engine selection change.

        Args:
            index: New combo box index.
        """
        self._populate_models()
        self._start_recognizer()

    @Slot(int)
    def _on_model_changed(self, index: int) -> None:
        """Handle model selection change.

        Args:
            index: New combo box index.
        """
        self._start_recognizer()

    @Slot(str)
    def _on_initial_prompt_changed(self, text: str) -> None:
        """Handle initial prompt change (Whisper only).

        Updates the running Whisper recognizer's initial_prompt if possible,
        otherwise restarts the recognizer.

        Args:
            text: New initial prompt text.
        """
        if self.speech_recognizer is not None and isinstance(self.speech_recognizer, WhisperSpeechRecognizer):
            self.speech_recognizer._initial_prompt = text.strip()

    @Slot(int)
    def _on_device_changed(self, index: int) -> None:
        """Handle device selection change.

        Args:
            index: New combo box index.
        """
        self._start_recognizer()

    @Slot(str)
    def _on_partial_result(self, text: str) -> None:
        """Handle partial recognition result.

        Args:
            text: Partial recognition text.
        """
        self.partial_result_edit.setText(text)

    @Slot(str)
    def _on_final_result(self, text: str) -> None:
        """Handle final recognition result.

        Args:
            text: Final recognition text.
        """
        self.final_result_edit.setText(text)
        self._update_similarity_scores(text)

    @Slot(str)
    def _on_error_occurred(self, error_msg: str) -> None:
        """Handle error messages.

        Args:
            error_msg: The error message string.
        """
        print(f"Speech Recognition Error: {error_msg}")

    @Slot(float)
    def _on_threshold_changed(self, value: float) -> None:
        """Handle threshold change.

        Re-evaluates all similarity scores with new threshold.

        Args:
            value: New threshold value.
        """
        # Re-evaluate colors with new threshold
        if self.final_result_edit.text():
            self._update_similarity_scores(self.final_result_edit.text())

    def closeEvent(self, event) -> None:
        """Handle window close event.

        Args:
            event: The close event.
        """
        if self.speech_recognizer is not None:
            self.speech_recognizer.stop()

        event.accept()


def main() -> None:
    """Main entry point for the test application."""
    # Check for Vosk models directory
    vosk_models_path = os.path.join(root_dir_path, "vosk_models")

    available_models: list[str] = []
    if os.path.exists(vosk_models_path):
        try:
            for item in os.listdir(vosk_models_path):
                item_path = os.path.join(vosk_models_path, item)
                if os.path.isdir(item_path) and item.startswith("vosk-model"):
                    available_models.append(item)
        except Exception as e:
            print(f"Warning: Failed to list vosk_models directory: {e}")

    if not available_models:
        print("No Vosk models found — Whisper engine will still be available.")
        print("To use Vosk, download a model from https://alphacephei.com/vosk/models")
        print("and extract it to the vosk_models/ directory.")
    else:
        # Sort models to find the best default
        default_model_index = 0
        small_models = [i for i, model in enumerate(available_models) if "small" in model.lower()]

        if small_models:
            default_model_index = small_models[0]
            print(f"Found {len(available_models)} Vosk model(s), selecting 'small' model: {available_models[default_model_index]}")
        else:
            # No "small" model found, find the smallest by directory size
            model_sizes = []
            for model in available_models:
                model_path = os.path.join(vosk_models_path, model)
                total_size = 0
                try:
                    for dirpath, dirnames, filenames in os.walk(model_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            total_size += os.path.getsize(filepath)
                    model_sizes.append((model, total_size))
                except Exception as e:
                    print(f"Warning: Could not calculate size for {model}: {e}")
                    model_sizes.append((model, float('inf')))

            model_sizes.sort(key=lambda x: x[1])
            smallest_model = model_sizes[0][0]
            default_model_index = available_models.index(smallest_model)
            print(f"Found {len(available_models)} Vosk model(s), selecting smallest model: {smallest_model} ({model_sizes[0][1] / (1024 * 1024):.1f} MB)")

        print(f"Available Vosk models: {', '.join(available_models)}")

    default_model_index = default_model_index if available_models else 0

    app = QApplication(sys.argv)

    window = SpeechRecognitionTestWindow(
        available_models=available_models,
        default_model_index=default_model_index
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
