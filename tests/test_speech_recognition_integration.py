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

This script provides a GUI application to test the SpeechRecognizer class with
real-time speech recognition and string similarity comparison.
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
    get_audio_input_devices,
    string_similarity
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
    """Main window for testing SpeechRecognizer with similarity matching."""

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
        self.speech_recognizer: SpeechRecognizer | None = None
        self.similarity_spinboxes: list[QDoubleSpinBox] = []
        self.threshold_spinbox: QDoubleSpinBox | None = None

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

        # Vosk model selection
        self.model_combo = QComboBox()
        for model in self.available_models:
            self.model_combo.addItem(model)
        # Set default model before connecting signal to avoid premature initialization
        self.model_combo.setCurrentIndex(self.default_model_index)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        config_layout.addRow("Vosk Model:", self.model_combo)

        # Audio device selection
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        config_layout.addRow("Audio Device:", self.device_combo)

        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

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
        """Update similarity scores for all reference strings.

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

    def _start_recognizer(self) -> None:
        """Start speech recognizer with current model and device selection."""
        # Stop existing recognizer
        if self.speech_recognizer is not None:
            self.speech_recognizer.stop()
            self.speech_recognizer = None

        # Get selected model
        model_name = self.model_combo.currentText()
        if not model_name:
            return

        model_path = os.path.join(self.vosk_models_path, model_name)

        # Get selected device index
        device_index = self.device_combo.currentData()
        if device_index is None:
            return

        # Start new recognizer with selected model and device
        try:
            self.speech_recognizer = SpeechRecognizer(
                model_path=model_path,
                device_index=device_index
            )
            self.speech_recognizer.partial_result.connect(self._on_partial_result)
            self.speech_recognizer.final_result.connect(self._on_final_result)
            self.speech_recognizer.error_occurred.connect(self._on_error_occurred)

            self.speech_recognizer.start()
            print(f"Started speech recognition with model '{model_name}' on device {device_index}")
        except Exception as e:
            print(f"Error starting speech recognizer: {e}")
            self.final_result_edit.setText(f"ERROR: {e}")

    @Slot(int)
    def _on_model_changed(self, index: int) -> None:
        """Handle model selection change.

        Args:
            index: New combo box index.
        """
        self._start_recognizer()

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

    if not os.path.exists(vosk_models_path):
        print(f"ERROR: Vosk models directory not found at {vosk_models_path}")
        print("Please create the vosk_models directory and download Vosk models.")
        print("See vosk_models/_add_vosk_models_here.txt for instructions.")
        print("Download from: https://alphacephei.com/vosk/models")
        return 1

    # Detect available models
    available_models = []
    try:
        for item in os.listdir(vosk_models_path):
            item_path = os.path.join(vosk_models_path, item)
            # Check if it's a directory and not the instruction file
            if os.path.isdir(item_path) and item.startswith("vosk-model"):
                available_models.append(item)
    except Exception as e:
        print(f"ERROR: Failed to list vosk_models directory: {e}")
        return 1

    if not available_models:
        print(f"ERROR: No Vosk models found in {vosk_models_path}")
        print("Please download a Vosk model and extract it to the vosk_models directory.")
        print("See vosk_models/_add_vosk_models_here.txt for instructions.")
        print("Download from: https://alphacephei.com/vosk/models")
        return 1

    # Sort models to find the best default
    # First, try to find a model with "small" in the name
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

        # Sort by size and get the smallest
        model_sizes.sort(key=lambda x: x[1])
        smallest_model = model_sizes[0][0]
        default_model_index = available_models.index(smallest_model)
        print(f"Found {len(available_models)} Vosk model(s), selecting smallest model: {smallest_model} ({model_sizes[0][1] / (1024 * 1024):.1f} MB)")

    print(f"Available models: {', '.join(available_models)}")

    app = QApplication(sys.argv)

    window = SpeechRecognitionTestWindow(
        available_models=available_models,
        default_model_index=default_model_index
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
