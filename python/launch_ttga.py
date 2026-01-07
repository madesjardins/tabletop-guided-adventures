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

import sys
import os

from PySide6 import QtWidgets, QtGui, QtCore

root_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_path = os.path.join(root_dir_path, "python")
if python_path not in sys.path:
    sys.path.append(python_path)

from ttga.constants import TEMP_DIR_PATH  # noqa: E402
from ttga.main_core import MainCore  # noqa: E402
from ttga.main_window import MainWindow  # noqa: E402

os.makedirs(TEMP_DIR_PATH, exist_ok=True)

app = QtWidgets.QApplication(sys.argv)

# Dark theme Palette
app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
palette = QtGui.QPalette()
silver = QtGui.QColor(192, 192, 192)
dark = QtGui.QColor(25, 25, 25)
palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(64, 64, 64))
palette.setColor(QtGui.QPalette.ColorRole.WindowText, silver)
palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(48, 48, 48))
palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(64, 64, 64))
palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, dark)
palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, silver)
palette.setColor(QtGui.QPalette.ColorRole.Text, silver)
palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(64, 64, 64))
palette.setColor(QtGui.QPalette.ColorRole.ButtonText, silver)
palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtCore.Qt.GlobalColor.red)
palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, dark)
app.setPalette(palette)

app.setStyleSheet(
    "QPushButton:disabled {background-color: rgb(86, 64, 64);}\n"
    "QComboBox:disabled {background-color: rgb(86, 64, 64);}\n"
    "QSlider:disabled {background-color: rgb(86, 64, 64);}\n"
    "QLineEdit:disabled {color: rgb(86, 112, 164);}\n"
)

# Create main core and window
core = MainCore()
window = MainWindow(core)
window.show()

# Run application
sys.exit(app.exec())
