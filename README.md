## Overview

This project aims to create an immersive tabletop gaming experience where a virtual narrator assists players in their adventures, enhancing traditional tabletop RPG solo sessions. The narrator uses at least one webcam to understand the game state and a projector to visually display its moves and storytelling elements.

# Installation

Install miniconda [from official website](https://www.anaconda.com/download).

Create the conda environment and activate it:
```bash
conda create -n ttga_env python=3.13 -c conda-forge
conda activate ttga_env
```

Navigate to the repository root and install the project-specific dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

## Download Required Assets

### Piper TTS Voices

Download Piper voice models for text-to-speech functionality. See `piper_voices/_add_piper_voices_here.txt` for detailed instructions and download links.

Place the downloaded `.onnx` and `.onnx.json` files in the `piper_voices/` directory.

### Camera Calibration Images

Download camera calibration test images from:
https://drive.google.com/drive/folders/1hNMYz9KV5h4m3BCLkFu0OBJlWMcWAwb7?usp=drive_link

Place all files from the Google Drive folder into the `tests/images/` directory. These images are required for running the camera calibration integration tests.
