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

## Apply PyBoof Bug Fix

**Important:** PyBoof has a bug in its Windows mmap implementation that affects MicroQR detection. Apply the fix by running:

```bash
apply_pyboof_patch.bat
```

This script will:
- Automatically detect your PyBoof installation location
- Check if the patch is already applied
- Create a backup of the original file
- Apply the fix to enable MicroQR code detection

The script is safe to run multiple times - it will detect if the patch is already applied and skip the operation.

## Download Required Assets

### Piper TTS Voices

Download Piper voice models for text-to-speech functionality. See `piper_voices/_add_piper_voices_here.txt` for detailed instructions and download links.

Place the downloaded `.onnx` and `.onnx.json` files in the `piper_voices/` directory.

### Test Images

Download test images (camera calibration, QR detection, etc.) from:
https://drive.google.com/drive/folders/1hNMYz9KV5h4m3BCLkFu0OBJlWMcWAwb7?usp=drive_link

See `tests/images/_add_test_images_here.txt` for detailed information about available test images.

Place all files from the Google Drive folder into the `tests/images/` directory. These images are required for running various integration tests.

### MicroQR

Download MicroQR folder (images to print) from:
https://drive.google.com/drive/folders/1XY2nEVhngzmPuN3-VRPTqTm4tHSvWqTZ?usp=drive_link

Place the downloaded MicroQR folder in 'images' directory.

### Vosk Speech Recognition Models

Download Vosk models for voice recognition functionality from:
https://alphacephei.com/vosk/models

See `vosk_models/_add_vosk_models_here.txt` for detailed instructions. Recommended models:
- **vosk-model-small-en-us-0.15** (40 MB) - Lightweight

Download and extract the model(s) directly into the `vosk_models/` directory.
