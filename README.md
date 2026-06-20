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

## LLM Narrator (optional, GPU-accelerated)

The virtual narrator can use a local large language model through
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python). The app remains
fully functional without it (it falls back to scripted narration), but a CUDA
build enables fast, fully in-process generation on an NVIDIA GPU.

### Prerequisites (Windows, NVIDIA GPU)

- An NVIDIA GPU with a recent driver.
- **CUDA Toolkit 12.x** (e.g. 12.6) — https://developer.nvidia.com/cuda-downloads
- **Visual Studio 2022** with the *Desktop development with C++* workload (MSVC).
- A modern **CMake** and **Ninja** inside the environment (the system CMake is
  often too old and will fail CUDA detection):
  ```bash
  pip install "cmake>=3.29" ninja
  ```

### Build and install

There are no reliable prebuilt CUDA wheels for Python 3.13, so build from source.
Run the following from a **"x64 Native Tools Command Prompt for VS 2022"** (this
puts the MSVC compiler on `PATH`):

```bat
conda activate ttga_env
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDACXX=%CUDA_PATH%\bin\nvcc.exe
set CMAKE_GENERATOR=Ninja
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89
pip install --upgrade --no-cache-dir --force-reinstall llama-cpp-python
```

Notes:
- `CMAKE_CUDA_ARCHITECTURES=89` targets Ada GPUs (RTX 40-series) for a faster
  build. Adjust for your card (`86` = RTX 30-series, `75` = RTX 20-series), or
  omit to build for all architectures.
- The **Ninja** generator is recommended; the Visual Studio generator can fail
  CUDA compiler detection.
- The compile can take 15-25 minutes.

### Verify

```bash
python -c "from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

This should print `True` and list your GPU.

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

### LLM Models (GGUF)

If you built `llama-cpp-python` (see *LLM Narrator* above), the narrator loads
local GGUF model files. Place `*.gguf` files in the `models/` directory. A
`Q4_K_M` quantization is a good balance for ~8 GB VRAM.

Recommended models:
- **Llama 3.1 8B Instruct** (quality) — https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
- **Qwen2.5 7B Instruct** (strong JSON/structured output) — https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF
- **Llama 3.2 3B Instruct** (latency-first) — https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

### Vosk Speech Recognition Models

Download Vosk models for voice recognition functionality from:
https://alphacephei.com/vosk/models

See `vosk_models/_add_vosk_models_here.txt` for detailed instructions. Recommended models:
- **vosk-model-small-en-us-0.15** (40 MB) - Lightweight

Download and extract the model(s) directly into the `vosk_models/` directory.
