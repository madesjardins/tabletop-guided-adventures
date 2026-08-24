## Overview

This project aims to create an immersive tabletop gaming experience where a virtual narrator assists players in their adventures, enhancing traditional tabletop RPG solo sessions. The narrator uses at least one webcam to understand the game state and a projector to visually display its moves and storytelling elements.

# Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager).

Navigate to the repository root and create the environment with all dependencies:
```bash
uv sync
```

This automatically uses Python 3.12 and installs all required packages into a local `.venv`.

## LLM Narrator (optional, GPU-accelerated)

The virtual narrator can use a local large language model through
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python). The app remains
fully functional without it (it falls back to scripted narration), but a CUDA
build enables fast, fully in-process generation on an NVIDIA GPU.

`llama-cpp-python` is listed in `pyproject.toml` and installed automatically by
`uv sync`. However, building it with CUDA support requires the MSVC compiler and
CUDA Toolkit to be available in the environment.

### Prerequisites

- An NVIDIA GPU with a recent driver (CUDA 12.x).
- CUDA Toolkit 12.x (e.g. at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`).
- Visual Studio 2022 (MSVC) with the C++ build tools.
- CMake >= 3.29 and Ninja (install via `uv pip install cmake ninja` if needed).

### Install (CUDA build)

Use the provided `build_llama_uv.bat` script, which loads the MSVC environment,
sets the CUDA build flags, and runs `uv sync`:

```bat
build_llama_uv.bat
```

Or set the environment variables manually before running `uv sync`:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDACXX=nvcc
set CMAKE_GENERATOR=Ninja
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89
uv sync
```

### Verify

```bash
uv run python -c "from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

This should print `True` and list your GPU.

## Apply PyBoof Bug Fix

**Important:** PyBoof has a bug in its Windows mmap implementation that affects MicroQR detection. Apply the fix by running:

```bat
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

If you installed `llama-cpp-python` (see *LLM Narrator* above), the narrator loads
local GGUF model files. Place `*.gguf` files in the `models/` directory. A
`Q4_K_M` quantization is a good balance for ~6 GB VRAM at 8B parameters.

See `models/_add_gguf_models_here.txt` for detailed instructions. Recommended models:
- **Llama 3.1 8B Instruct** (quality) — https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
- **Qwen2.5 7B Instruct** (strong JSON/structured output) — https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF
- **Llama 3.2 3B Instruct** (latency-first, ~3 GB VRAM) — https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

### Speech Recognition Models

The app supports two STT engines, selectable at runtime from the Speech
Recognition tab:

- **Vosk** (default, CPU-based): Download Vosk models from
  https://alphacephei.com/vosk/models and extract them into the `vosk_models/`
  directory. See `vosk_models/_add_vosk_models_here.txt` for details.
  Recommended: **vosk-model-small-en-us-0.15** (40 MB).

- **Whisper** (via `faster-whisper`, GPU-accelerated): Models are downloaded
  automatically from HuggingFace on first use. Available sizes: `tiny.en`,
  `base.en`, `small.en`, `medium.en`. Select the model size from the dropdown
  in the Speech Recognition tab. For faster downloads and higher rate limits,
  create a `.env` file (see `.env.example`) with a HuggingFace token:
  ```
  HF_TOKEN=hf_your_token_here
  ```
  Get a free read-only token at https://huggingface.co/settings/tokens.
