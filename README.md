# tabletop-guided-adventures

TTGA (Tabletop Guided Adventures) uses a virtual narrator to guide players through interactive storytelling sessions. The narrator uses at least one webcam to understand the game state and a projector to visually display its moves or storytelling elements.

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
