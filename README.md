# gnome-repro-structural

Faithful small-scale reproduction of the GNoME structural GNN
(Merchant et al., Nature 624, 2023) for predicting formation energies
of inorganic crystals from structure.

Target: 30–50 meV/atom MAE on the Matbench Discovery MP test split.

## Environment setup

Tested on Python 3.11, Linux + macOS.

```bash
# 1. Create venv
python -m venv .venv
source .venv/bin/activate         # (or .venv\Scripts\activate on Windows)

# 2. Install PyTorch (CPU build for laptop; pick CUDA build for GPU box)
# Laptop:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU machine (e.g. CUDA 12.6):
# pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3. Install this project + its ML extras
pip install -e ".[ml,dev]"

# 4. Configure local paths
cp .env.example .env
# edit .env if needed
```

## Data setup

```bash
python scripts/download_data.py
```

Downloads the Matbench Discovery MP relaxed structures + energies (~1.3 GB)
to the matbench-discovery package's local cache. This is a one-time step.

## Reproducing Phase 1

See `results/phase1_report.md` for headline numbers and ablations.
Training scripts in `scripts/`. Configs in `configs/`.