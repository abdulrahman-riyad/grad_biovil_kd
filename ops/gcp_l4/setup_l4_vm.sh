#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere on the GCP VM after the structured project has been copied.
# Example:
#   bash project_repo/ops/gcp_l4/setup_l4_vm.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/grad-biovil-l4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# CUDA 12.1 wheels work on standard GCP L4 driver images. If your image ships a
# different CUDA/driver stack, use the matching command from pytorch.org.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python -m pip install -r "$SCRIPT_DIR/requirements-gcp-l4.txt"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY
