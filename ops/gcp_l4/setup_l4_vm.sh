#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere on the GCP VM after the structured project has been copied.
# Example:
#   bash project_repo/ops/gcp_l4/setup_l4_vm.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/grad-biovil-l4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
  echo "python3 venv support is missing; installing base Python tools."
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-pip
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available; installing the Google-tested NVIDIA driver."
  echo "The installer can take several minutes and might require a reboot."
  sudo systemctl stop google-cloud-ops-agent || true
  mkdir -p /opt/google/cuda-installer
  cd /opt/google/cuda-installer
  curl -fSsL -O https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
  sudo python3 cuda_installer.pyz install_driver --installation-branch=lts || sudo python3 cuda_installer.pyz install_driver
  echo "Driver installation finished. Rebooting now; reconnect and rerun this script."
  sudo reboot
  exit 0
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# CUDA 12.1 wheels work with the Google-tested L4 driver installed above.
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
