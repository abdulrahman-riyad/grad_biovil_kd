#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh GCP L4/G2 VM for the grad_biovil_kd training pipeline.
#
# Run this on the VM, not in Cloud Shell:
#   curl -fsSL https://raw.githubusercontent.com/abdulrahman-riyad/grad_biovil_kd/main/ops/gcp_l4/bootstrap_l4_vm.sh -o bootstrap_l4_vm.sh
#   bash bootstrap_l4_vm.sh

BUCKET="${BUCKET:-project-e873d31b-f7ff-4085-bf7-ml-data-europe-west4}"
REPO_URL="${REPO_URL:-https://github.com/abdulrahman-riyad/grad_biovil_kd.git}"
BRANCH="${BRANCH:-main}"

GRAD_BIOVIL_ROOT="${GRAD_BIOVIL_ROOT:-$HOME/grad-biovil-kd}"
GRAD_BIOVIL_WORK="${GRAD_BIOVIL_WORK:-$HOME/grad-biovil-runs}"
MIMIC_CXR_ROOT="${MIMIC_CXR_ROOT:-$HOME/mimic-cxr-dataset}"
PROJECT_REPO="$GRAD_BIOVIL_ROOT/project_repo"

echo "=== Paths ==="
echo "BUCKET=$BUCKET"
echo "REPO_URL=$REPO_URL"
echo "BRANCH=$BRANCH"
echo "GRAD_BIOVIL_ROOT=$GRAD_BIOVIL_ROOT"
echo "GRAD_BIOVIL_WORK=$GRAD_BIOVIL_WORK"
echo "MIMIC_CXR_ROOT=$MIMIC_CXR_ROOT"

echo
echo "=== System packages ==="
sudo apt-get update
sudo apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  git \
  gnupg \
  python3-pip \
  python3-venv \
  rsync \
  tmux \
  unzip

if ! command -v gcloud >/dev/null 2>&1; then
  echo
  echo "=== Installing Google Cloud CLI ==="
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y google-cloud-cli
fi

echo
echo "=== Create directories ==="
mkdir -p \
  "$GRAD_BIOVIL_ROOT" \
  "$GRAD_BIOVIL_WORK" \
  "$MIMIC_CXR_ROOT"

echo
echo "=== Clone or update GitHub repo ==="
if [[ -d "$PROJECT_REPO/.git" ]]; then
  git -C "$PROJECT_REPO" fetch origin "$BRANCH"
  git -C "$PROJECT_REPO" checkout "$BRANCH"
  git -C "$PROJECT_REPO" pull --ff-only origin "$BRANCH"
else
  rm -rf "$PROJECT_REPO"
  git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_REPO"
fi

echo
echo "=== Sync structured artifacts from GCS ==="
gcloud storage rsync -r \
  "gs://$BUCKET/artifacts/grad-biovil-kd/checkpoints" \
  "$GRAD_BIOVIL_ROOT/checkpoints"

gcloud storage rsync -r \
  "gs://$BUCKET/artifacts/grad-biovil-kd/data_artifacts" \
  "$GRAD_BIOVIL_ROOT/data_artifacts"

gcloud storage rsync -r \
  "gs://$BUCKET/artifacts/grad-biovil-kd/models" \
  "$GRAD_BIOVIL_ROOT/models"

gcloud storage rsync -r \
  "gs://$BUCKET/artifacts/grad-biovil-kd/results_baseline" \
  "$GRAD_BIOVIL_ROOT/results"

echo
echo "=== Sync MIMIC-CXR extracted dataset from GCS ==="
gcloud storage rsync -r \
  "gs://$BUCKET/datasets/mimic-cxr-dataset/extracted" \
  "$MIMIC_CXR_ROOT"

echo
echo "=== Write environment file ==="
cat > "$HOME/grad_biovil_env.sh" <<EOF
export BUCKET="$BUCKET"
export GRAD_BIOVIL_ROOT="$GRAD_BIOVIL_ROOT"
export GRAD_BIOVIL_WORK="$GRAD_BIOVIL_WORK"
export MIMIC_CXR_ROOT="$MIMIC_CXR_ROOT"
export MIMIC_CXR_IMAGE_ROOT="$MIMIC_CXR_ROOT/official_data_iccv_final/files"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

echo
echo "=== Install Python environment and GPU dependencies ==="
source "$HOME/grad_biovil_env.sh"
bash "$PROJECT_REPO/ops/gcp_l4/setup_l4_vm.sh"

echo
echo "=== Bootstrap complete ==="
echo "Reconnect if the setup script installed drivers and rebooted the VM."
echo "Then run:"
echo "  source \"\$HOME/grad_biovil_env.sh\""
echo "  source \"\$HOME/venvs/grad-biovil-l4/bin/activate\""
echo "  python \"\$GRAD_BIOVIL_ROOT/project_repo/ops/gcp_l4/preflight_l4.py\" --output-json \"\$GRAD_BIOVIL_WORK/preflight_l4.json\""
