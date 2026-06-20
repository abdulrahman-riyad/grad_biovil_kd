#!/usr/bin/env bash
set -euo pipefail

# Run this from Google Cloud Shell to create the training VM.
#
# Example:
#   PROJECT_ID=project-e873d31b-f7ff-4085-bf7 bash create_l4_vm.sh

PROJECT_ID="${PROJECT_ID:-}"
INSTANCE_NAME="${INSTANCE_NAME:-grad-l4-train-01}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-300GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"

if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT_ID" ]]; then
  echo "ERROR: set PROJECT_ID or configure gcloud core/project." >&2
  exit 1
fi

gcloud config set project "$PROJECT_ID" >/dev/null

ZONES=(
  us-east1-b
  us-east1-c
  us-east1-d
  us-central1-a
  us-central1-b
  us-central1-c
  us-central1-f
  us-west1-a
  us-west1-b
  us-west1-c
  europe-west4-a
  europe-west4-b
  europe-west4-c
  europe-west1-b
  europe-west1-c
  asia-east1-a
  asia-east1-b
  asia-east1-c
)

STARTUP_SCRIPT="$(mktemp)"
cat > "$STARTUP_SCRIPT" <<'SCRIPT'
#!/bin/bash
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  exit 0
fi

sudo systemctl stop google-cloud-ops-agent || true
mkdir -p /opt/google/cuda-installer
cd /opt/google/cuda-installer
curl -fSsL -O https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
python3 cuda_installer.pyz install_driver --installation-branch=lts || python3 cuda_installer.pyz install_driver
SCRIPT

cleanup() {
  rm -f "$STARTUP_SCRIPT"
}
trap cleanup EXIT

for zone in "${ZONES[@]}"; do
  echo "===== Trying L4 / G2 in $zone ====="
  if gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$zone" \
    --machine-type="g2-standard-4" \
    --image-family="debian-12" \
    --image-project="debian-cloud" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --maintenance-policy="TERMINATE" \
    --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
    --scopes="https://www.googleapis.com/auth/cloud-platform" \
    --quiet; then
    echo
    echo "SUCCESS: $INSTANCE_NAME created in $zone"
    gcloud compute instances describe "$INSTANCE_NAME" \
      --project="$PROJECT_ID" \
      --zone="$zone" \
      --format="table(name,zone,status,machineType.basename())"
    echo
    echo "Use this zone for SSH and later commands:"
    echo "export PROJECT_ID=\"$PROJECT_ID\""
    echo "export ZONE=\"$zone\""
    echo "gcloud compute ssh \"$INSTANCE_NAME\" --project=\"\$PROJECT_ID\" --zone=\"\$ZONE\""
    exit 0
  fi
  echo "FAILED in $zone"
done

echo "ERROR: no L4 VM was created in the tested zones." >&2
exit 1
