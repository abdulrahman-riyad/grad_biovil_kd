# GCP GPU Runbook

This folder contains the GCP VM wrappers for the structured `grad-biovil-kd`
project. The active target machine is a single NVIDIA L4 24GB GPU:

```text
g2-standard-4
```

H100, RTX PRO 6000, and A100 80GB were not selected for this run because of
Flex-start waiting, stockout, or missing regional quota. Keep this runbook
L4-only unless a new GPU provisioning pass proves a better machine is available.

## Create The L4 VM

Run this from Google Cloud Shell:

```bash
export PROJECT_ID="project-e873d31b-f7ff-4085-bf7"
curl -fsSL \
  https://raw.githubusercontent.com/abdulrahman-riyad/grad_biovil_kd/main/ops/gcp_l4/create_l4_vm.sh \
  -o create_l4_vm.sh
bash create_l4_vm.sh
```

The script tries common G2/L4 zones and stops at the first successful VM. It
uses Debian 12 and installs the Google-tested NVIDIA driver from a startup
script. If the startup script has not finished by the time you SSH in, rerun the
setup step below; it will install the driver and reboot once if needed.

## Connect To The VM

Run this from Google Cloud Shell using the zone printed by the create script:

```bash
export PROJECT_ID="project-e873d31b-f7ff-4085-bf7"
export ZONE="us-east1-b"

gcloud compute ssh "grad-l4-train-01" \
  --project="$PROJECT_ID" \
  --zone="$ZONE"
```

Everything below is run inside the VM shell, not in Cloud Shell.

## Bootstrap The VM From GitHub And GCS

The bootstrap script installs system dependencies, installs Google Cloud CLI if
needed, clones or updates the GitHub repository, syncs the structured artifacts
from GCS, syncs the extracted MIMIC-CXR dataset from GCS, writes an environment
file, and installs the Python/CUDA environment.

Run this inside the VM:

```bash
export BUCKET="project-e873d31b-f7ff-4085-bf7-ml-data-europe-west4"

curl -fsSL \
  https://raw.githubusercontent.com/abdulrahman-riyad/grad_biovil_kd/main/ops/gcp_l4/bootstrap_l4_vm.sh \
  -o bootstrap_l4_vm.sh

bash bootstrap_l4_vm.sh
```

If the setup installs or changes the NVIDIA driver, the VM may reboot. Reconnect
with the SSH command above, then continue with:

```bash
source "$HOME/grad_biovil_env.sh"
source "$HOME/venvs/grad-biovil-l4/bin/activate"
```

## Expected Directory Layout

Set these environment variables on the VM:

```bash
export GRAD_BIOVIL_ROOT="$HOME/grad-biovil-kd"
export MIMIC_CXR_ROOT="$HOME/mimic-cxr-dataset"
export GRAD_BIOVIL_WORK="$HOME/grad-biovil-runs"
```

`GRAD_BIOVIL_ROOT` must point to the organized folder containing:

```text
checkpoints/
data_artifacts/
models/
project_repo/
results/
```

`MIMIC_CXR_ROOT` must point to the main MIMIC-CXR Kaggle dataset mirror. The
image root is expected at:

```text
$MIMIC_CXR_ROOT/official_data_iccv_final/files
```

If your image root is elsewhere, set:

```bash
export MIMIC_CXR_IMAGE_ROOT="/absolute/path/to/official_data_iccv_final/files"
```

## Environment Setup

If you used `bootstrap_l4_vm.sh`, this step has already been run. Use it only
when manually maintaining an already-synced VM.

```bash
cd "$GRAD_BIOVIL_ROOT"
bash project_repo/ops/gcp_l4/setup_l4_vm.sh
source "$HOME/venvs/grad-biovil-l4/bin/activate"
```

The setup script installs CUDA-enabled PyTorch and the project Python
dependencies. If `nvidia-smi` is missing, it installs the NVIDIA driver first
and reboots; reconnect and rerun the setup command. Confirm the final run prints
`cuda_available True` and an NVIDIA L4 device.

## Preflight

Run this before training. It validates the structured project, checkpoints,
teacher artifacts, split files, MIMIC image paths, imports, and CUDA visibility:

```bash
source "$HOME/grad_biovil_env.sh"
source "$HOME/venvs/grad-biovil-l4/bin/activate"

python project_repo/ops/gcp_l4/preflight_l4.py \
  --output-json "$GRAD_BIOVIL_WORK/preflight_l4.json"
```

Do not start training until `overall_ok` is `true`.

## Optional Smoke Test

Run a short sanity check only if the VM or dataset layout is new. This is not
part of the final training campaign:

```bash
source "$HOME/grad_biovil_env.sh"
source "$HOME/venvs/grad-biovil-l4/bin/activate"

python project_repo/ops/gcp_l4/run_hard_negative_l4.py \
  --run-key mobilevit_clinical_distilbert \
  --smoke
```

This also precomputes the teacher-guided hard-negative file if missing:

```text
$GRAD_BIOVIL_WORK/hard_negatives/biovil_teacher_train_top64_fn085_min060.npz
```

The hard-negative file name is parameter-specific. If you change the top-k or
thresholds, a new file is created.

## Full Training

Run the final 6-epoch campaign once for all six selected models:

```bash
source "$HOME/grad_biovil_env.sh"
source "$HOME/venvs/grad-biovil-l4/bin/activate"

python project_repo/ops/gcp_l4/run_hard_negative_l4.py \
  --run-key all \
  --epochs 6 \
  --hardware-profile l4_24gb
```

The six runs execute in this order:

```text
mobilevit_clinical_distilbert
repvit_clinical_distilbert
mobilevit_distil_biobert
repvit_distil_biobert
mobilevit_biovil_t
repvit_biovil_t
```

The BioViL-T teacher vision encoder + BioViL-T teacher text encoder is not
trained. It is evaluated as the seventh comparison row during evaluation using
the precomputed teacher image/text embeddings.

The launcher uses:

```text
init from previous non-hard-negative best.pt
denominator-based hard-negative InfoNCE
false-negative-aware hard-negative mining
soft multi-positive targets
simple disease/anatomy pseudo-label loss
longitudinal consistency when prior same-subject text exists
uncertainty regularization
5k retrieval logging after each epoch
last.pt, epoch_001.pt ... epoch_006.pt
best_val_loss.pt and best_5k_retrieval.pt checkpoint selection during training
CUDA AMP on the selected GPU
```

## Evaluation

After training:

```bash
source "$HOME/grad_biovil_env.sh"
source "$HOME/venvs/grad-biovil-l4/bin/activate"

python project_repo/ops/gcp_l4/evaluate_l4.py \
  --run-key all \
  --epochs 6 \
  --hardware-profile l4_24gb \
  --checkpoint-name best_5k_retrieval.pt \
  --candidate-pools 32,1000,5000,full \
  --seeds 42,43,44,45,46 \
  --similarity-chunk-size 1024
```

Outputs are written under:

```text
$GRAD_BIOVIL_WORK/eval_hard_negative_l4/
```

The compact summary table is:

```text
$GRAD_BIOVIL_WORK/eval_hard_negative_l4/table13_style_summary.csv
```

The evaluator writes:

```text
retrieval_summary_raw.csv          one row per run/pool/seed
retrieval_summary_aggregated.csv   mean/std grouped by run and pool
table13_style_summary.csv          compact reporting table
```

The sampled pools `32`, `1000`, and `5000` are evaluated over five seeds. The
full pool is evaluated once because it is deterministic and expensive.

## Hardware Profile

The launcher defaults to `l4_24gb`:

```text
batch-size: 16
num-workers: 4
epoch retrieval batch-size: 96
epoch retrieval num-workers: 4
AMP dtype: bfloat16
epoch retrieval pools: 5000
hard negatives per sample: 8
```

These values are intentionally conservative for a 24GB GPU. If the smoke test
and first full run are stable, you can try higher explicit overrides such as:

```bash
python project_repo/ops/gcp_l4/run_hard_negative_l4.py \
  --run-key mobilevit_clinical_distilbert \
  --epochs 6 \
  --batch-size 24 \
  --epoch-retrieval-batch-size 128
```

If this OOMs, return to the `l4_24gb` defaults. Keep all six selected models at
6 epochs. Training selects the best checkpoint by 5k retrieval; the final
evaluation then runs 32, 1000, 5000, and full-pool retrieval on the selected
checkpoints.
