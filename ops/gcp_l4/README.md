# GCP GPU Runbook

This folder contains the GCP VM wrappers for the structured `grad-biovil-kd`
project. The final target machine is a single NVIDIA H100 80GB GPU
(`a3-highgpu-1g`). The same scripts also support A100 80GB, full RTX PRO 6000
96GB, and L4 fallback profiles.

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

```bash
cd "$GRAD_BIOVIL_ROOT"
bash project_repo/ops/gcp_l4/setup_l4_vm.sh
source "$HOME/venvs/grad-biovil-l4/bin/activate"
```

The setup script installs CUDA-enabled PyTorch and the project Python
dependencies. Confirm it prints `cuda_available True` and an NVIDIA L4 device.

## Preflight

Run this before training. It validates the structured project, checkpoints,
teacher artifacts, split files, MIMIC image paths, imports, and CUDA visibility:

```bash
python project_repo/ops/gcp_l4/preflight_l4.py \
  --output-json "$GRAD_BIOVIL_WORK/preflight_l4.json"
```

Do not start training until `overall_ok` is `true`.

## Optional Smoke Test

Run a short sanity check only if the VM or dataset layout is new. This is not
part of the final training campaign:

```bash
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

Run the final 8-epoch campaign once for all six selected models:

```bash
python project_repo/ops/gcp_l4/run_hard_negative_l4.py \
  --run-key all \
  --epochs 8 \
  --hardware-profile h100_80gb
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
5k/full retrieval logging after each epoch
last.pt, epoch_001.pt ... epoch_008.pt
best_val_loss.pt, best_5k_retrieval.pt, and best_full_retrieval.pt checkpoint selection
CUDA AMP on the L4
```

## Evaluation

After training:

```bash
python project_repo/ops/gcp_l4/evaluate_l4.py \
  --run-key all \
  --epochs 8 \
  --hardware-profile h100_80gb \
  --checkpoint-name best_5k_retrieval.pt \
  --candidate-pools 32,1000,5000,full \
  --seeds 42,43,44,45,46
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

## Hardware Profiles

The launcher defaults to `h100_80gb`:

```text
batch-size: 64
num-workers: 8
epoch retrieval batch-size: 256
epoch retrieval num-workers: 8
AMP dtype: bfloat16
epoch retrieval pools: 5000,full
hard negatives per sample: 8
```

Available profiles:

```text
h100_80gb           H100 80GB / a3-highgpu-1g
a100_80gb           A100 80GB / a2-ultragpu-1g
rtx_pro_6000_96gb   full RTX PRO 6000 / g4-standard-48
l4_24gb             NVIDIA L4 / g2-standard-4
```

For the expected final H100 run, use the profile defaults first. If VRAM usage
is clearly below capacity and CPU/disk throughput is healthy, the first manual
tuning knob is `--batch-size 96`. If the dataloader is the bottleneck, try
`--num-workers 12`. Keep all six selected models at 8 epochs and select final
checkpoints by retrieval metrics rather than by the last epoch.
