# grad_biovil_kd

Code for the BioViL-T student distillation and image-text retrieval pipeline.

This repository intentionally contains code, notebooks, runbooks, and lightweight
documentation only. Large artifacts are stored separately in GCS:

- `checkpoints/`
- `data_artifacts/`
- `models/`
- training/evaluation outputs
- MIMIC-CXR image data

## GCP GPU Entry Point

Use the L4/G2 GCP runbook:

```bash
project_repo/ops/gcp_l4/README.md
```

The active VM target is:

```text
g2-standard-4 with 1x NVIDIA L4 24GB
```

Expected VM environment variables:

```bash
export GRAD_BIOVIL_ROOT="$HOME/grad-biovil-kd"
export MIMIC_CXR_ROOT="$HOME/mimic-cxr-dataset"
export GRAD_BIOVIL_WORK="$HOME/grad-biovil-runs"
```

The structured project root on the VM should contain:

```text
checkpoints/
data_artifacts/
models/
project_repo/
results/
```

The active training code is under:

```text
src/track_ab/
ops/gcp_l4/
```
