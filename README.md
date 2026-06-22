# grad_biovil_kd

Unified graduation project repository for efficient chest X-ray image-report
retrieval by distilling BioViL-T into compact student models.

The repository contains the maintained training and evaluation pipeline, project
baseline evaluations, earlier image-only and report-generation experiments, cloud
runbooks, and lightweight documentation. Large artifacts are stored outside git
in GCS or local experiment folders.

## Project Layout

```text
src/
  track_ab/                       Final image-report retrieval pipeline:
                                  contrastive KD, hard-negative training,
                                  teacher evaluation, student evaluation,
                                  efficiency, and Grad-CAM analysis.
  baselines/                      General-domain and medical-domain baseline
                                  retrieval evaluation scripts.
  student_training_experiments/   Full-student KD/HN training experiments
                                  exported from project notebook prototypes.
  kd_phase/                       Image-only knowledge-distillation experiments.
  week2_decoder/                  Image-to-report generation experiment.

ops/
  gcp_l4/                         GCP L4 VM creation, setup, preflight,
                                  training, evaluation, and sync runbooks.

docs/
  track_ab/                       Experiment notes and metric summaries.
  project_code_map.md             Unified map of source modules and notebook
                                  prototype provenance.
```

## Final GPU Target

The active cloud target is:

```text
g2-standard-4 with 1x NVIDIA L4 24GB
```

Use the L4/G2 runbook:

```bash
ops/gcp_l4/README.md
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

## Main Entry Points

- Final L4 training wrapper: `ops/gcp_l4/run_teacher_kd_hn_l4.py`
- Final L4 evaluation wrapper: `ops/gcp_l4/run_retrieval_eval_l4.py`
- Core full-student KD/HN trainer:
  `src/track_ab/train_teacher_kd_hn_full_student.py`
- Retrieval evaluator: `src/track_ab/evaluate_contrastive_retrieval.py`
- Baseline evaluators: `src/baselines/`
- Project notebook-derived full-student experiments:
  `src/student_training_experiments/`

## Artifact Policy

This git repository intentionally excludes large binary artifacts:

- checkpoints
- extracted embeddings
- data artifacts
- model weights
- training and evaluation outputs
- MIMIC-CXR image data

Store those artifacts in GCS or in the structured local project root documented
by the GCP runbooks.
