# grad_biovil_kd

Unified graduation project repository for efficient chest X-ray image-report
retrieval by distilling BioViL-T into compact student models.

The repository is organized by project function. The final thesis pipeline lives
in one package, and all four final student configurations are defined together
in `src/biovil_t_retrieval_distillation/configs/final_student_runs.py`.

## Project Layout

```text
src/
  biovil_t_retrieval_distillation/
    configs/final_student_runs.py        Four final thesis student runs.
    train_full_student_kd_hn.py          End-to-end Stage 1 + Stage 2 training.
    train_stage1_contrastive.py          Stage 1 contrastive KD training.
    train_stage2_hard_negative.py        Stage 2 hard-negative fine-tuning.
    evaluate_student_retrieval.py        Student retrieval evaluation.
    evaluate_teacher_retrieval.py        BioViL-T teacher retrieval evaluation.
    mine_hard_negatives.py               Teacher-guided hard-negative mining.
    benchmark_model_efficiency.py        FLOPs, params, and latency checks.
    explain_gradcam.py                   Grad-CAM++ analysis.
    notebook_training_exports/           Converted training notebooks retained
                                         as reference implementations.

  baseline_evaluations/                  General-domain and medical-domain
                                         retrieval baseline scripts.
  image_encoder_distillation/            Earlier image-only distillation work.
  report_generation/                     Image-to-report generation experiment.

ops/
  gcp_l4/                                L4 VM setup, preflight, training,
                                         evaluation, and sync wrappers.

docs/
  retrieval_distillation/                Notes and summaries for the final
                                         retrieval-distillation pipeline.
  project_code_map.md                    Source layout and provenance map.
```

## Final Student Runs

The final retrieval-distillation pipeline supports the four thesis student
configurations:

```text
mobilevit_clinical_distilbert  -> MobileViT + ClinicalDistilBERT
repvit_clinical_distilbert     -> RepViT + ClinicalDistilBERT
mobilevit_distil_biobert       -> MobileViT + DistilBioBERT
repvit_distil_biobert          -> RepViT + DistilBioBERT
```

These are the run keys used by the GCP L4 wrappers and by the final run
registry.

## GCP GPU Target

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

- Final L4 full-student training wrapper:
  `ops/gcp_l4/train_final_students_l4.py`
- L4 Stage 2 hard-negative wrapper:
  `ops/gcp_l4/train_stage2_hard_negatives_l4.py`
- L4 retrieval evaluation wrapper:
  `ops/gcp_l4/evaluate_retrieval_l4.py`
- Single-run L4 train/evaluate/sync wrapper:
  `ops/gcp_l4/run_single_model_l4.py`
- Final run registry:
  `src/biovil_t_retrieval_distillation/configs/final_student_runs.py`

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
