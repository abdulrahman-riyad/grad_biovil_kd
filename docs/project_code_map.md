# Project Code Map

This repository is organized by project function rather than by contributor,
week number, or notebook origin.

## Final BioViL-T Retrieval Distillation

`src/biovil_t_retrieval_distillation/` is the main thesis pipeline. It contains:

- dataset loading and image transforms;
- MobileViT and RepViT image-student loading;
- DistilBioBERT and ClinicalDistilBERT text-student loading;
- Stage 1 contrastive knowledge distillation;
- Stage 2 hard-negative fine-tuning;
- full-gallery student and teacher retrieval evaluation;
- efficiency benchmarking;
- Grad-CAM++ analysis;
- a central final-run registry for all four thesis student configurations.

The central run registry is:

```text
src/biovil_t_retrieval_distillation/configs/final_student_runs.py
```

It defines:

```text
MobileViT + ClinicalDistilBERT
RepViT + ClinicalDistilBERT
MobileViT + DistilBioBERT
RepViT + DistilBioBERT
```

The converted training notebooks are retained under
`src/biovil_t_retrieval_distillation/notebook_training_exports/` as reference
implementations. They are not the source of truth for model coverage; the run
registry and parameterized training scripts are.

## Baseline Evaluations

`src/baseline_evaluations/` contains the general-domain and medical-domain
retrieval baseline scripts:

- CLIP ViT-B/14, ViT-B/16, and ViT-B/32;
- MobileCLIP B, S0, S1, and S2;
- TinyCLIP variants;
- BiomedCLIP;
- MedCLIP;
- CXR-CLIP;
- MGCA;
- CheXzero and ConVIRT;
- embedding-folder retrieval aggregation.

Each converted script keeps the original source-notebook path in its header so
the implementation remains traceable.

## Earlier Project Components

`src/image_encoder_distillation/` contains the earlier image-only
knowledge-distillation experiments.

`src/report_generation/` contains the image-to-report generation experiment.

`notebooks/` contains retained historical notebooks. The project code lives
under `src/`.

## Cloud Operations

`ops/gcp_l4/` contains the L4/G2 VM workflow:

- VM creation and setup;
- preflight checks;
- final student training;
- Stage 2 hard-negative training;
- retrieval evaluation;
- single-run train/evaluate/sync execution.
