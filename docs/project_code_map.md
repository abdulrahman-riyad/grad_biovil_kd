# Project Code Map

This repository is organized by project function rather than by contributor or
notebook origin.

## Final Retrieval Pipeline

`src/track_ab/` contains the maintained image-report retrieval implementation:

- dataset loading and transforms;
- MobileViT and RepViT student loading;
- text-student selection;
- contrastive KD and teacher-feature imitation;
- hard-negative mining and fine-tuning;
- full-gallery retrieval evaluation;
- efficiency benchmarking;
- Grad-CAM analysis.

The GCP L4 wrappers in `ops/gcp_l4/` call this pipeline for the final cloud
training and evaluation runs.

## Baseline Evaluation Scripts

`src/baselines/` contains project baseline evaluation scripts converted from the
baseline notebook prototypes. These scripts cover:

- CLIP ViT-B/14, ViT-B/16, and ViT-B/32;
- MobileCLIP B, S0, S1, and S2;
- TinyCLIP variants;
- BiomedCLIP;
- MedCLIP;
- CXR-CLIP;
- MGCA;
- the retrieval aggregation/evaluation pipeline.

Each converted script keeps the original source-notebook path in its header so
results remain traceable.

## Student Training Experiments

`src/student_training_experiments/` contains project training experiments
converted from notebook prototypes for:

- MobileViT + DistilBioBERT KD/HN training;
- RepViT + DistilBioBERT KD/HN training.

These files preserve the original notebook sections as comments. The maintained
production-facing training entry point remains
`src/track_ab/train_teacher_kd_hn_full_student.py`.

## Earlier Project Phases

`src/kd_phase/` contains image-only knowledge-distillation experiments.

`src/week2_decoder/` contains the image-to-report generation experiment.

The notebooks under `notebooks/` are retained as historical project material;
the corresponding project code lives under `src/`.
