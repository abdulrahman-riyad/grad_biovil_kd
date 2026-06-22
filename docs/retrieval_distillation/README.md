# Retrieval Distillation Notes

This folder documents the final BioViL-T image-report retrieval distillation
pipeline.

The maintained implementation is:

```text
src/biovil_t_retrieval_distillation/
```

The final student run registry is:

```text
src/biovil_t_retrieval_distillation/configs/final_student_runs.py
```

## Final Student Configurations

```text
mobilevit_clinical_distilbert  -> MobileViT + ClinicalDistilBERT
repvit_clinical_distilbert     -> RepViT + ClinicalDistilBERT
mobilevit_distil_biobert       -> MobileViT + DistilBioBERT
repvit_distil_biobert          -> RepViT + DistilBioBERT
```

These four run keys are used by the L4 wrappers in `ops/gcp_l4/`.

## Main Scripts

```text
train_full_student_kd_hn.py       End-to-end Stage 1 + Stage 2 training.
train_stage1_contrastive.py       Stage 1 contrastive KD.
train_stage2_hard_negative.py     Stage 2 hard-negative fine-tuning.
mine_hard_negatives.py            Teacher-guided hard-negative mining.
evaluate_student_retrieval.py     Student retrieval evaluation.
evaluate_teacher_retrieval.py     BioViL-T teacher retrieval evaluation.
benchmark_model_efficiency.py     FLOPs, parameters, and latency.
explain_gradcam.py                Grad-CAM++ analysis.
export_retrieval_checkpoint.py    Deployment checkpoint export.
```

## Cloud Entry Points

Use the GCP L4 runbook at:

```text
ops/gcp_l4/README.md
```

Primary wrappers:

```text
train_final_students_l4.py
train_stage2_hard_negatives_l4.py
evaluate_retrieval_l4.py
run_single_model_l4.py
```

## Historical Notes

Older pilot and tracking notes are isolated under:

```text
docs/retrieval_distillation/history/
```

Those files preserve experiment provenance only. The final source layout and
run names in this README supersede older paths shown there.
