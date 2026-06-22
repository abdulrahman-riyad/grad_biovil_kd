# Week 3 Phase 2 Pilot Results

Historical note: this file preserves earlier experiment tracking. The current maintained pipeline and run names are documented in docs/retrieval_distillation/README.md.

Source folders:

- `week3/notebook_outputs/pilot_outputs/mobilevit_cxrbert_contrastive_pilot`
- `week3/notebook_outputs/pilot_outputs/repvit_cxrbert_contrastive_pilot`

Configuration:

- Text encoder: CXR-BERT-specialized
- Text source: impression
- Image encoders: frozen
- Text encoder: frozen
- Trainable parameters: 887,297
- Training rows: 4,096
- Validation rows: 1,024
- Batch size: 16
- Epochs: 3
- Learning rate: 1e-4

## Best Validation Metrics

| Student | Best Epoch | Val Loss | Image-to-Text R@1 | Image-to-Text R@5 | Text-to-Image R@1 | Text-to-Image R@5 |
|---|---:|---:|---:|---:|---:|---:|
| MobileViT-S | 2 | 1.8604 | 0.3232 | 0.8105 | 0.3516 | 0.8223 |
| RepViT-M1.1 | 2 | 1.8583 | 0.3389 | 0.8164 | 0.3418 | 0.8145 |

## Last Epoch Metrics

| Student | Epoch | Train Loss | Val Loss | Val Image-to-Text R@1 | Val Text-to-Image R@1 |
|---|---:|---:|---:|---:|---:|
| MobileViT-S | 3 | 1.3011 | 1.8752 | 0.3291 | 0.3389 |
| RepViT-M1.1 | 3 | 1.2439 | 1.8711 | 0.3428 | 0.3506 |

## Interpretation

- Both pilots are valid and stable.
- Training loss decreases across epochs for both students, so the projection heads are learning.
- Validation loss improves until epoch 2 and then slightly worsens at epoch 3, suggesting mild overfitting on the 4,096-row pilot subset.
- RepViT is slightly better on best validation loss and image-to-text R@1 in this pilot.
- MobileViT is slightly better on best text-to-image R@1/R@5 in this pilot.
- The differences are small; this is not enough to choose the final student.

## Historical Next Step

Run the same 4,096/1,024 pilot for the other two text encoders:

- BioViL-T text encoder
- BioClinical ModernBERT-base

Use the same frozen-encoder setting so the comparison is controlled.
