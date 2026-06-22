# Week 3 Text Encoder Pilot Comparison

Historical note: this file preserves earlier experiment tracking. The current maintained pipeline and run names are documented in docs/retrieval_distillation/README.md.

Source folders:

- `week3/notebook_outputs/pilot_outputs/mobilevit_cxrbert_contrastive_pilot`
- `week3/notebook_outputs/pilot_outputs/repvit_cxrbert_contrastive_pilot`
- `week3/notebook_outputs/pilot_outputs/pilots/runs/mobilevit_biovil_t_contrastive_pilot`
- `week3/notebook_outputs/pilot_outputs/pilots/runs/repvit_biovil_t_contrastive_pilot`
- `week3/notebook_outputs/pilot_outputs/pilots/runs/mobilevit_bioclinical_modernbert_contrastive_pilot`
- `week3/notebook_outputs/pilot_outputs/pilots/runs/repvit_bioclinical_modernbert_contrastive_pilot`

Shared pilot configuration:

- Train rows: 4,096
- Validation rows: 1,024
- Epochs: 3
- Batch size: 16
- Learning rate: 1e-4
- Frozen image encoder: yes
- Frozen text encoder: yes
- Trainable parameters: 887,297
- Text source: impression

## Best Validation Results

| Rank | Student | Text Encoder | Best Epoch | Val Loss | Image-to-Text R@1 | Image-to-Text R@5 | Text-to-Image R@1 | Text-to-Image R@5 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | MobileViT-S | BioViL-T text encoder | 3 | 1.8027 | 0.3779 | 0.8203 | 0.3818 | 0.8213 |
| 2 | RepViT-M1.1 | BioViL-T text encoder | 2 | 1.8129 | 0.3623 | 0.8232 | 0.3545 | 0.8359 |
| 3 | RepViT-M1.1 | CXR-BERT-specialized | 2 | 1.8583 | 0.3389 | 0.8164 | 0.3418 | 0.8145 |
| 4 | MobileViT-S | CXR-BERT-specialized | 2 | 1.8604 | 0.3232 | 0.8105 | 0.3516 | 0.8223 |
| 5 | RepViT-M1.1 | BioClinical ModernBERT-base | 3 | 2.0729 | 0.2666 | 0.7451 | 0.2813 | 0.7627 |
| 6 | MobileViT-S | BioClinical ModernBERT-base | 3 | 2.1538 | 0.2236 | 0.7188 | 0.2627 | 0.7422 |

## Decision

Use the BioViL-T text encoder for the next scaled Week 3 runs.

Rationale:

- It achieves the best validation loss for both image students.
- It achieves the best Image-to-Text R@1 and Text-to-Image R@1 overall.
- It is aligned with the original BioViL-T teacher family, which is appropriate because the image students were distilled from BioViL-T image embeddings.

Keep CXR-BERT-specialized as the secondary baseline.

Rationale:

- It is close to BioViL-T in validation loss and retrieval.
- It is directly chest-X-ray specialized.
- It is useful as a non-BioViL baseline text encoder.

Do not scale BioClinical ModernBERT-base in the current setup.

Rationale:

- It underperforms clearly on validation loss and both retrieval directions.
- It may still be useful later after domain-specific pooling, longer text, or fine-tuning, but it is not the right next scaling target.

## Next Run

Scale the two BioViL-T text encoder runs:

- MobileViT-S + BioViL-T text encoder
- RepViT-M1.1 + BioViL-T text encoder

Recommended first scale-up:

- Train rows: 16,384
- Validation rows: 4,096
- Epochs: 4
- Batch size: 32 if GPU memory allows, otherwise 16
- Learning rate: 1e-4
- Frozen image/text encoders

If validation keeps improving without instability, proceed to full train/validation split.

## Lightweight Text-Student Extension

Before moving to explainability, add three compact text-student candidates to the same pilot protocol:

| Preset | Hugging Face model | Reason |
|---|---|---|
| `distil_biobert` | `nlpie/distil-biobert` | Matches the project DistilBioBERT text-student baseline. |
| `clinical_distilbert` | `nlpie/clinical-distilbert` | Same compact family as DistilBioBERT, but clinically adapted. |
| `clinical_mobilebert` | `nlpie/clinical-mobilebert` | Smaller clinical encoder candidate for strongest efficiency baseline. |

Run `week3/503-lightweight-text-encoder-comparison.ipynb` after uploading the updated `week3/biovil_t_retrieval_distillation` folder to Kaggle. Compare the new pilot results against the six rows above before selecting any lightweight text student for a full run.
