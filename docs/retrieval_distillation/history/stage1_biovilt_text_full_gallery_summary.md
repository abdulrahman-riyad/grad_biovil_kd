# Full BioViL-T Contrastive Runs

Historical note: this file preserves earlier experiment tracking. The current maintained pipeline and run names are documented in docs/retrieval_distillation/README.md.

Source folder:

- `week3/notebook_outputs/pilot_outputs/contrastive_full_results`

Configuration:

- Text encoder: BioViL-T text encoder
- Text source: impression
- Image encoders: frozen
- Text encoder: frozen
- Trainable parameters: 887,297
- Train rows: 114,417
- Validation rows: 14,248
- Epochs: 8
- Batch size: 32

## Best Validation Results

| Student | Best Epoch | Val Loss | Image-to-Text R@1 | Image-to-Text R@5 | Text-to-Image R@1 | Text-to-Image R@5 |
|---|---:|---:|---:|---:|---:|---:|
| MobileViT-S | 8 | 2.0890 | 0.3348 | 0.7311 | 0.3426 | 0.7403 |
| RepViT-M1.1 | 5 | 2.0851 | 0.3363 | 0.7326 | 0.3404 | 0.7414 |

## Last Epoch Results

| Student | Epoch | Train Loss | Val Loss | Val Image-to-Text R@1 | Val Text-to-Image R@1 |
|---|---:|---:|---:|---:|---:|
| MobileViT-S | 8 | 1.5203 | 2.0890 | 0.3348 | 0.3426 |
| RepViT-M1.1 | 8 | 1.4339 | 2.0957 | 0.3339 | 0.3452 |

## Interpretation

- Both full runs are stable and completed successfully.
- RepViT-M1.1 has slightly better best validation loss.
- MobileViT-S reaches its best validation loss at epoch 8, while RepViT-M1.1 peaks earlier at epoch 5 and then slightly overfits.
- The two students are very close on batch-level validation retrieval.
- These metrics are batch-level contrastive retrieval over mini-batches, not final full-dataset retrieval metrics.

## Historical Interpretation Note

The full-run validation loss is higher than the 4k pilot loss because the full run uses:

- more validation rows,
- batch size 32 instead of 16,
- harder in-batch negatives.

Therefore, pilot and full losses are not directly comparable. Final model selection should use a dedicated full-split retrieval evaluation script that embeds all validation/test rows and computes retrieval metrics against a common candidate pool.

## Historical Next Step

Implement `evaluate_student_retrieval.py` to:

1. Load a trained contrastive checkpoint.
2. Embed validation/test image-text pairs.
3. Compute Image-to-Text and Text-to-Image retrieval against a fixed candidate pool.
4. Report R@1, R@5, R@10, median rank, and mean rank.
5. Save metrics, scores, and embeddings for both MobileViT-S and RepViT-M1.1.
