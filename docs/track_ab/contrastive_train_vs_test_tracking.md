# Week 3 Contrastive Training vs Test Tracking

Source folders:

- Training checkpoints and histories:
  - `week3/notebook_outputs/pilot_outputs/contrastive_full_results/runs/mobilevit_biovil_t_contrastive_full`
  - `week3/notebook_outputs/pilot_outputs/contrastive_full_results/runs/repvit_biovil_t_contrastive_full`
- Full-pool test retrieval outputs:
  - `week3/notebook_outputs/pilot_outputs/contrastive_tests/eval/mobilevit_biovil_t_contrastive_test`
  - `week3/notebook_outputs/pilot_outputs/contrastive_tests/eval/repvit_biovil_t_contrastive_test`

## Setup

| Item | Value |
|---|---|
| Text encoder | BioViL-T text encoder |
| Text source | Impression |
| Image encoders | Frozen |
| Text encoder | Frozen |
| Trainable parameters | 887,297 |
| Train rows | 114,417 |
| Validation rows | 14,248 |
| Test rows | 14,277 |
| Training epochs | 8 |
| Training batch size | 32 |
| Test batch size | 64 |

## Training/Validation Metrics

These are mini-batch contrastive metrics from training. They are useful for optimization tracking, but they are not final full-pool retrieval metrics.

| Student | Best Epoch | Train Loss | Val Loss | Val Image-to-Text R@1 | Val Image-to-Text R@5 | Val Text-to-Image R@1 | Val Text-to-Image R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileViT-S | 8 | 1.5203 | 2.0890 | 0.3348 | 0.7311 | 0.3426 | 0.7403 |
| RepViT-M1.1 | 5 | 1.4602 | 2.0851 | 0.3363 | 0.7326 | 0.3404 | 0.7414 |

## Full-Pool Test Retrieval

These are the proper retrieval metrics computed against all 14,277 test candidates.

| Student | Checkpoint Epoch | Image-to-Text R@1 | Image-to-Text R@5 | Image-to-Text R@10 | I2T Median Rank | I2T Mean Rank | Text-to-Image R@1 | Text-to-Image R@5 | Text-to-Image R@10 | T2I Median Rank | T2I Mean Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileViT-S | 8 | 0.0257 | 0.0773 | 0.1200 | 257 | 907.60 | 0.0223 | 0.0743 | 0.1156 | 264 | 1016.34 |
| RepViT-M1.1 | 5 | 0.0242 | 0.0761 | 0.1168 | 261 | 907.07 | 0.0205 | 0.0726 | 0.1130 | 278 | 1029.10 |

## Comparison

- Training/validation:
  - RepViT-M1.1 has slightly lower best validation loss.
  - MobileViT-S and RepViT-M1.1 are nearly tied on mini-batch validation retrieval.
- Full-pool test retrieval:
  - MobileViT-S is slightly better on all recall metrics in both retrieval directions.
  - RepViT-M1.1 has a marginally lower Image-to-Text mean rank, but worse median rank and worse Text-to-Image mean rank.
- Efficiency:
  - RepViT-M1.1 remains the efficiency-first student from the earlier benchmark.
  - MobileViT-S remains the quality-first student.

## Current Decision

Use MobileViT-S + BioViL-T text encoder as the quality-focused contrastive model.

Use RepViT-M1.1 + BioViL-T text encoder as the efficiency-focused contrastive model.

Do not claim a large quality gap between them. The full-pool test retrieval difference is small, but MobileViT-S is consistently ahead on recall.

## Notes

The full-pool test R@1 values are much lower than mini-batch validation R@1 values because the test evaluation retrieves against 14,277 candidates instead of only one mini-batch of 32 candidates. The full-pool metrics are the numbers to use for final retrieval claims.
