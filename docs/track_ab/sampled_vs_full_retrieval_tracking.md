# Sampled vs Full Retrieval Tracking

Source folders:

- Sampled candidate-pool outputs:
  - `week3/notebook_outputs/pilot_outputs/week3_sampled_retrieval_eval/week3/eval_sampled`
- Full-pool outputs:
  - `week3/notebook_outputs/pilot_outputs/contrastive_tests/eval/mobilevit_biovil_t_contrastive_test`
  - `week3/notebook_outputs/pilot_outputs/contrastive_tests/eval/repvit_biovil_t_contrastive_test`

## Evaluation Protocols

| Protocol | Candidate Pool | Purpose |
|---|---:|---|
| Mini sampled pool | 32 | Comparable to training mini-batch retrieval; very noisy, so averaged over 5 seeds |
| Sampled pool | 1,000 | Medium-difficulty retrieval estimate |
| Sampled pool | 5,000 | Harder sampled retrieval estimate |
| Full test pool | 14,277 | Final retrieval metric for reporting |

The full test pool is the primary research metric. Sampled pools are diagnostic and should be labeled with the candidate-pool size.

## Candidate Pool = 32

Mean over 5 random sampled pools.

| Student | I2T R@1 mean | I2T R@1 std | I2T R@5 mean | I2T R@10 mean | T2I R@1 mean | T2I R@1 std | T2I R@5 mean | T2I R@10 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileViT-S | 0.4750 | 0.0667 | 0.8188 | 0.9625 | 0.4563 | 0.1539 | 0.8125 | 0.9250 |
| RepViT-M1.1 | 0.4813 | 0.0852 | 0.8375 | 0.9813 | 0.4375 | 0.1135 | 0.8438 | 0.9625 |

## Candidate Pool = 1,000

| Student | I2T R@1 | I2T R@5 | I2T R@10 | I2T Median Rank | T2I R@1 | T2I R@5 | T2I R@10 | T2I Median Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileViT-S | 0.1390 | 0.3150 | 0.4200 | 17 | 0.1430 | 0.3180 | 0.4060 | 18 |
| RepViT-M1.1 | 0.1420 | 0.3240 | 0.4240 | 17 | 0.1380 | 0.3130 | 0.4100 | 18 |

## Candidate Pool = 5,000

| Student | I2T R@1 | I2T R@5 | I2T R@10 | I2T Median Rank | T2I R@1 | T2I R@5 | T2I R@10 | T2I Median Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileViT-S | 0.0498 | 0.1452 | 0.2066 | 92 | 0.0482 | 0.1394 | 0.2002 | 94 |
| RepViT-M1.1 | 0.0538 | 0.1482 | 0.2090 | 90 | 0.0420 | 0.1376 | 0.2052 | 96 |

## Candidate Pool = 14,277 Full Test

| Student | I2T R@1 | I2T R@5 | I2T R@10 | I2T Median Rank | T2I R@1 | T2I R@5 | T2I R@10 | T2I Median Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileViT-S | 0.0257 | 0.0773 | 0.1200 | 257 | 0.0223 | 0.0743 | 0.1156 | 264 |
| RepViT-M1.1 | 0.0242 | 0.0761 | 0.1168 | 261 | 0.0205 | 0.0726 | 0.1130 | 278 |

## Interpretation

- Retrieval metrics decrease as candidate-pool size increases. This is expected because the ranking problem becomes harder.
- Candidate pool 32 gives high R@1 values because the model only chooses among 32 possible matches.
- Candidate pool 1,000 and 5,000 provide useful intermediate views of retrieval quality.
- Full-pool test retrieval over 14,277 candidates is the strictest and should be used for final claims.
- MobileViT-S and RepViT-M1.1 remain close across all protocols.
- RepViT-M1.1 is slightly stronger in some sampled Image-to-Text settings.
- MobileViT-S is slightly stronger on the full-pool test recalls and remains the quality-focused choice.
