# Week 3 Track B Efficiency Benchmark Summary

Historical note: this file preserves earlier experiment tracking. The current maintained pipeline and run names are documented in docs/retrieval_distillation/README.md.

Benchmark source files:

- `week3/notebook_outputs/efficiency_metrics.json`
- `week3/notebook_outputs/efficiency_metrics.csv`

Runtime:

- Device: CUDA
- Batch size: 8
- Image size: 224 x 224
- Warmup iterations: 20
- Timed iterations: 100
- FLOPs tool: fvcore

## Results

| Model | Input | Params | Checkpoint | GFLOPs / batch | Latency / batch | Throughput | Peak Memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileViT-S KD | 1 image/view | 5.33M | 20.52 MB | 11.53 | 17.44 ms | 458.82 samples/s | 152.83 MB |
| MobileViT-S KD | 3-view study | 5.33M | 20.52 MB | 34.59 | 50.69 ms | 157.81 studies/s | 395.75 MB |
| RepViT-M1.1 KD | 1 image | 7.84M | 90.39 MB | 10.86 | 13.87 ms | 576.76 samples/s | 102.85 MB |

## Interpretation

- MobileViT-S remains the best embedding-quality student from Week 1 evaluation.
- RepViT-M1.1 is more efficient in single-image inference on this Kaggle GPU:
  - lower latency than MobileViT single-view,
  - higher throughput,
  - lower peak memory,
  - slightly lower measured GFLOPs per batch.
- MobileViT's 3-view study mode is more expensive because it forwards three images per study through the backbone and averages the view features.
- For Week 3, both models should continue:
  - MobileViT-S as the quality-first student.
  - RepViT-M1.1 as the efficiency-first student.

## Next Implementation Step

Implement Track A data/model plumbing:

1. Build an image-text contrastive dataset from the same MIMIC metadata and split files.
2. Add text encoder wrappers for:
   - BioViL-T text encoder,
   - CXR-BERT-specialized,
   - BioClinical ModernBERT-base.
3. Add projection heads so each image student and each text encoder map into the same latent space.
4. Train with symmetric image-text InfoNCE, while keeping the Week 1 teacher-space KD objective available as an auxiliary loss.
