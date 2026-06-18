# Week 3 Track A/B Implementation

This folder starts the MobileViT/RepViT continuation phase.

Current scope:

- Load the trained Week 1 MobileViT-Small and RepViT-M1.1 image students.
- Benchmark efficiency for Track B before adding text encoders and contrastive learning.
- Export reproducible JSON/CSV metrics for the report.
- Train lightweight image/text contrastive projection heads for Track A.

Run from the repository root:

```bash
python week3/track_ab/benchmark_efficiency.py \
  --mobilevit-checkpoint "week1/weeks output/week1/student_mobilevit/mobilevit_s_biovil_kd_checkpoint.pt" \
  --repvit-checkpoint "week1/kd_phase/RepViT/training_output/best.pt" \
  --repvit-root "week1/kd_phase/RepViT" \
  --batch-size 8 \
  --timed-iters 100
```

Outputs:

- `week3/efficiency_metrics.json`
- `week3/efficiency_metrics.csv`

If FLOP tools are unavailable, either install one of them:

```bash
pip install fvcore
```

or run with:

```bash
python week3/track_ab/benchmark_efficiency.py --skip-flops
```

## Phase 2 Contrastive Smoke Runs

Run from Kaggle after uploading this `track_ab` folder:

```bash
python /kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/track_ab/train_contrastive_student.py \
  --artifacts-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase" \
  --splits-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase/splits" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --output-dir "/kaggle/working/week3/runs/mobilevit_cxrbert_contrastive_smoke" \
  --image-student mobilevit \
  --mobilevit-checkpoint "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/mobilevit_s_biovil_kd_checkpoint.pt" \
  --text-encoder cxr_bert \
  --max-train-rows 64 \
  --max-val-rows 32 \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --batch-size 4 \
  --num-workers 2
```

RepViT smoke run:

```bash
python /kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/track_ab/train_contrastive_student.py \
  --artifacts-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase" \
  --splits-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase/splits" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --output-dir "/kaggle/working/week3/runs/repvit_cxrbert_contrastive_smoke" \
  --image-student repvit \
  --repvit-checkpoint "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase/RepViT/training_output/best.pt" \
  --repvit-root "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase/RepViT" \
  --text-encoder cxr_bert \
  --max-train-rows 64 \
  --max-val-rows 32 \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --batch-size 4 \
  --num-workers 2
```

Text encoder options:

- `biovil_t`: `microsoft/BiomedVLP-BioViL-T`
- `cxr_bert`: `microsoft/BiomedVLP-CXR-BERT-specialized`
- `bioclinical_modernbert`: `thomas-sounack/BioClinical-ModernBERT-base`
- `distil_biobert`: `nlpie/distil-biobert`
- `clinical_distilbert`: `nlpie/clinical-distilbert`
- `clinical_mobilebert`: `nlpie/clinical-mobilebert`

## Lightweight Text Encoder Comparison

Use `week3/503-lightweight-text-encoder-comparison.ipynb` to compare the three lightweight text-student candidates against the earlier Week 3 text encoder experiments. The notebook runs:

- MobileViT-S + DistilBioBERT
- RepViT-M1.1 + DistilBioBERT
- MobileViT-S + ClinicalDistilBERT
- RepViT-M1.1 + ClinicalDistilBERT
- MobileViT-S + ClinicalMobileBERT
- RepViT-M1.1 + ClinicalMobileBERT

The full protocol is:

- full train/validation split
- 10 epochs
- `torchrun --nproc_per_node=2` on Kaggle T4 x2 when two GPUs are available
- batch size 16 per GPU, effective batch size 32 on two GPUs
- frozen image encoder
- frozen text encoder
- trainable projection heads
- full-pool test retrieval
- sampled test retrieval with 5,000, 1,000, and 32 candidates

Run the optional smoke cell first. If all three text encoders load successfully, run the 10-epoch training and evaluation cells, then download `week3_lightweight_text_full10_eval.zip`.

## Full Retrieval Evaluation

After training full contrastive checkpoints, evaluate retrieval on a fixed validation or test candidate pool.

MobileViT:

```bash
python /kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/track_ab/evaluate_contrastive_retrieval.py \
  --checkpoint "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/contrastive_full_results/runs/mobilevit_biovil_t_contrastive_full/best.pt" \
  --artifacts-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase" \
  --splits-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase/splits" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --mobilevit-checkpoint "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/mobilevit_s_biovil_kd_checkpoint.pt" \
  --split test \
  --batch-size 64 \
  --similarity-chunk-size 512 \
  --output-dir "/kaggle/working/week3/eval/mobilevit_biovil_t_contrastive_test"
```

RepViT:

```bash
python /kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/track_ab/evaluate_contrastive_retrieval.py \
  --checkpoint "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/contrastive_full_results/runs/repvit_biovil_t_contrastive_full/best.pt" \
  --artifacts-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase" \
  --splits-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase/splits" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --repvit-checkpoint "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase/RepViT/training_output/best.pt" \
  --repvit-root "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase/RepViT" \
  --split test \
  --batch-size 64 \
  --similarity-chunk-size 512 \
  --output-dir "/kaggle/working/week3/eval/repvit_biovil_t_contrastive_test"
```

Outputs:

- `retrieval_metrics.json`
- `image_embeddings.npy`
- `text_embeddings.npy`
- `row_indices.npy`
- `subject_ids.npy`
- `study_ids.npy`
- `image_to_text_topk.csv`
- `text_to_image_topk.csv`

## Retrieval Example Analysis

Use the top-k CSV files to find exact successes and failure cases.

```bash
python /kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/track_ab/analyze_retrieval_examples.py \
  --metadata-csv "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase/biovil_t_fixed_metadata.csv" \
  --topk-csv "/kaggle/working/week3/eval/mobilevit_biovil_t_contrastive_test/image_to_text_topk.csv" \
  --direction image_to_text \
  --output-dir "/kaggle/working/week3/analysis/mobilevit_i2t_examples"
```

Outputs:

- `query_retrieval_summary.csv`
- `exact_top1_examples.csv`
- `top5_not_top1_examples.csv`
- `moderate_failures_examples.csv`
- `severe_failures_examples.csv`
- `analysis_stats.json`
- `analysis_report.md`

## Grad-CAM Explainability

Generate overlays from selected rows or from an examples CSV.

```bash
python /kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/track_ab/explain_contrastive_gradcam.py \
  --checkpoint "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/contrastive_full_best/mobilevit_biovil_t_contrastive_full/best.pt" \
  --metadata-csv "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/kd_phase/biovil_t_fixed_metadata.csv" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --mobilevit-checkpoint "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/mobilevit_s_biovil_kd_checkpoint.pt" \
  --cases-csv "/kaggle/working/week3/analysis/mobilevit_i2t_examples/severe_failures_examples.csv" \
  --output-dir "/kaggle/working/week3/explain/mobilevit_severe_failures" \
  --max-cases 12
```

## Deployable Contrastive Export

Package projection heads and metadata for deployment.

```bash
python /kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/track_ab/export_contrastive_checkpoint.py \
  --contrastive-checkpoint "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/contrastive_full_best/mobilevit_biovil_t_contrastive_full/best.pt" \
  --retrieval-metrics "/kaggle/working/week3/eval/mobilevit_biovil_t_contrastive_test/retrieval_metrics.json" \
  --image-checkpoint "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd/mobilevit_s_biovil_kd_checkpoint.pt" \
  --output-dir "/kaggle/working/week3/deploy/mobilevit_biovil_t_contrastive"
```
