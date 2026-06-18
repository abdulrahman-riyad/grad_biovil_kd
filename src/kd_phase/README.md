# KD Phase

This folder contains the implementation for the knowledge distillation phase after BioViL-T teacher embedding extraction.

## Current Inputs

Default artifact directory:

```text
weeks output/week1
```

Expected files:

```text
biovil_t_fixed_text_embeddings.npy
biovil_t_fixed_image_embeddings.npy
biovil_t_fixed_metadata.csv
biovil_t_fixed_study_scores.csv
biovil_t_fixed_metrics.json
```

## Step 1: Validate Teacher Artifacts

Run from the repository root:

```powershell
python kd_phase\data\validate_teacher_artifacts.py
```

This writes:

```text
kd_phase/outputs/artifact_validation_report.json
```

## Step 2: Create Subject-Level Splits

Run from the repository root:

```powershell
python kd_phase\data\make_subject_splits.py
```

This writes:

```text
kd_phase/splits/kd_train_indices.npy
kd_phase/splits/kd_val_indices.npy
kd_phase/splits/kd_test_indices.npy
kd_phase/splits/kd_train_metadata.csv
kd_phase/splits/kd_val_metadata.csv
kd_phase/splits/kd_test_metadata.csv
kd_phase/splits/split_report.json
```

The split is by `subject_id`, not by row, to avoid patient leakage.

## Next Implementation Target

After validation and splitting, the first implemented baseline is:

```text
ResNet-18 image student -> 128D normalized embedding
target = BioViL-T teacher image embedding
loss = cosine loss + MSE
```

## Step 3: Train First Image Student

Run this on Kaggle, because the metadata image paths point to the Kaggle dataset mount:

```bash
python kd_phase/train_image_student.py \
  --artifacts-dir "weeks output/week1" \
  --splits-dir "kd_phase/splits" \
  --output-dir "kd_phase/runs/resnet18_image_kd" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --epochs 5 \
  --batch-size 64 \
  --num-workers 2
```

For a fast smoke test before a full run:

```bash
python kd_phase/train_image_student.py \
  --artifacts-dir "weeks output/week1" \
  --splits-dir "kd_phase/splits" \
  --output-dir "kd_phase/runs/resnet18_image_kd_smoke" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --epochs 1 \
  --batch-size 32 \
  --max-train-batches 20 \
  --max-val-batches 10
```

Training outputs:

```text
kd_phase/runs/resnet18_image_kd/config.json
kd_phase/runs/resnet18_image_kd/history.json
kd_phase/runs/resnet18_image_kd/last.pt
kd_phase/runs/resnet18_image_kd/best.pt
```

## Step 4: Evaluate Image Student On Test Split

Run this on Kaggle after training:

```bash
python kd_phase/evaluate_image_student.py \
  --artifacts-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd" \
  --splits-dir "/kaggle/working/kd_phase/splits" \
  --checkpoint "/kaggle/working/kd_phase/runs/resnet18_image_kd/best.pt" \
  --output-dir "/kaggle/working/kd_phase/eval/resnet18_image_kd_test" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --batch-size 64 \
  --num-workers 2
```

If the checkpoint was downloaded locally and reuploaded as a Kaggle dataset, change `--checkpoint` to that dataset path.

Evaluation outputs:

```text
student_test_embeddings.npy
student_test_scores.csv
student_test_metrics.json
```

## Step 5: Train RepViT-M1.1 Student

Run this on Kaggle after copying `kd_phase` to `/kaggle/working/kd_phase`.

```bash
python /kaggle/working/kd_phase/train_repvit_student.py \
  --artifacts-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd" \
  --splits-dir "/kaggle/working/kd_phase/splits" \
  --output-dir "/kaggle/working/kd_phase/runs/repvit_m1_1_image_kd" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --repvit-root "/kaggle/working/kd_phase/RepViT" \
  --pretrained-checkpoint "/kaggle/working/kd_phase/RepViT/repvit_m1_1_distill_450e.pth" \
  --epochs 5 \
  --batch-size 64 \
  --num-workers 2
```

For a quick smoke test:

```bash
python /kaggle/working/kd_phase/train_repvit_student.py \
  --artifacts-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd" \
  --splits-dir "/kaggle/working/kd_phase/splits" \
  --output-dir "/kaggle/working/kd_phase/runs/repvit_m1_1_image_kd_smoke" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --repvit-root "/kaggle/working/kd_phase/RepViT" \
  --pretrained-checkpoint "/kaggle/working/kd_phase/RepViT/repvit_m1_1_distill_450e.pth" \
  --epochs 1 \
  --batch-size 32 \
  --num-workers 2 \
  --max-train-batches 20 \
  --max-val-batches 10
```

## Step 6: Evaluate RepViT-M1.1 On Test Split

```bash
python /kaggle/working/kd_phase/evaluate_repvit_student.py \
  --artifacts-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd" \
  --splits-dir "/kaggle/working/kd_phase/splits" \
  --checkpoint "/kaggle/working/kd_phase/runs/repvit_m1_1_image_kd/best.pt" \
  --output-dir "/kaggle/working/kd_phase/eval/repvit_m1_1_image_kd_test" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --batch-size 64 \
  --num-workers 2
```

## Step 7: Evaluate MobileViT-Small On Test Split

```bash
python /kaggle/working/kd_phase/evaluate_mobilevit_student.py \
  --artifacts-dir "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd" \
  --splits-dir "/kaggle/working/kd_phase/splits" \
  --checkpoint "/kaggle/working/kd_phase/mobileVit/e10_best_student.pth.zip" \
  --output-dir "/kaggle/working/kd_phase/eval/mobilevit_s_test" \
  --image-root "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files" \
  --batch-size 32 \
  --num-workers 2
```
