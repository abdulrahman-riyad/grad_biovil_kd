# Group Work Code Exports

This package collects the graduation-project code that originally lived in teammate Kaggle notebooks.
Each notebook was converted into a Python script so the repository contains the executable project code rather than notebook-only artifacts.

Conversion policy:
- one `.py` file per source notebook;
- markdown cells are retained as comments for provenance and section labels;
- notebook shell commands and magics are retained as comments because they are not valid Python syntax;
- source notebook paths and cell counts are recorded at the top of each exported file.

## Exports

| Group | Source notebook | Exported script | Code cells | Markdown cells |
|---|---|---|---:|---:|
| training | `mobilevit-distilbiobert-kd-hn-updated.ipynb` | `src/group_work/training/mobilevit_distilbiobert_kd_hn_updated.py` | 32 | 18 |
| training | `repvit-distilbiobert-kd-hn-updated.ipynb` | `src/group_work/training/repvit_distilbiobert_kd_hn_updated.py` | 32 | 17 |
| baselines | `base-models-evaluation-0d2772.ipynb` | `src/group_work/baselines/base_models_evaluation_0d2772.py` | 12 | 12 |
| baselines | `BiomedCLIP_PubMedBERT_256.ipynb` | `src/group_work/baselines/biomedclip_pubmedbert_256.py` | 8 | 5 |
| baselines | `CLIP_ViT_B14.ipynb` | `src/group_work/baselines/clip_vit_b14.py` | 6 | 5 |
| baselines | `CLIP_ViT_B16.ipynb` | `src/group_work/baselines/clip_vit_b16.py` | 6 | 5 |
| baselines | `CLIP_ViT_B32.ipynb` | `src/group_work/baselines/clip_vit_b32.py` | 6 | 5 |
| baselines | `cxr-clip.ipynb` | `src/group_work/baselines/cxr_clip.py` | 11 | 8 |
| baselines | `medclip-baseline.ipynb` | `src/group_work/baselines/medclip_baseline.py` | 9 | 9 |
| baselines | `MedCLIP_ViT_V1.ipynb` | `src/group_work/baselines/medclip_vit_v1.py` | 12 | 5 |
| baselines | `mgca-evaluation.ipynb` | `src/group_work/baselines/mgca_evaluation.py` | 19 | 11 |
| baselines | `MobileCLIP_b.ipynb` | `src/group_work/baselines/mobileclip_b.py` | 10 | 5 |
| baselines | `MobileCLIP_s0.ipynb` | `src/group_work/baselines/mobileclip_s0.py` | 9 | 5 |
| baselines | `MobileCLIP_s1.ipynb` | `src/group_work/baselines/mobileclip_s1.py` | 10 | 5 |
| baselines | `MobileCLIP_s2.ipynb` | `src/group_work/baselines/mobileclip_s2.py` | 9 | 5 |
| baselines | `retrival-evaluation-pipeline.ipynb` | `src/group_work/baselines/retrival_evaluation_pipeline.py` | 11 | 12 |
| baselines | `TinyCLIP.ipynb` | `src/group_work/baselines/tinyclip.py` | 10 | 8 |

The main maintained training/evaluation pipeline remains under `src/track_ab` and `ops/gcp_l4`.
These exports preserve the teammate baseline and training implementations for traceability, review, and future refactoring.
