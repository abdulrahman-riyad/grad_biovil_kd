# Auto-exported from project notebook.
# Source notebook: project baseline notebooks/TinyCLIP.ipynb
# Code cells: 10; markdown cells: 8
# Notebook shell commands and magics are preserved as comments.
# ruff: noqa
# pylint: skip-file

# %% [markdown] cell 1
# # Base Model Evaluation — TinyCLIP (All Variants)
#
# Pipeline:
# 1. Load the test set
# 2. Loop over all 11 TinyCLIP variants
# 3. Extract normalised image & text embeddings
# 4. Save per-variant tensors to disk

# %% [markdown] cell 2
# ## Installation

# %% code cell 3
# NOTEBOOK_COMMAND: !pip install -q --upgrade open-clip-torch huggingface_hub

# %% [markdown] cell 4
# ## Configuration

# %% code cell 5
import os
import re
import ast
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import login

# ── HuggingFace Authentication ──────────────────────────────────────────────
# Needed if access rules change or to avoid rate limits
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", "YOUR_HF_TOKEN_HERE")
if HF_TOKEN and HF_TOKEN != "YOUR_HF_TOKEN_HERE":
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in to HuggingFace Hub")
else:
    print("⚠️ No explicit token set. Proceeding with public access.")

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device Selection ────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Processing & Output Paths ───────────────────────────────────────────────
TEXT_BATCH_SIZE = 32
OUTPUT_DIR = Path('tinyclip_embeddings')
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Embeddings will be saved to: {OUTPUT_DIR.resolve()}\n")

# All variants re-mapped directly to official 'microsoft/' endpoints
TINYCLIP_VARIANTS = [
    {'repo_id': 'wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M', 'arch': 'TinyCLIP-ViT-40M-32-Text-19M', 'pretrained': 'LAION400M'},
    {'repo_id': 'wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M', 'arch': 'TinyCLIP-ViT-61M-32-Text-29M', 'pretrained': 'LAION400M'},
    {'repo_id': 'wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M', 'arch': 'TinyCLIP-ViT-8M-16-Text-3M-YFCC15M', 'pretrained': 'YFCC15M'},
    {'repo_id': 'wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M', 'arch': 'TinyCLIP-ViT-39M-16-Text-19M-YFCC15M', 'pretrained': 'YFCC15M'},
]

print(f"Total TinyCLIP variant targets mapped: {len(TINYCLIP_VARIANTS)}")

# %% [markdown] cell 6
# ## Prepare Test Set

# %% code cell 7
test_df = pd.read_csv('/kaggle/input/datasets/mohamed311ahmed/mimic-cxr-testsplit/kd_test_metadata.csv')
test_df.head()

# %% code cell 8
def clean_image_paths(val):
    """Normalises a column cell to a plain list-of-str paths."""
    if isinstance(val, str):
        cleaned = re.findall(r"PosixPath\(['\"](.+?)['\"]\)", val)
        if cleaned:
            return cleaned
        # plain string list repr: "['/a/b.png', '/c/d.png']"
        try:
            parsed = ast.literal_eval(val)
            return [str(p) for p in parsed]
        except Exception:
            return [val]
    elif isinstance(val, (list, tuple)):
        return [str(p) for p in val]
    return val

test_df['image_paths'] = test_df['image_paths'].apply(clean_image_paths)
print('Cleaned paths for the first row:', test_df['image_paths'].iloc[0])
print('Type:', type(test_df['image_paths'].iloc[0]))

# %% code cell 9
# ── Optionally limit rows for quick debugging ───────────────────────────
# test_df = test_df.head(20)   # ← uncomment to use a subset

image_lists = test_df['image_paths'].tolist()
text_list   = test_df['report_text'].tolist()   # swap to 'raw_report_text' if needed
print(f'Total rows: {len(test_df)}')

# %% [markdown] cell 10
# ## Embedding Extraction Function

# %% code cell 11
@torch.no_grad()
def extract_tinyclip_embeddings(
    image_paths_per_row: list,
    texts: list,
    repo_id: str,
    arch: str,
    pretrained: str,
    device: torch.device = DEVICE,
    text_batch_size: int = TEXT_BATCH_SIZE,
) -> dict:
    print(f"\n{'='*60}")
    print(f" Processing: {arch} | Pretrained: {pretrained}")
    print(f"{'='*60}\n")

    # ── 1. Load model & processor ────────────────────────────────────────────
    processor = CLIPProcessor.from_pretrained(repo_id)
    model     = CLIPModel.from_pretrained(repo_id).to(device).eval()

    # ── Helper: safe pooled output ───────────────────────────────────────────
    def _pool(encoder_output):
        """
        HuggingFace CLIP returns BaseModelOutputWithPooling.
        Depending on transformers version, pooler_output may be None
        for TinyCLIP — fall back to the CLS token from last_hidden_state.
        """
        if encoder_output.pooler_output is not None:
            return encoder_output.pooler_output          # [B, hidden]
        return encoder_output.last_hidden_state[:, 0]    # CLS token fallback

    # ── 2. Text embeddings ───────────────────────────────────────────────────
    text_feats = []
    for i in tqdm(range(0, len(texts), text_batch_size), desc="Text batches"):
        batch  = texts[i : i + text_batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        # Call text_model directly → bypass get_text_features() internals
        enc_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        pooled = _pool(enc_out)                          # [B, hidden]
        # Apply projection head manually
        projected = model.text_projection(pooled)        # [B, proj_dim]
        feats = F.normalize(projected, dim=-1)
        text_feats.append(feats.cpu())

    text_embeddings = torch.cat(text_feats, dim=0)       # [N, D]

    # ── 3. Image embeddings ──────────────────────────────────────────────────
    image_feats = []
    proj_dim = model.config.projection_dim

    for paths in tqdm(image_paths_per_row, desc="Processing Reports"):
        view_feats = []
        for p in paths:
            try:
                img    = Image.open(p).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)

                # Call vision_model directly → bypass get_image_features() internals
                enc_out = model.vision_model(
                    pixel_values=inputs["pixel_values"],
                )
                pooled = _pool(enc_out)                  # [1, hidden]
                # Apply projection head manually
                projected = model.visual_projection(pooled)  # [1, proj_dim]
                feats = F.normalize(projected, dim=-1)
                view_feats.append(feats)
            except Exception as e:
                print(f"  ⚠️  Skipping image {p}: {e}")

        if view_feats:
            pooled = torch.stack(view_feats, dim=0).mean(dim=0)
            pooled = F.normalize(pooled, dim=-1)
        else:
            pooled = torch.zeros(1, proj_dim, device=device)

        image_feats.append(pooled.cpu())

    image_embeddings = torch.cat(image_feats, dim=0)     # [N, D]

    # ── 4. Cleanup ───────────────────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()

    return {
        'text_embeddings':  text_embeddings,
        'image_embeddings': image_embeddings,
    }

# %% [markdown] cell 12
# ## Run All Variants

# %% code cell 13
results_summary = []

for variant in TINYCLIP_VARIANTS:
    repo_id = variant['repo_id']
    arch = variant['arch']
    pretrained = variant['pretrained']

    # Generate persistent filename tokens
    safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', arch)
    text_path = OUTPUT_DIR / f"{safe_name}_text_embeddings.pt"
    image_path = OUTPUT_DIR / f"{safe_name}_image_embeddings.pt"

    # Resume capability: Check if features already calculated on disk
    if text_path.exists() and image_path.exists():
        print(f"[SKIP] {arch} — features already saved on disk.")
        te = torch.load(text_path, map_location='cpu')
        ie = torch.load(image_path, map_location='cpu')
        sim = F.cosine_similarity(te, ie, dim=1).mean().item()
        results_summary.append({
            'arch': arch, 'pretrained': pretrained,
            'dim': te.shape[-1], 'mean_cosine_sim': sim,
            'text_path': str(text_path), 'image_path': str(image_path)
        })
        continue

    try:
        embeddings = extract_tinyclip_embeddings(
            image_paths_per_row=image_lists,
            texts=text_list,
            repo_id=repo_id,
            arch=arch,
            pretrained=pretrained,
            device=DEVICE
        )

        te = embeddings['text_embeddings']
        ie = embeddings['image_embeddings']

        # Save tensors out to disk
        torch.save(te, text_path)
        torch.save(ie, image_path)

        sim = F.cosine_similarity(te, ie, dim=1).mean().item()
        results_summary.append({
            'arch': arch, 'pretrained': pretrained,
            'dim': te.shape[-1], 'mean_cosine_sim': sim,
            'text_path': str(text_path), 'image_path': str(image_path)
        })
        print(f"✅ Extracted features saved successfully. Dimension: {te.shape[-1]} | Mean Cosine Sim: {sim:.4f}")

    except Exception as exc:
        print(f"❌ [ERROR] Processing skipped for {arch}: {exc}")
        results_summary.append({
            'arch': arch, 'pretrained': pretrained,
            'dim': None, 'mean_cosine_sim': None,
            'text_path': None, 'image_path': None,
            'error': str(exc)
        })

print("\n🎉 Embedding generation pipeline complete across all variants.")

# %% [markdown] cell 14
# ## Summary

# %% code cell 15
summary_df = pd.DataFrame(results_summary)
display(summary_df[['arch', 'pretrained', 'dim', 'mean_cosine_sim', 'text_path', 'image_path']])

# %% [markdown] cell 16
# ## Reload & Verify Any Saved Variant

# %% code cell 17
summary_df = pd.DataFrame(results_summary)
display(summary_df[['arch', 'pretrained', 'dim', 'mean_cosine_sim', 'text_path', 'image_path']])
