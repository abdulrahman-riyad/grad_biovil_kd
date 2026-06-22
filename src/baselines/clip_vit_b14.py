# Auto-exported from project notebook.
# Source notebook: week4_structured_project/Final_ppt/materials/new_teammates_baselines_notebooks/CLIP_ViT_B14.ipynb
# Code cells: 6; markdown cells: 5
# Notebook shell commands and magics are preserved as comments.
# ruff: noqa
# pylint: skip-file

# %% [markdown] cell 1
# # Base model evaulation noteBook
# the pipline is:
#  1.  get the test set
#  2.  get the model text encode and image encoder
#  3.  get the visual and textual embeddings
#  4.  save embeddings

# %% [markdown] cell 2
# ## Configrations

# %% code cell 3
import ast
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel

# ----------------------------
# Configuration
# ----------------------------
SEED = 42


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    major, minor = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)

    # Kaggle P100 runtimes expose sm_60, but the preinstalled PyTorch build here
    # only supports sm_70+. Fall back to CPU instead of failing later at runtime.
    if major < 7:
        print(
            f"CUDA device '{gpu_name}' has capability sm_{major}{minor}, "
            "which is unsupported by this PyTorch build. Falling back to CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda")


DEVICE = select_device()
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)



# CSV expected to contain columns like:
# text or text_list, plus PA, AP, Lateral


TEXT_BATCH_SIZE = 32
MAX_TEXT_TOKENS = 128

# Evaluation / throughputC
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Using device: {DEVICE}")

# %% [markdown] cell 4
# ## Prepare Testset

# %% code cell 5
test_df = pd.read_csv("/kaggle/input/datasets/mohamed311ahmed/mimic-cxr-testsplit/kd_test_metadata.csv")

test_df.head()

# %% code cell 6
import re
import pathlib
import pandas as pd

def clean_image_paths(val):
    """
    Cleans a dataframe cell containing either a string representation of
    PosixPaths or an actual list of pathlib objects into a clean list of string paths.
    """
    # Case 1: The data was read from a CSV and is a raw string "[PosixPath('...'), ...]"
    if isinstance(val, str):
        # Use regex to find everything inside the single or double quotes of PosixPath('...')
        cleaned_paths = re.findall(r"PosixPath\(['\"](.*?)['\"]\)", val)
        return cleaned_paths

    # Case 2: The data is already a live Python list of Path objects in your notebook's memory
    elif isinstance(val, (list, tuple)):
        return [str(p) for p in val]

    # Fallback: If it's already an empty list or unexpected format, return it as-is
    return val

# %% code cell 7
# Assuming your dataframe is named 'df'
# Apply the cleaning function in place to the same column
test_df['image_paths'] = test_df['image_paths'].apply(clean_image_paths)

# Verify the result
print("Cleaned paths for the first row:")
print(test_df['image_paths'].iloc[0])

print("\nData type of the column cell:")
print(type(test_df['image_paths'].iloc[0]))

# %% [markdown] cell 8
# ## Get the model text encoder and image encoder

# %% code cell 9
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm  # Import the progress bar library

def get_clip_embeddings_multi_image(image_paths_per_row, texts, model_name=CLIP_MODEL_ID):
    """
    Extracts CLIP embeddings for text and computes the MEAN embedding for multiple corresponding images.
    Uses tqdm to cleanly display progress.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and processor
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    all_text_embeds = []
    all_image_embeds = []

    total_rows = len(texts)

    # Wrap the loop with tqdm to generate a live progress bar
    for i in tqdm(range(total_rows), desc="Extracting CLIP Embeddings"):
        text = texts[i]
        paths = image_paths_per_row[i]

        # --- 1. Process Text ---
        text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_outputs = model.get_text_features(**text_inputs)
            if hasattr(text_outputs, "pooler_output"):
                text_embed = text_outputs.pooler_output.cpu()
            else:
                text_embed = text_outputs.cpu()

        all_text_embeds.append(text_embed)

        # --- 2. Process Multiple Images for this row ---
        valid_images = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                valid_images.append(img)
            except Exception as e:
                # Using tqdm.write instead of print keeps the progress bar from breaking apart
                tqdm.write(f"Row {i}: Error loading image at {path}: {e}")

        if len(valid_images) > 0:
            image_inputs = processor(images=valid_images, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

            with torch.no_grad():
                image_outputs = model.get_image_features(**image_inputs)
                if hasattr(image_outputs, "pooler_output"):
                    image_features = image_outputs.pooler_output.cpu()
                else:
                    image_features = image_outputs.cpu()

                mean_image_embed = image_features.mean(dim=0, keepdim=True)
        else:
            embedding_dim = model.config.projection_dim
            mean_image_embed = torch.zeros((1, embedding_dim))
            tqdm.write(f"Warning: Row {i} had no valid images. Filled embedding with zeros.")

        all_image_embeds.append(mean_image_embed)

    # Combine all individual row vectors into final single tensors
    final_text_features = torch.cat(all_text_embeds, dim=0)
    final_image_features = torch.cat(all_image_embeds, dim=0)

    return final_text_features, final_image_features

# %% [markdown] cell 10
# ## extract the visual and textual embeddings

# %% code cell 11
image_lists = test_df['image_paths'].tolist()
text_list = test_df['report_text'].tolist()  # You can also use 'raw_report_text'

# 3. Run your multi-image embedding function
print("Starting embedding extraction process...")
text_features, image_features = get_clip_embeddings_multi_image(
    image_paths_per_row=image_lists,
    texts=text_list,
)

# 4. Save the raw PyTorch tensors to disk
# This keeps the multi-dimensional arrays intact without altering or truncating them
torch.save(text_features, 'CLIP_ViT_B14_mimic_cxr_text_embeddings.pt')
torch.save(image_features, 'CLIP_ViT_B14_mimic_cxr_image_embeddings.pt')

print("\nAll done! Embeddings extracted and saved successfully.")
print(f"Saved Text Tensor Shape: {text_features.shape}")   # Expected: [num_rows, 512]
print(f"Saved Image Tensor Shape: {image_features.shape}") # Expected: [num_rows, 512]
