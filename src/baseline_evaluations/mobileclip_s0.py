# Auto-exported from project notebook.
# Source notebook: project baseline notebooks/MobileCLIP_s0.ipynb
# Code cells: 9; markdown cells: 5
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
# NOTEBOOK_COMMAND: !wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt

# %% code cell 4
# NOTEBOOK_COMMAND: !pip install git+https://github.com/apple/ml-mobileclip.git

# %% code cell 5
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

import mobileclip
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

model_name = "mobileclip_s0" # Options: mobileclip_s0, mobileclip_s1, mobileclip_s2, mobileclip_b
pretrained = "mobileclip_s0.pt"

model, _, preprocess = mobileclip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = mobileclip.get_tokenizer(model_name)

# CSV expected to contain columns like:
# text or text_list, plus PA, AP, Lateral


TEXT_BATCH_SIZE = 32
MAX_TEXT_TOKENS = 128

TEXT_EMBEDDING_PATH = "MoblileCLIP_s0_mimic_cxr_text_embeddings.pt"
IMAGE_EMBEDDING_PATH = "MobileCLIP_s0_mimic_cxr_image_embeddings.pt"

# Evaluation / throughputC
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Using device: {DEVICE}")

# %% [markdown] cell 6
# ## Prepare Testset

# %% code cell 7
test_df = pd.read_csv("/kaggle/input/datasets/mohamed311ahmed/mimic-cxr-testsplit/kd_test_metadata.csv")

test_df.head()

# %% code cell 8
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

# %% code cell 9
# Assuming your dataframe is named 'df'
# Apply the cleaning function in place to the same column
test_df['image_paths'] = test_df['image_paths'].apply(clean_image_paths)

# Verify the result
print("Cleaned paths for the first row:")
print(test_df['image_paths'].iloc[0])

print("\nData type of the column cell:")
print(type(test_df['image_paths'].iloc[0]))

# %% [markdown] cell 10
# ## Get the model text encoder and image encoder

# %% code cell 11
import torch
from PIL import Image
from tqdm import tqdm  # Import the progress bar library
import mobileclip      # Import the MobileCLIP library

def get_mobileclip_embeddings_multi_image(
    image_paths_per_row,
    texts,
    model_name=model_name,
    pretrained=pretrained
):
    """
    Extracts MobileCLIP embeddings for text and computes the MEAN embedding for multiple corresponding images.
    Uses tqdm to cleanly display progress.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load MobileCLIP model, preprocessor, and tokenizer
    # Available models: "mobileclip_s0", "mobileclip_s1", "mobileclip_s2", "mobileclip_b", "mobileclip_blt"
    model, _, preprocess = mobileclip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()

    tokenizer = mobileclip.get_tokenizer(model_name)

    all_text_embeds = []
    all_image_embeds = []

    total_rows = len(texts)
    embedding_dim = None # Dynamically determine projection dimension

    # Wrap the loop with tqdm to generate a live progress bar
    for i in tqdm(range(total_rows), desc=f"Extracting {model_name} Embeddings"):
        text = texts[i]
        paths = image_paths_per_row[i]

        # --- 1. Process Text ---
        # Tokenizer directly outputs a tensor for MobileCLIP
        text_tokens = tokenizer([text]).to(device)

        with torch.no_grad():
            text_embed = model.encode_text(text_tokens).cpu()

            # Capture the exact embedding dimension on the first pass (often 512)
            if embedding_dim is None:
                embedding_dim = text_embed.shape[-1]

        all_text_embeds.append(text_embed)

        # --- 2. Process Multiple Images for this row ---
        valid_image_tensors = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                # Preprocess immediately returns a single tensor for the image
                img_tensor = preprocess(img)
                valid_image_tensors.append(img_tensor)
            except Exception as e:
                # Using tqdm.write keeps the progress bar from breaking apart
                tqdm.write(f"Row {i}: Error loading image at {path}: {e}")

        if len(valid_image_tensors) > 0:
            # Stack individual preprocessed tensors into a batch tensor
            image_inputs = torch.stack(valid_image_tensors).to(device)

            with torch.no_grad():
                # Get embeddings for all images in the row
                image_features = model.encode_image(image_inputs).cpu()

                # Calculate the mean across the batch dimension (dim=0)
                # keepdim=True preserves the shape as (1, embedding_dim)
                mean_image_embed = image_features.mean(dim=0, keepdim=True)
        else:
            # Fallback for completely failed rows
            dim_to_use = embedding_dim if embedding_dim is not None else 512
            mean_image_embed = torch.zeros((1, dim_to_use))
            tqdm.write(f"Warning: Row {i} had no valid images. Filled embedding with zeros.")

        all_image_embeds.append(mean_image_embed)

    # Combine all individual row vectors into final single tensors
    final_text_features = torch.cat(all_text_embeds, dim=0)
    final_image_features = torch.cat(all_image_embeds, dim=0)

    return final_text_features, final_image_features

# %% [markdown] cell 12
# ## extract the visual and textual embeddings

# %% code cell 13
image_lists = test_df['image_paths'].tolist()
text_list = test_df['report_text'].tolist()  # You can also use 'raw_report_text'

# 3. Run your multi-image embedding function
print("Starting embedding extraction process...")
text_features, image_features = get_mobileclip_embeddings_multi_image(
    image_paths_per_row=image_lists,
    texts=text_list,
)

# 4. Save the raw PyTorch tensors to disk
# This keeps the multi-dimensional arrays intact without altering or truncating them
torch.save(text_features, TEXT_EMBEDDING_PATH)
torch.save(image_features, IMAGE_EMBEDDING_PATH)

print("\nAll done! Embeddings extracted and saved successfully.")
print(f"Saved Text Tensor Shape: {text_features.shape}")   # Expected: [num_rows, 512]
print(f"Saved Image Tensor Shape: {image_features.shape}") # Expected: [num_rows, 512]

# %% code cell 14
import torch
import torch.nn.functional as F

# 1. Load your saved tensors
text_features = torch.load(TEXT_EMBEDDING_PATH)
image_features = torch.load(IMAGE_EMBEDDING_PATH)

# 2. Compute the cosine similarity between paired rows
# dim=1 ensures it computes similarity across the 512 embedding dimensions for each row
paired_similarities = F.cosine_similarity(text_features, image_features, dim=1)

print("Paired Similarities Tensor Shape:", paired_similarities.shape) # Expected: [num_rows]
print("Similarity of row 0:", paired_similarities[0].item())
