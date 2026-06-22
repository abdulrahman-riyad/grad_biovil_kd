# Auto-exported from project notebook.
# Source notebook: project baseline notebooks/MedCLIP_ViT_V1.ipynb
# Code cells: 12; markdown cells: 5
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
# NOTEBOOK_CELL_MAGIC: %%writefile requirements.txt
# numpy
# pandas
# Pillow
# requests
# tqdm
# wget
# nltk>=3.7
# scikit_learn>=1.1.2
# textaugment>=1.3.4
# timm>=0.6.11
# torch>=1.12.1
# torchvision>=0.13.1
# transformers>=4.23.1,<=4.24.0

# %% code cell 4
# NOTEBOOK_COMMAND: !pip install -r requirements.txt --force-reinstall

# %% code cell 5
# NOTEBOOK_COMMAND: !pip install medclip

# %% code cell 6
import ast
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional
import transformers
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from kaggle_secrets import UserSecretsClient
secret_label = "HF_TOKEN"
secret_value = UserSecretsClient().get_secret(secret_label)
os.environ["HF_TOKEN"] = secret_value
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

vision_backbone = "vit"

DEVICE = select_device()
# CSV expected to contain columns like:
# text or text_list, plus PA, AP, Lateral


TEXT_BATCH_SIZE = 32
MAX_TEXT_TOKENS = 128

# Evaluation / throughputC
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Using device: {DEVICE}")

# %% [markdown] cell 7
# ## Prepare Testset

# %% code cell 8
test_df = pd.read_csv("/kaggle/input/datasets/mohamed311ahmed/mimic-cxr-testsplit/kd_test_metadata.csv")

test_df.head()

# %% code cell 9
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

# %% code cell 10
# Assuming your dataframe is named 'df'
# Apply the cleaning function in place to the same column
test_df['image_paths'] = test_df['image_paths'].apply(clean_image_paths)

# Verify the result
print("Cleaned paths for the first row:")
print(test_df['image_paths'].iloc[0])

print("\nData type of the column cell:")
print(type(test_df['image_paths'].iloc[0]))

# %% [markdown] cell 11
# ## Get the model text encoder and image encoder

# %% code cell 12
def get_medclip_embeddings_multi_image(
    image_paths_per_row,
    texts,
    vision_backbone=vision_backbone
):
    """
    Extracts MedCLIP embeddings for text and computes the MEAN embedding for multiple corresponding images.

    Parameters:
    - image_paths_per_row: List of lists containing image file paths.
    - texts: List of text descriptions.
    - vision_backbone: 'vit' for Swin Transformer (default) or 'resnet' for ResNet-50.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialize the correct MedCLIP variant backbone
    if vision_backbone.lower() == "vit":
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    elif vision_backbone.lower() == "resnet":
        from medclip import MedCLIPVisionModel
        model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
    else:
        raise ValueError("vision_backbone must be either 'vit' or 'resnet'")

    # Load pretrained weights and send to device
    model.from_pretrained()
    model = model.to(device)
    model.eval()

    # 2. Initialize the unified processor (handles both images and text)
    processor = MedCLIPProcessor()

    all_text_embeds = []
    all_image_embeds = []

    total_rows = len(texts)

    # Wrap the loop with tqdm to generate a live progress bar
    for i in tqdm(range(total_rows), desc="Extracting MedCLIP Embeddings"):
        text = texts[i]
        paths = image_paths_per_row[i]

        # --- 1. Process Text ---
        # MedCLIP processor automatically handles text padding and truncation
        text_inputs = processor(text=[text], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_embed = model.encode_text(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            ).cpu()

        all_text_embeds.append(text_embed)

        # --- 2. Process Multiple Images for this row ---
        valid_images = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                valid_images.append(img)
            except Exception as e:
                # Using tqdm.write keeps the progress bar layout from breaking apart
                tqdm.write(f"Row {i}: Error loading image at {path}: {e}")

        if len(valid_images) > 0:
            # MedCLIPProcessor natively processes lists of images into a single batched tensor
            image_inputs = processor(images=valid_images, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].to(device)

            with torch.no_grad():
                image_features = model.encode_image(pixel_values).cpu()
                mean_image_embed = image_features.mean(dim=0, keepdim=True)
        else:
            # Dynamically grab the correct embedding dimension directly from the text features
            embedding_dim = text_embed.shape[-1]
            mean_image_embed = torch.zeros((1, embedding_dim))
            tqdm.write(f"Warning: Row {i} had no valid images. Filled embedding with zeros.")

        all_image_embeds.append(mean_image_embed)

    # Combine all individual row vectors into final single tensors
    final_text_features = torch.cat(all_text_embeds, dim=0)
    final_image_features = torch.cat(all_image_embeds, dim=0)

    return final_text_features, final_image_features

# %% [markdown] cell 13
# ## extract the visual and textual embeddings

# %% code cell 14
test_df = test_df.head(20)

# %% code cell 15
image_lists = test_df['image_paths'].tolist()
text_list = test_df['report_text'].tolist()  # You can also use 'raw_report_text'

# 3. Run your multi-image embedding function
print("Starting embedding extraction process...")
text_features, image_features = get_medclip_embeddings_multi_image(
    image_paths_per_row=image_lists,
    texts=text_list,
)

# 4. Save the raw PyTorch tensors to disk
# This keeps the multi-dimensional arrays intact without altering or truncating them
torch.save(text_features, 'medCLIP_ViT_mimic_cxr_text_embeddings.pt')
torch.save(image_features, 'medCLIP_ViT_mimic_cxr_image_embeddings.pt')

print("\nAll done! Embeddings extracted and saved successfully.")
print(f"Saved Text Tensor Shape: {text_features.shape}")   # Expected: [num_rows, 512]
print(f"Saved Image Tensor Shape: {image_features.shape}") # Expected: [num_rows, 512]

# %% code cell 16
import torch
import torch.nn.functional as F

# 1. Load your saved tensors
text_features = torch.load('/kaggle/working/medCLIP_ViT_mimic_cxr_text_embeddings.pt')
image_features = torch.load('/kaggle/working/medCLIP_ViT_mimic_cxr_image_embeddings.pt')

# 2. Compute the cosine similarity between paired rows
# dim=1 ensures it computes similarity across the 512 embedding dimensions for each row
paired_similarities = F.cosine_similarity(text_features, image_features, dim=1)

print("Paired Similarities Tensor Shape:", paired_similarities.shape) # Expected: [num_rows]
print("Similarity of row 0:", paired_similarities[0].item())
