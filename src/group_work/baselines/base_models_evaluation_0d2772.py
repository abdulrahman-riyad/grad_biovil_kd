# Auto-exported from teammate notebook.
# Source notebook: week4_structured_project/Final_ppt/materials/new_teammates_baselines_notebooks/base-models-evaluation-0d2772.ipynb
# Code cells: 12; markdown cells: 12
# Notebook shell commands and magics are preserved as comments.
# ruff: noqa
# pylint: skip-file

# %% [markdown] cell 1
# # Base Model Evaluation — CheXzero-small & ConVIRT
# Compares **CheXzero-small (ViT-B/32)** and **ConVIRT** on MIMIC-CXR.
#
# Pipeline:
# 1. Load test set
# 2. Extract batched text + image embeddings for each model
# 3. Save embeddings per model
# 4. Evaluate with paired cosine similarity and Recall@K

# %% [markdown] cell 2
# ## 1 · Installation

# %% code cell 3
# CheXzero — uses openai/CLIP as backbone
# NOTEBOOK_COMMAND: !pip install -q git+https://github.com/openai/CLIP.git

# ConVIRT — ClinicalBERT text encoder + ResNet-50 image encoder
# NOTEBOOK_COMMAND: !pip install -q transformers timm

# %% [markdown] cell 4
# ## 2 · Imports & Global Configuration

# %% code cell 5
import os
import re
import random
import glob
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms as T
import matplotlib.pyplot as plt

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, _ = torch.cuda.get_device_capability(0)
    if major < 7:
        print(f"GPU sm_{major}x unsupported — falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda")

DEVICE = select_device()
print(f"Using device: {DEVICE}")

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_PATH = "/kaggle/input/datasets/mohamed311ahmed/mimic-cxr-testsplit/kd_test_metadata.csv"

# Update these to your uploaded Kaggle dataset slugs
CHEXZERO_WEIGHTS_DIR = "/kaggle/input/models/marawanmogeb/chexzero-small/pytorch/default/1"   # folder containing .pt file(s)
CONVIRT_CKPT         = "/kaggle/input/models/marawanmogeb/convirt/pytorch/default/1/ConVIRT.pth"  # e.g. "/kaggle/input/convirt-weights/convirt_mimic.pt"

EMBED_DIR = "/kaggle/working/embeddings"
os.makedirs(EMBED_DIR, exist_ok=True)

# ── Hyper-params ──────────────────────────────────────────────────────────────
IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE  = 64
EMBED_DIM        = 512
CLINICALBERT     = "emilyalsentzer/Bio_ClinicalBERT"

# %% [markdown] cell 6
# ## 3 · Resolve CheXzero Checkpoint
# Auto-detects whichever `.pt` file exists in the uploaded Kaggle dataset folder.

# %% code cell 7
ckpt_files = glob.glob(f"{CHEXZERO_WEIGHTS_DIR}/*.pt")

if not ckpt_files:
    raise FileNotFoundError(
        f"No .pt files found in {CHEXZERO_WEIGHTS_DIR}.\n"
        "Upload your CheXzero weights as a Kaggle dataset and set CHEXZERO_WEIGHTS_DIR above."
    )

# If multiple checkpoints exist, pick the one with the highest AUC score in its filename
CHEXZERO_CKPT = sorted(ckpt_files)[-1]
print(f"Found {len(ckpt_files)} checkpoint(s). Using: {CHEXZERO_CKPT}")

# %% [markdown] cell 8
# ## 4 · Data Loading

# %% code cell 9
def clean_image_paths(val):
    """Handle PosixPath strings that survive CSV serialisation."""
    if isinstance(val, str):
        return re.findall(r"PosixPath\(['\"](.+?)['\"]\)", val)
    elif isinstance(val, (list, tuple)):
        return [str(p) for p in val]
    return val

test_df = pd.read_csv(CSV_PATH)
test_df["image_paths"] = test_df["image_paths"].apply(clean_image_paths)

image_lists = test_df["image_paths"].tolist()
text_list   = test_df["report_text"].tolist()

print(f"Test set size : {len(test_df):,} rows")
print(f"Sample paths  : {image_lists[0]}")
test_df.head(3)

# %% [markdown] cell 10
# ## 5 · Abstract Base Extractor

# %% code cell 11
class BaseEmbeddingExtractor(ABC):
    """Shared interface and batched inference logic for all extractors."""

    def __init__(self, device: torch.device, name: str):
        self.device = device
        self.name   = name

    @abstractmethod
    def _load_model(self): ...

    @abstractmethod
    def encode_text_batch(self, texts: List[str]) -> torch.Tensor: ...

    @abstractmethod
    def encode_image_batch(self, paths: List[str]) -> torch.Tensor: ...

    def extract(
        self,
        texts: List[str],
        image_path_lists: List[List[str]],
        text_batch_size: int = TEXT_BATCH_SIZE,
        image_batch_size: int = IMAGE_BATCH_SIZE,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print(f"\n[{self.name}] Extracting text embeddings…")
        text_embeds = self._run_text_batches(texts, text_batch_size)

        print(f"[{self.name}] Extracting image embeddings…")
        image_embeds = self._run_image_batches(image_path_lists, image_batch_size)

        return F.normalize(text_embeds, dim=-1), F.normalize(image_embeds, dim=-1)

    def _run_text_batches(self, texts, batch_size):
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="text batches"):
            with torch.no_grad():
                emb = self.encode_text_batch(texts[i : i + batch_size]).cpu()
            all_embeds.append(emb)
        return torch.cat(all_embeds, dim=0)

    def _run_image_batches(self, image_path_lists, batch_size):
        all_embeds, embed_dim = [], None
        for i, paths in enumerate(tqdm(image_path_lists, desc="image rows")):
            valid = [p for p in paths if os.path.isfile(p)]
            if not valid:
                tqdm.write(f"Row {i}: no valid images — zero vector inserted.")
                all_embeds.append(torch.zeros(1, embed_dim or EMBED_DIM))
                continue

            row_embeds = []
            for j in range(0, len(valid), batch_size):
                with torch.no_grad():
                    emb = self.encode_image_batch(valid[j : j + batch_size]).cpu()
                row_embeds.append(emb)
                embed_dim = emb.shape[-1]
            all_embeds.append(torch.cat(row_embeds, dim=0).mean(dim=0, keepdim=True))

        return torch.cat(all_embeds, dim=0)

    def save(self, text_feats: torch.Tensor, image_feats: torch.Tensor):
        prefix = self.name.replace(" ", "_").replace("-", "_").lower()
        torch.save(text_feats,  f"{EMBED_DIR}/{prefix}_text.pt")
        torch.save(image_feats, f"{EMBED_DIR}/{prefix}_image.pt")
        print(f"[{self.name}] Saved embeddings → {EMBED_DIR}/{prefix}_{{text,image}}.pt")

# %% [markdown] cell 12
# ## 6 · Model A — CheXzero-small (ViT-B/32 fine-tuned on MIMIC-CXR)

# %% code cell 13
import clip

class CheXzeroExtractor(BaseEmbeddingExtractor):
    """
    CheXzero-small: CLIP ViT-B/32 fine-tuned on MIMIC-CXR radiology reports.
    Uses CLIP's own preprocess (224×224) to match the positional embedding table.
    """
    CONTEXT_LEN = 77

    def __init__(self, device, ckpt_path: str):
        super().__init__(device, "CheXzero-small")
        self.ckpt_path = ckpt_path
        self._load_model()

    def _load_model(self):
        # clip.load returns the correct preprocess for ViT-B/32 (224×224)
        self.model, clip_preprocess = clip.load("ViT-B/32", device=self.device, jit=False)

        state = torch.load(self.ckpt_path, map_location=self.device)
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[CheXzero-small] Loaded weights from {self.ckpt_path}")

        # Prepend grayscale→RGB so CXR images work with CLIP's normalisation
        self.preprocess = T.Compose([
            T.Grayscale(num_output_channels=3),
            clip_preprocess,
        ])

    def encode_text_batch(self, texts):
        tokens = clip.tokenize(texts, context_length=self.CONTEXT_LEN, truncate=True).to(self.device)
        return self.model.encode_text(tokens).float()

    def encode_image_batch(self, paths):
        tensors = [self.preprocess(Image.open(p).convert("RGB")) for p in paths]
        return self.model.encode_image(torch.stack(tensors).to(self.device)).float()


chexzero_extractor = CheXzeroExtractor(DEVICE, CHEXZERO_CKPT)

# %% code cell 14
results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

text_feats, image_feats = chexzero_extractor.extract(text_list, image_lists)
chexzero_extractor.save(text_feats, image_feats)
results[chexzero_extractor.name] = (text_feats, image_feats)
print(f"  text  shape : {text_feats.shape}")
print(f"  image shape : {image_feats.shape}")

# %% [markdown] cell 15
# ## 7 · Model B — ConVIRT (ResNet-50 + ClinicalBERT)
#
# **Weights**: Set `CONVIRT_CKPT` at the top to your Kaggle dataset path.
# If `None`, runs with ImageNet + ClinicalBERT initialisations (untrained baseline).

# %% code cell 16
import glob, os
for f in glob.glob(f"{EMBED_DIR}/convirt_*.pt"):
    os.remove(f)
    print(f"Deleted: {f}")

# %% code cell 17
import timm
from transformers import AutoTokenizer, AutoModel


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
    def forward(self, x): return self.net(x)


class ConVIRTModel(nn.Module):
    TEXT_ENCODER_NAME = "allenai/biomed_roberta_base"

    def __init__(self, img_proj_dims, txt_proj_dims):
        super().__init__()
        self.image_encoder = timm.create_model("resnet50", pretrained=False, num_classes=0, global_pool="avg")
        self.image_proj    = ProjectionMLP(*img_proj_dims)
        self.text_encoder  = AutoModel.from_pretrained(self.TEXT_ENCODER_NAME)
        self.text_proj     = ProjectionMLP(*txt_proj_dims)

    def encode_image(self, pixel_values):
        return self.image_proj(self.image_encoder(pixel_values))

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_proj(out.pooler_output)


def remap_benchx_state_dict(state: dict, verbose: bool = True) -> dict:
    if verbose:
        prefixes = sorted(set(".".join(k.split(".")[:2]) for k in state.keys()))
        print("[ConVIRT] Checkpoint key prefixes:")
        for p in prefixes:
            sample = next(k for k in state.keys() if k.startswith(p))
            print(f"  {p}  (e.g. {sample})")

    remap_rules = [
        ("linguistic.encoder.", "text_encoder."),
        ("lin_proj.",           "text_proj.net."),
        ("visual.model.",       "image_encoder."),
        ("vis_proj.",           "image_proj.net."),
    ]

    remapped, skipped = {}, []
    for k, v in state.items():
        new_key = None
        for src, dst in remap_rules:
            if k.startswith(src):
                new_key = dst + k[len(src):]
                break
        if new_key:
            remapped[new_key] = v
        else:
            skipped.append(k)

    if verbose:
        print(f"\n[ConVIRT] Remapped : {len(remapped)} keys")
        print(f"[ConVIRT] Skipped  : {len(skipped)} keys")

    return remapped


def infer_proj_dims(state: dict, proj_prefix: str) -> tuple:
    w0 = state[f"{proj_prefix}.net.0.weight"]
    w2 = state[f"{proj_prefix}.net.2.weight"]
    return w0.shape[1], w0.shape[0], w2.shape[0]


class ConVIRTExtractor(BaseEmbeddingExtractor):
    MAX_TEXT_LEN = 128

    def __init__(self, device, ckpt_path=None):
        super().__init__(device, "ConVIRT")
        self.ckpt_path = ckpt_path
        self._load_model()

    def _load_model(self):
        if self.ckpt_path and os.path.isfile(self.ckpt_path):
            raw   = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            state = remap_benchx_state_dict(raw["model"], verbose=True)

            img_proj_dims = infer_proj_dims(state, "image_proj")
            txt_proj_dims = infer_proj_dims(state, "text_proj")
            print(f"[ConVIRT] image_proj dims : {img_proj_dims}")
            print(f"[ConVIRT] text_proj dims  : {txt_proj_dims}")

            self.model = ConVIRTModel(img_proj_dims, txt_proj_dims).to(self.device)

            missing, unexpected = self.model.load_state_dict(state, strict=False)
            unexpected = [k for k in unexpected if "position_ids" not in k]
            if missing:
                print(f"[ConVIRT] ⚠ Still missing : {missing[:5]}{'…' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"[ConVIRT] ⚠ Still unexpected: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
            if not missing and not unexpected:
                print("[ConVIRT] ✓ All keys matched perfectly.")
            print(f"[ConVIRT] Loaded checkpoint from {self.ckpt_path}")
        else:
            print("[ConVIRT] ⚠ No checkpoint — using random + BioMed-RoBERTa initialisations.")
            self.model = ConVIRTModel(
                img_proj_dims=(2048, 768, 768),
                txt_proj_dims=(768,  768, 768),
            ).to(self.device)

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(ConVIRTModel.TEXT_ENCODER_NAME)

        # Exact BenchX preprocessing from config: resize=256, crop_size=224, ImageNet stats
        # No Grayscale transform — .convert("RGB") in encode_image_batch handles CXR images
        self.preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[ConVIRT] Ready.")

    def encode_text_batch(self, texts):
        enc = self.tokenizer(
            texts, max_length=self.MAX_TEXT_LEN,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return self.model.encode_text(
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device)
        )

    def encode_image_batch(self, paths):
        # .convert("RGB") correctly replicates single-channel CXR to 3 channels
        tensors = [self.preprocess(Image.open(p).convert("RGB")) for p in paths]
        return self.model.encode_image(torch.stack(tensors).to(self.device))


convirt_extractor = ConVIRTExtractor(DEVICE, CONVIRT_CKPT)
text_feats, image_feats = convirt_extractor.extract(text_list, image_lists)
convirt_extractor.save(text_feats, image_feats)
results["ConVIRT"] = (text_feats, image_feats)

# %% [markdown] cell 18
# ## 8 · Run Extraction

# %% [markdown] cell 19
# results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
#
# for extractor in [chexzero_extractor, convirt_extractor]:
#     text_feats, image_feats = extractor.extract(text_list, image_lists)
#     extractor.save(text_feats, image_feats)
#     results[extractor.name] = (text_feats, image_feats)
#     print(f"  text  shape : {text_feats.shape}")
#     print(f"  image shape : {image_feats.shape}")

# %% [markdown] cell 20
# ## 9 · Evaluation

# %% code cell 21
def recall_at_k(text_feats, image_feats, ks=(1, 5, 10)):
    """Image→Text Recall@K. Assumes L2-normalised inputs."""
    sim = F.normalize(image_feats, dim=-1) @ F.normalize(text_feats, dim=-1).T
    N      = sim.shape[0]
    labels = torch.arange(N)
    return {
        f"R@{k}": (sim.topk(min(k, N), dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        for k in ks
    }

def paired_cosine_sim(tf, imf):
    return F.cosine_similarity(tf, imf, dim=1)


eval_rows = []
for model_name, (tf, imf) in results.items():
    sims = paired_cosine_sim(tf, imf)
    row  = {
        "Model"         : model_name,
        "Mean Cos Sim"  : sims.mean().item(),
        "Median Cos Sim": sims.median().item(),
        **recall_at_k(tf, imf),
    }
    eval_rows.append(row)
    print(f"\n{model_name}")
    for k, v in row.items():
        if k != "Model":
            print(f"  {k:20s}: {v:.4f}")

eval_df = pd.DataFrame(eval_rows).set_index("Model")
display(eval_df.style.highlight_max(color="lightgreen", axis=0))

# %% code cell 22
raw = torch.load(CONVIRT_CKPT, map_location="cpu", weights_only=False)

cfg = raw["config"]
print(type(cfg))
print(cfg)

# %% [markdown] cell 23
# ## 10 · Visualisation

# %% code cell 24
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Cosine similarity distributions
ax = axes[0]
for model_name, (tf, imf) in results.items():
    ax.hist(paired_cosine_sim(tf, imf).numpy(), bins=50, alpha=0.6, label=model_name, density=True)
ax.set_xlabel("Paired Cosine Similarity")
ax.set_ylabel("Density")
ax.set_title("Paired Cosine Similarity Distribution")
ax.legend()

# (b) Recall@K
ax      = axes[1]
r_cols  = [c for c in eval_df.columns if c.startswith("R@")]
x       = np.arange(len(r_cols))
width   = 0.35
for i, (model_name, row) in enumerate(eval_df.iterrows()):
    ax.bar(x + i * width, [row[c] for c in r_cols], width=width, label=model_name, alpha=0.85)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(r_cols)
ax.set_ylabel("Recall")
ax.set_title("Image→Text Recall@K")
ax.legend()

plt.tight_layout()
plt.savefig(f"{EMBED_DIR}/evaluation_comparison.png", dpi=150)
plt.show()

eval_df.to_csv(f"{EMBED_DIR}/evaluation_summary.csv")
print(f"Saved → {EMBED_DIR}/evaluation_{{comparison.png,summary.csv}}")
