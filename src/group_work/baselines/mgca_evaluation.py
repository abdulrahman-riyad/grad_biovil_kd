# Auto-exported from teammate notebook.
# Source notebook: week4_structured_project/Final_ppt/materials/new_teammates_baselines_notebooks/mgca-evaluation.ipynb
# Code cells: 19; markdown cells: 11
# Notebook shell commands and magics are preserved as comments.
# ruff: noqa
# pylint: skip-file

# %% [markdown] cell 1
# # MGCA Evaluation on MIMIC-CXR
# Evaluates **MGCA-ResNet50** and **MGCA-ViT** (Multi-Granularity Cross-modal Alignment) on MIMIC-CXR.
#
# Checkpoints: [youngzhou12/MGCA-resnet50](https://huggingface.co/youngzhou12/MGCA-resnet50) and [youngzhou12/MGCA-vit](https://huggingface.co/youngzhou12/MGCA-vit) (BenchX re-trained on MIMIC-CXR).
#
# Pipeline:
# 1. Load test set
# 2. Inspect checkpoint key structure
# 3. Extract batched text + image embeddings for each variant
# 4. Evaluate with paired cosine similarity and Recall@K

# %% [markdown] cell 2
# ## 1 · Installation

# %% code cell 3
# NOTEBOOK_COMMAND: !pip install -q transformers timm

# %% [markdown] cell 4
# ## 2 · Imports & Global Configuration

# %% code cell 5
import os
import re
import random
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

# Set these to your uploaded Kaggle dataset paths
MGCA_RESNET_CKPT = "/kaggle/input/models/marawanmogeb/mgca/pytorch/default/1/MGCA-resnet50.pth"
MGCA_VIT_CKPT    = "/kaggle/input/models/marawanmogeb/mgca/pytorch/default/1/MGCA-vit.pth"

EMBED_DIR = "/kaggle/working/embeddings"
os.makedirs(EMBED_DIR, exist_ok=True)

# ── Hyper-params ──────────────────────────────────────────────────────────────
IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE  = 64
EMBED_DIM        = 512    # will be overridden from checkpoint
CLINICALBERT     = "emilyalsentzer/Bio_ClinicalBERT"

# %% [markdown] cell 6
# ## 3 · Data Loading

# %% code cell 7
def clean_image_paths(val):
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

# %% [markdown] cell 8
# ## 4 · Abstract Base Extractor

# %% code cell 9
class BaseEmbeddingExtractor(ABC):
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
        print(f"[{self.name}] Saved → {EMBED_DIR}/{prefix}_{{text,image}}.pt")

# %% [markdown] cell 10
# ## 5 · Inspect Checkpoint Key Structure
# Run this before defining the model to confirm BenchX key names and projection dims.

# %% code cell 11
def inspect_checkpoint(ckpt_path: str, label: str, n_keys: int = 25):
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Top-level keys : {list(raw.keys())}")

    # Print saved config if present
    if "config" in raw:
        print(f"\nConfig:\n{raw['config']}")

    state = raw.get("state_dict", raw.get("model", raw))

    prefixes = sorted(set(".".join(k.split(".")[:2]) for k in state.keys()))
    print(f"\nKey prefixes ({len(prefixes)} total):")
    for p in prefixes:
        sample = next(k for k in state.keys() if k.startswith(p))
        shape  = tuple(state[sample].shape)
        print(f"  {p:<40s}  e.g. {sample}  {shape}")

    print(f"\nFirst {n_keys} keys:")
    for k in list(state.keys())[:n_keys]:
        print(f"  {k:<60s}  {tuple(state[k].shape)}")

inspect_checkpoint(MGCA_RESNET_CKPT, "MGCA-ResNet50")
inspect_checkpoint(MGCA_VIT_CKPT,    "MGCA-ViT")

# %% code cell 12
def audit_state_dict_loading(model, state, name):
    loaded_by_module = {}
    for k in state:
        top = k.split(".")[0]
        loaded_by_module[top] = loaded_by_module.get(top, 0) + 1

    expected_by_module = {}
    for k, _ in model.named_parameters():
        top = k.split(".")[0]
        expected_by_module[top] = expected_by_module.get(top, 0) + 1
    for k, _ in model.named_buffers():
        top = k.split(".")[0]
        expected_by_module[top] = expected_by_module.get(top, 0) + 1

    print(f"\n[{name}] Key audit:")
    print(f"  {'Module':<20s}  {'In checkpoint':>15}  {'Expected by model':>18}")
    for mod in sorted(expected_by_module):
        ckpt_count = loaded_by_module.get(mod, 0)
        exp_count  = expected_by_module[mod]
        flag = "✓" if ckpt_count >= exp_count * 0.9 else "⚠ MISMATCH"
        print(f"  {mod:<20s}  {ckpt_count:>15}  {exp_count:>18}  {flag}")

# %% [markdown] cell 13
# ## 6 · MGCA Model & Extractor
#
# **Architecture** (Shih et al., NeurIPS 2022):
# - Image encoder: ResNet-50 **or** ViT-B/16 → global pool → projection MLP → embed_dim
# - Text encoder: ClinicalBERT → pooler_output → projection MLP → embed_dim
# - Multi-granularity alignment: instance + prototype + token level
#
# For embedding extraction we only need the encoder + projection head (no prototype/token heads needed at inference).
#
# > **Update `remap_rules` below** after running the inspection cell above if key names differ.

# %% code cell 14
import timm
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torchvision.models as tv_models
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    """2-layer projection head — dims inferred from checkpoint."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
    def forward(self, x): return self.net(x)



class MGCAModel(nn.Module):
    TEXT_ENCODER_NAME = "emilyalsentzer/Bio_ClinicalBERT"

    def __init__(self, backbone: str, img_proj_dims: tuple, txt_proj_dims: tuple):
        super().__init__()
        self.backbone_name = backbone

        # Keep global_pool="" to extract full feature maps matching the checkpoint layout
        self.image_encoder = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="")

        # --- FIX 1: Force HuggingFace to instantiate only 6 layers to match MGCA ---
        config = AutoConfig.from_pretrained(self.TEXT_ENCODER_NAME)
        config.num_hidden_layers = 6
        self.text_encoder = AutoModel.from_pretrained(self.TEXT_ENCODER_NAME, config=config)

        self.image_proj = ProjectionMLP(*img_proj_dims)
        self.text_proj  = ProjectionMLP(*txt_proj_dims)

    def encode_image(self, x):
        feat = self.image_encoder(x)

        # --- FIX 2: Vectorize/Pool raw feature maps safely before projection head ---
        if "resnet" in self.backbone_name.lower():
            # Convert 4D tensor [B, 2048, 7, 7] -> Global Average Pool -> [B, 2048]
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        else:
            # Convert 3D ViT tokens [B, 197, 768] -> Extract [CLS] token at index 0 -> [B, 768]
            feat = feat[:, 0, :]

        return self.image_proj(feat)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # --- FIX 3: Extract the raw [CLS] token from last_hidden_state instead of pooler_output ---
        cls_token = out.last_hidden_state[:, 0, :]
        return self.text_proj(cls_token)

def remap_benchx_state_dict(state: dict, remap_rules: list, verbose: bool = True) -> dict:
    if verbose:
        prefixes = sorted(set(".".join(k.split(".")[:2]) for k in state.keys()))
        print("Checkpoint key prefixes:")
        for p in prefixes:
            sample = next(k for k in state.keys() if k.startswith(p))
            print(f"  {p}  (e.g. {sample})")

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
        print(f"\nRemapped : {len(remapped)} keys")
        print(f"Skipped  : {len(skipped)} keys (prototype/token heads, not needed for inference)")
        if skipped:
            print(f"  e.g. {skipped[:3]}")
    return remapped


def infer_proj_dims(state: dict, proj_prefix: str) -> tuple:
    """Dynamically read dimensions from the first and last weight tensors found."""
    w_keys = sorted([k for k in state.keys() if k.startswith(f"{proj_prefix}.net.") and k.endswith(".weight")])
    if not w_keys:
        raise KeyError(f"No weights found for prefix: {proj_prefix}")

    w_first = state[w_keys[0]]
    w_last  = state[w_keys[-1]]

    # Returns: (in_dim, hidden_dim, out_dim)
    return w_first.shape[1], w_first.shape[0], w_last.shape[0]


class MGCAExtractor(BaseEmbeddingExtractor):
    MAX_TEXT_LEN = 128

    # ── Update these after running the inspection cell ──────────────────────
    # Maps BenchX checkpoint key prefixes → our module attribute names.
    # Pattern from BenchX ConVIRT: linguistic/visual → text/image encoder.
    # MGCA likely uses img_encoder_q / text_encoder_q — adjust if needed.
    REMAP_RULES = [
        ("img_encoder_q.global_embed.head.", "image_proj.net."),
        ("text_encoder_q.global_embed.head.", "text_proj.net."),
        ("img_encoder_q.model.", "image_encoder."),
        ("text_encoder_q.model.", "text_encoder."),
    ]

    def __init__(self, device, ckpt_path: str, backbone: str, name: str):
        super().__init__(device, name)
        self.ckpt_path = ckpt_path
        self.backbone  = backbone
        self._load_model()

    def _load_model(self):
        raw = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        state_dict = raw.get("state_dict", raw.get("model", raw))

        # Strip any leading 'model.' prefixes if present in lightning states
        cleaned_state = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}

        state = remap_benchx_state_dict(cleaned_state, self.REMAP_RULES, verbose=False)
        audit_state_dict_loading(MGCAModel(self.backbone, (2048,2048,128), (768,2048,128)), state, self.name)

        # Force checkpoint projection keys to match our ProjectionMLP (.0 and .2)
        for proj in ["image_proj", "text_proj"]:
            w_keys = sorted([k for k in state.keys() if k.startswith(f"{proj}.net.") and k.endswith(".weight")])
            b_keys = sorted([k for k in state.keys() if k.startswith(f"{proj}.net.") and k.endswith(".bias")])
            if len(w_keys) >= 2:
                w_last = w_keys[-1]
                if w_last != f"{proj}.net.2.weight":
                    state[f"{proj}.net.2.weight"] = state.pop(w_last)
                if len(b_keys) >= 2:
                    b_last = b_keys[-1]
                    if b_last != f"{proj}.net.2.bias":
                        state[f"{proj}.net.2.bias"] = state.pop(b_last)

        img_proj_dims = infer_proj_dims(state, "image_proj")
        txt_proj_dims = infer_proj_dims(state, "text_proj")
        print(f"[{self.name}] image_proj dims : {img_proj_dims}")
        print(f"[{self.name}] text_proj dims  : {txt_proj_dims}")

        self.model = MGCAModel(self.backbone, img_proj_dims, txt_proj_dims).to(self.device)

        # Load weights and observe missing keys
        missing, unexpected = self.model.load_state_dict(state, strict=False)

        # Filter out harmless metadata keys
        missing = [k for k in missing if not k.endswith(".position_ids") and "num_batches_tracked" not in k]
        unexpected = [k for k in unexpected if not k.endswith(".position_ids")]

        if missing:
            print(f"[{self.name}] ⚠ Missing Layer Keys (Untrained!): {len(missing)} keys left unassigned.")
            # --- ADD THIS LOOP TO PRINT THE RELEVANT KEYS ---
            for k in missing:
                print(f"   -> Missing: {k}")
        if not missing:
            print(f"[{self.name}] ✓ All core backbone layers successfully mapped and loaded!")

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MGCAModel.TEXT_ENCODER_NAME)

        self.preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        print(f"[{self.name}] Ready.")

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
        tensors = [self.preprocess(Image.open(p).convert("RGB")) for p in paths]
        return self.model.encode_image(torch.stack(tensors).to(self.device))


mgca_resnet_extractor = MGCAExtractor(
    DEVICE, MGCA_RESNET_CKPT, backbone="resnet50", name="MGCA-ResNet50"
)
mgca_vit_extractor = MGCAExtractor(
    DEVICE, MGCA_VIT_CKPT, backbone="vit_base_patch16_224", name="MGCA-ViT"
)

# %% [markdown] cell 15
# ## 7 · Run Extraction

# %% code cell 16
results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

for extractor in [mgca_resnet_extractor, mgca_vit_extractor]:
    text_feats, image_feats = extractor.extract(text_list, image_lists)
    extractor.save(text_feats, image_feats)
    results[extractor.name] = (text_feats, image_feats)
    print(f"  text  shape : {text_feats.shape}")
    print(f"  image shape : {image_feats.shape}")

# %% [markdown] cell 17
# ## 8 · Evaluation

# %% code cell 18
import numpy as np

def recall_at_k(text_feats, image_feats, text_list, ks=(1, 5, 10)):
    """Image→Text Recall@K with mandatory L2 normalization."""
    # --- CRITICAL FIX: L2 Normalize features along the embedding dimension ---
    image_feats_norm = F.normalize(image_feats, p=2, dim=1)
    text_feats_norm  = F.normalize(text_feats, p=2, dim=1)

    # Compute true cosine similarity matrix for all pairs
    sim = image_feats_norm @ text_feats_norm.T

    # Get top-K predicted indices
    topk_indices = sim.topk(max(ks), dim=1).indices

    text_arr = np.array(text_list)
    target_texts = text_arr[:, None] # Shape: [N, 1]

    results = {}
    for k in ks:
        k_indices = topk_indices[:, :k]
        retrieved_texts = text_arr[k_indices.cpu().numpy()]

        # Check if any retrieved report matches the target report string
        matches = (retrieved_texts == target_texts).any(axis=1)
        results[f"R@{k}"] = matches.mean()

    return results

def paired_cosine_sim(tf, imf):
    return F.cosine_similarity(tf, imf, dim=1)

# Recalculate metrics
eval_rows = []
for model_name, (tf, imf) in results.items():
    sims = paired_cosine_sim(tf, imf)
    row  = {
        "Model"          : model_name,
        "Mean Cos Sim"   : sims.mean().item(),
        "Median Cos Sim" : sims.median().item(),
        **recall_at_k(tf, imf, text_list),
    }
    eval_rows.append(row)
    print(f"\n{model_name}")
    for k, v in row.items():
        if k != "Model":
            print(f"  {k:20s}: {v:.4f}")

eval_df = pd.DataFrame(eval_rows).set_index("Model")
display(eval_df.style.highlight_max(color="lightgreen", axis=0))

# %% [markdown] cell 19
# ## 9 · Visualisation

# %% code cell 20
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
ax     = axes[1]
r_cols = [c for c in eval_df.columns if c.startswith("R@")]
x      = np.arange(len(r_cols))
width  = 0.35
for i, (model_name, row) in enumerate(eval_df.iterrows()):
    ax.bar(x + i * width, [row[c] for c in r_cols], width=width, label=model_name, alpha=0.85)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(r_cols)
ax.set_ylabel("Recall")
ax.set_title("Image→Text Recall@K")
ax.legend()

plt.tight_layout()
plt.savefig(f"{EMBED_DIR}/mgca_evaluation.png", dpi=150)
plt.show()

eval_df.to_csv(f"{EMBED_DIR}/mgca_evaluation_summary.csv")
print(f"Saved → {EMBED_DIR}/mgca_evaluation{{.png,_summary.csv}}")

# %% code cell 21
print("--- 500 Sample Diagnostic Test ---")
for model_name, (tf, imf) in results.items():
    # Slice the first 500 embeddings and texts
    subset_tf = tf[:500]
    subset_imf = imf[:500]
    subset_texts = text_list[:500]

    # Run the exact same metric
    metrics = recall_at_k(subset_tf, subset_imf, subset_texts)

    print(f"\n{model_name} (N=500):")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

# %% [markdown] cell 22
# ## Second Implementation

# %% code cell 23
# ── Cell 1: Imports and Configuration ──
import os
import re
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as T
import timm
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Device and execution setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT_BATCH_SIZE = 32
OUTPUT_DIR = Path('mgca_embeddings_robust')
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Using device: {DEVICE}")
print(f"Embeddings will be saved to: {OUTPUT_DIR.resolve()}")

# ── UPDATED CHECKPOINT PATHS ──
MGCA_VARIANTS = [
    {
        'name': 'MGCA-ResNet50',
        'ckpt': '/kaggle/input/models/marawanmogeb/mgca/pytorch/default/1/MGCA-resnet50.pth',
        'backbone': 'resnet50'
    },
    {
        'name': 'MGCA-ViT',
        'ckpt': '/kaggle/input/models/marawanmogeb/mgca/pytorch/default/1/MGCA-vit.pth',
        'backbone': 'vit_base_patch16_224'
    }
]

# %% code cell 24
# ── Cell 2: Load and Clean Dataset ──
test_df = pd.read_csv('/kaggle/input/datasets/mohamed311ahmed/mimic-cxr-testsplit/kd_test_metadata.csv')

def clean_image_paths(val):
    if isinstance(val, str):
        cleaned = re.findall(r"PosixPath\(['\"](.+?)['\"]\)", val)
        if cleaned: return cleaned
        try:
            parsed = ast.literal_eval(val)
            return [str(p) for p in parsed]
        except Exception:
            return [val]
    elif isinstance(val, (list, tuple)):
        return [str(p) for p in val]
    return val

test_df['image_paths'] = test_df['image_paths'].apply(clean_image_paths)

# Sync lists directly from dataframe
image_lists = test_df['image_paths'].tolist()
text_list   = test_df['report_text'].tolist()

print(f"Loaded {len(test_df)} studies.")

# %% code cell 25
def audit_state_dict_loading(model, state, name):
    loaded_by_module = {}
    for k in state:
        top = k.split(".")[0]
        loaded_by_module[top] = loaded_by_module.get(top, 0) + 1

    expected_by_module = {}
    for k, _ in model.named_parameters():
        top = k.split(".")[0]
        expected_by_module[top] = expected_by_module.get(top, 0) + 1
    for k, _ in model.named_buffers():
        top = k.split(".")[0]
        expected_by_module[top] = expected_by_module.get(top, 0) + 1

    print(f"\n[{name}] Key audit:")
    print(f"  {'Module':<20s}  {'In checkpoint':>15}  {'Expected by model':>18}")
    for mod in sorted(expected_by_module):
        ckpt_count = loaded_by_module.get(mod, 0)
        exp_count  = expected_by_module[mod]
        flag = "✓" if ckpt_count >= exp_count * 0.9 else "⚠ MISMATCH"
        print(f"  {mod:<20s}  {ckpt_count:>15}  {exp_count:>18}  {flag}")

# %% code cell 26
# ── Cell 3: MGCA Architecture & Checkpoint Loader ──
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
    def forward(self, x): return self.net(x)

class MGCAModel(nn.Module):
    TEXT_ENCODER_NAME = "emilyalsentzer/Bio_ClinicalBERT"

    def __init__(self, backbone: str, img_proj_dims: tuple, txt_proj_dims: tuple):
        super().__init__()
        self.backbone_name = backbone
        self.image_encoder = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="")

        # Truncate text encoder to exactly 6 layers
        config = AutoConfig.from_pretrained(self.TEXT_ENCODER_NAME)
        config.num_hidden_layers = 6
        self.text_encoder = AutoModel.from_pretrained(self.TEXT_ENCODER_NAME, config=config)

        self.image_proj = ProjectionMLP(*img_proj_dims)
        self.text_proj  = ProjectionMLP(*txt_proj_dims)

    def encode_image(self, x):
        feat = self.image_encoder(x)
        if "resnet" in self.backbone_name.lower():
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        else:
            feat = feat[:, 0, :]
        return self.image_proj(feat)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :] # Raw [CLS] Token
        return self.text_proj(cls_token)

def load_mgca_checkpoint(ckpt_path: str, backbone: str, device: torch.device):
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = raw.get("state_dict", raw.get("model", raw))

    state = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    remap_rules = [
        ("img_encoder_q.global_embed.head.", "image_proj.net."),
        ("text_encoder_q.global_embed.head.", "text_proj.net."),
        ("img_encoder_q.model.", "image_encoder."),
        ("text_encoder_q.model.", "text_encoder."),
    ]

    remapped = {}
    for k, v in state.items():
        new_key = k
        for src, dst in remap_rules:
            if k.startswith(src):
                new_key = dst + k[len(src):]
                break
        remapped[new_key] = v
    state = remapped

    # Align Projection Head Keys
    for proj in ["image_proj", "text_proj"]:
        w_keys = sorted([k for k in state.keys() if k.startswith(f"{proj}.net.") and k.endswith(".weight")])
        b_keys = sorted([k for k in state.keys() if k.startswith(f"{proj}.net.") and k.endswith(".bias")])
        if len(w_keys) >= 2 and w_keys[-1] != f"{proj}.net.2.weight":
            state[f"{proj}.net.2.weight"] = state.pop(w_keys[-1])
        if len(b_keys) >= 2 and b_keys[-1] != f"{proj}.net.2.bias":
            state[f"{proj}.net.2.bias"] = state.pop(b_keys[-1])

    w0 = state["image_proj.net.0.weight"]
    w2 = state["image_proj.net.2.weight"]
    img_proj_dims = (w0.shape[1], w0.shape[0], w2.shape[0])

    w0_t = state["text_proj.net.0.weight"]
    w2_t = state["text_proj.net.2.weight"]
    txt_proj_dims = (w0_t.shape[1], w0_t.shape[0], w2_t.shape[0])

    model = MGCAModel(backbone, img_proj_dims, txt_proj_dims).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# %% code cell 27
# ── Cell 4: Robust Extraction Function ──
@torch.no_grad()
def extract_mgca_embeddings(
    image_paths_per_row: list, texts: list, model_name: str, ckpt_path: str, backbone: str, device: torch.device
) -> dict:
    print(f"\nProcessing: {model_name}...")
    model = load_mgca_checkpoint(ckpt_path, backbone, device)
    tokenizer = AutoTokenizer.from_pretrained(MGCAModel.TEXT_ENCODER_NAME)
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    text_feats = []
    for i in tqdm(range(0, len(texts), TEXT_BATCH_SIZE), desc="Text batches"):
        batch = texts[i : i + TEXT_BATCH_SIZE]
        enc = tokenizer(batch, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        projected = model.encode_text(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        text_feats.append(F.normalize(projected, p=2, dim=-1).cpu())
    text_embeddings = torch.cat(text_feats, dim=0)

    image_feats = []
    for paths in tqdm(image_paths_per_row, desc="Image Reports"):
        view_feats = []
        for p in paths:
            try:
                tensor = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                projected = model.encode_image(tensor)
                view_feats.append(F.normalize(projected, p=2, dim=-1))
            except Exception as e:
                pass # Skip corrupted single images silently to preserve the run

        if view_feats:
            pooled = torch.stack(view_feats, dim=0).mean(dim=0)
            pooled = F.normalize(pooled, p=2, dim=-1) # Re-normalize after averaging
        else:
            pooled = torch.zeros(1, model.image_proj.net[-1].out_features, device=device)

        image_feats.append(pooled.cpu())

    image_embeddings = torch.cat(image_feats, dim=0)
    del model
    torch.cuda.empty_cache()
    return {'text': text_embeddings, 'image': image_embeddings}

# %% code cell 28
# ── Cell 5: Execution Loop ──
results_summary = []

for variant in MGCA_VARIANTS:
    name, ckpt = variant['name'], variant['ckpt']

    if not Path(ckpt).exists():
        print(f"⚠️ Checkpoint missing for {name}. Ensure {ckpt} exists.")
        continue

    t_path = OUTPUT_DIR / f"{name}_text.pt"
    i_path = OUTPUT_DIR / f"{name}_image.pt"

    if t_path.exists() and i_path.exists():
        print(f"[SKIP] {name} — already saved.")
        results_summary.append({'name': name, 't_path': t_path, 'i_path': i_path})
        continue

    try:
        embs = extract_mgca_embeddings(image_lists, text_list, name, ckpt, variant['backbone'], DEVICE)
        torch.save(embs['text'], t_path)
        torch.save(embs['image'], i_path)
        results_summary.append({'name': name, 't_path': t_path, 'i_path': i_path})
        print(f"✅ Saved features for {name}.")
    except Exception as exc:
        print(f"❌ [ERROR] Processing skipped for {name}: {exc}")

# %% code cell 29
# ── Cell 6: Evaluation Metrics (Recall@K) ──
def recall_at_k(text_feats, image_feats, text_targets, ks=(1, 5, 10)):
    """L2 Normalization & Exact String Match Retrieval."""
    image_feats_norm = F.normalize(image_feats, p=2, dim=1)
    text_feats_norm  = F.normalize(text_feats, p=2, dim=1)

    sim = image_feats_norm @ text_feats_norm.T
    topk_indices = sim.topk(max(ks), dim=1).indices

    text_arr = np.array(text_targets)
    target_texts = text_arr[:, None]

    results = {}
    for k in ks:
        k_indices = topk_indices[:, :k]
        retrieved_texts = text_arr[k_indices.cpu().numpy()]
        matches = (retrieved_texts == target_texts).any(axis=1)
        results[f"R@{k}"] = matches.mean()
    return results

eval_rows = []

for variant in results_summary:
    tf = torch.load(variant['t_path'], map_location='cpu')
    imf = torch.load(variant['i_path'], map_location='cpu')
    subset_texts = text_list

    sims = F.cosine_similarity(tf, imf, dim=1)
    metrics = recall_at_k(tf, imf, subset_texts)

    row = {
        "Model": variant['name'],
        "Mean Cos Sim": sims.mean().item(),
        **metrics
    }
    eval_rows.append(row)

eval_df = pd.DataFrame(eval_rows).set_index("Model")
display(eval_df.style.highlight_max(color="lightgreen", axis=0))
