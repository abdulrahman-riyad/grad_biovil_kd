# Auto-exported from teammate notebook.
# Source notebook: week4_structured_project/Final_ppt/materials/new_teammates_training_notebooks/mobilevit-distilbiobert-kd-hn-updated.ipynb
# Code cells: 32; markdown cells: 18
# Notebook shell commands and magics are preserved as comments.
# ruff: noqa
# pylint: skip-file

# %% code cell 1
# NOTEBOOK_COMMAND: !pip install -q calflops
# NOTEBOOK_COMMAND: !pip install hi-ml-multimodal

# %% code cell 2
import numpy as np
import pandas as pd
import os, pickle, json, time
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from transformers import AutoTokenizer, AutoModel
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from calflops import calculate_flops
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.text.utils import get_biovil_t_bert

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown] cell 3
# # Distillation Dataset

# %% code cell 4
TRAIN_PKL_1 = '/kaggle/input/datasets/shahdammar/distillationdataset/biovil_t_fixed_full.pkl'
TRAIN_PKL_2 = '/kaggle/input/datasets/shahdammar/distillationdataset/biovil_t_fixed_full2.pkl'
VAL_PKL = "/kaggle/input/datasets/shahdammar/distillationdataset/biovil_t_validation_full.pkl"

# %% code cell 5
OUT_DIR = '/kaggle/working/contrastive_kd'
os.makedirs(OUT_DIR, exist_ok=True)
STAGE1_IMG_CKPT = f'{OUT_DIR}/stage1_mobilevit.pth'
STAGE1_TXT_CKPT = f'{OUT_DIR}/stage1_distilbiobert.pth'
STAGE2_IMG_CKPT = f'{OUT_DIR}/stage2_mobilevit_hn.pth'
STAGE2_TXT_CKPT = f'{OUT_DIR}/stage2_distilbiobert_hn.pth'

# %% code cell 6
TEACHER_DIM = 128
MAX_VIEWS = 3
BATCH_SIZE = 32
LR = 1e-4
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 5
TEMPERATURE = 0.07    # InfoNCE temperature
LAMBDA_KD = 0.25      # Weight for KD loss terms
MAX_TEXT_LEN = 128    # Max tokens for report text
HN_POOL_SIZE = 25000  # Hard negative candidate pool size
TOP_K_HN = 5          # Number of hard negatives per sample

# %% [markdown] cell 7
# ## Exploring Dataset

# %% code cell 8
with open(TRAIN_PKL_1, 'rb') as f:
    data = pickle.load(f)

print(f"Data type: {type(data)}")

# %% code cell 9
data.head()

# %% code cell 10
data.columns

# %% code cell 11
with open(VAL_PKL, 'rb') as f:
    val_df = pickle.load(f)

print(len(val_df))

# %% [markdown] cell 12
# ## Merge Training Dataset

# %% code cell 13
with open(TRAIN_PKL_1, 'rb') as f:
    df1 = pickle.load(f)

with open(TRAIN_PKL_2, 'rb') as f:
    df2 = pickle.load(f)

train_df = pd.concat([df1, df2], ignore_index=True)

# %% code cell 14
with open(VAL_PKL, 'rb') as f:
    val_df = pickle.load(f)

print(f"Train: {len(train_df):,} studies")
print(f"Val: {len(val_df):,} studies")
print(f"Columns: {list(train_df.columns)}")

# %% [markdown] cell 15
# ## Splitting Train to train set and test set

# %% code cell 16
all_subjects = sorted(train_df['subject_id'].unique())

split_idx = int(len(all_subjects) * 0.9)
train_subj = all_subjects[:split_idx]
test_subj = all_subjects[split_idx:]

new_train_df = train_df[train_df['subject_id'].isin(train_subj)].reset_index(drop=True)
test_df = train_df[train_df['subject_id'].isin(test_subj)].reset_index(drop=True)
train_df = new_train_df

print(f"Train: {len(train_df):,} studies")
print(f"Val: {len(val_df):,} studies")
print(f"Test: {len(test_df):,} studies")

# %% [markdown] cell 17
# ## Dataset Class
# <font size='4'>Some studies include multiple views. The maximum number of views to process is set to 3, as most of the dataset has 3 views or fewer.</font>

# %% code cell 18
class ContrastiveDistillationDataset(Dataset):
    """
    Returns per study:
      stacked_images : [MAX_VIEWS, 3, 224, 224]
      num_views : int  — how many real views (rest are zero-padded)
      teacher_img_emb : [128] — BioViL-T teacher image embedding
      teacher_txt_emb : [128] — BioViL-T teacher text embedding
      report_text : str  — raw report text for the text student
    """
    def __init__(self, dataframe, max_views=MAX_VIEWS):
        self.df = dataframe.reset_index(drop=True)
        self.max_views = max_views
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        paths = row['image_paths']
        num_to_process = min(len(paths), self.max_views)

        # Load image views (zero-pad missing)
        view_tensors = []
        for i in range(self.max_views):
            if i < num_to_process:
                try:
                    img = Image.open(paths[i]).convert('RGB')
                    view_tensors.append(self.transform(img))
                except:
                    view_tensors.append(torch.zeros(3, 224, 224))
            else:
                view_tensors.append(torch.zeros(3, 224, 224))

        stacked_images = torch.stack(view_tensors)                          # [MAX_VIEWS, 3, 224, 224]
        teacher_img_emb = torch.tensor(row['image_embedding'],  dtype=torch.float32)  # [128]
        teacher_txt_emb = torch.tensor(row['report_embedding'], dtype=torch.float32)  # [128]
        report_text = str(row['report_text'])

        return stacked_images, num_to_process, teacher_img_emb, teacher_txt_emb, report_text


def collate_fn(batch):
    """Custom collate to handle variable-length text in the same batch."""
    images, counts, t_img, t_txt, texts = zip(*batch)
    return (
        torch.stack(images),
        torch.tensor(counts, dtype=torch.long),
        torch.stack(t_img),
        torch.stack(t_txt),
        list(texts)
    )


train_ds = ContrastiveDistillationDataset(train_df)
val_ds   = ContrastiveDistillationDataset(val_df)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, collate_fn=collate_fn, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, collate_fn=collate_fn, pin_memory=True)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# %% [markdown] cell 19
# # Student
#
# <font size='4'> **Vision Student:** MobileViT-Small provides a hybrid CNN-Transformer architecture for global and local feature awareness.
# The model is designed to handle a variable number of X-ray views per study by processing images individually and performing Late Fusion (averaging).
#
# <font size='4'> **Text Student:** DistilBioBERT
#
# <font size='4'> **Projection Head:** A multi-layer mapper that aligns the high-dimensional latent features from both student backbones with the 128-dimensional BioViL teacher space.

# %% code cell 20
class MobileViTStudent(nn.Module):
    def __init__(self, teacher_dim=TEACHER_DIM):
        super().__init__()
        self.backbone = timm.create_model('mobilevit_s', pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features

        self.mapper = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, teacher_dim)
        )

    def forward(self, x, counts):
        """
        x: [B, MAX_VIEWS, 3, 224, 224]
        counts: [B] — number of real views per study
        """
        B, V, C, H, W = x.shape
        x = x.view(-1, C, H, W) # [B*V, 3, 224, 224]
        features = self.backbone(x) # [B*V, 640]
        per_view_emb = self.mapper(features) # [B*V, 128]
        per_view_emb = per_view_emb.view(B, V, -1) # [B, V, 128]

        # Masked mean (ignore zero-padded views)
        mask = torch.arange(V, device=x.device).expand(B, V)
        mask = (mask < counts.unsqueeze(1)).float().unsqueeze(-1)  # [B, V, 1]
        sum_emb = (per_view_emb * mask).sum(dim=1) # [B, 128]
        mean_emb = sum_emb / counts.view(-1, 1).float()

        return F.normalize(mean_emb, p=2, dim=-1) # [B, 128]

# %% code cell 21
class DistilBioBERTStudent(nn.Module):
    """
    Text student: DistilBioBERT backbone + projection head.
    Maps radiology reports → 128D BioViL-T teacher text space.

    Why DistilBioBERT:
      - 40% smaller than CXR-BERT teacher (~66M vs ~110M params)
      - Biomedically pretrained: understands clinical vocabulary
      - Good starting point because BioViL-T text encoder was itself
        initialized from PubMedBERT (same domain family)
    """
    MODEL_NAME = 'nlpie/distil-biobert'

    def __init__(self, teacher_dim=TEACHER_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(self.MODEL_NAME)
        hidden = self.backbone.config.hidden_size   # 768
        self.projection = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, teacher_dim)
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # [B, 768] — CLS token
        return F.normalize(self.projection(cls), p=2, dim=-1)  # [B, 128]


# Shared tokenizer
tokenizer = AutoTokenizer.from_pretrained(DistilBioBERTStudent.MODEL_NAME)

def tokenize_batch(texts):
    """Tokenize a list of report strings → input_ids, attention_mask on device."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors='pt'
    )
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)

# %% code cell 22
img_student = MobileViTStudent().to(device)
txt_student = DistilBioBERTStudent().to(device)

# Parameter counts
img_params = sum(p.numel() for p in img_student.parameters()) / 1e6
txt_params = sum(p.numel() for p in txt_student.parameters()) / 1e6
print(f"Image student params : {img_params:.1f}M")
print(f"Text  student params : {txt_params:.1f}M")
print(f"Total student params : {img_params + txt_params:.1f}M")
print(f"(Teacher: ResNet50 ~25M + CXR-BERT ~110M = ~135M)")

# %% [markdown] cell 23
# # Loss Functions

# %% [markdown] cell 24
# ### InfoNCE Loss
# Trains the model to identify the correct image-report pair from a batch of negatives.
# For each image, the correct report is the positive; all other reports in the batch
# are negatives. Temperature=0.07 (standard from CLIP).
#
# ### Combined Loss
# ```
# L = InfoNCE(student_img, student_txt) ← cross-modal alignment
#   + λ * [MSE + cosine](student_img, teacher_img) ← image imitation
#   + λ * [MSE + cosine](student_txt, teacher_txt) ← text imitation
# ```

# %% code cell 25
def infonce_loss(img_emb, txt_emb, temperature=TEMPERATURE):
    """
    img_emb : [B, 128] — L2-normalized student image embeddings
    txt_emb : [B, 128] — L2-normalized student text embeddings
    Returns scalar loss.
    """
    logits = img_emb @ txt_emb.T / temperature

    labels = torch.arange(len(logits), device=logits.device)

    # Image→Text direction + Text→Image direction
    loss_i2t = F.cross_entropy(logits,   labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2.0


def kd_loss(student_emb, teacher_emb):
    """
    Feature imitation: MSE + cosine loss.
    """
    mse      = F.mse_loss(student_emb, teacher_emb)
    cos_loss = 1 - F.cosine_similarity(student_emb, teacher_emb).mean()
    return mse + cos_loss


def stage1_loss(s_img, s_txt, t_img, t_txt, lam=LAMBDA_KD):
    """
    Full combined loss.
    s_img: student image embedding [B, 128]
    s_txt: student text  embedding [B, 128]
    t_img: teacher image embedding [B, 128]
    t_txt: teacher text  embedding [B, 128]
    """
    l_infonce = infonce_loss(s_img, s_txt)
    l_kd_img  = kd_loss(s_img, t_img)
    l_kd_txt  = kd_loss(s_txt, t_txt)
    return l_infonce + lam * (l_kd_img + l_kd_txt)

# %% code cell 26
# retrieval helpers defined EARLY so both training stages can select checkpoints on Recall@1.
@torch.no_grad()
def extract_embeddings(img_model, txt_model, loader):
    """Extract all student embeddings from a dataloader."""
    img_model.eval()
    txt_model.eval()
    all_img, all_txt = [], []

    for imgs, counts, _, _, texts in tqdm(loader, desc='Extracting embeddings'):
        imgs, counts = imgs.to(device), counts.to(device)
        input_ids, attn_mask = tokenize_batch(texts)

        s_img = img_model(imgs, counts)
        s_txt = txt_model(input_ids, attn_mask)

        all_img.append(s_img.cpu())
        all_txt.append(s_txt.cpu())

    return torch.cat(all_img), torch.cat(all_txt)



def compute_retrieval_metrics(img_embs, txt_embs, topk=(1, 5, 10)):
    """
    Compute retrieval metrics on [N, 128] embedding pairs.
    Ground truth: index i in images matches index i in texts.
    """
    N = img_embs.shape[0]
    results = {}

    # Similarity matrix [N, N]
    sims = img_embs @ txt_embs.T   # cosine (embeddings are L2-normalized)

    for direction, query, gallery in [('Image→Text', sims, None),
                                       ('Text→Image', sims.T, None)]:
        q = sims if direction == 'Image→Text' else sims.T
        ranks = []
        for i in range(N):
            row = q[i]
            sorted_idx = row.argsort(descending=True).tolist()
            rank = sorted_idx.index(i) + 1   # 1-indexed
            ranks.append(rank)

        ranks = np.array(ranks)
        res = {'Median Rank': float(np.median(ranks)),
               'Mean Rank':   float(np.mean(ranks))}
        for k in topk:
            res[f'R@{k}'] = float((ranks <= k).mean())
        results[direction] = res

    return results

# %% [markdown] cell 27
# # Stage 1: Contrastive Distillation

# %% code cell 28
def train_one_epoch_stage1(img_model, txt_model, loader, optimizer, scaler):
    img_model.train()
    txt_model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Train S1', leave=False)
    for imgs, counts, t_img, t_txt, texts in pbar:
        imgs, counts = imgs.to(device), counts.to(device)
        t_img, t_txt = t_img.to(device), t_txt.to(device)
        input_ids, attn_mask = tokenize_batch(texts)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            s_img = img_model(imgs, counts)              # [B, 128]
            s_txt = txt_model(input_ids, attn_mask)      # [B, 128]
            loss  = stage1_loss(s_img, s_txt, t_img, t_txt)

        if not torch.isfinite(loss):
            print("Non-finite loss detected — skipping batch")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(img_model.parameters()) +
                                       list(txt_model.parameters()), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(loader)


# Returns R@1 (Image->Text) as the primary selection score, plus the full metrics dict
@torch.no_grad()
def validate_retrieval(img_model, txt_model, loader, sample_n=5000, seed=42):
    """Selection metric = Recall@1 on the val set (the objective we actually care about)."""
    img_embs, txt_embs = extract_embeddings(img_model, txt_model, loader)

    # Fixed sampled pool (seeded) so the score is comparable across epochs.
    n = min(sample_n, len(img_embs))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(img_embs), n, replace=False)
    metrics = compute_retrieval_metrics(img_embs[idx], txt_embs[idx])

    # Primary selection score: Image->Text R@1 (higher = better).
    r_at_1 = metrics['Image→Text']['R@1']
    return r_at_1, metrics


# validate_stage1 uses stage1_loss (InfoNCE + KD)
@torch.no_grad()
def validate_stage1(img_model, txt_model, loader):
    img_model.eval()
    txt_model.eval()
    total_loss, total_img_cos, total_txt_cos = 0, 0, 0

    for imgs, counts, t_img, t_txt, texts in tqdm(loader, desc='Val S1', leave=False):
        imgs, counts = imgs.to(device), counts.to(device)
        t_img, t_txt = t_img.to(device), t_txt.to(device)
        input_ids, attn_mask = tokenize_batch(texts)

        s_img = img_model(imgs, counts)
        s_txt = txt_model(input_ids, attn_mask)

        # matched loss (InfoNCE + KD), same objective as training — for logging only.
        total_loss   += stage1_loss(s_img, s_txt, t_img, t_txt).item()
        total_img_cos += F.cosine_similarity(s_img, t_img).mean().item()
        total_txt_cos += F.cosine_similarity(s_txt, t_txt).mean().item()

    n = len(loader)
    return total_loss / n, total_img_cos / n, total_txt_cos / n


def run_stage1(img_model, txt_model, train_loader, val_loader, epochs=STAGE1_EPOCHS):
    print("\n" + "="*60)
    print("STAGE 1: Contrastive Distillation")
    print("="*60)

    optimizer = torch.optim.AdamW(
        list(img_model.parameters()) + list(txt_model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler('cuda')

    # select on Recall@1 (higher is better) instead of val InfoNCE.
    best_val_r1 = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_r1': [],
               'val_img_cos': [], 'val_txt_cos': []}

    for epoch in range(epochs):
        train_loss = train_one_epoch_stage1(img_model, txt_model, train_loader, optimizer, scaler)

        # matched loss kept for LOGGING only.
        val_loss, val_img_cos, val_txt_cos = validate_stage1(img_model, txt_model, val_loader)
        # R@1 is the SELECTION metric.
        val_r1, val_metrics = validate_retrieval(img_model, txt_model, val_loader)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r1'].append(val_r1)
        history['val_img_cos'].append(val_img_cos)
        history['val_txt_cos'].append(val_txt_cos)

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        print(f" Train Loss  : {train_loss:.4f}")
        print(f" Val Loss    : {val_loss:.4f}  (InfoNCE+KD, logging only)")
        print(f" Val R@1     : {val_r1:.4f}  (selection metric, higher = better)")
        print(f" Val Img-Teacher ↑ : {val_img_cos:.4f}  cosine")
        print(f" Val Txt-Teacher ↑ : {val_txt_cos:.4f}  cosine")

        # save when R@1 improves.
        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            torch.save({'epoch': epoch,
                        'model_state_dict': img_model.state_dict(),
                        'val_r1': val_r1,
                        'val_img_cos': val_img_cos}, STAGE1_IMG_CKPT)
            torch.save({'epoch': epoch,
                        'model_state_dict': txt_model.state_dict(),
                        'val_r1': val_r1,
                        'val_txt_cos': val_txt_cos}, STAGE1_TXT_CKPT)
            print(f"New best saved (R@1={val_r1:.4f})")

    print("\nStage 1 complete.")
    return history

# %% code cell 29
stage1_history = run_stage1(img_student, txt_student, train_loader, val_loader)

# %% [markdown] cell 30
# ### Paths to Stage 1 Models

# %% code cell 31
STAGE1_IMG_CKPT = '/kaggle/input/datasets/yasmeen550098/mobilevit-stage1/stage1_mobilevit.pth'
STAGE1_TXT_CKPT = '/kaggle/input/datasets/yasmeen550098/mobilevit-stage1/stage1_distilbiobert.pth'

# %% [markdown] cell 32
# # Stage 2: Hard Negative Fine-Tuning
#
# ### What are Hard Negatives?
# Random negatives = whatever else is in the batch (easy to distinguish).
# Hard negatives = reports that are clinically similar to the query but belong to a different patient such as two pneumonia cases.
#
# We select hard negatives using the **teacher text embeddings** `report_embedding` column. For each sample, we find the top-K most similar reports that are NOT the correct match.
#
# ### InfoNCE with Hard Negatives
# Same InfoNCE formula, but we explicitly add the hard negatives into the denominator alongside the random batch negatives.

# %% code cell 33
class StudentPoolTensor(torch.Tensor):
    """
    Typed wrapper that tags a tensor as built from STUDENT embeddings.
    Passed to get_hard_negatives() so the function can assert at runtime
    that it is not accidentally receiving teacher embeddings.
    """
    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data)


@torch.no_grad()
def build_student_pool(txt_model, train_df, pool_size, tokenize_batch_fn, seed=None):
    """
    Encode a random sample of training reports through the STUDENT text encoder
    to build a pool of student text embeddings for HN mining.

    txt_model         : student text encoder
    train_df          : full training DataFrame with 'report' text column
    pool_size         : number of samples to encode
    tokenize_batch_fn : existing tokenize_batch helper
    seed              : for reproducibility (None = random each call)

    Returns StudentPoolTensor [pool_size, 128] — L2-normalized, on CPU.
    """
    txt_model.eval()
    pool_df = train_df.sample(
        n=min(pool_size, len(train_df)),
        random_state=seed
    ).reset_index(drop=True)

    all_embs = []
    encode_bs = 64
    for start in range(0, len(pool_df), encode_bs):
        batch_texts = pool_df['report_text'].iloc[start:start + encode_bs].tolist()
        input_ids, attn_mask = tokenize_batch_fn(batch_texts)
        with torch.amp.autocast('cuda'):
            embs = txt_model(input_ids, attn_mask)   # [b, 128]
        all_embs.append(embs.detach().cpu())

    pool_tensor = torch.cat(all_embs, dim=0)                    # [pool_size, 128]
    pool_norm   = F.normalize(pool_tensor, p=2, dim=-1)         # L2-normalize
    return StudentPoolTensor(pool_norm)                          # typed, stays on CPU


def get_hard_negatives(student_txt_emb_batch, student_pool_embs, top_k=TOP_K_HN):
    """
    Mine hard negatives using pure STUDENT-to-STUDENT cosine similarity.

    Finds samples the student currently confuses with the query — i.e. what is
    genuinely hard for the student at this point in training, not what was hard
    for the stronger teacher.

    student_txt_emb_batch : [B, 128] — student text embs for current batch (on device)
    student_pool_embs     : StudentPoolTensor [P, 128] — student pool (CPU, L2-normalized)
                            Must be built with build_student_pool().
    Returns hard_neg_embs : [B, top_k, 128] on same device as student_txt_emb_batch
    """
    assert isinstance(student_pool_embs, StudentPoolTensor), (
        "student_pool_embs must be a StudentPoolTensor. "
        "Build it with build_student_pool()"
    )

    # L2-normalize both sides to get valid cosine similarities
    batch_cpu  = F.normalize(student_txt_emb_batch.detach().cpu(), p=2, dim=-1)  # [B, 128]
    pool_norm  = F.normalize(student_pool_embs, p=2, dim=-1)                     # [P, 128]

    sims = batch_cpu @ pool_norm.T   # [B, P] — student-to-student cosine similarity

    # Mask out self-matches (a sample should not be its own hard negative)
    sims[sims > 0.99] = -1.0

    # Top-k most similar in student space → these are hard for the student right now
    _, top_idx = sims.topk(top_k, dim=-1)   # [B, top_k]

    hard_negs = pool_norm[top_idx]           # [B, top_k, 128] — student embeddings
    return hard_negs.to(student_txt_emb_batch.device)


def infonce_with_hard_negatives(img_emb, txt_emb, hard_neg_embs, temperature=TEMPERATURE):
    """
    InfoNCE where the denominator includes both batch negatives AND hard negatives.

    img_emb : [B, 128]
    txt_emb : [B, 128] — positive text embeddings
    hard_neg_embs : [B, K, 128] — hard negative text embeddings
    """
    B, K, D = hard_neg_embs.shape

    # Standard batch similarities: [B, B]
    batch_sims = img_emb @ txt_emb.T / temperature

    # Hard negative similarities: [B, K]
    # For each image i, similarity with its K hard negatives
    hn_sims = torch.bmm(img_emb.unsqueeze(1),
                        hard_neg_embs.transpose(1, 2)).squeeze(1) / temperature  # [B, K]

    # Concatenate: [B, B+K] — batch negatives + hard negatives in denominator
    logits = torch.cat([batch_sims, hn_sims], dim=1)  # [B, B+K]

    # Labels: diagonal of the batch part (positions 0..B-1)
    labels = torch.arange(B, device=img_emb.device)

    return F.cross_entropy(logits, labels)


def stage2_loss(s_img, s_txt, t_img, t_txt, hard_negs, lam=LAMBDA_KD):
    """
    Stage 2 loss = InfoNCE with hard negatives + KD regularization.
    """
    l_infonce = infonce_with_hard_negatives(s_img, s_txt, hard_negs)
    l_kd_img  = kd_loss(s_img, t_img)
    l_kd_txt  = kd_loss(s_txt, t_txt)
    return l_infonce + lam * (l_kd_img + l_kd_txt)

# %% code cell 34
# How often to rebuild the student pool (in epochs).
# Every 2 epochs: the student's representations shift meaningfully enough
# that a fresh pool reflects its current confusion landscape, without the
# instability of rebuilding every epoch on a still-noisy student.
HN_REFRESH_EPOCHS = 2


def run_stage2(img_model, txt_model, train_loader, val_loader,
               train_df, epochs=STAGE2_EPOCHS):
    print("\n" + "="*60)
    print("STAGE 2: Hard Negative Fine-Tuning")
    print("="*60)

    img_model.load_state_dict(torch.load(STAGE1_IMG_CKPT)['model_state_dict'])
    txt_model.load_state_dict(torch.load(STAGE1_TXT_CKPT)['model_state_dict'])
    print("Loaded Stage 1 best checkpoints.")

    optimizer = torch.optim.AdamW(
        list(img_model.parameters()) + list(txt_model.parameters()),
        lr=LR * 0.3,
        weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler('cuda')

    # Fixed seeded validation pool — built ONCE from student embeddings at
    # the start of Stage 2, used only for comparable logging across epochs.
    print("Building seeded validation pool from student embeddings...")
    val_pool = build_student_pool(txt_model, train_df, HN_POOL_SIZE,
                                  tokenize_batch, seed=42)
    txt_model.train()   # restore train mode after pool build
    print(f"Val pool: {val_pool.shape}")

    best_val_r1 = -1.0
    pool_embs   = None   # will be built/refreshed at epoch 0, 2, 4, ...
    history     = {'train_loss': [], 'val_loss': [], 'val_r1': [],
                   'val_img_cos': [], 'val_txt_cos': []}

    for epoch in range(epochs):
        img_model.train()
        txt_model.train()
        total_loss = 0

        # ── Rebuild student pool every HN_REFRESH_EPOCHS epochs ──────────────
        # Epoch 0 always builds. Subsequent rebuilds at 2, 4, ...
        # Each rebuild encodes a fresh random sample of training reports through
        # the current student, so HNs reflect the student's evolving confusion
        # landscape rather than a stale snapshot.
        if epoch % HN_REFRESH_EPOCHS == 0:
            print(f"\n[Epoch {epoch+1}] Rebuilding student pool for HN mining "                  f"(pool_size={HN_POOL_SIZE})...")
            pool_embs = build_student_pool(txt_model, train_df, HN_POOL_SIZE,
                                           tokenize_batch, seed=None)
            txt_model.train()   # restore train mode after pool build
            print(f"Student pool ready: {pool_embs.shape}")

        pbar = tqdm(train_loader, desc=f'Train S2 E{epoch+1}', leave=False)
        for batch_idx, (imgs, counts, t_img, t_txt, texts) in enumerate(pbar):
            imgs, counts = imgs.to(device), counts.to(device)
            t_img, t_txt = t_img.to(device), t_txt.to(device)
            input_ids, attn_mask = tokenize_batch(texts)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                s_img = img_model(imgs, counts)
                s_txt = txt_model(input_ids, attn_mask)

                # Mine hard negatives in student-to-student space.
                # pool_embs is a StudentPoolTensor — get_hard_negatives will
                # assert this at runtime to prevent accidental teacher pool use.
                hard_negs = get_hard_negatives(s_txt, pool_embs, top_k=TOP_K_HN)

                loss = stage2_loss(s_img, s_txt, t_img, t_txt, hard_negs)

            if not torch.isfinite(loss):
                print("Non-finite loss — skipping batch")
                continue

            # Diagnostic: log every 100 batches to verify HN quality
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    s_txt_norm = F.normalize(s_txt.detach().cpu(), p=2, dim=-1)
                    # HN similarity in student space (hard_negs are student embs)
                    hn_sims = (s_txt_norm.unsqueeze(1) @
                               hard_negs.detach().cpu().transpose(1, 2)).squeeze(1)
                    # Batch negative sims (student-student, off-diagonal)
                    batch_neg_sims = s_txt_norm @ s_txt_norm.T
                    batch_neg_sims.fill_diagonal_(0)
                    l_infonce = infonce_with_hard_negatives(s_img, s_txt, hard_negs)
                    l_kd = LAMBDA_KD * (kd_loss(s_img, t_img) + kd_loss(s_txt, t_txt))
                    print(f"  [B{batch_idx}] HN sim (student): {hn_sims.mean().item():.3f} | "
                          f"Batch neg sim: {batch_neg_sims.mean().item():.3f} | "
                          f"InfoNCE: {l_infonce.item():.4f} | KD: {l_kd.item():.4f}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(img_model.parameters()) + list(txt_model.parameters()),
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_train_loss = total_loss / len(train_loader)

        # Validation: use the fixed seeded student val_pool for comparable logging.
        val_loss, val_img_cos, val_txt_cos = validate_stage2(
            img_model, txt_model, val_loader, val_pool)
        val_r1, val_metrics = validate_retrieval(img_model, txt_model, val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_r1'].append(val_r1)
        history['val_img_cos'].append(val_img_cos)
        history['val_txt_cos'].append(val_txt_cos)

        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        print(f"Train Loss  : {avg_train_loss:.4f}")
        print(f"Val Loss    : {val_loss:.4f}  (HN InfoNCE+KD, logging only)")
        print(f"Val R@1     : {val_r1:.4f}  (selection metric, higher = better)")
        print(f"Val Img-Teacher ↑ : {val_img_cos:.4f}")
        print(f"Val Txt-Teacher ↑ : {val_txt_cos:.4f}")

        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            torch.save({'epoch': epoch,
                        'model_state_dict': img_model.state_dict(),
                        'val_r1': val_r1}, STAGE2_IMG_CKPT)
            torch.save({'epoch': epoch,
                        'model_state_dict': txt_model.state_dict(),
                        'val_r1': val_r1}, STAGE2_TXT_CKPT)
            print(f"New best saved (R@1={val_r1:.4f}).")

    print("\nStage 2 complete.")
    return history


# Stage 2 validation using the SAME objective as training.
# val_pool is a StudentPoolTensor built from student embeddings at Stage 2 startup.
@torch.no_grad()
def validate_stage2(img_model, txt_model, loader, val_pool):
    img_model.eval()
    txt_model.eval()
    total_loss, total_img_cos, total_txt_cos = 0, 0, 0

    for imgs, counts, t_img, t_txt, texts in tqdm(loader, desc='Val S2', leave=False):
        imgs, counts = imgs.to(device), counts.to(device)
        t_img, t_txt = t_img.to(device), t_txt.to(device)
        input_ids, attn_mask = tokenize_batch(texts)

        s_img = img_model(imgs, counts)
        s_txt = txt_model(input_ids, attn_mask)

        # val_pool is StudentPoolTensor — assertion in get_hard_negatives will pass
        hard_negs = get_hard_negatives(s_txt, val_pool, top_k=TOP_K_HN)
        total_loss    += stage2_loss(s_img, s_txt, t_img, t_txt, hard_negs).item()
        total_img_cos += F.cosine_similarity(s_img, t_img).mean().item()
        total_txt_cos += F.cosine_similarity(s_txt, t_txt).mean().item()

    n = len(loader)
    return total_loss / n, total_img_cos / n, total_txt_cos / n


stage2_history = run_stage2(img_student, txt_student, train_loader, val_loader, train_df)

# %% [markdown] cell 35
# # Evaluations

# %% [markdown] cell 36
# ## Retrieval Evaluation
# Compute Recall@1, R@5, R@10, Median Rank on the validation set.

# %% code cell 37
@torch.no_grad()
# Teacher (BioViL-T) embeddings come straight from the
# precomputed per-study tensors the loader already yields
# (image_embedding / report_embedding).
@torch.no_grad()
def extract_teacher_embeddings(loader):
    """BioViL-T teacher embeddings (precomputed) for the same studies."""
    all_img, all_txt = [], []
    for imgs, counts, t_img, t_txt, texts in tqdm(loader, desc='Teacher embeddings'):
        all_img.append(t_img.cpu())   # [B,128] BioViL-T image
        all_txt.append(t_txt.cpu())   # [B,128] BioViL-T text
    return torch.cat(all_img), torch.cat(all_txt)



# sample_n=None  -> use the FULL set (most honest benchmark)
# sample_n=K      -> seeded random K-study gallery
def _sample_indices(n_total, sample_n, seed=42):
    if sample_n is None or sample_n >= n_total:
        return np.arange(n_total)            # full set, deterministic
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, sample_n, replace=False)


def run_retrieval_eval(img_model, txt_model, loader,
                       checkpoint_img=None, checkpoint_txt=None,
                       label='Stage 2 (Hard Negatives)',
                       sample_n=None, seed=42):
    """Student retrieval eval. sample_n=None scores the FULL set."""
    if checkpoint_img:
        img_model.load_state_dict(torch.load(checkpoint_img)['model_state_dict'], strict=False)
    if checkpoint_txt:
        txt_model.load_state_dict(torch.load(checkpoint_txt)['model_state_dict'], strict=False)

    img_embs, txt_embs = extract_embeddings(img_model, txt_model, loader)

    idx = _sample_indices(len(img_embs), sample_n, seed)
    metrics = compute_retrieval_metrics(img_embs[idx], txt_embs[idx])

    print(f"\n{'='*60}")
    print(f"Retrieval Evaluation — {label}  (gallery = {len(idx)} studies)")
    print(f"{'='*60}")
    for direction, m in metrics.items():
        print(f"\n  {direction}:")
        for k, v in m.items():
            print(f"    {k:15s}: {v:.4f}")

    return metrics


# teacher retrieval through the SAME metric function,
# SAME gallery indices (same seed / sample_n) as the students.
def run_teacher_retrieval_eval(loader, label='BioViL-T Teacher',
                               sample_n=None, seed=42):
    img_embs, txt_embs = extract_teacher_embeddings(loader)
    idx = _sample_indices(len(img_embs), sample_n, seed)
    metrics = compute_retrieval_metrics(img_embs[idx], txt_embs[idx])

    print(f"\n{'='*60}")
    print(f"Retrieval Evaluation — {label}  (gallery = {len(idx)} studies)")
    print(f"{'='*60}")
    for direction, m in metrics.items():
        print(f"\n  {direction}:")
        for k, v in m.items():
            print(f"    {k:15s}: {v:.4f}")
    return metrics


# Validation retrieval (1,201 studies)
metrics_val_s1 = run_retrieval_eval(img_student, txt_student, val_loader,
                                    STAGE1_IMG_CKPT, STAGE1_TXT_CKPT,
                                    label='Stage 1 (Contrastive KD) — VAL',
                                    sample_n=None)

metrics_val_s2 = run_retrieval_eval(img_student, txt_student, val_loader,
                                    STAGE2_IMG_CKPT, STAGE2_TXT_CKPT,
                                    label='Stage 2 (Hard Negatives) — VAL',
                                    sample_n=None)

# %% [markdown] cell 38
# ## Save All Results

# %% code cell 39
# Save training histories
"""with open(f'{OUT_DIR}/stage1_history.json', 'w') as f:
    json.dump(stage1_history, f, indent=2)"""

with open(f'{OUT_DIR}/stage2_history.json', 'w') as f:
    json.dump(stage2_history, f, indent=2)

# Save retrieval metrics
with open(f'{OUT_DIR}/retrieval_metrics_val_s1.json', 'w') as f:
    json.dump(metrics_val_s1, f, indent=2)

with open(f'{OUT_DIR}/retrieval_metrics_val_s2.json', 'w') as f:
    json.dump(metrics_val_s2, f, indent=2)

print(f"All outputs saved to: {OUT_DIR}")
print("\nFiles saved:")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(f'{OUT_DIR}/{f}') / 1e6
    print(f"  {f}  ({size:.1f} MB)")

# %% [markdown] cell 40
# ## Evaluation on Test Dataset

# %% code cell 41
print("Running final evaluation on held-out TEST set")

test_ds = ContrastiveDistillationDataset(test_df)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, collate_fn=collate_fn, pin_memory=True)

# TEST is the headline benchmark. All models scored on the
# SAME gallery. sample_n=None -> full 14,018-study
TEST_SAMPLE_N = None
TEST_SEED     = 42

# ── Students ──
metrics_test_s1 = run_retrieval_eval(
    img_student, txt_student, test_loader,
    STAGE1_IMG_CKPT, STAGE1_TXT_CKPT,
    label='Stage 1 — TEST', sample_n=TEST_SAMPLE_N, seed=TEST_SEED)

metrics_test_s2 = run_retrieval_eval(
    img_student, txt_student, test_loader,
    STAGE2_IMG_CKPT, STAGE2_TXT_CKPT,
    label='Stage 2 — TEST', sample_n=TEST_SAMPLE_N, seed=TEST_SEED)

# ── Teacher anchor ──
metrics_test_teacher = run_teacher_retrieval_eval(
    test_loader, label='BioViL-T Teacher — TEST',
    sample_n=TEST_SAMPLE_N, seed=TEST_SEED)

# ── Teacher anchor on VAL too ──
metrics_val_teacher = run_teacher_retrieval_eval(
    val_loader, label='BioViL-T Teacher — VAL',
    sample_n=None, seed=TEST_SEED)

with open(f'{OUT_DIR}/retrieval_metrics_test_s1.json', 'w') as f:
    json.dump(metrics_test_s1, f, indent=2)
with open(f'{OUT_DIR}/retrieval_metrics_test_s2.json', 'w') as f:
    json.dump(metrics_test_s2, f, indent=2)
with open(f'{OUT_DIR}/retrieval_metrics_test_teacher.json', 'w') as f:
    json.dump(metrics_test_teacher, f, indent=2)

# %% [markdown] cell 42
# ## Efficiency Table — FLOPs / Params / Latency
# Compares teacher vs all students on computational cost.

# %% code cell 43
# Loads BioViL-T Encoders
tokenizer, biovil_t_txt_model = get_biovil_t_bert()
biovil_t_txt_model = biovil_t_txt_model.to(device)

image_inference = get_image_inference(ImageModelType.BIOVIL_T)
biovil_t_img_model = image_inference.model.to(device)

# Load final Stage 2 checkpoints
img_student.load_state_dict(torch.load(STAGE2_IMG_CKPT)['model_state_dict'], strict=False)
txt_student.load_state_dict(torch.load(STAGE2_TXT_CKPT)['model_state_dict'], strict=False)

# %% code cell 44
def count_params(model):
    """Accurate parameter count in Millions (M)."""
    return f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"

# %% code cell 45
def measure_latency(model, dummy_input, n_runs=200, n_warmup=50):
    """
    Research-grade GPU latency measurement.
    Returns mean and std in milliseconds.
    """
    model = model.to(device)
    model.eval()

    # Fix cuDNN state for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        # Warmup (not recorded)
        for _ in range(n_warmup):
            if isinstance(dummy_input, (list, tuple)):
                model(*dummy_input)
            else:
                model(dummy_input)
        torch.cuda.synchronize()

        # Timed runs — measure each individually for std
        for _ in range(n_runs):
            starter.record()
            if isinstance(dummy_input, (list, tuple)):
                model(*dummy_input)
            else:
                model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

    mean_ms = np.mean(timings)
    std_ms  = np.std(timings)
    return mean_ms, std_ms

# %% code cell 46
def get_flops(model, model_inputs):
    """
    Uses calflops to calculate FLOPs.
    Handles multiple inputs and custom wrappers.
    """
    model.eval()

    if isinstance(model_inputs, (list, tuple)):
        input_args = list(model_inputs)
    else:
        input_args = [model_inputs]

    flops, macs, params = calculate_flops(
        model=model,
        args=input_args,
        output_as_string=False,
        print_results=False,
        print_detailed=False
    )

    # Convert to GFLOPs (1 GFLOP = 1e9 FLOPs)
    return f"{flops / 1e9:.3f} G"


def get_flops_text(model):
    """
    FLOPs for HuggingFace-style models (DistilBioBERT, CXR-BERT).
    Uses kwargs instead of args so calflops routes input_ids and
    attention_mask correctly during tracing.
    """
    model.eval()
    flops, _, _ = calculate_flops(
        model=model,
        kwargs={'input_ids': dummy_ids, 'attention_mask': dummy_mask},
        output_as_string=False,
        print_results=False,
        print_detailed=False
    )
    return f"{flops / 1e9:.3f} G"

# %% code cell 47
# Dummy inputs
dummy_img    = torch.randn(1, 3, 224, 224).to(device)
dummy_view1  = torch.randn(1, 1, 3, 224, 224).to(device)
dummy_views3 = torch.randn(1, 3, 3, 224, 224).to(device)
dummy_count1 = torch.tensor([1]).to(device)
dummy_count3 = torch.tensor([3]).to(device)

vocab_size = txt_student.backbone.config.vocab_size
dummy_ids  = torch.randint(0, vocab_size, (1, MAX_TEXT_LEN)).to(device)
dummy_mask = torch.ones(1, MAX_TEXT_LEN, dtype=torch.long).to(device)


rows = []
print("Calculating efficiency metrics... (this may take a minute)")


# 1. MobileViT-S backbone only
rows.append({
    'Model     ': 'MobileViT-S (backbone only)     ',
    'FLOPs     ': get_flops(img_student.backbone, dummy_img)+'     ',
    'Params     ': count_params(img_student.backbone)+'     ',
    'Latency (ms)     ': '{:.2f} ± {:.2f}'.format(*measure_latency(img_student.backbone, dummy_img))
})

# 2. MobileViT-S Student — 1-view (fair apples-to-apples vs. teacher)
rows.append({
    'Model     ': 'MobileViT-S Student (1-view)     ',
    'FLOPs     ': get_flops(img_student, [dummy_view1, dummy_count1])+'     ',
    'Params     ': count_params(img_student)+'     ',
    'Latency (ms)     ': '{:.2f} ± {:.2f}'.format(*measure_latency(img_student, [dummy_view1, dummy_count1]))
})

# 3. MobileViT-S Student — 3-view (real operational cost)
rows.append({
    'Model     ': 'MobileViT-S Student (3-view)     ',
    'FLOPs     ': get_flops(img_student, [dummy_views3, dummy_count3])+'     ',
    'Params     ': count_params(img_student)+'     ',
    'Latency (ms)     ': '{:.2f} ± {:.2f}'.format(*measure_latency(img_student, [dummy_views3, dummy_count3]))
})


# 4. DistilBioBERT backbone only
rows.append({
    'Model     ': 'DistilBioBERT (backbone only)     ',
    'FLOPs     ': get_flops_text(txt_student.backbone) +'     ',
    'Params     ': count_params(txt_student.backbone)+'     ',
    'Latency (ms)     ': '{:.2f} ± {:.2f}'.format(*measure_latency(txt_student.backbone, [dummy_ids, dummy_mask]))
})

# 5. DistilBioBERT Student full
rows.append({
    'Model     ': 'DistilBioBERT Student (full)     ',
    'FLOPs     ': get_flops_text(txt_student)+'     ',
    'Params     ': count_params(txt_student) +'     ',
    'Latency (ms)     ': '{:.2f} ± {:.2f}'.format(*measure_latency(txt_student, [dummy_ids, dummy_mask]))
})


# 6. BioViL-T Image Encoder (teacher)
rows.append({
    'Model     ': 'BioViL-T Image Encoder (single-image)     ',
    'FLOPs     ': get_flops(biovil_t_img_model, dummy_img) + '     ',
    'Params     ': count_params(biovil_t_img_model) + '     ',
    'Latency (ms)     ': '{:.2f} ± {:.2f}'.format(*measure_latency(biovil_t_img_model, dummy_img))
})

# 7. BioViL-T Text Encoder / CXR-BERT (teacher)  ← kwargs fix applied
rows.append({
    'Model     ': 'BioViL-T Text Encoder / CXR-BERT     ',
    'FLOPs     ': get_flops_text(biovil_t_txt_model) + '     ',
    'Params     ': count_params(biovil_t_txt_model) + '     ',
    'Latency (ms)     ': '{:.2f} ± {:.2f}'.format(*measure_latency(biovil_t_txt_model, [dummy_ids, dummy_mask]))
})


# Table
efficiency_df = pd.DataFrame(rows)
print("\nEfficiency Comparison Table")
print(efficiency_df.to_string(index=False))
efficiency_df.to_csv(f'{OUT_DIR}/efficiency_table.csv', index=False)

# %% [markdown] cell 48
# # Summary

# %% code cell 49
def _print_block(title, metrics):
    print(f"\n {title}")
    for direction, m in metrics.items():
        print(f"  {direction}: R@1={m['R@1']:.4f}  R@5={m['R@5']:.4f}  "
              f"R@10={m['R@10']:.4f}  MedRank={m['Median Rank']:.0f} MeanRank={m['Mean Rank']:.2f}")

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print("### NOTE: VAL and TEST use different gallery sizes and are NOT directly comparable.")

print("\n### VALIDATION SET")
_print_block("BioViL-T Teacher — Validation", metrics_val_teacher)
_print_block("Stage 1 (Contrastive KD) — Validation", metrics_val_s1)
_print_block("Stage 2 (Hard Negatives) — Validation", metrics_val_s2)

print("\n### TEST SET")
_print_block("BioViL-T Teacher — TEST", metrics_test_teacher)
_print_block("Stage 1 (Contrastive KD) — TEST", metrics_test_s1)
_print_block("Stage 2 (Hard Negatives) — TEST", metrics_test_s2)

# teacher's retrieval
def _recovery(student_m, teacher_m, key='Image→Text', metric='R@1'):
    t = teacher_m[key][metric]
    s = student_m[key][metric]
    return (s / t * 100.0) if t > 0 else float('nan')

print("\n### DISTILLATION RECOVERY on TEST (student R@1 as % of teacher R@1)")
for name, m in [('Stage 1', metrics_test_s1), ('Stage 2 (HN)', metrics_test_s2)]:
    i2t = _recovery(m, metrics_test_teacher, 'Image→Text', 'R@1')
    t2i = _recovery(m, metrics_test_teacher, 'Text→Image', 'R@1')
    print(f"  {name:14s}: I→T {i2t:5.1f}%   T→I {t2i:5.1f}%   (of teacher R@1)")

print("\n Efficiency")
print(efficiency_df.to_string(index=False))

print("\n Checkpoints")
print(f"Stage 1 image: {STAGE1_IMG_CKPT}")
print(f"Stage 1 text : {STAGE1_TXT_CKPT}")
print(f"Stage 2 image: {STAGE2_IMG_CKPT}")
print(f"Stage 2 text : {STAGE2_TXT_CKPT}")

# %% [markdown] cell 50
# ## Thoughts
# <font size='4'> With the successful distillation of BioViL features into MobileViT & DistilBioBERT, we are ready to build a generative VLM. By appending a projection layer and a decoder, the model will be trained to synthesize medical reports from medical images. The upcoming phase involves fine-tuning a decoder on the reports to ensure clinical accuracy. </font>
