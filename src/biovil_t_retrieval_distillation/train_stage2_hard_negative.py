from __future__ import annotations

import argparse
import json
import os
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from data.image_text_dataset import (
    ImageTextContrastiveDataset,
    collate_image_text,
    clean_report_text,
    extract_impression,
)
from data.transforms import build_image_transform
from losses import symmetric_info_nce
from models.image_text_retrieval_model import ImageTextContrastiveModel
from models.student_loaders import load_mobilevit_student, load_repvit_student
from models.text_encoders import TEXT_ENCODER_PRESETS, build_text_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileViT/RepViT image-text contrastive projection heads with teacher-guided hard negatives."
    )
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hard-negative-file", required=True)
    parser.add_argument(
        "--init-contrastive-checkpoint",
        default=None,
        help="Optional prior non-hard-negative contrastive best.pt used to initialize the shared-space model.",
    )
    parser.add_argument("--metadata-file", default="biovil_t_fixed_metadata.csv")
    parser.add_argument("--teacher-image-embeddings-file", default="biovil_t_fixed_image_embeddings.npy")
    parser.add_argument("--teacher-text-embeddings-file", default="biovil_t_fixed_text_embeddings.npy")
    parser.add_argument("--image-student", choices=["mobilevit", "repvit"], required=True)
    parser.add_argument("--mobilevit-checkpoint", default=None)
    parser.add_argument("--repvit-checkpoint", default=None)
    parser.add_argument("--repvit-root", default=None)
    parser.add_argument("--text-encoder", choices=list(TEXT_ENCODER_PRESETS.keys()), default="biovil_t")
    parser.add_argument("--text-model-id", default=None, help="Optional custom Hugging Face model id.")
    parser.add_argument("--text-source", choices=["impression", "report"], default="impression")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-hidden-dim", type=int, default=None)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--freeze-image-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-text-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-views", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA autocast mixed precision. Recommended on supported NVIDIA GPUs.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Autocast dtype used when --amp is enabled.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help="Optional lower LR for unfrozen image/text encoder parameters. Defaults to --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--hard-negatives-per-sample", type=int, default=8)
    parser.add_argument("--hard-negative-weight", type=float, default=0.5)
    parser.add_argument(
        "--hard-negative-mode",
        choices=["denominator", "auxiliary"],
        default="denominator",
        help="denominator appends mined texts to the InfoNCE candidate denominator; auxiliary keeps the old separate loss.",
    )
    parser.add_argument("--kd-image-weight", type=float, default=0.0)
    parser.add_argument("--kd-text-weight", type=float, default=0.0)
    parser.add_argument("--kd-relational-weight", type=float, default=0.1)
    parser.add_argument("--kd-temperature", type=float, default=0.07)
    parser.add_argument("--soft-positive-weight", type=float, default=0.25)
    parser.add_argument("--soft-positive-threshold", type=float, default=0.85)
    parser.add_argument("--soft-positive-temperature", type=float, default=0.07)
    parser.add_argument("--label-soft-positive-weight", type=float, default=0.15)
    parser.add_argument("--anatomy-soft-positive-weight", type=float, default=0.05)
    parser.add_argument(
        "--pseudo-label-weight",
        type=float,
        default=0.0,
        help="Optional BCE loss from image embeddings to simple report-derived CXR pseudo-labels.",
    )
    parser.add_argument("--longitudinal-weight", type=float, default=0.0)
    parser.add_argument("--uncertainty-weight", type=float, default=0.0)
    parser.add_argument(
        "--epoch-retrieval-pool-sizes",
        default="",
        help="Optional comma-separated retrieval pools to log after each epoch, e.g. '5000,full'. Empty disables.",
    )
    parser.add_argument("--epoch-retrieval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--epoch-retrieval-batch-size", type=int, default=64)
    parser.add_argument("--epoch-retrieval-num-workers", type=int, default=4)
    parser.add_argument("--epoch-retrieval-chunk-size", type=int, default=512)
    parser.add_argument("--epoch-retrieval-seed", type=int, default=42)
    parser.add_argument(
        "--retrieval-selection-pool",
        default="5000",
        help="Pool key used for best retrieval checkpointing, usually 5000 or full.",
    )
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA for the nccl backend.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return distributed, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def make_loader(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    image_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    max_views: int,
    text_source: str,
    distributed: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = ImageTextContrastiveDataset(
        metadata=metadata,
        indices=indices,
        image_root=image_root,
        transform=build_image_transform(image_size=image_size, train=train),
        max_views=max_views,
        text_source=text_source,
        view_sampling="random" if train else "first",
        skip_empty_text=True,
    )
    sampler = DistributedSampler(dataset, shuffle=train, drop_last=train) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        sampler=sampler,
        collate_fn=collate_image_text,
    )
    return loader, sampler


def load_image_encoder(args: argparse.Namespace, teacher_dim: int = 128) -> tuple[torch.nn.Module, int]:
    if args.image_student == "mobilevit":
        if not args.mobilevit_checkpoint:
            raise ValueError("--mobilevit-checkpoint is required for --image-student mobilevit.")
        return load_mobilevit_student(args.mobilevit_checkpoint, teacher_dim=teacher_dim), teacher_dim

    if not args.repvit_checkpoint or not args.repvit_root:
        raise ValueError("--repvit-checkpoint and --repvit-root are required for --image-student repvit.")
    return load_repvit_student(args.repvit_checkpoint, args.repvit_root, teacher_dim=teacher_dim), teacher_dim


def build_text_encoder_serialized(
    text_encoder_name: str,
    max_text_length: int,
    distributed: bool,
    rank: int,
) -> torch.nn.Module:
    barrier_kwargs = {"device_ids": [torch.cuda.current_device()]} if torch.cuda.is_available() else {}
    if distributed and rank != 0:
        dist.barrier(**barrier_kwargs)
    text_encoder = build_text_encoder(text_encoder_name, max_length=max_text_length)
    if distributed and rank == 0:
        dist.barrier(**barrier_kwargs)
    return text_encoder


class HardNegativeTextSampler:
    def __init__(
        self,
        metadata: pd.DataFrame,
        hard_negative_file: str | Path,
        text_source: str,
        seed: int,
    ) -> None:
        payload = np.load(hard_negative_file, allow_pickle=False)
        self.query_row_indices = payload["query_row_indices"].astype(np.int64)
        self.hard_negative_row_indices = payload["hard_negative_row_indices"].astype(np.int64)
        self.hard_negative_scores = (
            payload["hard_negative_scores"].astype(np.float32)
            if "hard_negative_scores" in payload.files
            else None
        )
        self.row_to_position = {int(row): pos for pos, row in enumerate(self.query_row_indices.tolist())}
        self.metadata = metadata
        self.text_source = text_source
        self.rng = np.random.default_rng(seed)

    def text_for_row(self, row_index: int) -> str:
        row = self.metadata.iloc[int(row_index)]
        if self.text_source == "report":
            return clean_report_text(row.get("report_text", ""))
        return extract_impression(row.get("report_text", ""))

    def sample_texts(self, row_indices: torch.Tensor, negatives_per_sample: int) -> list[str]:
        texts: list[str] = []
        for row_tensor in row_indices.detach().cpu():
            row_index = int(row_tensor.item())
            position = self.row_to_position.get(row_index)
            if position is None:
                raise KeyError(f"Row {row_index} is missing from the hard-negative file.")
            candidates = self.hard_negative_row_indices[position]
            if self.hard_negative_scores is not None:
                finite_candidates = candidates[np.isfinite(self.hard_negative_scores[position])]
                if len(finite_candidates) > 0:
                    candidates = finite_candidates
            if len(candidates) == 0:
                raise RuntimeError(f"No hard-negative candidates available for row {row_index}.")
            if negatives_per_sample > len(candidates):
                selected = self.rng.choice(candidates, size=negatives_per_sample, replace=True)
            elif negatives_per_sample == len(candidates):
                selected = candidates
            else:
                selected = self.rng.choice(candidates, size=negatives_per_sample, replace=False)
            texts.extend(self.text_for_row(int(row)) for row in selected)
        return texts


DISEASE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "atelectasis": ("atelectasis", "subsegmental atelectatic"),
    "cardiomegaly": ("cardiomegaly", "enlarged cardiac", "cardiac enlargement"),
    "consolidation": ("consolidation", "airspace opacity", "air space opacity"),
    "edema": ("edema", "pulmonary edema", "vascular congestion"),
    "enlarged_cardiomediastinum": ("enlarged cardiomediastinum", "widened mediastinum"),
    "fracture": ("fracture", "fractured"),
    "lung_lesion": ("nodule", "mass", "lesion"),
    "lung_opacity": ("opacity", "opacities", "infiltrate"),
    "pleural_effusion": ("pleural effusion", "effusion"),
    "pleural_other": ("pleural thickening", "pleural calcification", "pleural scarring"),
    "pneumonia": ("pneumonia", "infectious process"),
    "pneumothorax": ("pneumothorax",),
    "support_devices": ("tube", "catheter", "line", "pacemaker", "support device"),
    "no_finding": ("no acute cardiopulmonary", "no focal consolidation", "no pleural effusion", "no pneumothorax"),
}

ANATOMY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "left_lung": ("left lung", "left base", "left lower", "left upper", "left hemithorax"),
    "right_lung": ("right lung", "right base", "right lower", "right upper", "right hemithorax"),
    "bilateral_lungs": ("bilateral", "both lungs", "bibasal", "bibasilar"),
    "pleura": ("pleura", "pleural", "costophrenic"),
    "heart": ("heart", "cardiac", "cardiomediastinal", "cardiomediastinum"),
    "mediastinum": ("mediastinum", "mediastinal", "hilar", "hilum"),
    "bones": ("rib", "clavicle", "spine", "osseous", "bone", "fracture"),
    "devices": ("tube", "catheter", "line", "pacemaker", "device"),
}

NEGATION_PREFIX = re.compile(
    r"\b(no|not|without|absent|negative for|free of|lack of|no evidence of|no focal)\b.{0,45}$",
    flags=re.IGNORECASE,
)


def keyword_is_positive(text: str, keyword: str) -> bool:
    for match in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
        prefix = text[max(0, match.start() - 64) : match.start()]
        if not NEGATION_PREFIX.search(prefix):
            return True
    return False


def labels_from_texts(
    texts: list[str],
    label_keywords: dict[str, tuple[str, ...]],
    device: torch.device,
) -> torch.Tensor:
    labels = torch.zeros((len(texts), len(label_keywords)), dtype=torch.float32, device=device)
    for row, text in enumerate(texts):
        lowered = text.lower()
        for col, keywords in enumerate(label_keywords.values()):
            labels[row, col] = float(any(keyword_is_positive(lowered, keyword) for keyword in keywords))
    return labels


def pseudo_labels_for_texts(texts: list[str], device: torch.device) -> torch.Tensor:
    disease = labels_from_texts(texts, DISEASE_KEYWORDS, device)
    anatomy = labels_from_texts(texts, ANATOMY_KEYWORDS, device)
    return torch.cat([disease, anatomy], dim=1)


def jaccard_similarity_matrix(labels: torch.Tensor) -> torch.Tensor:
    labels = (labels > 0).float()
    intersection = labels @ labels.T
    counts = labels.sum(dim=1, keepdim=True)
    union = counts + counts.T - intersection
    return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.zeros_like(intersection))


def load_teacher_embeddings(
    artifacts_dir: Path,
    image_file: str,
    text_file: str,
) -> tuple[np.ndarray, np.ndarray]:
    image_embeddings = np.load(artifacts_dir / image_file, mmap_mode="r")
    text_embeddings = np.load(artifacts_dir / text_file, mmap_mode="r")
    if image_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError(
            f"Teacher image/text row counts differ: {image_embeddings.shape} vs {text_embeddings.shape}"
        )
    return image_embeddings, text_embeddings


def teacher_batch(
    row_indices: torch.Tensor,
    teacher_image_embeddings: np.ndarray,
    teacher_text_embeddings: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = row_indices.detach().cpu().numpy().astype(np.int64)
    teacher_images = torch.as_tensor(np.asarray(teacher_image_embeddings[rows]), dtype=torch.float32, device=device)
    teacher_texts = torch.as_tensor(np.asarray(teacher_text_embeddings[rows]), dtype=torch.float32, device=device)
    return F.normalize(teacher_images, p=2, dim=1), F.normalize(teacher_texts, p=2, dim=1)


def build_prior_text_lookup(metadata: pd.DataFrame, text_source: str) -> dict[int, str]:
    prior_by_row: dict[int, str] = {}
    sort_cols = [col for col in ["subject_id", "study_id"] if col in metadata.columns]
    if len(sort_cols) < 2:
        return prior_by_row
    sorted_meta = metadata.reset_index().sort_values(sort_cols)
    previous_text_by_subject: dict[int, str] = {}
    for row in sorted_meta.itertuples(index=False):
        original_index = int(getattr(row, "index"))
        subject_id = int(getattr(row, "subject_id"))
        if subject_id in previous_text_by_subject:
            prior_by_row[original_index] = previous_text_by_subject[subject_id]
        report_text = getattr(row, "report_text", "")
        current_text = clean_report_text(report_text) if text_source == "report" else extract_impression(report_text)
        if current_text:
            previous_text_by_subject[subject_id] = current_text
    return prior_by_row


def prior_texts_for_batch(row_indices: torch.Tensor, prior_text_by_row: dict[int, str]) -> list[str | None]:
    texts: list[str | None] = []
    for row_tensor in row_indices.detach().cpu():
        texts.append(prior_text_by_row.get(int(row_tensor.item())))
    return texts


def relational_kd_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    teacher_image_embeddings: torch.Tensor,
    teacher_text_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    student_i2t = F.normalize(image_embeddings, p=2, dim=1) @ F.normalize(text_embeddings, p=2, dim=1).T
    teacher_i2t = teacher_image_embeddings @ teacher_text_embeddings.T
    student_i2t = student_i2t / temperature
    teacher_i2t = teacher_i2t / temperature
    loss_i2t = F.kl_div(
        F.log_softmax(student_i2t, dim=1),
        F.softmax(teacher_i2t, dim=1),
        reduction="batchmean",
    )
    loss_t2i = F.kl_div(
        F.log_softmax(student_i2t.T, dim=1),
        F.softmax(teacher_i2t.T, dim=1),
        reduction="batchmean",
    )
    return 0.5 * (loss_i2t + loss_t2i)


def direct_kd_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    if student_embeddings.shape[1] != teacher_embeddings.shape[1]:
        zero = student_embeddings.sum() * 0.0
        return zero
    return 1.0 - F.cosine_similarity(
        F.normalize(student_embeddings, p=2, dim=1),
        F.normalize(teacher_embeddings, p=2, dim=1),
        dim=1,
    ).mean()


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def build_soft_positive_targets(
    row_indices: torch.Tensor,
    texts: list[str],
    teacher_text_embeddings: np.ndarray | None,
    device: torch.device,
    teacher_threshold: float,
    teacher_temperature: float,
    soft_positive_weight: float,
    label_weight: float,
    anatomy_weight: float,
) -> torch.Tensor:
    batch_size = len(texts)
    targets = torch.eye(batch_size, dtype=torch.float32, device=device)
    if batch_size == 0:
        return targets

    soft_scores = torch.zeros((batch_size, batch_size), dtype=torch.float32, device=device)
    if teacher_text_embeddings is not None and soft_positive_weight > 0:
        rows = row_indices.detach().cpu().numpy().astype(np.int64)
        teacher_texts = torch.as_tensor(np.asarray(teacher_text_embeddings[rows]), dtype=torch.float32, device=device)
        teacher_texts = F.normalize(teacher_texts, p=2, dim=1)
        teacher_scores = teacher_texts @ teacher_texts.T
        threshold_mask = teacher_scores >= teacher_threshold
        teacher_scaled = torch.exp((teacher_scores - teacher_threshold) / max(teacher_temperature, 1e-6))
        soft_scores = soft_scores + soft_positive_weight * teacher_scaled * threshold_mask.float()

    if label_weight > 0:
        disease_labels = labels_from_texts(texts, DISEASE_KEYWORDS, device)
        soft_scores = soft_scores + label_weight * jaccard_similarity_matrix(disease_labels)
    if anatomy_weight > 0:
        anatomy_labels = labels_from_texts(texts, ANATOMY_KEYWORDS, device)
        soft_scores = soft_scores + anatomy_weight * jaccard_similarity_matrix(anatomy_labels)

    soft_scores.fill_diagonal_(0.0)
    targets = targets + soft_scores
    return targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)


def soft_symmetric_info_nce(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * (image_embeddings @ text_embeddings.T)
    image_to_text_loss = soft_cross_entropy(logits, targets)
    text_to_image_loss = soft_cross_entropy(logits.T, targets.T)
    loss = 0.5 * (image_to_text_loss + text_to_image_loss)
    with torch.no_grad():
        diag = logits.diag()
        i2t_ranks = (logits > diag.unsqueeze(1)).sum(dim=1) + 1
        t2i_ranks = (logits.T > diag.unsqueeze(1)).sum(dim=1) + 1
    batch_size = image_embeddings.shape[0]
    return loss, {
        "loss": float(loss.detach().cpu()),
        "image_to_text_loss": float(image_to_text_loss.detach().cpu()),
        "text_to_image_loss": float(text_to_image_loss.detach().cpu()),
        "image_to_text_r1": float((i2t_ranks <= 1).float().mean().detach().cpu()),
        "image_to_text_r5": float((i2t_ranks <= min(5, batch_size)).float().mean().detach().cpu()),
        "text_to_image_r1": float((t2i_ranks <= 1).float().mean().detach().cpu()),
        "text_to_image_r5": float((t2i_ranks <= min(5, batch_size)).float().mean().detach().cpu()),
        "logit_scale": float(scale.detach().cpu()),
    }


def uncertainty_loss(
    model: torch.nn.Module,
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    image_logvar, text_logvar = raw_model.embedding_log_variances(image_embeddings, text_embeddings)
    pair_logvar = 0.5 * (image_logvar + text_logvar)
    cosine_distance = 1.0 - F.cosine_similarity(image_embeddings, text_embeddings, dim=1)
    loss = (torch.exp(-pair_logvar) * cosine_distance + pair_logvar).mean()
    return loss, {
        "uncertainty_loss": float(loss.detach().cpu()),
        "image_logvar_mean": float(image_logvar.detach().mean().cpu()),
        "text_logvar_mean": float(text_logvar.detach().mean().cpu()),
    }


def longitudinal_consistency_loss(
    model: torch.nn.Module,
    image_embeddings: torch.Tensor,
    row_indices: torch.Tensor,
    prior_text_by_row: dict[int, str] | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not prior_text_by_row:
        zero = image_embeddings.sum() * 0.0
        return zero, {"longitudinal_loss": 0.0, "longitudinal_pairs": 0.0}
    prior_items = prior_texts_for_batch(row_indices, prior_text_by_row)
    valid_positions = [idx for idx, text in enumerate(prior_items) if text]
    if not valid_positions:
        zero = image_embeddings.sum() * 0.0
        return zero, {"longitudinal_loss": 0.0, "longitudinal_pairs": 0.0}
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    prior_texts = [prior_items[idx] for idx in valid_positions if prior_items[idx] is not None]
    prior_embeddings = raw_model.encode_texts(prior_texts)
    selected_images = image_embeddings[torch.as_tensor(valid_positions, dtype=torch.long, device=image_embeddings.device)]
    loss = 1.0 - F.cosine_similarity(
        F.normalize(selected_images, p=2, dim=1),
        F.normalize(prior_embeddings, p=2, dim=1),
        dim=1,
    ).mean()
    return loss, {"longitudinal_loss": float(loss.detach().cpu()), "longitudinal_pairs": float(len(valid_positions))}


def load_init_checkpoint(model: ImageTextContrastiveModel, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        print(
            json.dumps(
                {
                    "init_checkpoint": str(checkpoint_path),
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                },
                indent=2,
            )
        )
    return checkpoint


def optimizer_param_groups(
    model: ImageTextContrastiveModel,
    lr: float,
    encoder_lr: float | None,
    weight_decay: float,
) -> list[dict[str, Any]]:
    encoder_lr = lr if encoder_lr is None else encoder_lr
    head_params: list[torch.nn.Parameter] = []
    encoder_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("image_encoder.") or name.startswith("text_encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)
    groups: list[dict[str, Any]] = []
    if head_params:
        groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
    return groups


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def amp_dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def cuda_autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype) -> Any:
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def make_grad_scaler(enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def image_to_positive_and_hard_text_loss(
    model: torch.nn.Module,
    image_embeddings: torch.Tensor,
    positive_text_embeddings: torch.Tensor,
    hard_negative_texts: list[str],
    negatives_per_sample: int,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    batch_size = image_embeddings.shape[0]
    if negatives_per_sample <= 0:
        zero = image_embeddings.sum() * 0.0
        return zero, {"hard_negative_loss": 0.0, "hard_negative_acc": 0.0}

    hard_text_embeddings = raw_model.encode_texts(hard_negative_texts)
    hard_text_embeddings = hard_text_embeddings.reshape(batch_size, negatives_per_sample, -1)

    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    positive_text_embeddings = F.normalize(positive_text_embeddings, p=2, dim=1)
    hard_text_embeddings = F.normalize(hard_text_embeddings, p=2, dim=2)

    positive_logits = torch.sum(image_embeddings * positive_text_embeddings, dim=1, keepdim=True)
    hard_logits = torch.einsum("bd,bkd->bk", image_embeddings, hard_text_embeddings)
    logits = torch.cat([positive_logits, hard_logits], dim=1) * logit_scale.exp().clamp(max=100.0)
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=1) == 0).float().mean()
    return loss, {
        "hard_negative_loss": float(loss.detach().cpu()),
        "hard_negative_acc": float(accuracy.detach().cpu()),
    }


def denominator_hard_negative_info_nce(
    model: torch.nn.Module,
    image_embeddings: torch.Tensor,
    positive_text_embeddings: torch.Tensor,
    hard_negative_texts: list[str],
    negatives_per_sample: int,
    logit_scale: torch.Tensor,
    soft_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    batch_size = image_embeddings.shape[0]
    if negatives_per_sample <= 0:
        if soft_targets is not None:
            return soft_symmetric_info_nce(image_embeddings, positive_text_embeddings, logit_scale, soft_targets)
        return symmetric_info_nce(image_embeddings, positive_text_embeddings, logit_scale)

    hard_text_embeddings = raw_model.encode_texts(hard_negative_texts)
    hard_text_embeddings = hard_text_embeddings.reshape(batch_size, negatives_per_sample, -1)

    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    positive_text_embeddings = F.normalize(positive_text_embeddings, p=2, dim=1)
    hard_text_embeddings = F.normalize(hard_text_embeddings, p=2, dim=2)
    scale = logit_scale.exp().clamp(max=100.0)

    batch_logits = scale * (image_embeddings @ positive_text_embeddings.T)
    hard_logits = scale * torch.einsum("bd,bkd->bk", image_embeddings, hard_text_embeddings)
    i2t_logits = torch.cat([batch_logits, hard_logits], dim=1)
    labels = torch.arange(batch_size, dtype=torch.long, device=image_embeddings.device)
    if soft_targets is None:
        image_to_text_loss = F.cross_entropy(i2t_logits, labels)
        text_to_image_loss = F.cross_entropy(batch_logits.T, labels)
    else:
        hard_zero_targets = torch.zeros(
            (batch_size, negatives_per_sample),
            dtype=soft_targets.dtype,
            device=soft_targets.device,
        )
        image_to_text_loss = soft_cross_entropy(i2t_logits, torch.cat([soft_targets, hard_zero_targets], dim=1))
        text_to_image_loss = soft_cross_entropy(batch_logits.T, soft_targets.T)
    loss = 0.5 * (image_to_text_loss + text_to_image_loss)

    with torch.no_grad():
        i2t_ranks = (batch_logits > batch_logits.diag().unsqueeze(1)).sum(dim=1) + 1
        t2i_ranks = (batch_logits.T > batch_logits.diag().unsqueeze(1)).sum(dim=1) + 1
        hard_negative_acc = (i2t_logits.argmax(dim=1) == labels).float().mean()

    return loss, {
        "loss": float(loss.detach().cpu()),
        "image_to_text_loss": float(image_to_text_loss.detach().cpu()),
        "text_to_image_loss": float(text_to_image_loss.detach().cpu()),
        "image_to_text_r1": float((i2t_ranks <= 1).float().mean().detach().cpu()),
        "image_to_text_r5": float((i2t_ranks <= min(5, batch_size)).float().mean().detach().cpu()),
        "text_to_image_r1": float((t2i_ranks <= 1).float().mean().detach().cpu()),
        "text_to_image_r5": float((t2i_ranks <= min(5, batch_size)).float().mean().detach().cpu()),
        "logit_scale": float(scale.detach().cpu()),
        "hard_negative_loss": float(image_to_text_loss.detach().cpu()),
        "hard_negative_acc": float(hard_negative_acc.detach().cpu()),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
    gradient_clip_norm: float,
    hard_negative_sampler: HardNegativeTextSampler | None,
    negatives_per_sample: int,
    hard_negative_weight: float,
    hard_negative_mode: str,
    teacher_image_embeddings: np.ndarray | None,
    teacher_text_embeddings: np.ndarray | None,
    kd_image_weight: float,
    kd_text_weight: float,
    kd_relational_weight: float,
    kd_temperature: float,
    soft_positive_weight: float,
    soft_positive_threshold: float,
    soft_positive_temperature: float,
    label_soft_positive_weight: float,
    anatomy_soft_positive_weight: float,
    pseudo_label_weight: float,
    longitudinal_weight: float,
    prior_text_by_row: dict[int, str] | None,
    uncertainty_weight: float,
    distributed: bool,
    rank: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: Any | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    if not any(param.requires_grad for param in raw_model.image_encoder.parameters()):
        raw_model.image_encoder.eval()
    if not any(param.requires_grad for param in raw_model.text_encoder.parameters()):
        raw_model.text_encoder.eval()

    totals: dict[str, float] = {}
    steps = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress = tqdm(loader, desc="train" if is_train else "val", disable=not is_main_process(rank))
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with cuda_autocast_context(device, amp_enabled, amp_dtype):
                image_embeddings, text_embeddings = model(batch)
            soft_targets = build_soft_positive_targets(
                row_indices=batch["row_index"],
                texts=batch["text"],
                teacher_text_embeddings=teacher_text_embeddings,
                device=device,
                teacher_threshold=soft_positive_threshold,
                teacher_temperature=soft_positive_temperature,
                soft_positive_weight=soft_positive_weight,
                label_weight=label_soft_positive_weight,
                anatomy_weight=anatomy_soft_positive_weight,
            )
            base_loss, base_metrics = soft_symmetric_info_nce(
                image_embeddings,
                text_embeddings,
                raw_model.logit_scale,
                soft_targets,
            )
            loss = base_loss
            metrics = dict(base_metrics)
            hard_loss = image_embeddings.sum() * 0.0
            hard_metrics = {"hard_negative_loss": 0.0, "hard_negative_acc": 0.0}

            if hard_negative_sampler is not None and negatives_per_sample > 0 and hard_negative_weight > 0:
                hard_texts = hard_negative_sampler.sample_texts(batch["row_index"], negatives_per_sample)
                if hard_negative_mode == "denominator":
                    hard_loss, hard_metrics = denominator_hard_negative_info_nce(
                        model=model,
                        image_embeddings=image_embeddings,
                        positive_text_embeddings=text_embeddings,
                        hard_negative_texts=hard_texts,
                        negatives_per_sample=negatives_per_sample,
                        logit_scale=raw_model.logit_scale,
                        soft_targets=soft_targets,
                    )
                    loss = hard_loss
                    metrics.update(
                        {
                            key: value
                            for key, value in hard_metrics.items()
                            if key not in {"hard_negative_loss", "hard_negative_acc"}
                        }
                    )
                else:
                    hard_loss, hard_metrics = image_to_positive_and_hard_text_loss(
                        model=model,
                        image_embeddings=image_embeddings,
                        positive_text_embeddings=text_embeddings,
                        hard_negative_texts=hard_texts,
                        negatives_per_sample=negatives_per_sample,
                        logit_scale=raw_model.logit_scale,
                    )
                    loss = base_loss + hard_negative_weight * hard_loss

            kd_image_loss = image_embeddings.sum() * 0.0
            kd_text_loss = image_embeddings.sum() * 0.0
            kd_relational_loss = image_embeddings.sum() * 0.0
            if teacher_image_embeddings is not None and teacher_text_embeddings is not None:
                teacher_images, teacher_texts = teacher_batch(
                    batch["row_index"], teacher_image_embeddings, teacher_text_embeddings, device
                )
                if kd_relational_weight > 0:
                    kd_relational_loss = relational_kd_loss(
                        image_embeddings=image_embeddings,
                        text_embeddings=text_embeddings,
                        teacher_image_embeddings=teacher_images,
                        teacher_text_embeddings=teacher_texts,
                        temperature=kd_temperature,
                    )
                    loss = loss + kd_relational_weight * kd_relational_loss
                if kd_image_weight > 0:
                    image_features = raw_model.encode_image_features(batch)
                    kd_image_loss = direct_kd_loss(image_features, teacher_images)
                    loss = loss + kd_image_weight * kd_image_loss
                if kd_text_weight > 0 and getattr(raw_model.text_encoder, "model_id", "") != "microsoft/BiomedVLP-BioViL-T":
                    text_features = raw_model.encode_text_features(batch["text"])
                    kd_text_loss = direct_kd_loss(text_features, teacher_texts)
                    loss = loss + kd_text_weight * kd_text_loss

            pseudo_label_loss = image_embeddings.sum() * 0.0
            if pseudo_label_weight > 0:
                if not hasattr(raw_model, "pseudo_label_head"):
                    raise RuntimeError("pseudo_label_head was not initialized before optimizer creation.")
                labels = pseudo_labels_for_texts(batch["text"], device=device)
                logits = raw_model.pseudo_label_head(image_embeddings)
                pseudo_label_loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = loss + pseudo_label_weight * pseudo_label_loss

            longitudinal_loss, longitudinal_metrics = longitudinal_consistency_loss(
                model=model,
                image_embeddings=image_embeddings,
                row_indices=batch["row_index"],
                prior_text_by_row=prior_text_by_row,
            )
            if longitudinal_weight > 0:
                loss = loss + longitudinal_weight * longitudinal_loss

            uncertainty_reg_loss = image_embeddings.sum() * 0.0
            uncertainty_metrics = {
                "uncertainty_loss": 0.0,
                "image_logvar_mean": 0.0,
                "text_logvar_mean": 0.0,
            }
            if uncertainty_weight > 0:
                uncertainty_reg_loss, uncertainty_metrics = uncertainty_loss(model, image_embeddings, text_embeddings)
                loss = loss + uncertainty_weight * uncertainty_reg_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite contrastive loss: {float(loss.detach().cpu())}")

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if gradient_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [param for param in model.parameters() if param.requires_grad],
                            max_norm=gradient_clip_norm,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [param for param in model.parameters() if param.requires_grad],
                            max_norm=gradient_clip_norm,
                        )
                    optimizer.step()

            metrics["base_loss"] = metrics["loss"]
            metrics["loss"] = float(loss.detach().cpu())
            metrics.update(hard_metrics)
            metrics["kd_image_loss"] = float(kd_image_loss.detach().cpu())
            metrics["kd_text_loss"] = float(kd_text_loss.detach().cpu())
            metrics["kd_relational_loss"] = float(kd_relational_loss.detach().cpu())
            metrics["pseudo_label_loss"] = float(pseudo_label_loss.detach().cpu())
            metrics.update(longitudinal_metrics)
            metrics.update(uncertainty_metrics)
            metrics["soft_positive_offdiag_mass"] = float((1.0 - soft_targets.diag().mean()).detach().cpu())
            metrics["soft_positive_nonzero"] = float(((soft_targets > 0).float().sum(dim=1).mean()).detach().cpu())
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            steps += 1
            progress.set_postfix(
                loss=totals["loss"] / steps,
                i2t_r1=totals["image_to_text_r1"] / steps,
                hn_acc=totals["hard_negative_acc"] / steps,
            )

            if max_batches is not None and steps >= max_batches:
                break

    if steps == 0:
        raise RuntimeError("No batches were processed.")
    if distributed:
        keys = sorted(totals)
        values = torch.tensor([totals[key] for key in keys] + [float(steps)], dtype=torch.float64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        reduced_steps = max(float(values[-1].item()), 1.0)
        return {key: float(values[index].item() / reduced_steps) for index, key in enumerate(keys)}
    return {key: value / steps for key, value in totals.items()}


def parse_epoch_retrieval_pools(value: str) -> list[int | None]:
    pools: list[int | None] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item in {"full", "all"}:
            pools.append(None)
        else:
            pools.append(int(item))
    return pools


def load_split_indices(splits_dir: Path, split: str) -> np.ndarray:
    path = splits_dir / f"kd_{split}_indices.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing split indices: {path}")
    return np.load(path).astype(np.int64)


def maybe_sample_indices(indices: np.ndarray, candidate_pool_size: int | None, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if candidate_pool_size is None or candidate_pool_size <= 0 or candidate_pool_size >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    sampled = rng.choice(indices, size=candidate_pool_size, replace=False)
    return np.asarray(sorted(sampled.tolist()), dtype=np.int64)


@torch.no_grad()
def collect_retrieval_embeddings(
    model: ImageTextContrastiveModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    image_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    row_indices: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="epoch-retrieval-embed"):
        batch = move_batch_to_device(batch, device)
        image_batch, text_batch = model(batch)
        image_embeddings.append(image_batch.detach().cpu().float())
        text_embeddings.append(text_batch.detach().cpu().float())
        row_indices.append(batch["row_index"].detach().cpu())

    if not image_embeddings:
        raise RuntimeError("No retrieval embeddings were generated.")

    return {
        "image_embeddings": torch.cat(image_embeddings, dim=0),
        "text_embeddings": torch.cat(text_embeddings, dim=0),
        "row_indices": torch.cat(row_indices, dim=0).numpy(),
    }


def subset_payload(payload: dict[str, Any], selected_row_indices: np.ndarray) -> dict[str, Any]:
    positions_by_row = {int(row): pos for pos, row in enumerate(payload["row_indices"].tolist())}
    positions = np.asarray([positions_by_row[int(row)] for row in selected_row_indices], dtype=np.int64)
    return {
        "image_embeddings": payload["image_embeddings"][positions],
        "text_embeddings": payload["text_embeddings"][positions],
        "row_indices": payload["row_indices"][positions],
    }


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= min(k, ranks.numel())).float().mean().item())


@torch.no_grad()
def compute_retrieval_metrics(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> dict[str, float]:
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
    candidates = candidate_embeddings.to(device)
    num_items = query_embeddings.shape[0]
    ranks: list[torch.Tensor] = []

    for start in tqdm(range(0, num_items, chunk_size), desc="epoch-retrieval-score"):
        end = min(start + chunk_size, num_items)
        queries = query_embeddings[start:end].to(device)
        logits = queries @ candidates.T
        local_targets = torch.arange(start, end, device=device)
        target_scores = logits[torch.arange(end - start, device=device), local_targets]
        rank_batch = (logits > target_scores[:, None]).sum(dim=1) + 1
        ranks.append(rank_batch.detach().cpu())

    all_ranks = torch.cat(ranks).float()
    return {
        "R@1": recall_at_k(all_ranks, 1),
        "R@5": recall_at_k(all_ranks, 5),
        "R@10": recall_at_k(all_ranks, 10),
        "MedianRank": float(all_ranks.median().item()),
        "MeanRank": float(all_ranks.mean().item()),
        "NumQueries": int(num_items),
    }


def run_epoch_retrieval_eval(
    model: ImageTextContrastiveModel,
    metadata: pd.DataFrame,
    splits_dir: Path,
    split: str,
    image_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_views: int,
    text_source: str,
    pool_sizes: list[int | None],
    seed: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if not pool_sizes:
        return {}

    split_indices = load_split_indices(splits_dir, split)
    include_full = any(pool_size is None for pool_size in pool_sizes)
    results: dict[str, Any] = {}

    if include_full:
        full_loader, _ = make_loader(
            metadata=metadata,
            indices=split_indices,
            image_root=image_root,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            train=False,
            max_views=max_views,
            text_source=text_source,
            distributed=False,
        )
        full_payload = collect_retrieval_embeddings(model, full_loader, device)
        for pool_size in pool_sizes:
            selected_indices = maybe_sample_indices(split_indices, pool_size, seed)
            payload = full_payload if pool_size is None else subset_payload(full_payload, selected_indices)
            label = "full" if pool_size is None else str(int(pool_size))
            results[label] = {
                "candidate_pool_size": int(len(payload["row_indices"])),
                "image_to_text": compute_retrieval_metrics(
                    payload["image_embeddings"],
                    payload["text_embeddings"],
                    device=device,
                    chunk_size=chunk_size,
                ),
                "text_to_image": compute_retrieval_metrics(
                    payload["text_embeddings"],
                    payload["image_embeddings"],
                    device=device,
                    chunk_size=chunk_size,
                ),
            }
        return results

    for pool_size in pool_sizes:
        selected_indices = maybe_sample_indices(split_indices, pool_size, seed)
        loader, _ = make_loader(
            metadata=metadata,
            indices=selected_indices,
            image_root=image_root,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            train=False,
            max_views=max_views,
            text_source=text_source,
            distributed=False,
        )
        payload = collect_retrieval_embeddings(model, loader, device)
        label = str(int(pool_size)) if pool_size is not None else "full"
        results[label] = {
            "candidate_pool_size": int(len(payload["row_indices"])),
            "image_to_text": compute_retrieval_metrics(
                payload["image_embeddings"],
                payload["text_embeddings"],
                device=device,
                chunk_size=chunk_size,
            ),
            "text_to_image": compute_retrieval_metrics(
                payload["text_embeddings"],
                payload["image_embeddings"],
                device=device,
                chunk_size=chunk_size,
            ),
        }
    return results


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def retrieval_avg_r1(retrieval_eval: dict[str, Any], pool_key: str) -> float | None:
    if not retrieval_eval or pool_key not in retrieval_eval:
        return None
    pool = retrieval_eval[pool_key]
    return 0.5 * (
        float(pool["image_to_text"]["R@1"]) + float(pool["text_to_image"]["R@1"])
    )


def checkpoint_without_auxiliary_modules(model: ImageTextContrastiveModel) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("pseudo_label_head.")
    }


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = init_distributed()
    seed_everything(args.seed + rank)

    artifacts_dir = Path(args.artifacts_dir)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(artifacts_dir / args.metadata_file)
    train_indices = np.load(splits_dir / "kd_train_indices.npy")
    val_indices = np.load(splits_dir / "kd_val_indices.npy")
    if args.max_train_rows is not None:
        train_indices = train_indices[: args.max_train_rows]
    if args.max_val_rows is not None:
        val_indices = val_indices[: args.max_val_rows]

    train_loader, train_sampler = make_loader(
        metadata,
        train_indices,
        args.image_root,
        args.image_size,
        args.batch_size,
        args.num_workers,
        train=True,
        max_views=args.max_views,
        text_source=args.text_source,
        distributed=distributed,
    )
    val_loader, val_sampler = make_loader(
        metadata,
        val_indices,
        args.image_root,
        args.image_size,
        args.batch_size,
        args.num_workers,
        train=False,
        max_views=args.max_views,
        text_source=args.text_source,
        distributed=distributed,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = amp_dtype_from_name(args.amp_dtype)
    scaler = make_grad_scaler(amp_enabled and amp_dtype == torch.float16)
    image_encoder, image_feature_dim = load_image_encoder(args)
    text_encoder_name = args.text_model_id or args.text_encoder
    text_encoder = build_text_encoder_serialized(
        text_encoder_name=text_encoder_name,
        max_text_length=args.max_text_length,
        distributed=distributed,
        rank=rank,
    )

    model = ImageTextContrastiveModel(
        image_encoder=image_encoder,
        image_arch=args.image_student,
        image_feature_dim=image_feature_dim,
        text_encoder=text_encoder,
        text_feature_dim=text_encoder.output_dim,
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dropout=args.projection_dropout,
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_text_encoder=args.freeze_text_encoder,
    ).to(device)

    init_checkpoint: dict[str, Any] | None = None
    if args.init_contrastive_checkpoint:
        init_checkpoint = load_init_checkpoint(model, args.init_contrastive_checkpoint)
        model.to(device)

    if args.pseudo_label_weight > 0:
        model.pseudo_label_head = torch.nn.Linear(
            args.projection_dim,
            len(DISEASE_KEYWORDS) + len(ANATOMY_KEYWORDS),
            device=device,
        )

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    raw_model_for_optimizer = model.module if isinstance(model, DistributedDataParallel) else model
    param_groups = optimizer_param_groups(
        raw_model_for_optimizer,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
    )
    trainable_params = [param for group in param_groups for param in group["params"]]
    optimizer = torch.optim.AdamW(param_groups)
    hard_negative_sampler = HardNegativeTextSampler(
        metadata=metadata,
        hard_negative_file=args.hard_negative_file,
        text_source=args.text_source,
        seed=args.seed + rank,
    )
    teacher_image_embeddings: np.ndarray | None = None
    teacher_text_embeddings: np.ndarray | None = None
    if (
        args.kd_image_weight > 0
        or args.kd_text_weight > 0
        or args.kd_relational_weight > 0
        or args.soft_positive_weight > 0
    ):
        teacher_image_embeddings, teacher_text_embeddings = load_teacher_embeddings(
            artifacts_dir=artifacts_dir,
            image_file=args.teacher_image_embeddings_file,
            text_file=args.teacher_text_embeddings_file,
        )
    prior_text_by_row = build_prior_text_lookup(metadata, args.text_source) if args.longitudinal_weight > 0 else None

    config = vars(args) | {
        "device": str(device),
        "train_rows": int(len(train_loader.dataset)),
        "val_rows": int(len(val_loader.dataset)),
        "image_feature_dim": int(image_feature_dim),
        "text_feature_dim": int(text_encoder.output_dim),
        "trainable_parameters": int(sum(param.numel() for param in trainable_params)),
        "distributed": distributed,
        "world_size": world_size,
        "per_process_batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * world_size,
        "amp_enabled": amp_enabled,
        "amp_dtype": args.amp_dtype if amp_enabled else None,
        "hard_negative_mode": args.hard_negative_mode,
        "hard_negative_mining": "teacher_image_to_teacher_text_i2t_with_false_negative_filtering",
        "init_checkpoint_epoch": None if init_checkpoint is None else init_checkpoint.get("epoch"),
        "disease_labels": list(DISEASE_KEYWORDS),
        "anatomy_labels": list(ANATOMY_KEYWORDS),
        "num_prior_text_rows": 0 if prior_text_by_row is None else len(prior_text_by_row),
    }
    epoch_retrieval_pool_sizes = parse_epoch_retrieval_pools(args.epoch_retrieval_pool_sizes)
    if is_main_process(rank):
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=json_safe), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_5k_retrieval = float("-inf")
    best_full_retrieval = float("-inf")
    try:
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                max_batches=args.max_train_batches,
                gradient_clip_norm=args.gradient_clip_norm,
                hard_negative_sampler=hard_negative_sampler,
                negatives_per_sample=args.hard_negatives_per_sample,
                hard_negative_weight=args.hard_negative_weight,
                hard_negative_mode=args.hard_negative_mode,
                teacher_image_embeddings=teacher_image_embeddings,
                teacher_text_embeddings=teacher_text_embeddings,
                kd_image_weight=args.kd_image_weight,
                kd_text_weight=args.kd_text_weight,
                kd_relational_weight=args.kd_relational_weight,
                kd_temperature=args.kd_temperature,
                soft_positive_weight=args.soft_positive_weight,
                soft_positive_threshold=args.soft_positive_threshold,
                soft_positive_temperature=args.soft_positive_temperature,
                label_soft_positive_weight=args.label_soft_positive_weight,
                anatomy_soft_positive_weight=args.anatomy_soft_positive_weight,
                pseudo_label_weight=args.pseudo_label_weight,
                longitudinal_weight=args.longitudinal_weight,
                prior_text_by_row=prior_text_by_row,
                uncertainty_weight=args.uncertainty_weight,
                distributed=distributed,
                rank=rank,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                max_batches=args.max_val_batches,
                gradient_clip_norm=args.gradient_clip_norm,
                hard_negative_sampler=None,
                negatives_per_sample=0,
                hard_negative_weight=0.0,
                hard_negative_mode=args.hard_negative_mode,
                teacher_image_embeddings=teacher_image_embeddings,
                teacher_text_embeddings=teacher_text_embeddings,
                kd_image_weight=args.kd_image_weight,
                kd_text_weight=args.kd_text_weight,
                kd_relational_weight=args.kd_relational_weight,
                kd_temperature=args.kd_temperature,
                soft_positive_weight=args.soft_positive_weight,
                soft_positive_threshold=args.soft_positive_threshold,
                soft_positive_temperature=args.soft_positive_temperature,
                label_soft_positive_weight=args.label_soft_positive_weight,
                anatomy_soft_positive_weight=args.anatomy_soft_positive_weight,
                pseudo_label_weight=args.pseudo_label_weight,
                longitudinal_weight=args.longitudinal_weight,
                prior_text_by_row=prior_text_by_row,
                uncertainty_weight=args.uncertainty_weight,
                distributed=distributed,
                rank=rank,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                scaler=None,
            )

            if is_main_process(rank):
                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                retrieval_eval: dict[str, Any] = {}
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": checkpoint_without_auxiliary_modules(raw_model),
                    "config": config,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "retrieval_eval": retrieval_eval,
                }
                torch.save(checkpoint, output_dir / "last.pt")
                torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
                retrieval_eval = run_epoch_retrieval_eval(
                    model=raw_model,
                    metadata=metadata,
                    splits_dir=splits_dir,
                    split=args.epoch_retrieval_split,
                    image_root=args.image_root,
                    image_size=args.image_size,
                    batch_size=args.epoch_retrieval_batch_size,
                    num_workers=args.epoch_retrieval_num_workers,
                    max_views=args.max_views,
                    text_source=args.text_source,
                    pool_sizes=epoch_retrieval_pool_sizes,
                    seed=args.epoch_retrieval_seed,
                    chunk_size=args.epoch_retrieval_chunk_size,
                    device=device,
                )

                record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                if retrieval_eval:
                    record["retrieval_eval"] = {
                        "split": args.epoch_retrieval_split,
                        "seed": args.epoch_retrieval_seed,
                        "pools": retrieval_eval,
                    }
                history.append(record)
                print(json.dumps(record, indent=2, default=json_safe))
                (output_dir / "history.json").write_text(json.dumps(history, indent=2, default=json_safe), encoding="utf-8")

                checkpoint["retrieval_eval"] = retrieval_eval
                torch.save(checkpoint, output_dir / "last.pt")
                torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save(checkpoint, output_dir / "best.pt")
                    torch.save(checkpoint, output_dir / "best_val_loss.pt")
                selected_score = retrieval_avg_r1(retrieval_eval, args.retrieval_selection_pool)
                if selected_score is not None and selected_score > best_5k_retrieval:
                    best_5k_retrieval = selected_score
                    checkpoint["selection_metric"] = {
                        "name": f"{args.retrieval_selection_pool}_avg_i2t_t2i_r1",
                        "value": selected_score,
                    }
                    torch.save(checkpoint, output_dir / "best_5k_retrieval.pt")
                full_score = retrieval_avg_r1(retrieval_eval, "full")
                if full_score is not None and full_score > best_full_retrieval:
                    best_full_retrieval = full_score
                    checkpoint["selection_metric"] = {
                        "name": "full_avg_i2t_t2i_r1",
                        "value": full_score,
                    }
                    torch.save(checkpoint, output_dir / "best_full_retrieval.pt")
            if distributed:
                barrier_kwargs = {"device_ids": [torch.cuda.current_device()]} if torch.cuda.is_available() else {}
                dist.barrier(**barrier_kwargs)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
