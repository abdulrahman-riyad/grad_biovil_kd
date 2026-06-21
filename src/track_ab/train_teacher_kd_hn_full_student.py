from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.contrastive_dataset import (
    ImageTextContrastiveDataset,
    clean_report_text,
    collate_image_text,
    extract_impression,
)
from data.transforms import build_image_transform
from models.contrastive_model import ImageTextContrastiveModel
from models.student_loaders import load_mobilevit_student, load_repvit_student, torch_load
from models.text_encoders import TEXT_ENCODER_PRESETS, build_text_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train full image/text students with the teammate-style two-stage "
            "InfoNCE + BioViL-T teacher KD + student-mined hard-negative recipe, "
            "but using the fixed kd_train/kd_val/kd_test split files."
        )
    )
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-file", default="metadata/biovil_t_fixed_metadata.csv")
    parser.add_argument("--teacher-image-embeddings-file", default="teacher/biovil_t_fixed_image_embeddings.npy")
    parser.add_argument("--teacher-text-embeddings-file", default="teacher/biovil_t_fixed_text_embeddings.npy")
    parser.add_argument("--image-student", choices=["mobilevit", "repvit"], required=True)
    parser.add_argument("--mobilevit-checkpoint", default=None)
    parser.add_argument("--repvit-checkpoint", default=None)
    parser.add_argument("--repvit-root", default=None)
    parser.add_argument("--text-encoder", choices=list(TEXT_ENCODER_PRESETS.keys()), required=True)
    parser.add_argument("--text-source", choices=["impression", "report"], default="impression")
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-hidden-dim", type=int, default=None)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-views", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stage2-lr-multiplier", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--kd-weight", type=float, default=0.25)
    parser.add_argument("--hn-pool-size", type=int, default=25000)
    parser.add_argument("--hn-top-k", type=int, default=5)
    parser.add_argument("--hn-refresh-epochs", type=int, default=2)
    parser.add_argument("--pool-encode-batch-size", type=int, default=64)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
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


def metadata_path(artifacts_dir: Path, metadata_file: str) -> Path:
    path = Path(metadata_file)
    return path if path.is_absolute() else artifacts_dir / path


def artifact_path(artifacts_dir: Path, filename: str) -> Path:
    path = Path(filename)
    return path if path.is_absolute() else artifacts_dir / path


def text_for_row(row: pd.Series, text_source: str) -> str:
    if text_source == "report":
        return clean_report_text(row.get("report_text", ""))
    return extract_impression(row.get("report_text", ""))


def load_image_encoder(args: argparse.Namespace, teacher_dim: int = 128) -> tuple[nn.Module, int]:
    if args.image_student == "mobilevit":
        if not args.mobilevit_checkpoint:
            raise ValueError("--mobilevit-checkpoint is required for --image-student mobilevit.")
        return load_mobilevit_student(args.mobilevit_checkpoint, teacher_dim=teacher_dim), teacher_dim
    if not args.repvit_checkpoint or not args.repvit_root:
        raise ValueError("--repvit-checkpoint and --repvit-root are required for --image-student repvit.")
    return load_repvit_student(args.repvit_checkpoint, args.repvit_root, teacher_dim=teacher_dim), teacher_dim


def make_loader(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    args: argparse.Namespace,
    train: bool,
) -> DataLoader:
    dataset = ImageTextContrastiveDataset(
        metadata=metadata,
        indices=indices,
        image_root=args.image_root,
        transform=build_image_transform(image_size=args.image_size, train=train),
        max_views=args.max_views,
        text_source=args.text_source,
        view_sampling="random" if train else "first",
        skip_empty_text=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        collate_fn=collate_image_text,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    if device.type != "cuda":
        return torch.amp.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def fixed_temperature_info_nce(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = F.normalize(image_embeddings, p=2, dim=1) @ F.normalize(text_embeddings, p=2, dim=1).T
    logits = logits / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    i2t = F.cross_entropy(logits, labels)
    t2i = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (i2t + t2i)
    ranks_i2t = target_ranks(logits)
    ranks_t2i = target_ranks(logits.T)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "image_to_text_loss": float(i2t.detach().cpu()),
        "text_to_image_loss": float(t2i.detach().cpu()),
        "image_to_text_r1": recall_at_k(ranks_i2t, 1),
        "image_to_text_r5": recall_at_k(ranks_i2t, 5),
        "text_to_image_r1": recall_at_k(ranks_t2i, 1),
        "text_to_image_r5": recall_at_k(ranks_t2i, 5),
    }


def target_ranks(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    order = logits.argsort(dim=1, descending=True)
    return order.eq(labels[:, None]).float().argmax(dim=1) + 1


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= min(k, int(ranks.numel()))).float().mean().detach().cpu())


def kd_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    student_embeddings = F.normalize(student_embeddings, p=2, dim=1)
    teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)
    mse = F.mse_loss(student_embeddings, teacher_embeddings)
    cosine = 1.0 - F.cosine_similarity(student_embeddings, teacher_embeddings, dim=1).mean()
    return mse + cosine


def hard_negative_info_nce(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    hard_negative_text_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    hard_negative_text_embeddings = F.normalize(hard_negative_text_embeddings, p=2, dim=2)
    batch_logits = image_embeddings @ text_embeddings.T / temperature
    hard_logits = torch.bmm(
        image_embeddings.unsqueeze(1),
        hard_negative_text_embeddings.transpose(1, 2),
    ).squeeze(1) / temperature
    logits = torch.cat([batch_logits, hard_logits], dim=1)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def hard_negatives_from_pool(
    text_embeddings: torch.Tensor,
    pool_embeddings_cpu: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    query = F.normalize(text_embeddings.detach().cpu(), p=2, dim=1)
    pool = F.normalize(pool_embeddings_cpu, p=2, dim=1)
    sims = query @ pool.T
    sims[sims > 0.99] = -1.0
    _, top_idx = sims.topk(k=min(top_k, pool.shape[0]), dim=1)
    return pool[top_idx].to(text_embeddings.device, non_blocking=True)


@torch.no_grad()
def build_student_text_pool(
    model: ImageTextContrastiveModel,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int | None,
) -> torch.Tensor:
    raw_indices = np.asarray(train_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if len(raw_indices) > args.hn_pool_size:
        raw_indices = rng.choice(raw_indices, size=args.hn_pool_size, replace=False)
    texts = [text_for_row(metadata.iloc[int(idx)], args.text_source) for idx in raw_indices]
    texts = [text for text in texts if text]
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), args.pool_encode_batch_size), desc="build-hn-text-pool"):
        batch_texts = texts[start : start + args.pool_encode_batch_size]
        with autocast_context(device, args.amp, args.amp_dtype):
            embeddings = model.encode_texts(batch_texts)
        chunks.append(F.normalize(embeddings.detach().cpu(), p=2, dim=1))
    if not chunks:
        raise RuntimeError("Hard-negative text pool is empty.")
    return torch.cat(chunks, dim=0)


def run_epoch(
    model: ImageTextContrastiveModel,
    loader: DataLoader,
    teacher_image_embeddings: np.ndarray,
    teacher_text_embeddings: np.ndarray,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    device: torch.device,
    stage: int,
    hard_negative_pool: torch.Tensor | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {}
    steps = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    max_batches = args.max_train_batches if is_train else args.max_val_batches
    scaler_enabled = bool(args.amp and device.type == "cuda" and args.amp_dtype == "float16" and is_train)
    scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)

    with context:
        progress = tqdm(loader, desc=("train-s1" if stage == 1 else "train-s2") if is_train else "val")
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            row_index = batch["row_index"].detach().cpu().numpy()
            teacher_img = torch.from_numpy(teacher_image_embeddings[row_index]).to(device=device, dtype=torch.float32)
            teacher_txt = torch.from_numpy(teacher_text_embeddings[row_index]).to(device=device, dtype=torch.float32)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, args.amp, args.amp_dtype):
                image_embeddings, text_embeddings = model(batch)
                if stage == 1:
                    base_loss, base_metrics = fixed_temperature_info_nce(
                        image_embeddings,
                        text_embeddings,
                        args.temperature,
                    )
                else:
                    if hard_negative_pool is None:
                        raise ValueError("Stage 2 requires a hard-negative text pool.")
                    hard_negs = hard_negatives_from_pool(text_embeddings, hard_negative_pool, args.hn_top_k)
                    base_loss = hard_negative_info_nce(
                        image_embeddings,
                        text_embeddings,
                        hard_negs,
                        args.temperature,
                    )
                    _, base_metrics = fixed_temperature_info_nce(image_embeddings, text_embeddings, args.temperature)

                image_kd = kd_loss(image_embeddings.float(), teacher_img.float())
                text_kd = kd_loss(text_embeddings.float(), teacher_txt.float())
                loss = base_loss + args.kd_weight * (image_kd + text_kd)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {float(loss.detach().cpu())}")

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        args.gradient_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()

            metrics = dict(base_metrics)
            metrics.update(
                {
                    "loss": float(loss.detach().cpu()),
                    "base_loss": float(base_loss.detach().cpu()),
                    "image_kd_loss": float(image_kd.detach().cpu()),
                    "text_kd_loss": float(text_kd.detach().cpu()),
                }
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            steps += 1
            progress.set_postfix(loss=totals["loss"] / steps, i2t_r1=totals["image_to_text_r1"] / steps)

            if max_batches is not None and steps >= max_batches:
                break

    if steps == 0:
        raise RuntimeError("No batches were processed.")
    return {key: value / steps for key, value in totals.items()}


@torch.no_grad()
def collect_embeddings(
    model: ImageTextContrastiveModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    image_parts: list[torch.Tensor] = []
    text_parts: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="retrieval-embed"):
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, args.amp, args.amp_dtype):
            image_embeddings, text_embeddings = model(batch)
        image_parts.append(F.normalize(image_embeddings.detach().cpu().float(), p=2, dim=1))
        text_parts.append(F.normalize(text_embeddings.detach().cpu().float(), p=2, dim=1))
    return torch.cat(image_parts, dim=0), torch.cat(text_parts, dim=0)


def retrieval_metrics(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, chunk_size: int = 512) -> dict[str, Any]:
    n = image_embeddings.shape[0]
    labels = torch.arange(n)

    def ranks_for(query: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
        ranks: list[torch.Tensor] = []
        gallery_t = gallery.T
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sims = query[start:end] @ gallery_t
            order = sims.argsort(dim=1, descending=True)
            local_labels = labels[start:end]
            ranks.append(order.eq(local_labels[:, None]).float().argmax(dim=1) + 1)
        return torch.cat(ranks, dim=0)

    def summarize(ranks: torch.Tensor) -> dict[str, float]:
        ranks_f = ranks.float()
        return {
            "R@1": float((ranks <= 1).float().mean()),
            "R@5": float((ranks <= 5).float().mean()),
            "R@10": float((ranks <= 10).float().mean()),
            "MedianRank": float(ranks_f.median()),
            "MeanRank": float(ranks_f.mean()),
            "NumQueries": int(n),
        }

    i2t = ranks_for(image_embeddings, text_embeddings)
    t2i = ranks_for(text_embeddings, image_embeddings)
    return {"image_to_text": summarize(i2t), "text_to_image": summarize(t2i)}


def save_checkpoint(
    path: Path,
    model: ImageTextContrastiveModel,
    args: argparse.Namespace,
    config: dict[str, Any],
    epoch: int,
    stage: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "config": config,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.artifacts_dir)
    splits_dir = Path(args.splits_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = pd.read_csv(metadata_path(artifacts_dir, args.metadata_file))
    train_indices = np.load(splits_dir / "kd_train_indices.npy")
    val_indices = np.load(splits_dir / "kd_val_indices.npy")
    if args.max_train_rows is not None:
        train_indices = train_indices[: args.max_train_rows]
    if args.max_val_rows is not None:
        val_indices = val_indices[: args.max_val_rows]

    teacher_image_embeddings = np.load(artifact_path(artifacts_dir, args.teacher_image_embeddings_file)).astype("float32")
    teacher_text_embeddings = np.load(artifact_path(artifacts_dir, args.teacher_text_embeddings_file)).astype("float32")

    train_loader = make_loader(metadata, train_indices, args, train=True)
    val_loader = make_loader(metadata, val_indices, args, train=False)

    image_encoder, image_feature_dim = load_image_encoder(args, teacher_dim=128)
    text_encoder = build_text_encoder(args.text_encoder, max_length=args.max_text_length)
    model = ImageTextContrastiveModel(
        image_encoder=image_encoder,
        image_arch=args.image_student,
        image_feature_dim=image_feature_dim,
        text_encoder=text_encoder,
        text_feature_dim=int(text_encoder.output_dim),
        projection_dim=args.projection_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dropout=args.projection_dropout,
        freeze_image_encoder=False,
        freeze_text_encoder=False,
    ).to(device)

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "train_rows": int(len(train_loader.dataset)),
            "val_rows": int(len(val_loader.dataset)),
            "image_feature_dim": int(image_feature_dim),
            "text_feature_dim": int(text_encoder.output_dim),
            "trainable_parameters": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
            "methodology": "teammate_style_full_student_kd_hn_fixed_split",
            "freeze_image_encoder": False,
            "freeze_text_encoder": False,
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_stage1_r1 = float("-inf")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.stage1_epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            teacher_image_embeddings,
            teacher_text_embeddings,
            optimizer,
            args,
            device,
            stage=1,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            teacher_image_embeddings,
            teacher_text_embeddings,
            None,
            args,
            device,
            stage=1,
        )
        image_emb, text_emb = collect_embeddings(model, val_loader, device, args)
        val_retrieval = retrieval_metrics(image_emb, text_emb)
        val_r1 = 0.5 * (val_retrieval["image_to_text"]["R@1"] + val_retrieval["text_to_image"]["R@1"])
        record = {
            "stage": 1,
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "val_retrieval": val_retrieval,
            "selection_avg_r1": val_r1,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        save_checkpoint(output_dir / "stage1_last.pt", model, args, config, epoch, 1, record)
        if val_r1 > best_stage1_r1:
            best_stage1_r1 = val_r1
            save_checkpoint(output_dir / "stage1_best.pt", model, args, config, epoch, 1, record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    stage1_best = torch_load(output_dir / "stage1_best.pt")
    model.load_state_dict(stage1_best["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * args.stage2_lr_multiplier, weight_decay=args.weight_decay)
    best_stage2_r1 = float("-inf")
    hard_negative_pool: torch.Tensor | None = None

    for epoch in range(1, args.stage2_epochs + 1):
        if hard_negative_pool is None or (epoch - 1) % args.hn_refresh_epochs == 0:
            hard_negative_pool = build_student_text_pool(
                model,
                metadata,
                train_indices,
                args,
                device,
                seed=None if epoch > 1 else args.seed,
            )
            model.train()
        train_metrics = run_epoch(
            model,
            train_loader,
            teacher_image_embeddings,
            teacher_text_embeddings,
            optimizer,
            args,
            device,
            stage=2,
            hard_negative_pool=hard_negative_pool,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            teacher_image_embeddings,
            teacher_text_embeddings,
            None,
            args,
            device,
            stage=2,
            hard_negative_pool=hard_negative_pool,
        )
        image_emb, text_emb = collect_embeddings(model, val_loader, device, args)
        val_retrieval = retrieval_metrics(image_emb, text_emb)
        val_r1 = 0.5 * (val_retrieval["image_to_text"]["R@1"] + val_retrieval["text_to_image"]["R@1"])
        record = {
            "stage": 2,
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "val_retrieval": val_retrieval,
            "selection_avg_r1": val_r1,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        save_checkpoint(output_dir / "stage2_last.pt", model, args, config, epoch, 2, record)
        if val_r1 > best_stage2_r1:
            best_stage2_r1 = val_r1
            save_checkpoint(output_dir / "stage2_best.pt", model, args, config, epoch, 2, record)
            save_checkpoint(output_dir / "best.pt", model, args, config, epoch, 2, record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
