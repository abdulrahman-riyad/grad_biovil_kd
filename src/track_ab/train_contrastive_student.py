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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from data.contrastive_dataset import ImageTextContrastiveDataset, collate_image_text
from data.transforms import build_image_transform
from losses import symmetric_info_nce
from models.contrastive_model import ImageTextContrastiveModel
from models.student_loaders import load_mobilevit_student, load_repvit_student
from models.text_encoders import TEXT_ENCODER_PRESETS, build_text_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileViT/RepViT image-text contrastive projection heads.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-file", default="biovil_t_fixed_metadata.csv")
    parser.add_argument("--image-student", choices=["mobilevit", "repvit"], required=True)
    parser.add_argument("--mobilevit-checkpoint", default=None)
    parser.add_argument("--repvit-checkpoint", default=None)
    parser.add_argument("--repvit-root", default=None)
    parser.add_argument("--text-encoder", choices=list(TEXT_ENCODER_PRESETS.keys()), default="cxr_bert")
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
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
        dist.init_process_group(backend="nccl")
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
    """Avoid concurrent checkpoint materialization from multiple DDP ranks."""
    barrier_kwargs = {"device_ids": [torch.cuda.current_device()]} if torch.cuda.is_available() else {}
    if distributed and rank != 0:
        dist.barrier(**barrier_kwargs)
    text_encoder = build_text_encoder(text_encoder_name, max_length=max_text_length)
    if distributed and rank == 0:
        dist.barrier(**barrier_kwargs)
    return text_encoder


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
    gradient_clip_norm: float,
    distributed: bool,
    rank: int,
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

            image_embeddings, text_embeddings = model(batch)
            loss, metrics = symmetric_info_nce(image_embeddings, text_embeddings, raw_model.logit_scale)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite contrastive loss: {float(loss.detach().cpu())}")

            if is_train:
                loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        max_norm=gradient_clip_norm,
                    )
                optimizer.step()

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            steps += 1
            progress.set_postfix(loss=totals["loss"] / steps, i2t_r1=totals["image_to_text_r1"] / steps)

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


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


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
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

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
    }
    if is_main_process(rank):
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=json_safe), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
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
                distributed=distributed,
                rank=rank,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                max_batches=args.max_val_batches,
                gradient_clip_norm=args.gradient_clip_norm,
                distributed=distributed,
                rank=rank,
            )

            if is_main_process(rank):
                record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                history.append(record)
                print(json.dumps(record, indent=2, default=json_safe))
                (output_dir / "history.json").write_text(json.dumps(history, indent=2, default=json_safe), encoding="utf-8")

                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "config": config,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }
                torch.save(checkpoint, output_dir / "last.pt")
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save(checkpoint, output_dir / "best.pt")
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
