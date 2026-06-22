from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from data.image_text_dataset import ImageTextContrastiveDataset, collate_image_text
from data.transforms import build_image_transform
from models.image_text_retrieval_model import ImageTextContrastiveModel
from models.text_encoders import TEXT_ENCODER_PRESETS, build_text_encoder
from train_stage2_hard_negative import (
    ANATOMY_KEYWORDS,
    DISEASE_KEYWORDS,
    HardNegativeTextSampler,
    build_prior_text_lookup,
    build_soft_positive_targets,
    checkpoint_without_auxiliary_modules,
    direct_kd_loss,
    json_safe,
    load_image_encoder,
    load_init_checkpoint,
    load_split_indices,
    load_teacher_embeddings,
    maybe_sample_indices,
    optimizer_param_groups,
    parse_epoch_retrieval_pools,
    pseudo_labels_for_texts,
    relational_kd_loss,
    retrieval_avg_r1,
    run_epoch_retrieval_eval,
    teacher_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TPU v5e-8 PyTorch/XLA fine-tuning for integrated hard-negative contrastive training."
    )
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hard-negative-file", required=True)
    parser.add_argument("--init-contrastive-checkpoint", default=None)
    parser.add_argument("--metadata-file", default="biovil_t_fixed_metadata.csv")
    parser.add_argument("--teacher-image-embeddings-file", default="biovil_t_fixed_image_embeddings.npy")
    parser.add_argument("--teacher-text-embeddings-file", default="biovil_t_fixed_text_embeddings.npy")
    parser.add_argument("--image-student", choices=["mobilevit", "repvit"], required=True)
    parser.add_argument("--mobilevit-checkpoint", default=None)
    parser.add_argument("--repvit-checkpoint", default=None)
    parser.add_argument("--repvit-root", default=None)
    parser.add_argument("--text-encoder", choices=list(TEXT_ENCODER_PRESETS.keys()), default="biovil_t")
    parser.add_argument("--text-model-id", default=None)
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
    parser.add_argument("--batch-size", type=int, default=8, help="Per-TPU-core batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per TPU process.")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--hard-negatives-per-sample", type=int, default=8)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-mode", choices=["denominator"], default="denominator")
    parser.add_argument("--kd-image-weight", type=float, default=0.0)
    parser.add_argument("--kd-text-weight", type=float, default=0.0)
    parser.add_argument("--kd-relational-weight", type=float, default=0.1)
    parser.add_argument("--kd-temperature", type=float, default=0.07)
    parser.add_argument("--soft-positive-weight", type=float, default=0.25)
    parser.add_argument("--soft-positive-threshold", type=float, default=0.85)
    parser.add_argument("--soft-positive-temperature", type=float, default=0.07)
    parser.add_argument("--label-soft-positive-weight", type=float, default=0.15)
    parser.add_argument("--anatomy-soft-positive-weight", type=float, default=0.05)
    parser.add_argument("--pseudo-label-weight", type=float, default=0.05)
    parser.add_argument("--longitudinal-weight", type=float, default=0.03)
    parser.add_argument("--uncertainty-weight", type=float, default=0.01)
    parser.add_argument("--epoch-retrieval-pool-sizes", default="")
    parser.add_argument("--epoch-retrieval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--epoch-retrieval-batch-size", type=int, default=64)
    parser.add_argument("--epoch-retrieval-num-workers", type=int, default=2)
    parser.add_argument("--epoch-retrieval-chunk-size", type=int, default=512)
    parser.add_argument("--epoch-retrieval-seed", type=int, default=42)
    parser.add_argument("--retrieval-selection-pool", default="5000")
    parser.add_argument(
        "--num-tpu-cores",
        type=int,
        default=8,
        help=(
            "Requested TPU devices. With the PJRT runtime, values greater than 1 are "
            "implemented by passing nprocs=None to xmp.spawn, which uses all visible devices."
        ),
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


def world_size() -> int:
    try:
        return int(xm.xrt_world_size())
    except Exception:
        return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    try:
        return int(xm.get_ordinal())
    except Exception:
        return int(os.environ.get("RANK", "0"))


def is_master() -> bool:
    try:
        return bool(xm.is_master_ordinal())
    except Exception:
        return rank() == 0


def make_xla_loader(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    image_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    max_views: int,
    text_source: str,
) -> DataLoader:
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
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size(),
        rank=rank(),
        shuffle=train,
        drop_last=train,
    )
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=train,
        collate_fn=collate_image_text,
        **kwargs,
    )


def reduce_metrics(totals: dict[str, torch.Tensor], steps: int, device: torch.device) -> dict[str, float]:
    keys = sorted(totals)
    values = torch.stack([totals[key] for key in keys] + [torch.tensor(float(steps), device=device)])
    reduced = xm.all_reduce(xm.REDUCE_SUM, values)
    reduced_cpu = reduced.detach().cpu().float()
    denom = max(float(reduced_cpu[-1].item()), 1.0)
    return {key: float(reduced_cpu[index].item() / denom) for index, key in enumerate(keys)}


def tpu_soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def tpu_alignment_loss(
    model: ImageTextContrastiveModel,
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    hard_negative_texts: list[str] | None,
    negatives_per_sample: int,
    soft_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    scale = model.logit_scale.exp().clamp(max=100.0)
    batch_size = image_embeddings.shape[0]
    batch_logits = scale * (image_embeddings @ text_embeddings.T)

    if hard_negative_texts and negatives_per_sample > 0:
        hard_text_embeddings = model.encode_texts(hard_negative_texts)
        hard_text_embeddings = F.normalize(
            hard_text_embeddings.reshape(batch_size, negatives_per_sample, -1),
            p=2,
            dim=2,
        )
        hard_logits = scale * torch.einsum("bd,bkd->bk", image_embeddings, hard_text_embeddings)
        hard_zero_targets = torch.zeros(
            (batch_size, negatives_per_sample),
            dtype=soft_targets.dtype,
            device=soft_targets.device,
        )
        image_to_text_loss = tpu_soft_cross_entropy(
            torch.cat([batch_logits, hard_logits], dim=1),
            torch.cat([soft_targets, hard_zero_targets], dim=1),
        )
        hard_negative_acc = (
            torch.cat([batch_logits, hard_logits], dim=1).argmax(dim=1)
            == torch.arange(batch_size, device=image_embeddings.device)
        ).float().mean()
    else:
        image_to_text_loss = tpu_soft_cross_entropy(batch_logits, soft_targets)
        hard_negative_acc = torch.zeros((), device=image_embeddings.device)

    text_to_image_loss = tpu_soft_cross_entropy(batch_logits.T, soft_targets.T)
    loss = 0.5 * (image_to_text_loss + text_to_image_loss)
    with torch.no_grad():
        diag = batch_logits.diag()
        i2t_ranks = (batch_logits > diag.unsqueeze(1)).sum(dim=1) + 1
        t2i_ranks = (batch_logits.T > diag.unsqueeze(1)).sum(dim=1) + 1
    return loss, {
        "loss": loss.detach(),
        "image_to_text_loss": image_to_text_loss.detach(),
        "text_to_image_loss": text_to_image_loss.detach(),
        "image_to_text_r1": (i2t_ranks <= 1).float().mean(),
        "image_to_text_r5": (i2t_ranks <= min(5, batch_size)).float().mean(),
        "text_to_image_r1": (t2i_ranks <= 1).float().mean(),
        "text_to_image_r5": (t2i_ranks <= min(5, batch_size)).float().mean(),
        "logit_scale": scale.detach(),
        "hard_negative_loss": image_to_text_loss.detach(),
        "hard_negative_acc": hard_negative_acc.detach(),
    }


def tpu_uncertainty_loss(
    model: ImageTextContrastiveModel,
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    image_logvar, text_logvar = model.embedding_log_variances(image_embeddings, text_embeddings)
    pair_logvar = 0.5 * (image_logvar + text_logvar)
    cosine_distance = 1.0 - F.cosine_similarity(image_embeddings, text_embeddings, dim=1)
    loss = (torch.exp(-pair_logvar) * cosine_distance + pair_logvar).mean()
    return loss, {
        "uncertainty_loss": loss.detach(),
        "image_logvar_mean": image_logvar.detach().mean(),
        "text_logvar_mean": text_logvar.detach().mean(),
    }


def tpu_longitudinal_consistency_loss(
    model: ImageTextContrastiveModel,
    image_embeddings: torch.Tensor,
    row_indices: torch.Tensor,
    prior_text_by_row: dict[int, str] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = image_embeddings.sum() * 0.0
    if not prior_text_by_row:
        return zero, {
            "longitudinal_loss": torch.zeros((), device=image_embeddings.device),
            "longitudinal_pairs": torch.zeros((), device=image_embeddings.device),
        }
    prior_items = [prior_text_by_row.get(int(row.item())) for row in row_indices.detach().cpu()]
    valid_positions = [idx for idx, text in enumerate(prior_items) if text]
    if not valid_positions:
        return zero, {
            "longitudinal_loss": torch.zeros((), device=image_embeddings.device),
            "longitudinal_pairs": torch.zeros((), device=image_embeddings.device),
        }
    prior_texts = [prior_items[idx] for idx in valid_positions if prior_items[idx] is not None]
    prior_embeddings = model.encode_texts(prior_texts)
    selected_images = image_embeddings[torch.as_tensor(valid_positions, dtype=torch.long, device=image_embeddings.device)]
    loss = 1.0 - F.cosine_similarity(
        F.normalize(selected_images, p=2, dim=1),
        F.normalize(prior_embeddings, p=2, dim=1),
        dim=1,
    ).mean()
    return loss, {
        "longitudinal_loss": loss.detach(),
        "longitudinal_pairs": torch.tensor(float(len(valid_positions)), device=image_embeddings.device),
    }


def run_xla_epoch(
    model: ImageTextContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
    args: argparse.Namespace,
    hard_negative_sampler: HardNegativeTextSampler | None,
    teacher_image_embeddings: np.ndarray | None,
    teacher_text_embeddings: np.ndarray | None,
    prior_text_by_row: dict[int, str] | None,
    epoch: int,
    phase: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if not any(param.requires_grad for param in model.image_encoder.parameters()):
        model.image_encoder.eval()
    if not any(param.requires_grad for param in model.text_encoder.parameters()):
        model.text_encoder.eval()

    totals: dict[str, torch.Tensor] = {}
    steps = 0
    device_loader = pl.MpDeviceLoader(loader, device)
    progress = tqdm(device_loader, desc=f"{phase}-xla", disable=not is_master())
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in progress:
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            image_embeddings, text_embeddings = model(batch)
            soft_targets = build_soft_positive_targets(
                row_indices=batch["row_index"],
                texts=batch["text"],
                teacher_text_embeddings=teacher_text_embeddings,
                device=device,
                teacher_threshold=args.soft_positive_threshold,
                teacher_temperature=args.soft_positive_temperature,
                soft_positive_weight=args.soft_positive_weight,
                label_weight=args.label_soft_positive_weight,
                anatomy_weight=args.anatomy_soft_positive_weight,
            )
            hard_texts: list[str] | None = None
            if is_train and hard_negative_sampler is not None and args.hard_negatives_per_sample > 0:
                hard_texts = hard_negative_sampler.sample_texts(batch["row_index"], args.hard_negatives_per_sample)
            loss, metrics = tpu_alignment_loss(
                model=model,
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                hard_negative_texts=hard_texts,
                negatives_per_sample=args.hard_negatives_per_sample if hard_texts else 0,
                soft_targets=soft_targets,
            )
            base_loss = loss

            kd_image_loss = image_embeddings.sum() * 0.0
            kd_text_loss = image_embeddings.sum() * 0.0
            kd_relational_loss = image_embeddings.sum() * 0.0
            if teacher_image_embeddings is not None and teacher_text_embeddings is not None:
                teacher_images, teacher_texts = teacher_batch(
                    batch["row_index"], teacher_image_embeddings, teacher_text_embeddings, device
                )
                if args.kd_relational_weight > 0:
                    kd_relational_loss = relational_kd_loss(
                        image_embeddings=image_embeddings,
                        text_embeddings=text_embeddings,
                        teacher_image_embeddings=teacher_images,
                        teacher_text_embeddings=teacher_texts,
                        temperature=args.kd_temperature,
                    )
                    loss = loss + args.kd_relational_weight * kd_relational_loss
                if args.kd_image_weight > 0:
                    kd_image_loss = direct_kd_loss(model.encode_image_features(batch), teacher_images)
                    loss = loss + args.kd_image_weight * kd_image_loss
                if args.kd_text_weight > 0 and getattr(model.text_encoder, "model_id", "") != "microsoft/BiomedVLP-BioViL-T":
                    kd_text_loss = direct_kd_loss(model.encode_text_features(batch["text"]), teacher_texts)
                    loss = loss + args.kd_text_weight * kd_text_loss

            pseudo_label_loss = image_embeddings.sum() * 0.0
            if args.pseudo_label_weight > 0:
                labels = pseudo_labels_for_texts(batch["text"], device=device)
                logits = model.pseudo_label_head(image_embeddings)
                pseudo_label_loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = loss + args.pseudo_label_weight * pseudo_label_loss

            long_loss, long_metrics = tpu_longitudinal_consistency_loss(
                model=model,
                image_embeddings=image_embeddings,
                row_indices=batch["row_index"],
                prior_text_by_row=prior_text_by_row,
            )
            if args.longitudinal_weight > 0:
                loss = loss + args.longitudinal_weight * long_loss

            unc_loss = image_embeddings.sum() * 0.0
            unc_metrics = {
                "uncertainty_loss": torch.zeros((), device=device),
                "image_logvar_mean": torch.zeros((), device=device),
                "text_logvar_mean": torch.zeros((), device=device),
            }
            if args.uncertainty_weight > 0:
                unc_loss, unc_metrics = tpu_uncertainty_loss(model, image_embeddings, text_embeddings)
                loss = loss + args.uncertainty_weight * unc_loss

            if is_train:
                loss.backward()
                if args.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        max_norm=args.gradient_clip_norm,
                    )
                xm.optimizer_step(optimizer, barrier=True)
                xm.mark_step()

            metrics["base_loss"] = base_loss.detach()
            metrics["loss"] = loss.detach()
            metrics["kd_image_loss"] = kd_image_loss.detach()
            metrics["kd_text_loss"] = kd_text_loss.detach()
            metrics["kd_relational_loss"] = kd_relational_loss.detach()
            metrics["pseudo_label_loss"] = pseudo_label_loss.detach()
            metrics.update(long_metrics)
            metrics.update(unc_metrics)
            metrics["soft_positive_offdiag_mass"] = (1.0 - soft_targets.diag().mean()).detach()
            metrics["soft_positive_nonzero"] = ((soft_targets > 0).float().sum(dim=1).mean()).detach()
            for key, value in metrics.items():
                totals[key] = totals.get(key, torch.zeros((), device=device)) + value.to(device)
            steps += 1
            if is_master():
                progress.set_postfix(step=steps)
            if max_batches is not None and steps >= max_batches:
                break

    if steps == 0:
        raise RuntimeError("No TPU batches were processed.")
    return reduce_metrics(totals, steps, device=device)


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[ImageTextContrastiveModel, dict[str, Any] | None]:
    image_encoder, image_feature_dim = load_image_encoder(args)
    text_encoder_name = args.text_model_id or args.text_encoder
    text_encoder = build_text_encoder(text_encoder_name, max_length=args.max_text_length)
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
    )
    init_checkpoint = None
    if args.init_contrastive_checkpoint:
        init_checkpoint = load_init_checkpoint(model, args.init_contrastive_checkpoint)
    if args.pseudo_label_weight > 0:
        model.pseudo_label_head = torch.nn.Linear(
            args.projection_dim,
            len(DISEASE_KEYWORDS) + len(ANATOMY_KEYWORDS),
        )
    return model.to(device), init_checkpoint


def _mp_fn(index: int, args_dict: dict[str, Any]) -> None:
    args = argparse.Namespace(**args_dict)
    seed_everything(args.seed + index)
    device = xm.xla_device()
    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)
    splits_dir = Path(args.splits_dir)
    if is_master():
        output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(artifacts_dir / args.metadata_file)
    train_indices = load_split_indices(splits_dir, "train")
    val_indices = load_split_indices(splits_dir, "val")
    if args.max_train_rows is not None:
        train_indices = train_indices[: args.max_train_rows]
    if args.max_val_rows is not None:
        val_indices = val_indices[: args.max_val_rows]

    train_loader = make_xla_loader(
        metadata=metadata,
        indices=train_indices,
        image_root=args.image_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
        max_views=args.max_views,
        text_source=args.text_source,
    )
    val_loader = make_xla_loader(
        metadata=metadata,
        indices=val_indices,
        image_root=args.image_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
        max_views=args.max_views,
        text_source=args.text_source,
    )

    model, init_checkpoint = build_model(args, device)
    param_groups = optimizer_param_groups(model, lr=args.lr, encoder_lr=args.encoder_lr, weight_decay=args.weight_decay)
    trainable_params = [param for group in param_groups for param in group["params"]]
    optimizer = torch.optim.AdamW(param_groups)

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
    hard_negative_sampler = HardNegativeTextSampler(
        metadata=metadata,
        hard_negative_file=args.hard_negative_file,
        text_source=args.text_source,
        seed=args.seed + rank(),
    )

    config = vars(args) | {
        "device": str(device),
        "xla_world_size": world_size(),
        "xla_rank": rank(),
        "train_rows": int(len(train_loader.dataset)),
        "val_rows": int(len(val_loader.dataset)),
        "image_feature_dim": 128,
        "text_feature_dim": int(model.text_encoder.output_dim),
        "trainable_parameters": int(sum(param.numel() for param in trainable_params)),
        "effective_batch_size": int(args.batch_size * world_size()),
        "init_checkpoint_epoch": None if init_checkpoint is None else init_checkpoint.get("epoch"),
        "disease_labels": list(DISEASE_KEYWORDS),
        "anatomy_labels": list(ANATOMY_KEYWORDS),
        "num_prior_text_rows": 0 if prior_text_by_row is None else len(prior_text_by_row),
    }
    epoch_retrieval_pool_sizes = parse_epoch_retrieval_pools(args.epoch_retrieval_pool_sizes)
    if is_master():
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=json_safe), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_5k_retrieval = float("-inf")
    best_full_retrieval = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)  # type: ignore[union-attr]
        val_loader.sampler.set_epoch(epoch)  # type: ignore[union-attr]
        train_metrics = run_xla_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_train_batches,
            args=args,
            hard_negative_sampler=hard_negative_sampler,
            teacher_image_embeddings=teacher_image_embeddings,
            teacher_text_embeddings=teacher_text_embeddings,
            prior_text_by_row=prior_text_by_row,
            epoch=epoch,
            phase="train",
        )
        val_metrics = run_xla_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            max_batches=args.max_val_batches,
            args=args,
            hard_negative_sampler=None,
            teacher_image_embeddings=teacher_image_embeddings,
            teacher_text_embeddings=teacher_text_embeddings,
            prior_text_by_row=prior_text_by_row,
            epoch=epoch,
            phase="val",
        )

        retrieval_eval: dict[str, Any] = {}
        if is_master() and epoch_retrieval_pool_sizes:
            retrieval_eval = run_epoch_retrieval_eval(
                model=model,
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
        xm.rendezvous(f"after_retrieval_epoch_{epoch}")

        if is_master():
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
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": checkpoint_without_auxiliary_modules(model),
                "config": config,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "retrieval_eval": retrieval_eval,
            }
            xm.save(checkpoint, output_dir / "last.pt")
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                xm.save(checkpoint, output_dir / "best.pt")
                xm.save(checkpoint, output_dir / "best_val_loss.pt")
            selected_score = retrieval_avg_r1(retrieval_eval, args.retrieval_selection_pool)
            if selected_score is not None and selected_score > best_5k_retrieval:
                best_5k_retrieval = selected_score
                checkpoint["selection_metric"] = {
                    "name": f"{args.retrieval_selection_pool}_avg_i2t_t2i_r1",
                    "value": selected_score,
                }
                xm.save(checkpoint, output_dir / "best_5k_retrieval.pt")
            full_score = retrieval_avg_r1(retrieval_eval, "full")
            if full_score is not None and full_score > best_full_retrieval:
                best_full_retrieval = full_score
                checkpoint["selection_metric"] = {"name": "full_avg_i2t_t2i_r1", "value": full_score}
                xm.save(checkpoint, output_dir / "best_full_retrieval.pt")
        xm.rendezvous(f"after_checkpoint_epoch_{epoch}")


def main() -> None:
    args = parse_args()
    # Newer PyTorch/XLA PJRT runtimes reject nprocs=8 on TPU. Passing None is the
    # supported way to spawn across all visible TPU devices; nprocs=1 remains useful
    # for debugging a single TPU core.
    nprocs = 1 if args.num_tpu_cores == 1 else None
    if args.num_tpu_cores > 1:
        os.environ.setdefault("TPU_NUM_DEVICES", str(args.num_tpu_cores))
    xmp.spawn(_mp_fn, args=(vars(args),), nprocs=nprocs, start_method="fork")


if __name__ == "__main__":
    main()
