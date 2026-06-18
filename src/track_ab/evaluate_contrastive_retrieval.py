from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.contrastive_dataset import ImageTextContrastiveDataset, collate_image_text
from data.transforms import build_image_transform
from models.contrastive_model import ImageTextContrastiveModel
from models.student_loaders import load_mobilevit_student, load_repvit_student, torch_load
from models.text_encoders import build_text_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-pool retrieval for Week 3 contrastive checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Path to contrastive best.pt or last.pt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--mobilevit-checkpoint", default=None)
    parser.add_argument("--repvit-checkpoint", default=None)
    parser.add_argument("--repvit-root", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--text-source", choices=["impression", "report"], default=None)
    parser.add_argument("--max-text-length", type=int, default=None)
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Optional sampled pool size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-chunk-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-topk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def config_value(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    return config.get(key, default)


def load_indices(splits_dir: Path, split: str) -> np.ndarray:
    path = splits_dir / f"kd_{split}_indices.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing split indices: {path}")
    return np.load(path)


def maybe_sample_indices(indices: np.ndarray, candidate_pool_size: int | None, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if candidate_pool_size is None or candidate_pool_size <= 0 or candidate_pool_size >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    sampled = rng.choice(indices, size=candidate_pool_size, replace=False)
    return np.asarray(sorted(sampled.tolist()), dtype=np.int64)


def make_loader(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    image_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_views: int,
    text_source: str,
) -> DataLoader:
    dataset = ImageTextContrastiveDataset(
        metadata=metadata,
        indices=indices,
        image_root=image_root,
        transform=build_image_transform(image_size=image_size, train=False),
        max_views=max_views,
        text_source=text_source,
        view_sampling="first",
        skip_empty_text=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_image_text,
    )


def load_image_encoder(config: dict[str, Any], args: argparse.Namespace) -> tuple[torch.nn.Module, int, str]:
    image_student = str(config["image_student"])
    image_feature_dim = int(config.get("image_feature_dim", 128))

    if image_student == "mobilevit":
        checkpoint = args.mobilevit_checkpoint or config.get("mobilevit_checkpoint")
        if not checkpoint:
            raise ValueError("MobileViT evaluation requires --mobilevit-checkpoint or config mobilevit_checkpoint.")
        return load_mobilevit_student(checkpoint, teacher_dim=image_feature_dim), image_feature_dim, image_student

    if image_student == "repvit":
        checkpoint = args.repvit_checkpoint or config.get("repvit_checkpoint")
        repvit_root = args.repvit_root or config.get("repvit_root")
        if not checkpoint or not repvit_root:
            raise ValueError("RepViT evaluation requires --repvit-checkpoint and --repvit-root.")
        return load_repvit_student(checkpoint, repvit_root, teacher_dim=image_feature_dim), image_feature_dim, image_student

    raise ValueError(f"Unsupported image_student: {image_student}")


def build_model(checkpoint: dict[str, Any], args: argparse.Namespace) -> ImageTextContrastiveModel:
    config = checkpoint["config"]
    image_encoder, image_feature_dim, image_student = load_image_encoder(config, args)
    text_encoder_name = config.get("text_model_id") or config["text_encoder"]
    max_text_length = int(config_value(args, config, "max_text_length", 256))
    text_encoder = build_text_encoder(text_encoder_name, max_length=max_text_length)

    model = ImageTextContrastiveModel(
        image_encoder=image_encoder,
        image_arch=image_student,
        image_feature_dim=image_feature_dim,
        text_encoder=text_encoder,
        text_feature_dim=int(config["text_feature_dim"]),
        projection_dim=int(config["projection_dim"]),
        projection_hidden_dim=config.get("projection_hidden_dim"),
        projection_dropout=float(config.get("projection_dropout", 0.0)),
        freeze_image_encoder=bool(config.get("freeze_image_encoder", True)),
        freeze_text_encoder=bool(config.get("freeze_text_encoder", True)),
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing_prefixes = ("pseudo_label_head.",)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected or missing:
        print(
            json.dumps(
                {
                    "checkpoint_load_warning": str(args.checkpoint),
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                },
                indent=2,
            )
        )
    return model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def collect_embeddings(
    model: ImageTextContrastiveModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    image_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    row_indices: list[torch.Tensor] = []
    subject_ids: list[torch.Tensor] = []
    study_ids: list[torch.Tensor] = []
    texts: list[str] = []

    for batch in tqdm(loader, desc="embed"):
        batch = move_batch_to_device(batch, device)
        image_batch, text_batch = model(batch)
        image_embeddings.append(image_batch.detach().cpu().float())
        text_embeddings.append(text_batch.detach().cpu().float())
        row_indices.append(batch["row_index"].detach().cpu())
        subject_ids.append(batch["subject_id"].detach().cpu())
        study_ids.append(batch["study_id"].detach().cpu())
        texts.extend(batch["text"])

    if not image_embeddings:
        raise RuntimeError("No embeddings were generated.")

    return {
        "image_embeddings": torch.cat(image_embeddings, dim=0),
        "text_embeddings": torch.cat(text_embeddings, dim=0),
        "row_indices": torch.cat(row_indices, dim=0).numpy(),
        "subject_ids": torch.cat(subject_ids, dim=0).numpy(),
        "study_ids": torch.cat(study_ids, dim=0).numpy(),
        "texts": texts,
    }


def compute_retrieval(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    row_indices: np.ndarray,
    direction: str,
    device: torch.device,
    chunk_size: int,
    top_k: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    candidate_embeddings = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=1)
    candidates = candidate_embeddings.to(device)
    num_items = query_embeddings.shape[0]
    top_k = min(top_k, num_items)

    ranks: list[torch.Tensor] = []
    topk_rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, num_items, chunk_size), desc=f"score-{direction}"):
        end = min(start + chunk_size, num_items)
        queries = query_embeddings[start:end].to(device)
        logits = queries @ candidates.T
        local_targets = torch.arange(start, end, device=device)
        target_scores = logits[torch.arange(end - start, device=device), local_targets]
        rank_batch = (logits > target_scores[:, None]).sum(dim=1) + 1
        ranks.append(rank_batch.detach().cpu())

        values, indices = logits.topk(k=top_k, dim=1)
        values = values.detach().cpu()
        indices = indices.detach().cpu()
        rank_cpu = rank_batch.detach().cpu()
        for local_index in range(end - start):
            query_position = start + local_index
            for retrieved_rank in range(top_k):
                retrieved_position = int(indices[local_index, retrieved_rank])
                topk_rows.append(
                    {
                        "direction": direction,
                        "query_position": query_position,
                        "query_row_index": int(row_indices[query_position]),
                        "target_position": query_position,
                        "target_row_index": int(row_indices[query_position]),
                        "target_rank": int(rank_cpu[local_index]),
                        "retrieved_rank": retrieved_rank + 1,
                        "retrieved_position": retrieved_position,
                        "retrieved_row_index": int(row_indices[retrieved_position]),
                        "score": float(values[local_index, retrieved_rank]),
                        "is_target": retrieved_position == query_position,
                    }
                )

    all_ranks = torch.cat(ranks).float()
    metrics = {
        "R@1": recall_at_k(all_ranks, 1),
        "R@5": recall_at_k(all_ranks, 5),
        "R@10": recall_at_k(all_ranks, 10),
        "MedianRank": float(all_ranks.median().item()),
        "MeanRank": float(all_ranks.mean().item()),
        "NumQueries": int(num_items),
    }
    return metrics, topk_rows


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= min(k, ranks.numel())).float().mean().item())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_safe), encoding="utf-8")


def write_topk_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    args = parse_args()
    checkpoint = torch_load(args.checkpoint)
    config = checkpoint["config"]

    artifacts_dir = Path(config_value(args, config, "artifacts_dir"))
    splits_dir = Path(config_value(args, config, "splits_dir"))
    image_root = str(config_value(args, config, "image_root"))
    metadata_file = str(config_value(args, config, "metadata_file", "biovil_t_fixed_metadata.csv"))
    image_size = int(config_value(args, config, "image_size", 224))
    max_views = int(config_value(args, config, "max_views", 3))
    text_source = str(config_value(args, config, "text_source", "impression"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(artifacts_dir / metadata_file)
    indices = load_indices(splits_dir, args.split)
    selected_indices = maybe_sample_indices(indices, args.candidate_pool_size, args.seed)
    loader = make_loader(
        metadata=metadata,
        indices=selected_indices,
        image_root=image_root,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_views=max_views,
        text_source=text_source,
    )

    device = torch.device(args.device)
    model = build_model(checkpoint, args).to(device)
    payload = collect_embeddings(model, loader, device)
    image_embeddings = payload["image_embeddings"]
    text_embeddings = payload["text_embeddings"]
    row_indices = payload["row_indices"]

    i2t_metrics, i2t_topk = compute_retrieval(
        image_embeddings,
        text_embeddings,
        row_indices,
        direction="image_to_text",
        device=device,
        chunk_size=args.similarity_chunk_size,
        top_k=args.top_k,
    )
    t2i_metrics, t2i_topk = compute_retrieval(
        text_embeddings,
        image_embeddings,
        row_indices,
        direction="text_to_image",
        device=device,
        chunk_size=args.similarity_chunk_size,
        top_k=args.top_k,
    )

    metrics = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "image_student": config.get("image_student"),
        "text_encoder": config.get("text_encoder"),
        "text_model_id": config.get("text_model_id"),
        "num_rows": int(len(row_indices)),
        "candidate_pool_size": int(len(row_indices)),
        "device": str(device),
        "image_to_text": i2t_metrics,
        "text_to_image": t2i_metrics,
        "config": {
            "batch_size": args.batch_size,
            "similarity_chunk_size": args.similarity_chunk_size,
            "top_k": args.top_k,
            "image_size": image_size,
            "max_views": max_views,
            "text_source": text_source,
            "metadata_file": metadata_file,
        },
    }
    write_json(output_dir / "retrieval_metrics.json", metrics)

    if args.save_embeddings:
        np.save(output_dir / "image_embeddings.npy", image_embeddings.numpy())
        np.save(output_dir / "text_embeddings.npy", text_embeddings.numpy())
        np.save(output_dir / "row_indices.npy", row_indices)
        np.save(output_dir / "subject_ids.npy", payload["subject_ids"])
        np.save(output_dir / "study_ids.npy", payload["study_ids"])

    if args.save_topk:
        write_topk_csv(output_dir / "image_to_text_topk.csv", i2t_topk)
        write_topk_csv(output_dir / "text_to_image_topk.csv", t2i_topk)

    print(json.dumps(metrics, indent=2, default=json_safe))


if __name__ == "__main__":
    main()
