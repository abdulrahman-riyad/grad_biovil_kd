from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BioViL-T teacher image/text embeddings for retrieval.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-file", default="metadata/biovil_t_fixed_metadata.csv")
    parser.add_argument("--image-embeddings-file", default="teacher/biovil_t_fixed_image_embeddings.npy")
    parser.add_argument("--text-embeddings-file", default="teacher/biovil_t_fixed_text_embeddings.npy")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-chunk-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-topk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_indices(splits_dir: Path, split: str) -> np.ndarray:
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


def normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features, p=2, dim=1)


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= min(k, ranks.numel())).float().mean().item())


def compute_retrieval(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    row_indices: np.ndarray,
    direction: str,
    device: torch.device,
    chunk_size: int,
    top_k: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    query_embeddings = normalize(query_embeddings)
    candidate_embeddings = normalize(candidate_embeddings)
    candidates = candidate_embeddings.to(device)
    num_items = query_embeddings.shape[0]
    top_k = min(top_k, num_items)
    ranks: list[torch.Tensor] = []
    topk_rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, num_items, chunk_size), desc=f"teacher-score-{direction}"):
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
    return {
        "R@1": recall_at_k(all_ranks, 1),
        "R@5": recall_at_k(all_ranks, 5),
        "R@10": recall_at_k(all_ranks, 10),
        "MedianRank": float(all_ranks.median().item()),
        "MeanRank": float(all_ranks.mean().item()),
        "NumQueries": int(num_items),
    }, topk_rows


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


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


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(artifacts_dir / args.metadata_file, usecols=["subject_id", "study_id"])
    split_indices = load_indices(splits_dir, args.split)
    selected_indices = maybe_sample_indices(split_indices, args.candidate_pool_size, args.seed)

    image_embeddings = np.load(artifacts_dir / args.image_embeddings_file, mmap_mode="r")
    text_embeddings = np.load(artifacts_dir / args.text_embeddings_file, mmap_mode="r")
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError(f"Teacher embedding shapes differ: {image_embeddings.shape} vs {text_embeddings.shape}")

    device = torch.device(args.device)
    selected_image_embeddings = torch.as_tensor(np.asarray(image_embeddings[selected_indices]), dtype=torch.float32)
    selected_text_embeddings = torch.as_tensor(np.asarray(text_embeddings[selected_indices]), dtype=torch.float32)

    i2t_metrics, i2t_topk = compute_retrieval(
        selected_image_embeddings,
        selected_text_embeddings,
        selected_indices,
        direction="image_to_text",
        device=device,
        chunk_size=args.similarity_chunk_size,
        top_k=args.top_k,
    )
    t2i_metrics, t2i_topk = compute_retrieval(
        selected_text_embeddings,
        selected_image_embeddings,
        selected_indices,
        direction="text_to_image",
        device=device,
        chunk_size=args.similarity_chunk_size,
        top_k=args.top_k,
    )

    selected_metadata = metadata.iloc[selected_indices]
    metrics = {
        "split": args.split,
        "checkpoint": "teacher_precomputed_embeddings",
        "checkpoint_epoch": None,
        "image_student": "biovil_t_teacher_vision",
        "text_encoder": "biovil_t_teacher_text",
        "text_model_id": "microsoft/BiomedVLP-BioViL-T",
        "num_rows": int(len(selected_indices)),
        "candidate_pool_size": int(len(selected_indices)),
        "device": str(device),
        "image_to_text": i2t_metrics,
        "text_to_image": t2i_metrics,
        "config": {
            "similarity_chunk_size": args.similarity_chunk_size,
            "top_k": args.top_k,
            "metadata_file": args.metadata_file,
            "image_embeddings_file": args.image_embeddings_file,
            "text_embeddings_file": args.text_embeddings_file,
            "seed": args.seed,
        },
    }
    write_json(output_dir / "retrieval_metrics.json", metrics)

    if args.save_embeddings:
        np.save(output_dir / "image_embeddings.npy", selected_image_embeddings.numpy())
        np.save(output_dir / "text_embeddings.npy", selected_text_embeddings.numpy())
        np.save(output_dir / "row_indices.npy", selected_indices)
        np.save(output_dir / "subject_ids.npy", selected_metadata["subject_id"].to_numpy(dtype=np.int64))
        np.save(output_dir / "study_ids.npy", selected_metadata["study_id"].to_numpy(dtype=np.int64))

    if args.save_topk:
        write_topk_csv(output_dir / "image_to_text_topk.csv", i2t_topk)
        write_topk_csv(output_dir / "text_to_image_topk.csv", t2i_topk)

    print(json.dumps(metrics, indent=2, default=json_safe))


if __name__ == "__main__":
    main()
