from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute teacher-guided hard negatives for Week 3 contrastive training.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--metadata-file", default="biovil_t_fixed_metadata.csv")
    parser.add_argument("--image-embeddings-file", default="biovil_t_fixed_image_embeddings.npy")
    parser.add_argument("--text-embeddings-file", default="biovil_t_fixed_text_embeddings.npy")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--candidate-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exclude-same-study", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--false-negative-text-threshold",
        type=float,
        default=0.85,
        help=(
            "Mask candidate reports whose teacher text cosine with the query report is at least this value. "
            "This avoids treating semantically near-identical reports as hard negatives. Use <=0 to disable."
        ),
    )
    parser.add_argument(
        "--min-hard-text-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional lower teacher text cosine bound for candidate reports. "
            "Use 0.60 to keep only clinically related hard negatives; default keeps all non-masked candidates."
        ),
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def load_split_indices(splits_dir: Path, split: str) -> np.ndarray:
    path = splits_dir / f"kd_{split}_indices.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return np.load(path).astype(np.int64)


def normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features, p=2, dim=1)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    splits_dir = Path(args.splits_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(artifacts_dir / args.metadata_file, usecols=["study_id"])
    query_indices = load_split_indices(splits_dir, args.split)
    candidate_indices = load_split_indices(splits_dir, args.candidate_split)

    image_embeddings = np.load(artifacts_dir / args.image_embeddings_file, mmap_mode="r")
    text_embeddings = np.load(artifacts_dir / args.text_embeddings_file, mmap_mode="r")

    device = torch.device(args.device)
    torch_dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32

    candidate_text = torch.as_tensor(np.asarray(text_embeddings[candidate_indices]), dtype=torch.float32)
    candidate_text = normalize(candidate_text).to(device=device, dtype=torch_dtype)
    candidate_study_ids = torch.as_tensor(
        metadata.iloc[candidate_indices]["study_id"].to_numpy(dtype=np.int64),
        device=device,
    )
    candidate_row_indices = torch.as_tensor(candidate_indices, device=device)

    top_k = min(args.top_k, max(int(len(candidate_indices)) - 1, 1))
    hard_negative_rows = np.empty((len(query_indices), top_k), dtype=np.int64)
    hard_negative_scores = np.empty((len(query_indices), top_k), dtype=np.float32)

    for start in tqdm(range(0, len(query_indices), args.chunk_size), desc="mine-hard-negatives"):
        end = min(start + args.chunk_size, len(query_indices))
        query_rows = query_indices[start:end]
        query_images = torch.as_tensor(np.asarray(image_embeddings[query_rows]), dtype=torch.float32)
        query_images = normalize(query_images).to(device=device, dtype=torch_dtype)

        scores = query_images @ candidate_text.T
        same_row = candidate_row_indices.unsqueeze(0).eq(
            torch.as_tensor(query_rows, device=device).unsqueeze(1)
        )
        scores = scores.masked_fill(same_row, float("-inf"))

        if args.exclude_same_study:
            query_study_ids = torch.as_tensor(
                metadata.iloc[query_rows]["study_id"].to_numpy(dtype=np.int64),
                device=device,
            )
            same_study = candidate_study_ids.unsqueeze(0).eq(query_study_ids.unsqueeze(1))
            scores = scores.masked_fill(same_study, float("-inf"))

        if args.false_negative_text_threshold > 0 or args.min_hard_text_threshold > 0:
            query_text = torch.as_tensor(np.asarray(text_embeddings[query_rows]), dtype=torch.float32)
            query_text = normalize(query_text).to(device=device, dtype=torch_dtype)
            text_text_scores = query_text @ candidate_text.T
            if args.false_negative_text_threshold > 0:
                likely_false_negative = text_text_scores >= args.false_negative_text_threshold
                scores = scores.masked_fill(likely_false_negative, float("-inf"))
            if args.min_hard_text_threshold > 0:
                too_unrelated = text_text_scores < args.min_hard_text_threshold
                scores = scores.masked_fill(too_unrelated, float("-inf"))

        values, positions = torch.topk(scores, k=top_k, dim=1)
        selected_rows = candidate_row_indices[positions]
        hard_negative_rows[start:end] = selected_rows.detach().cpu().numpy()
        hard_negative_scores[start:end] = values.detach().cpu().float().numpy()

    config = {
        "artifacts_dir": str(artifacts_dir),
        "splits_dir": str(splits_dir),
        "metadata_file": args.metadata_file,
        "image_embeddings_file": args.image_embeddings_file,
        "text_embeddings_file": args.text_embeddings_file,
        "split": args.split,
        "candidate_split": args.candidate_split,
        "top_k": int(top_k),
        "chunk_size": args.chunk_size,
        "device": str(device),
        "exclude_same_study": args.exclude_same_study,
        "false_negative_text_threshold": args.false_negative_text_threshold,
        "min_hard_text_threshold": args.min_hard_text_threshold,
        "dtype": args.dtype,
        "num_queries": int(len(query_indices)),
        "num_candidates": int(len(candidate_indices)),
    }

    np.savez_compressed(
        output_file,
        query_row_indices=query_indices,
        hard_negative_row_indices=hard_negative_rows,
        hard_negative_scores=hard_negative_scores,
        config_json=json.dumps(config, default=json_safe),
    )
    print(json.dumps(config | {"output_file": str(output_file)}, indent=2, default=json_safe))


if __name__ == "__main__":
    main()
