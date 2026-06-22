import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.image_kd_dataset import ImageTeacherKDDataset
from data.transforms import build_image_transform
from models.image_student import ResNet18ImageStudent
from train_image_student import collate_batch, json_safe


def make_test_loader(
    metadata: pd.DataFrame,
    teacher_embeddings: np.ndarray,
    indices: np.ndarray,
    image_root: str | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = ImageTeacherKDDataset(
        metadata=metadata,
        teacher_image_embeddings=teacher_embeddings,
        indices=indices,
        image_root=image_root,
        transform=build_image_transform(image_size=image_size, train=False),
        view_sampling="first",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_batch,
    )


def load_student(checkpoint_path: Path, embedding_dim: int, device: torch.device) -> tuple[ResNet18ImageStudent, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = ResNet18ImageStudent(
        embedding_dim=embedding_dim,
        pretrained=False,
        dropout=float(config.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def compute_retrieval_metrics(similarity: np.ndarray) -> dict[str, float]:
    ranks: list[int] = []
    for i in range(similarity.shape[0]):
        order = np.argsort(-similarity[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)

    ranks_array = np.asarray(ranks, dtype=np.int64)
    return {
        "R@1": float(np.mean(ranks_array <= 1)),
        "R@5": float(np.mean(ranks_array <= 5)),
        "R@10": float(np.mean(ranks_array <= 10)),
        "MedianRank": float(np.median(ranks_array)),
        "MeanRank": float(np.mean(ranks_array)),
    }


def sampled_retrieval(
    student_image_embeddings: np.ndarray,
    teacher_text_embeddings: np.ndarray,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    n = len(student_image_embeddings)
    if sample_size <= 0:
        return {"enabled": False}

    rng = np.random.default_rng(seed)
    if n > sample_size:
        sample_indices = np.sort(rng.choice(n, size=sample_size, replace=False))
        sampled = True
    else:
        sample_indices = np.arange(n)
        sampled = False

    image_sample = student_image_embeddings[sample_indices]
    text_sample = teacher_text_embeddings[sample_indices]
    similarity = image_sample @ text_sample.T
    return {
        "enabled": True,
        "sampled": bool(sampled),
        "candidate_pool_size": int(len(sample_indices)),
        "image_to_text": compute_retrieval_metrics(similarity),
        "text_to_image": compute_retrieval_metrics(similarity.T),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained image student against BioViL-T teacher embeddings.")
    parser.add_argument("--artifacts-dir", default="weeks output/week1")
    parser.add_argument("--splits-dir", default="image_encoder_distillation/splits")
    parser.add_argument("--checkpoint", default="weeks output/week1/student_resnet18/best.pt")
    parser.add_argument("--output-dir", default="image_encoder_distillation/eval/resnet18_image_kd_test")
    parser.add_argument("--prefix", default="biovil_t_fixed")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--retrieval-sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(artifacts_dir / f"{args.prefix}_metadata.csv")
    teacher_image_embeddings = np.load(artifacts_dir / f"{args.prefix}_image_embeddings.npy", mmap_mode="r")
    teacher_text_embeddings_path = artifacts_dir / f"{args.prefix}_text_embeddings.npy"
    split_indices = np.load(splits_dir / f"kd_{args.split}_indices.npy")

    if len(metadata) != len(teacher_image_embeddings):
        raise ValueError("Metadata and teacher image embedding row counts do not match.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_student(
        checkpoint_path=Path(args.checkpoint),
        embedding_dim=int(teacher_image_embeddings.shape[1]),
        device=device,
    )
    loader = make_test_loader(
        metadata=metadata,
        teacher_embeddings=teacher_image_embeddings,
        indices=split_indices,
        image_root=args.image_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    student_batches: list[np.ndarray] = []
    teacher_batches: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    subject_ids: list[np.ndarray] = []
    study_ids: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate-{args.split}"):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target_embedding"].to(device, non_blocking=True)
            student = model(images)

            student_batches.append(student.cpu().numpy().astype(np.float32))
            teacher_batches.append(targets.cpu().numpy().astype(np.float32))
            row_indices.append(batch["row_index"].numpy())
            subject_ids.append(batch["subject_id"].numpy())
            study_ids.append(batch["study_id"].numpy())

    student_embeddings = np.concatenate(student_batches, axis=0)
    teacher_embeddings = np.concatenate(teacher_batches, axis=0)
    row_indices_array = np.concatenate(row_indices, axis=0)
    subject_ids_array = np.concatenate(subject_ids, axis=0)
    study_ids_array = np.concatenate(study_ids, axis=0)

    cosine = np.sum(student_embeddings * teacher_embeddings, axis=1)
    mse = np.mean((student_embeddings - teacher_embeddings) ** 2, axis=1)
    l2 = np.linalg.norm(student_embeddings - teacher_embeddings, axis=1)

    scores = pd.DataFrame(
        {
            "row_index": row_indices_array,
            "subject_id": subject_ids_array,
            "study_id": study_ids_array,
            "student_teacher_cosine": cosine,
            "student_teacher_mse": mse,
            "student_teacher_l2": l2,
        }
    )

    metrics: dict[str, Any] = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "num_rows": int(len(scores)),
        "device": str(device),
        "student_teacher_cosine": summarize(cosine),
        "student_teacher_mse": summarize(mse),
        "student_teacher_l2": summarize(l2),
    }

    if teacher_text_embeddings_path.exists() and args.retrieval_sample_size > 0:
        teacher_text_embeddings = np.load(teacher_text_embeddings_path, mmap_mode="r")[row_indices_array]
        metrics["student_image_to_teacher_text_retrieval"] = sampled_retrieval(
            student_image_embeddings=student_embeddings,
            teacher_text_embeddings=np.asarray(teacher_text_embeddings, dtype=np.float32),
            sample_size=args.retrieval_sample_size,
            seed=args.seed,
        )

    np.save(output_dir / f"student_{args.split}_embeddings.npy", student_embeddings)
    scores.to_csv(output_dir / f"student_{args.split}_scores.csv", index=False)
    (output_dir / f"student_{args.split}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=json_safe),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, default=json_safe))


if __name__ == "__main__":
    main()
