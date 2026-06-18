import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.multiview_image_kd_dataset import MultiViewImageTeacherKDDataset
from data.transforms import build_image_transform
from evaluate_image_student import sampled_retrieval, summarize
from models.mobilevit_student import MobileViTStudent
from train_image_student import json_safe


def collate_multiview(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["images"] for item in batch]),
        "counts": torch.tensor([item["count"] for item in batch], dtype=torch.long),
        "target_embedding": torch.stack([item["target_embedding"] for item in batch]),
        "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        "subject_id": torch.tensor([item["subject_id"] for item in batch], dtype=torch.long),
        "study_id": torch.tensor([item["study_id"] for item in batch], dtype=torch.long),
    }


def make_loader(
    metadata: pd.DataFrame,
    teacher_embeddings: np.ndarray,
    indices: np.ndarray,
    image_root: str | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_views: int,
) -> DataLoader:
    dataset = MultiViewImageTeacherKDDataset(
        metadata=metadata,
        teacher_image_embeddings=teacher_embeddings,
        indices=indices,
        image_root=image_root,
        transform=build_image_transform(image_size=image_size, train=False),
        max_views=max_views,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_multiview,
    )


def load_mobilevit(checkpoint_path: Path, embedding_dim: int, device: torch.device) -> tuple[MobileViTStudent, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = MobileViTStudent(teacher_dim=embedding_dim, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MobileViT-Small student against BioViL-T teacher embeddings.")
    parser.add_argument("--artifacts-dir", default=".")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--checkpoint", default="mobileVit/e10_best_student.pth.zip")
    parser.add_argument("--output-dir", default="eval/mobilevit_s_test")
    parser.add_argument("--prefix", default="biovil_t_fixed")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-views", type=int, default=3)
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
    model, checkpoint = load_mobilevit(Path(args.checkpoint), int(teacher_image_embeddings.shape[1]), device)
    loader = make_loader(
        metadata,
        teacher_image_embeddings,
        split_indices,
        args.image_root,
        args.image_size,
        args.batch_size,
        args.num_workers,
        args.max_views,
    )

    student_batches: list[np.ndarray] = []
    teacher_batches: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    subject_ids: list[np.ndarray] = []
    study_ids: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate-mobilevit-{args.split}"):
            images = batch["images"].to(device, non_blocking=True)
            counts = batch["counts"].to(device, non_blocking=True)
            targets = batch["target_embedding"].to(device, non_blocking=True)
            student = model(images, counts)

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
        "student_arch": "mobilevit_s",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_cos": float(checkpoint.get("val_cos", -1.0)),
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
