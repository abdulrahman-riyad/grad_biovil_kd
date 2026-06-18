import argparse
import json
import random
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
from training.losses import image_embedding_kd_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target_embedding": torch.stack([item["target_embedding"] for item in batch]),
        "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        "subject_id": torch.tensor([item["subject_id"] for item in batch], dtype=torch.long),
        "study_id": torch.tensor([item["study_id"] for item in batch], dtype=torch.long),
        "image_path": [item["image_path"] for item in batch],
    }


def make_loader(
    metadata: pd.DataFrame,
    teacher_embeddings: np.ndarray,
    indices: np.ndarray,
    image_root: str | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    view_sampling: str,
) -> DataLoader:
    dataset = ImageTeacherKDDataset(
        metadata=metadata,
        teacher_image_embeddings=teacher_embeddings,
        indices=indices,
        image_root=image_root,
        transform=build_image_transform(image_size=image_size, train=train),
        view_sampling=view_sampling,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        collate_fn=collate_batch,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cosine_weight: float,
    mse_weight: float,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "cosine_loss": 0.0, "mse_loss": 0.0, "cosine": 0.0}
    steps = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress = tqdm(loader, desc="train" if is_train else "val")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target_embedding"].to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            student_embeddings = model(images)
            loss, metrics = image_embedding_kd_loss(
                student_embeddings,
                targets,
                cosine_weight=cosine_weight,
                mse_weight=mse_weight,
            )

            if is_train:
                loss.backward()
                optimizer.step()

            for key in totals:
                totals[key] += metrics[key]
            steps += 1
            progress.set_postfix(loss=totals["loss"] / steps, cosine=totals["cosine"] / steps)

            if max_batches is not None and steps >= max_batches:
                break

    if steps == 0:
        raise RuntimeError("No batches were processed.")
    return {key: value / steps for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-18 image student using BioViL-T image embeddings.")
    parser.add_argument("--artifacts-dir", default="weeks output/week1")
    parser.add_argument("--splits-dir", default="kd_phase/splits")
    parser.add_argument("--output-dir", default="kd_phase/runs/resnet18_image_kd")
    parser.add_argument("--prefix", default="biovil_t_fixed")
    parser.add_argument("--image-root", default=None, help="Path to official_data_iccv_final/files. Optional on Kaggle.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-view-sampling", choices=["first", "random"], default="random")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug limit only.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Debug limit only.")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.artifacts_dir)
    splits_dir = Path(args.splits_dir)
    metadata = pd.read_csv(artifacts_dir / f"{args.prefix}_metadata.csv")
    teacher_embeddings = np.load(artifacts_dir / f"{args.prefix}_image_embeddings.npy", mmap_mode="r")
    train_indices = np.load(splits_dir / "kd_train_indices.npy")
    val_indices = np.load(splits_dir / "kd_val_indices.npy")

    if len(metadata) != len(teacher_embeddings):
        raise ValueError("Metadata and teacher embedding row counts do not match.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(
        metadata=metadata,
        teacher_embeddings=teacher_embeddings,
        indices=train_indices,
        image_root=args.image_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
        view_sampling=args.train_view_sampling,
    )
    val_loader = make_loader(
        metadata=metadata,
        teacher_embeddings=teacher_embeddings,
        indices=val_indices,
        image_root=args.image_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
        view_sampling="first",
    )

    model = ResNet18ImageStudent(
        embedding_dim=teacher_embeddings.shape[1],
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_config = vars(args) | {
        "device": str(device),
        "num_train_rows": int(len(train_indices)),
        "num_val_rows": int(len(val_indices)),
        "embedding_dim": int(teacher_embeddings.shape[1]),
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=json_safe), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
            max_batches=args.max_val_batches,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2, default=json_safe), encoding="utf-8")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": run_config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, output_dir / "best.pt")

        print(json.dumps(record, indent=2, default=json_safe))


if __name__ == "__main__":
    main()
