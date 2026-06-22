import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.repvit_student import RepViTM11ImageStudent
from train_image_student import json_safe, make_loader, run_epoch, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RepViT-M1.1 image student using BioViL-T image embeddings.")
    parser.add_argument("--artifacts-dir", default=".")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--output-dir", default="runs/repvit_m1_1_image_kd")
    parser.add_argument("--prefix", default="biovil_t_fixed")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--repvit-root", default="RepViT")
    parser.add_argument("--pretrained-checkpoint", default="RepViT/repvit_m1_1_distill_450e.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-view-sampling", choices=["first", "random"], default="random")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
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
        metadata,
        teacher_embeddings,
        train_indices,
        args.image_root,
        args.image_size,
        args.batch_size,
        args.num_workers,
        train=True,
        view_sampling=args.train_view_sampling,
    )
    val_loader = make_loader(
        metadata,
        teacher_embeddings,
        val_indices,
        args.image_root,
        args.image_size,
        args.batch_size,
        args.num_workers,
        train=False,
        view_sampling="first",
    )

    model = RepViTM11ImageStudent(
        embedding_dim=int(teacher_embeddings.shape[1]),
        repvit_root=args.repvit_root,
        pretrained_checkpoint=args.pretrained_checkpoint,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_config = vars(args) | {
        "student_arch": "repvit_m1_1",
        "device": str(device),
        "num_train_rows": int(len(train_indices)),
        "num_val_rows": int(len(val_indices)),
        "embedding_dim": int(teacher_embeddings.shape[1]),
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=json_safe), encoding="utf-8")

    history = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.cosine_weight, args.mse_weight, args.max_train_batches)
        val_metrics = run_epoch(model, val_loader, None, device, args.cosine_weight, args.mse_weight, args.max_val_batches)
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
