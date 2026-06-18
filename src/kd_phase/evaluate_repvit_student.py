import argparse
from pathlib import Path
from typing import Any

import torch

from evaluate_image_student import main as evaluate_main
from models.repvit_student import RepViTM11ImageStudent


def load_student(checkpoint_path: Path, embedding_dim: int, device: torch.device) -> tuple[RepViTM11ImageStudent, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = RepViTM11ImageStudent(
        embedding_dim=embedding_dim,
        repvit_root=config.get("repvit_root", "RepViT"),
        pretrained_checkpoint=None,
        dropout=float(config.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    import evaluate_image_student

    evaluate_image_student.load_student = load_student
    evaluate_main()
