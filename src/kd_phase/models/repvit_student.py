from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _load_repvit_factory(repvit_root: str | Path):
    root = Path(repvit_root).resolve()
    if not (root / "model" / "repvit.py").exists():
        raise FileNotFoundError(f"Could not find RepViT model code under: {root}")
    sys.path.insert(0, str(root))
    from model.repvit import repvit_m1_1

    return repvit_m1_1


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported RepViT checkpoint format.")


class RepViTM11ImageStudent(nn.Module):
    """RepViT-M1.1 image student that predicts a normalized 128D teacher embedding."""

    def __init__(
        self,
        embedding_dim: int = 128,
        repvit_root: str | Path = "RepViT",
        pretrained_checkpoint: str | Path | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        repvit_m1_1 = _load_repvit_factory(repvit_root)
        self.backbone = repvit_m1_1(pretrained=False, num_classes=0, distillation=False)
        self.feature_dim = 512

        layers: list[nn.Module] = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.feature_dim, embedding_dim))
        self.projection = nn.Sequential(*layers)

        if pretrained_checkpoint:
            checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
            state_dict = _state_dict_from_checkpoint(checkpoint)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            ignored = [key for key in unexpected if key.startswith("classifier.")]
            real_unexpected = [key for key in unexpected if key not in ignored]
            if real_unexpected:
                raise RuntimeError(f"Unexpected RepViT checkpoint keys: {real_unexpected[:10]}")
            if missing:
                print(f"RepViT pretrained load: missing {len(missing)} keys, unexpected {len(unexpected)} keys.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)
