from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def torch_load(path: str | Path) -> Any:
    """Load trusted local checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format. Expected a state_dict or model_state_dict.")


class MobileViTStudent(nn.Module):
    """MobileViT-Small student matching the Week 1 KD checkpoint architecture."""

    def __init__(self, teacher_dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model("mobilevit_s", pretrained=pretrained, num_classes=0)
        self.mapper = nn.Sequential(
            nn.Linear(640, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, teacher_dim),
        )

    def extract_backbone_features(self, images: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        batch_size, max_views, channels, height, width = images.shape
        flat_images = images.view(-1, channels, height, width)
        per_view_features = self.backbone(flat_images).view(batch_size, max_views, -1)

        mask = torch.arange(max_views, device=images.device).expand(batch_size, max_views)
        mask = (mask < counts.unsqueeze(1)).float().unsqueeze(-1)
        summed = torch.sum(per_view_features * mask, dim=1)
        return summed / counts.clamp_min(1).view(-1, 1).float()

    def forward(self, images: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        features = self.extract_backbone_features(images, counts)
        return F.normalize(self.mapper(features), p=2, dim=1)


def load_mobilevit_student(checkpoint_path: str | Path, teacher_dim: int = 128) -> MobileViTStudent:
    checkpoint = torch_load(checkpoint_path)
    model = MobileViTStudent(teacher_dim=teacher_dim, pretrained=False)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint))
    return model


def _load_repvit_factory(repvit_root: str | Path):
    root = Path(repvit_root).resolve()
    if not (root / "model" / "repvit.py").exists():
        raise FileNotFoundError(f"Could not find RepViT model code under: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from model.repvit import repvit_m1_1

    return repvit_m1_1


class RepViTM11ImageStudent(nn.Module):
    """RepViT-M1.1 student matching the Week 1 KD checkpoint architecture."""

    def __init__(
        self,
        embedding_dim: int = 128,
        repvit_root: str | Path = "RepViT",
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)


def load_repvit_student(
    checkpoint_path: str | Path,
    repvit_root: str | Path,
    teacher_dim: int = 128,
) -> RepViTM11ImageStudent:
    checkpoint = torch_load(checkpoint_path)
    model = RepViTM11ImageStudent(embedding_dim=teacher_dim, repvit_root=repvit_root)
    model.load_state_dict(state_dict_from_checkpoint(checkpoint))
    return model
