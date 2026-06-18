import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18ImageStudent(nn.Module):
    """ResNet-18 image student that predicts a normalized 128D teacher embedding."""

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        layers: list[nn.Module] = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, embedding_dim))
        self.projection = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)
