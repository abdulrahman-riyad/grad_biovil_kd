import torch
from torch import nn
from torch.nn import functional as F


class MobileViTStudent(nn.Module):
    """MobileViT-Small student matching the image-distillation checkpoint architecture."""

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

    def forward(self, images: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        batch_size, max_views, channels, height, width = images.shape
        flat_images = images.view(-1, channels, height, width)
        features = self.backbone(flat_images)
        per_view_embeddings = self.mapper(features).view(batch_size, max_views, -1)

        mask = torch.arange(max_views, device=images.device).expand(batch_size, max_views)
        mask = (mask < counts.unsqueeze(1)).float().unsqueeze(-1)
        summed = torch.sum(per_view_embeddings * mask, dim=1)
        averaged = summed / counts.clamp_min(1).view(-1, 1).float()
        return F.normalize(averaged, p=2, dim=1)
