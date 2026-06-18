from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), p=2, dim=1)


class ImageTextContrastiveModel(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        image_arch: str,
        image_feature_dim: int,
        text_encoder: nn.Module,
        text_feature_dim: int,
        projection_dim: int = 256,
        projection_hidden_dim: int | None = None,
        projection_dropout: float = 0.0,
        freeze_image_encoder: bool = True,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        if image_arch not in {"mobilevit", "repvit"}:
            raise ValueError("image_arch must be 'mobilevit' or 'repvit'.")

        self.image_encoder = image_encoder
        self.image_arch = image_arch
        self.text_encoder = text_encoder
        self.image_projection = ProjectionHead(
            image_feature_dim,
            output_dim=projection_dim,
            hidden_dim=projection_hidden_dim,
            dropout=projection_dropout,
        )
        self.text_projection = ProjectionHead(
            text_feature_dim,
            output_dim=projection_dim,
            hidden_dim=projection_hidden_dim,
            dropout=projection_dropout,
        )
        self.image_log_variance = nn.Linear(projection_dim, 1)
        self.text_log_variance = nn.Linear(projection_dim, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()

    def encode_image_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.image_arch == "mobilevit":
            return self.image_encoder(batch["images"], batch["counts"])
        return self.image_encoder(batch["image"])

    def encode_text_features(self, texts: list[str]) -> torch.Tensor:
        return self.text_encoder(texts)

    def encode_images(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.encode_image_features(batch)
        return self.image_projection(features)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        features = self.encode_text_features(texts)
        return self.text_projection(features)

    def forward(self, batch: dict[str, torch.Tensor | list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.encode_images(batch)  # type: ignore[arg-type]
        text_embeddings = self.encode_texts(batch["text"])  # type: ignore[arg-type]
        return image_embeddings, text_embeddings

    def embedding_log_variances(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_logvar = self.image_log_variance(image_embeddings).squeeze(1).clamp(min=-5.0, max=5.0)
        text_logvar = self.text_log_variance(text_embeddings).squeeze(1).clamp(min=-5.0, max=5.0)
        return image_logvar, text_logvar
