from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class MobileViTStudent(nn.Module):
    """MobileViT-Small student matching the Week 1 checkpoint architecture."""

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


class VisualPrefixAdapter(nn.Module):
    """Project image features into Granite token-embedding space."""

    def __init__(
        self,
        image_feature_dim: int,
        decoder_hidden_size: int,
        num_visual_tokens: int = 16,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.decoder_hidden_size = decoder_hidden_size
        self.net = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_visual_tokens * decoder_hidden_size),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        prefix = self.net(image_features)
        return prefix.view(image_features.shape[0], self.num_visual_tokens, self.decoder_hidden_size)


class MobileViTGranitePrefixDecoder(nn.Module):
    """Frozen MobileViT + learned visual prefix + Granite causal decoder."""

    def __init__(
        self,
        image_encoder: MobileViTStudent,
        decoder: nn.Module,
        image_feature_source: str = "kd_embedding",
        num_visual_tokens: int = 16,
        adapter_hidden_dim: int = 1024,
        adapter_dropout: float = 0.0,
        freeze_mobilevit: bool = True,
        freeze_granite: bool = True,
    ) -> None:
        super().__init__()
        if image_feature_source not in {"kd_embedding", "backbone"}:
            raise ValueError("image_feature_source must be 'kd_embedding' or 'backbone'.")

        self.image_encoder = image_encoder
        self.decoder = decoder
        self.image_feature_source = image_feature_source
        image_feature_dim = 128 if image_feature_source == "kd_embedding" else 640
        hidden_size = int(decoder.config.hidden_size)
        self.visual_adapter = VisualPrefixAdapter(
            image_feature_dim=image_feature_dim,
            decoder_hidden_size=hidden_size,
            num_visual_tokens=num_visual_tokens,
            hidden_dim=adapter_hidden_dim,
            dropout=adapter_dropout,
        )

        if freeze_mobilevit:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()

        if freeze_granite:
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.decoder.eval()

    @property
    def num_visual_tokens(self) -> int:
        return self.visual_adapter.num_visual_tokens

    def decoder_device(self) -> torch.device:
        return self.decoder.get_input_embeddings().weight.device

    def decoder_dtype(self) -> torch.dtype:
        return self.decoder.get_input_embeddings().weight.dtype

    def encode_images(self, images: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        if self.image_feature_source == "backbone":
            features = self.image_encoder.extract_backbone_features(images, counts)
        else:
            features = self.image_encoder(images, counts)
        return features

    def visual_prefix(self, images: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_images(images, counts)
        prefix = self.visual_adapter(image_features)
        return prefix.to(device=self.decoder_device(), dtype=self.decoder_dtype())

    def forward(
        self,
        images: torch.Tensor,
        counts: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        prefix = self.visual_prefix(images, counts)
        decoder_device = self.decoder_device()

        input_ids = input_ids.to(decoder_device)
        attention_mask = attention_mask.to(decoder_device)
        labels = labels.to(decoder_device)

        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, text_embeds], dim=1)

        prefix_mask = torch.ones(
            (attention_mask.shape[0], self.num_visual_tokens),
            dtype=attention_mask.dtype,
            device=decoder_device,
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        prefix_labels = torch.full(
            (labels.shape[0], self.num_visual_tokens),
            fill_value=-100,
            dtype=labels.dtype,
            device=decoder_device,
        )
        full_labels = torch.cat([prefix_labels, labels], dim=1)

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        counts: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        prefix = self.visual_prefix(images, counts)
        decoder_device = self.decoder_device()

        prompt_input_ids = prompt_input_ids.to(decoder_device)
        prompt_attention_mask = prompt_attention_mask.to(decoder_device)
        prompt_embeds = self.decoder.get_input_embeddings()(prompt_input_ids)
        inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)

        prefix_mask = torch.ones(
            (prompt_attention_mask.shape[0], self.num_visual_tokens),
            dtype=prompt_attention_mask.dtype,
            device=decoder_device,
        )
        attention_mask = torch.cat([prefix_mask, prompt_attention_mask], dim=1)

        return self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )


def load_mobilevit_checkpoint(checkpoint_path: str, teacher_dim: int = 128) -> MobileViTStudent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model = MobileViTStudent(teacher_dim=teacher_dim, pretrained=False)
    model.load_state_dict(state_dict)
    return model
