from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def symmetric_info_nce(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * image_embeddings @ text_embeddings.T
    labels = torch.arange(logits.shape[0], device=logits.device)

    image_to_text_loss = F.cross_entropy(logits, labels)
    text_to_image_loss = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (image_to_text_loss + text_to_image_loss)

    metrics = contrastive_retrieval_metrics(logits)
    metrics.update(
        {
            "loss": float(loss.detach().cpu()),
            "image_to_text_loss": float(image_to_text_loss.detach().cpu()),
            "text_to_image_loss": float(text_to_image_loss.detach().cpu()),
            "logit_scale": float(scale.detach().cpu()),
        }
    )
    return loss, metrics


def contrastive_retrieval_metrics(logits: torch.Tensor) -> dict[str, float]:
    labels = torch.arange(logits.shape[0], device=logits.device)
    image_ranks = _target_ranks(logits, labels)
    text_ranks = _target_ranks(logits.T, labels)

    return {
        "image_to_text_r1": _recall_at_k(image_ranks, 1),
        "image_to_text_r5": _recall_at_k(image_ranks, 5),
        "text_to_image_r1": _recall_at_k(text_ranks, 1),
        "text_to_image_r5": _recall_at_k(text_ranks, 5),
    }


def _target_ranks(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    order = logits.argsort(dim=1, descending=True)
    matches = order.eq(labels[:, None])
    return matches.float().argmax(dim=1) + 1


def _recall_at_k(ranks: torch.Tensor, k: int) -> float:
    capped_k = min(k, int(ranks.numel()))
    return float((ranks <= capped_k).float().mean().detach().cpu())


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value
