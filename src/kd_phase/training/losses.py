import torch
from torch.nn import functional as F


def image_embedding_kd_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    cosine_weight: float = 1.0,
    mse_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)
    cosine_loss = 1.0 - F.cosine_similarity(student_embeddings, teacher_embeddings, dim=1).mean()
    mse_loss = F.mse_loss(student_embeddings, teacher_embeddings)
    total = cosine_weight * cosine_loss + mse_weight * mse_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "cosine_loss": float(cosine_loss.detach().cpu()),
        "mse_loss": float(mse_loss.detach().cpu()),
        "cosine": float((1.0 - cosine_loss).detach().cpu()),
    }
    return total, metrics
