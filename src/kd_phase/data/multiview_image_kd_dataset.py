from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from data.image_kd_dataset import parse_image_paths, resolve_image_path


class MultiViewImageTeacherKDDataset(Dataset):
    """KD dataset that returns up to max_views images per study for late-fusion students."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        teacher_image_embeddings: np.ndarray,
        indices: np.ndarray,
        image_root: str | Path | None,
        transform: Callable | None,
        max_views: int = 3,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.teacher_image_embeddings = teacher_image_embeddings
        self.indices = np.asarray(indices, dtype=np.int64)
        self.image_root = image_root
        self.transform = transform
        self.max_views = max_views

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | int | list[str]]:
        row_index = int(self.indices[item])
        row = self.metadata.iloc[row_index]
        raw_paths = parse_image_paths(row["image_paths"])[: self.max_views]
        if not raw_paths:
            raise FileNotFoundError("No image paths found in metadata row.")

        images: list[torch.Tensor] = []
        resolved_paths: list[str] = []
        for raw_path in raw_paths:
            image_path = resolve_image_path(raw_path, self.image_root)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
            resolved_paths.append(str(image_path))

        count = len(images)
        while len(images) < self.max_views:
            images.append(torch.zeros_like(images[0]))

        target = torch.from_numpy(np.array(self.teacher_image_embeddings[row_index], dtype=np.float32, copy=True))
        return {
            "images": torch.stack(images),
            "count": int(count),
            "target_embedding": target,
            "row_index": row_index,
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "image_paths": resolved_paths,
        }
