import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


DEFAULT_KAGGLE_IMAGE_ROOT = (
    "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/"
    "official_data_iccv_final/files"
)
FILES_MARKER = "official_data_iccv_final/files/"


def parse_image_paths(value: object) -> list[str]:
    """Parse metadata image_paths strings like [PosixPath('/kaggle/...jpg')]."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if pd.isna(value):
        return []

    text = str(value)
    path_matches = re.findall(r"(?:PosixPath|WindowsPath)\('([^']+)'\)", text)
    if path_matches:
        return path_matches

    quoted_matches = re.findall(r"'([^']+\.(?:jpg|jpeg|png))'", text, flags=re.IGNORECASE)
    if quoted_matches:
        return quoted_matches

    if re.search(r"\.(jpg|jpeg|png)$", text, flags=re.IGNORECASE):
        return [text]

    return []


def resolve_image_path(raw_path: str, image_root: str | Path | None = None) -> Path:
    """Map a Kaggle absolute image path to the configured image root when needed."""
    raw = str(raw_path).replace("\\", "/")
    if image_root is None:
        return Path(raw_path)

    root = Path(image_root)
    if raw.startswith(DEFAULT_KAGGLE_IMAGE_ROOT):
        rel = raw[len(DEFAULT_KAGGLE_IMAGE_ROOT):].lstrip("/")
        return root / rel

    marker_index = raw.find(FILES_MARKER)
    if marker_index >= 0:
        rel = raw[marker_index + len(FILES_MARKER):].lstrip("/")
        return root / rel

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root / path


class ImageTeacherKDDataset(Dataset):
    """Image-only student KD dataset with BioViL-T image embeddings as targets."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        teacher_image_embeddings: np.ndarray,
        indices: np.ndarray,
        image_root: str | Path | None,
        transform: Callable | None,
        view_sampling: str = "first",
    ) -> None:
        if view_sampling not in {"first", "random"}:
            raise ValueError("view_sampling must be either 'first' or 'random'.")

        self.metadata = metadata.reset_index(drop=True)
        self.teacher_image_embeddings = teacher_image_embeddings
        self.indices = np.asarray(indices, dtype=np.int64)
        self.image_root = image_root
        self.transform = transform
        self.view_sampling = view_sampling

    def __len__(self) -> int:
        return int(len(self.indices))

    def _choose_path(self, paths: list[str]) -> str:
        if not paths:
            raise FileNotFoundError("No image paths found in metadata row.")
        if self.view_sampling == "random" and len(paths) > 1:
            return paths[int(np.random.randint(0, len(paths)))]
        return paths[0]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str | int]:
        row_index = int(self.indices[item])
        row = self.metadata.iloc[row_index]
        paths = parse_image_paths(row["image_paths"])
        selected_path = resolve_image_path(self._choose_path(paths), self.image_root)

        if not selected_path.exists():
            raise FileNotFoundError(
                f"Image not found: {selected_path}. "
                "If running outside Kaggle, pass --image-root pointing to official_data_iccv_final/files."
            )

        image = Image.open(selected_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.from_numpy(np.array(self.teacher_image_embeddings[row_index], dtype=np.float32, copy=True))
        return {
            "image": image,
            "target_embedding": target,
            "row_index": row_index,
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "image_path": str(selected_path),
        }
