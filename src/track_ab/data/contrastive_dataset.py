from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable

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


def clean_report_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def extract_impression(value: object) -> str:
    text = clean_report_text(value)
    if not text:
        return ""

    match = re.search(r"\bimpression\s*:\s*(.+)$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


class ImageTextContrastiveDataset(Dataset):
    """MIMIC-CXR image/text pairs for shared latent-space contrastive training."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        indices: np.ndarray,
        image_root: str | Path | None,
        transform: Callable | None,
        max_views: int = 3,
        text_source: str = "impression",
        view_sampling: str = "random",
        skip_empty_text: bool = True,
    ) -> None:
        if text_source not in {"impression", "report"}:
            raise ValueError("text_source must be 'impression' or 'report'.")
        if view_sampling not in {"first", "random"}:
            raise ValueError("view_sampling must be 'first' or 'random'.")

        self.metadata = metadata.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.max_views = max_views
        self.text_source = text_source
        self.view_sampling = view_sampling

        candidate_indices = np.asarray(indices, dtype=np.int64)
        if skip_empty_text:
            kept: list[int] = []
            for row_index in candidate_indices:
                if self._text_for_row(self.metadata.iloc[int(row_index)]):
                    kept.append(int(row_index))
            self.indices = np.asarray(kept, dtype=np.int64)
        else:
            self.indices = candidate_indices

    def __len__(self) -> int:
        return int(len(self.indices))

    def _text_for_row(self, row: pd.Series) -> str:
        if self.text_source == "report":
            return clean_report_text(row.get("report_text", ""))
        return extract_impression(row.get("report_text", ""))

    def _choose_single_image(self, paths: list[str]) -> str:
        if not paths:
            raise FileNotFoundError("No image paths found in metadata row.")
        if self.view_sampling == "random" and len(paths) > 1:
            return paths[int(np.random.randint(0, len(paths)))]
        return paths[0]

    def _load_image(self, raw_path: str) -> tuple[torch.Tensor, str]:
        path = resolve_image_path(raw_path, self.image_root)
        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}. Pass --image-root pointing to official_data_iccv_final/files."
            )
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row_index = int(self.indices[item])
        row = self.metadata.iloc[row_index]
        raw_paths = parse_image_paths(row["image_paths"])
        selected_raw_path = self._choose_single_image(raw_paths)
        single_image, selected_path = self._load_image(selected_raw_path)

        multi_images: list[torch.Tensor] = []
        multi_paths: list[str] = []
        for raw_path in raw_paths[: self.max_views]:
            image, resolved_path = self._load_image(raw_path)
            multi_images.append(image)
            multi_paths.append(resolved_path)

        count = len(multi_images)
        while len(multi_images) < self.max_views:
            multi_images.append(torch.zeros_like(single_image))

        return {
            "image": single_image,
            "images": torch.stack(multi_images),
            "count": int(count),
            "row_index": row_index,
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "image_path": selected_path,
            "image_paths": multi_paths,
            "text": self._text_for_row(row),
        }


def collate_image_text(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "images": torch.stack([item["images"] for item in batch]),
        "counts": torch.tensor([item["count"] for item in batch], dtype=torch.long),
        "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        "subject_id": torch.tensor([item["subject_id"] for item in batch], dtype=torch.long),
        "study_id": torch.tensor([item["study_id"] for item in batch], dtype=torch.long),
        "image_path": [item["image_path"] for item in batch],
        "image_paths": [item["image_paths"] for item in batch],
        "text": [item["text"] for item in batch],
    }
