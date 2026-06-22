from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from data.image_text_dataset import ImageTextContrastiveDataset
from data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_image_transform
from evaluate_student_retrieval import build_model
from models.student_loaders import torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for contrastive retrieval cases.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases-csv", default=None, help="CSV from analyze_retrieval_examples.py.")
    parser.add_argument("--row-indices", default=None, help="Comma-separated metadata row indices.")
    parser.add_argument("--mobilevit-checkpoint", default=None)
    parser.add_argument("--repvit-checkpoint", default=None)
    parser.add_argument("--repvit-root", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--text-source", choices=["impression", "report"], default=None)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class GradCAM:
    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = layer.register_forward_hook(self._forward_hook)
        self._backward_handle = layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def _forward_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        self.activations = output

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0]

    def cam(self, activation_index: int = 0) -> torch.Tensor:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        acts = self.activations[activation_index]
        grads = self.gradients[activation_index]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0))
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-6)
        return cam.detach().cpu()


def find_last_conv2d(module: nn.Module) -> tuple[str, nn.Conv2d]:
    last_name = None
    last_layer = None
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            last_name = name
            last_layer = child
    if last_layer is None or last_name is None:
        raise ValueError("Could not find a Conv2d layer for Grad-CAM.")
    return last_name, last_layer


def row_indices_from_args(args: argparse.Namespace) -> list[int]:
    rows: list[int] = []
    if args.row_indices:
        rows.extend(int(item.strip()) for item in args.row_indices.split(",") if item.strip())
    if args.cases_csv:
        cases = pd.read_csv(args.cases_csv)
        if "query_row_index" not in cases.columns:
            raise ValueError("--cases-csv must contain a query_row_index column.")
        rows.extend(int(value) for value in cases["query_row_index"].head(args.max_cases).tolist())
    deduped: list[int] = []
    seen: set[int] = set()
    for row in rows:
        if row not in seen:
            deduped.append(row)
            seen.add(row)
    return deduped[: args.max_cases]


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image.cpu() * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def save_overlay(image: np.ndarray, cam: torch.Tensor, path: Path) -> None:
    import matplotlib.pyplot as plt

    cam_resized = torch.nn.functional.interpolate(
        cam[None, None],
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.imshow(cam_resized, cmap="jet", alpha=0.42)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close()


def make_single_case_dataset(
    metadata: pd.DataFrame,
    row_index: int,
    image_root: str,
    image_size: int,
    max_views: int,
    text_source: str,
) -> ImageTextContrastiveDataset:
    return ImageTextContrastiveDataset(
        metadata=metadata,
        indices=np.asarray([row_index], dtype=np.int64),
        image_root=image_root,
        transform=build_image_transform(image_size=image_size, train=False),
        max_views=max_views,
        text_source=text_source,
        view_sampling="first",
        skip_empty_text=False,
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch_load(args.checkpoint)
    config = checkpoint["config"]
    image_size = int(args.image_size or config.get("image_size", 224))
    max_views = int(args.max_views or config.get("max_views", 3))
    text_source = str(args.text_source or config.get("text_source", "impression"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata_csv)
    row_indices = row_indices_from_args(args)
    if not row_indices:
        raise ValueError("No row indices provided. Use --row-indices or --cases-csv.")

    device = torch.device(args.device)
    model = build_model(checkpoint, args).to(device)
    model.eval()
    layer_name, target_layer = find_last_conv2d(model.image_encoder)
    gradcam = GradCAM(target_layer)

    manifest: list[dict[str, Any]] = []
    try:
        for row_index in row_indices:
            dataset = make_single_case_dataset(metadata, row_index, args.image_root, image_size, max_views, text_source)
            item = dataset[0]
            batch: dict[str, Any] = {
                "image": item["image"].unsqueeze(0).to(device),
                "images": item["images"].unsqueeze(0).to(device),
                "counts": torch.tensor([item["count"]], dtype=torch.long, device=device),
                "text": [item["text"]],
            }

            model.zero_grad(set_to_none=True)
            image_embedding = model.encode_images(batch)
            text_embedding = model.encode_texts(batch["text"])
            score = (image_embedding * text_embedding).sum()
            score.backward()

            cam = gradcam.cam(activation_index=0)
            image_np = denormalize(item["image"])
            out_path = output_dir / f"row_{row_index}_gradcam.png"
            save_overlay(image_np, cam, out_path)

            manifest.append(
                {
                    "row_index": int(row_index),
                    "study_id": int(item["study_id"]),
                    "subject_id": int(item["subject_id"]),
                    "score": float(score.detach().cpu()),
                    "target_layer": layer_name,
                    "output": str(out_path.name),
                    "text": " ".join(str(item["text"]).split())[:700],
                    "image_path": item["image_path"],
                }
            )
    finally:
        gradcam.close()

    (output_dir / "gradcam_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"target_layer": layer_name, "num_cases": len(manifest), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
