from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from models.student_loaders import torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a deployable Week 3 contrastive checkpoint package.")
    parser.add_argument("--contrastive-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retrieval-metrics", default=None)
    parser.add_argument("--image-checkpoint", default=None)
    parser.add_argument("--repvit-root", default=None)
    parser.add_argument("--include-full-state", action="store_true")
    return parser.parse_args()


def filter_projection_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keep_prefixes = ("image_projection.", "text_projection.", "logit_scale")
    return {key: value for key, value in state_dict.items() if key.startswith(keep_prefixes)}


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    checkpoint = torch_load(args.contrastive_checkpoint)
    config = dict(checkpoint["config"])
    state_dict = checkpoint["model_state_dict"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projection_package = {
        "epoch": checkpoint.get("epoch"),
        "config": config,
        "projection_state_dict": filter_projection_state(state_dict),
    }
    torch.save(projection_package, output_dir / "contrastive_projection_heads.pt")

    if args.include_full_state:
        torch.save(
            {
                "epoch": checkpoint.get("epoch"),
                "config": config,
                "model_state_dict": state_dict,
            },
            output_dir / "contrastive_full_model_state.pt",
        )

    manifest = {
        "export_type": "week3_contrastive_retrieval_package",
        "source_checkpoint": args.contrastive_checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "image_student": config.get("image_student"),
        "text_encoder": config.get("text_encoder"),
        "text_model_id": config.get("text_model_id"),
        "projection_dim": config.get("projection_dim"),
        "text_source": config.get("text_source"),
        "image_size": config.get("image_size"),
        "max_views": config.get("max_views"),
        "max_text_length": config.get("max_text_length"),
        "image_checkpoint": args.image_checkpoint or config.get("mobilevit_checkpoint") or config.get("repvit_checkpoint"),
        "repvit_root": args.repvit_root or config.get("repvit_root"),
        "files": {
            "projection_heads": "contrastive_projection_heads.pt",
            "full_state": "contrastive_full_model_state.pt" if args.include_full_state else None,
            "retrieval_metrics": "retrieval_metrics.json" if args.retrieval_metrics else None,
        },
    }

    if args.retrieval_metrics:
        shutil.copy2(args.retrieval_metrics, output_dir / "retrieval_metrics.json")
        metrics = json.loads(Path(args.retrieval_metrics).read_text(encoding="utf-8"))
        manifest["retrieval_summary"] = {
            "split": metrics.get("split"),
            "candidate_pool_size": metrics.get("candidate_pool_size"),
            "image_to_text": metrics.get("image_to_text"),
            "text_to_image": metrics.get("text_to_image"),
        }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_safe), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Week 3 Contrastive Retrieval Export",
                "",
                "This folder packages the selected contrastive projection heads and deployment metadata.",
                "",
                "Files:",
                "",
                "- `contrastive_projection_heads.pt`: image/text projection heads and learned logit scale.",
                "- `manifest.json`: model IDs, preprocessing settings, checkpoint references, and retrieval metadata.",
                "- `retrieval_metrics.json`: copied evaluation metrics when provided.",
                "",
                "The frozen image student checkpoint and Hugging Face text encoder are referenced in `manifest.json`.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, default=json_safe))


if __name__ == "__main__":
    main()
