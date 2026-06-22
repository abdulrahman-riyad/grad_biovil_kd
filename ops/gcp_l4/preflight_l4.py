from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_stage2_hard_negatives_l4 import RUNS, common_paths, image_root_from_args


FILES_MARKER = "official_data_iccv_final/files/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight validation for the GCP L4 structured project.")
    parser.add_argument("--project-root", default=os.environ.get("GRAD_BIOVIL_ROOT"))
    parser.add_argument("--mimic-root", default=os.environ.get("MIMIC_CXR_ROOT"))
    parser.add_argument("--image-root", default=os.environ.get("MIMIC_CXR_IMAGE_ROOT"))
    parser.add_argument("--work-root", default=os.environ.get("GRAD_BIOVIL_WORK", str(Path.home() / "grad-biovil-runs")))
    parser.add_argument("--sample-images", type=int, default=25)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def status(exists: bool, detail: str | None = None) -> dict[str, Any]:
    return {"ok": bool(exists), "detail": detail}


def resolve_image_path(raw_path: str, image_root: Path) -> Path:
    raw = str(raw_path).replace("\\", "/")
    marker_index = raw.find(FILES_MARKER)
    if marker_index >= 0:
        return image_root / raw[marker_index + len(FILES_MARKER) :].lstrip("/")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return image_root / path


def parse_image_paths(value: object) -> list[str]:
    import re

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


def check_cuda() -> dict[str, Any]:
    try:
        import torch
        from packaging.version import Version

        info: dict[str, Any] = {
            "torch_version": torch.__version__,
            "torch_ge_2_6": Version(torch.__version__.split("+", 1)[0]) >= Version("2.6.0"),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "vram_gb": round(props.total_memory / 1024**3, 2),
                }
            )
        return info
    except Exception as exc:
        return {"cuda_available": False, "error": repr(exc)}


def main() -> None:
    args = parse_args()
    if not args.project_root:
        args.project_root = str(Path(__file__).resolve().parents[3])

    paths = common_paths(args)
    image_root = image_root_from_args(args)
    retrieval_distillation_dir = paths["retrieval_distillation_dir"]
    if str(retrieval_distillation_dir) not in sys.path:
        sys.path.insert(0, str(retrieval_distillation_dir))

    checks: dict[str, Any] = {
        "project_root": status(paths["project_root"].exists(), str(paths["project_root"])),
        "work_root": status(True, str(paths["work_root"])),
        "image_root": status(image_root.exists(), str(image_root)),
        "cuda": check_cuda(),
        "paths": {},
        "checkpoint_initializers": {},
        "teacher_artifacts": {},
        "splits": {},
        "metadata": {},
        "sample_images": {},
        "imports": {},
    }

    required_paths = {
        "retrieval_distillation_dir": paths["retrieval_distillation_dir"],
        "artifacts_dir": paths["artifacts_dir"],
        "splits_dir": paths["splits_dir"],
        "mobilevit_checkpoint": paths["mobilevit_checkpoint"],
        "repvit_checkpoint": paths["repvit_checkpoint"],
        "repvit_root": paths["repvit_root"],
        "repvit_model_py": paths["repvit_root"] / "model" / "repvit.py",
    }
    for name, path in required_paths.items():
        checks["paths"][name] = status(path.exists(), str(path))

    for key, config in RUNS.items():
        candidates = [paths["project_root"] / relative for relative in config.stage1_checkpoint_candidates]
        existing = [path for path in candidates if path.exists()]
        checks["checkpoint_initializers"][key] = {
            "ok": bool(existing),
            "existing": [str(path) for path in existing],
            "candidates": [str(path) for path in candidates],
        }

    artifacts_dir = paths["artifacts_dir"]
    teacher_image_path = artifacts_dir / "teacher" / "biovil_t_fixed_image_embeddings.npy"
    teacher_text_path = artifacts_dir / "teacher" / "biovil_t_fixed_text_embeddings.npy"
    metadata_path = artifacts_dir / "metadata" / "biovil_t_fixed_metadata.csv"
    checks["teacher_artifacts"]["image_embeddings"] = status(teacher_image_path.exists(), str(teacher_image_path))
    checks["teacher_artifacts"]["text_embeddings"] = status(teacher_text_path.exists(), str(teacher_text_path))
    checks["metadata"]["metadata_csv"] = status(metadata_path.exists(), str(metadata_path))

    if teacher_image_path.exists() and teacher_text_path.exists():
        image_embeddings = np.load(teacher_image_path, mmap_mode="r")
        text_embeddings = np.load(teacher_text_path, mmap_mode="r")
        checks["teacher_artifacts"]["image_shape"] = tuple(int(v) for v in image_embeddings.shape)
        checks["teacher_artifacts"]["text_shape"] = tuple(int(v) for v in text_embeddings.shape)
        checks["teacher_artifacts"]["same_shape"] = image_embeddings.shape == text_embeddings.shape

    for split in ["train", "val", "test"]:
        split_path = paths["splits_dir"] / f"kd_{split}_indices.npy"
        if split_path.exists():
            values = np.load(split_path, mmap_mode="r")
            checks["splits"][split] = {
                "ok": True,
                "path": str(split_path),
                "count": int(values.shape[0]),
            }
        else:
            checks["splits"][split] = {"ok": False, "path": str(split_path)}

    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        checks["metadata"]["rows"] = int(len(metadata))
        checks["metadata"]["columns"] = sorted(metadata.columns.tolist())
        image_checks: list[dict[str, Any]] = []
        for row in metadata.head(max(args.sample_images, 0)).itertuples(index=False):
            raw_paths = parse_image_paths(getattr(row, "image_paths", ""))
            if not raw_paths:
                image_checks.append({"ok": False, "reason": "no image_paths"})
                continue
            resolved = resolve_image_path(raw_paths[0], image_root)
            image_checks.append({"ok": resolved.exists(), "path": str(resolved)})
        checks["sample_images"]["checked"] = len(image_checks)
        checks["sample_images"]["missing"] = [item for item in image_checks if not item["ok"]]

    for module in [
        "data.image_text_dataset",
        "models.image_text_retrieval_model",
        "models.student_loaders",
        "models.text_encoders",
    ]:
        try:
            __import__(module)
            checks["imports"][module] = status(True)
        except Exception as exc:
            checks["imports"][module] = status(False, repr(exc))

    failures: list[str] = []

    def collect_failures(prefix: str, payload: Any) -> None:
        if isinstance(payload, dict):
            if payload.get("ok") is False:
                failures.append(prefix)
            for key, value in payload.items():
                collect_failures(f"{prefix}.{key}" if prefix else str(key), value)
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                collect_failures(f"{prefix}[{index}]", value)

    collect_failures("", checks)
    if not checks["cuda"].get("torch_ge_2_6"):
        failures.append("cuda.torch_ge_2_6")
    checks["overall_ok"] = (
        len(failures) == 0
        and bool(checks["cuda"].get("cuda_available"))
        and bool(checks["cuda"].get("torch_ge_2_6"))
    )
    checks["failures"] = failures

    output = json.dumps(checks, indent=2)
    print(output)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    if not checks["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
