from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.student_loaders import load_mobilevit_student, load_repvit_student


class MobileViTBenchmarkWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        counts = torch.full(
            (images.shape[0],),
            fill_value=images.shape[1],
            dtype=torch.long,
            device=images.device,
        )
        return self.model(images, counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Week 3 MobileViT and RepViT students for Track B efficiency reporting."
    )
    parser.add_argument(
        "--mobilevit-checkpoint",
        default="week1/weeks output/week1/student_mobilevit/mobilevit_s_biovil_kd_checkpoint.pt",
        help="Path to the exported MobileViT KD checkpoint.",
    )
    parser.add_argument(
        "--repvit-checkpoint",
        default="week1/image_encoder_distillation/RepViT/training_output/best.pt",
        help="Path to the trained RepViT-M1.1 KD checkpoint.",
    )
    parser.add_argument(
        "--repvit-root",
        default="week1/image_encoder_distillation/RepViT",
        help="Path to the cloned RepViT repository folder containing model/repvit.py.",
    )
    parser.add_argument("--teacher-dim", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mobilevit-max-views", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--timed-iters", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--output-json", default="week3/efficiency_metrics.json")
    parser.add_argument("--output-csv", default="week3/efficiency_metrics.csv")
    return parser.parse_args()


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def file_size_mb(path: str | Path) -> float | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size / (1024 * 1024)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
) -> dict[str, float]:
    model.eval()
    input_tensor = input_tensor.to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(input_tensor)
        synchronize(device)

        start = time.perf_counter()
        for _ in range(timed_iters):
            _ = model(input_tensor)
        synchronize(device)
        elapsed = time.perf_counter() - start

    latency_ms = (elapsed / timed_iters) * 1000.0
    throughput = input_tensor.shape[0] / (latency_ms / 1000.0)
    result = {
        "latency_ms_per_forward": latency_ms,
        "throughput_samples_per_sec": throughput,
    }
    if device.type == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return result


def measure_flops(model: nn.Module, input_tensor: torch.Tensor) -> dict[str, Any]:
    model.eval()
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, input_tensor).total()
        return {
            "tool": "fvcore",
            "flops": int(flops),
            "gflops": float(flops / 1e9),
            "error": None,
        }
    except Exception as fvcore_error:
        try:
            from thop import profile

            macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
            return {
                "tool": "thop",
                "macs": int(macs),
                "gmacs": float(macs / 1e9),
                "error": None,
            }
        except Exception as thop_error:
            return {
                "tool": None,
                "flops": None,
                "gflops": None,
                "error": f"fvcore failed: {fvcore_error}; thop failed: {thop_error}",
            }


def benchmark_entry(
    name: str,
    model: nn.Module,
    input_shape: tuple[int, ...],
    checkpoint_path: str | Path,
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
    skip_flops: bool,
) -> dict[str, Any]:
    model = model.to(device)
    dummy = torch.randn(*input_shape)
    metrics: dict[str, Any] = {
        "model": name,
        "input_shape": list(input_shape),
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_mb": file_size_mb(checkpoint_path),
        "parameters": count_parameters(model),
    }

    if skip_flops:
        metrics["compute"] = {"tool": None, "flops": None, "gflops": None, "error": "Skipped by flag."}
    else:
        metrics["compute"] = measure_flops(model.cpu(), dummy)
        model = model.to(device)

    metrics["latency"] = measure_latency(
        model=model,
        input_tensor=dummy,
        device=device,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
    )
    return metrics


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: str | Path, entries: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "input_shape",
        "checkpoint_size_mb",
        "total_parameters",
        "trainable_parameters",
        "compute_tool",
        "gflops",
        "gmacs",
        "latency_ms_per_forward",
        "throughput_samples_per_sec",
        "peak_memory_mb",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            row = {
                "model": entry["model"],
                "input_shape": json.dumps(entry["input_shape"]),
                "checkpoint_size_mb": entry["checkpoint_size_mb"],
                "total_parameters": entry["parameters"]["total"],
                "trainable_parameters": entry["parameters"]["trainable"],
                "compute_tool": entry["compute"].get("tool"),
                "gflops": entry["compute"].get("gflops"),
                "gmacs": entry["compute"].get("gmacs"),
                "latency_ms_per_forward": entry["latency"].get("latency_ms_per_forward"),
                "throughput_samples_per_sec": entry["latency"].get("throughput_samples_per_sec"),
                "peak_memory_mb": entry["latency"].get("peak_memory_mb"),
            }
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    image_size = args.image_size

    mobilevit = MobileViTBenchmarkWrapper(
        load_mobilevit_student(args.mobilevit_checkpoint, teacher_dim=args.teacher_dim)
    )
    repvit = load_repvit_student(
        args.repvit_checkpoint,
        repvit_root=args.repvit_root,
        teacher_dim=args.teacher_dim,
    )

    entries = [
        benchmark_entry(
            name="mobilevit_s_kd_single_view",
            model=mobilevit,
            input_shape=(args.batch_size, 1, 3, image_size, image_size),
            checkpoint_path=args.mobilevit_checkpoint,
            device=device,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            skip_flops=args.skip_flops,
        ),
        benchmark_entry(
            name=f"mobilevit_s_kd_{args.mobilevit_max_views}_view_study",
            model=mobilevit,
            input_shape=(args.batch_size, args.mobilevit_max_views, 3, image_size, image_size),
            checkpoint_path=args.mobilevit_checkpoint,
            device=device,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            skip_flops=args.skip_flops,
        ),
        benchmark_entry(
            name="repvit_m1_1_kd_single_image",
            model=repvit,
            input_shape=(args.batch_size, 3, image_size, image_size),
            checkpoint_path=args.repvit_checkpoint,
            device=device,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            skip_flops=args.skip_flops,
        ),
    ]

    payload = {
        "device": str(device),
        "config": vars(args),
        "entries": entries,
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, entries)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
