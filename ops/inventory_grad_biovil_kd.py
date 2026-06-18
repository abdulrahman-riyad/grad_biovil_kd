#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = "/kaggle/input/datasets/abdulrahmanriyad/grad-biovil-kd"
DEFAULT_OUTPUT_DIR = "/kaggle/working/grad_biovil_kd_inventory"


@dataclass
class FileRecord:
    rel_path: str
    abs_path: str
    parent: str
    name: str
    extension: str
    size_bytes: int
    size_mb: float
    category: str
    sha256: str
    hash_status: str
    mtime_utc: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory the grad-biovil-kd Kaggle dataset before restructuring it for GCP/GCS. "
            "Outputs a full file tree, MANIFEST.csv, summary JSON, hashed-file list, and huge-file list."
        )
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Dataset root to inventory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where inventory files are written.")
    parser.add_argument(
        "--hash-threshold-mb",
        type=float,
        default=512.0,
        help="Files up to this size are SHA256 hashed. Larger files are marked skip_huge.",
    )
    parser.add_argument("--hash-chunk-mb", type=float, default=8.0, help="Read chunk size for SHA256 hashing.")
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=None,
        help="Optional max depth for dataset_tree_limited.txt. Full tree is always written separately.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise an error if --root does not exist.",
    )
    return parser.parse_args()


def utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def normalize_extension(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else "[none]"


def infer_category(rel_path: str, size_bytes: int) -> str:
    p = rel_path.replace("\\", "/").lower()
    name = Path(p).name
    ext = Path(p).suffix.lower()

    if "/track_ab/" in p or p.startswith("track_ab/"):
        return "project_repo_track_ab"
    if "/week2" in p or p.startswith("week2"):
        return "project_repo_week2"
    if ext in {".py", ".ipynb", ".md", ".tex", ".sh", ".yaml", ".yml", ".toml"}:
        return "project_repo_or_docs"
    if "/splits/" in p or name.startswith("kd_train") or name.startswith("kd_val") or name.startswith("kd_test"):
        return "data_artifact_splits"
    if "biovil_t_fixed" in name and ext in {".npy", ".csv", ".json", ".pkl"}:
        return "data_artifact_teacher"
    if "hard_negative" in p or "hard_negatives" in p or ext == ".npz":
        return "data_artifact_hard_negatives"
    if ext in {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}:
        return "checkpoint_or_model_weight"
    if "repvit" in p and ("/model/" in p or "/data/" in p or "/detection/" in p or "/segmentation/" in p or "/sam/" in p):
        return "external_model_repo_repvit"
    if ext in {".json", ".csv"} and ("eval" in p or "result" in p or "metrics" in p or "summary" in p):
        return "result_or_metric"
    if ext in {".npy"} and ("embedding" in name or "indices" in name or "ids" in name):
        return "data_artifact_array"
    if ext in {".zip", ".tar", ".gz", ".7z"}:
        return "archive"
    if ext in {".pdf", ".png", ".jpg", ".jpeg", ".txt"}:
        return "documentation_or_media"
    if size_bytes >= 512 * 1024 * 1024:
        return "large_binary_uncategorized"
    return "uncategorized"


def sha256_file(path: Path, chunk_size_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
        if path.is_file():
            yield path


def collect_records(root: Path, hash_threshold_bytes: int, chunk_size_bytes: int) -> list[FileRecord]:
    records: list[FileRecord] = []
    for path in iter_files(root):
        stat = path.stat()
        rel = path.relative_to(root).as_posix()
        size_bytes = int(stat.st_size)
        if size_bytes <= hash_threshold_bytes:
            try:
                file_hash = sha256_file(path, chunk_size_bytes)
                hash_status = "hashed"
            except OSError as exc:
                file_hash = ""
                hash_status = f"hash_error:{type(exc).__name__}"
        else:
            file_hash = ""
            hash_status = "skip_huge"

        records.append(
            FileRecord(
                rel_path=rel,
                abs_path=str(path),
                parent=Path(rel).parent.as_posix() if Path(rel).parent.as_posix() != "." else "",
                name=path.name,
                extension=normalize_extension(path),
                size_bytes=size_bytes,
                size_mb=round(size_bytes / (1024 * 1024), 6),
                category=infer_category(rel, size_bytes),
                sha256=file_hash,
                hash_status=hash_status,
                mtime_utc=utc_from_timestamp(stat.st_mtime),
            )
        )
    return records


def write_manifest(records: list[FileRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else list(FileRecord.__annotations__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_hash_files(records: list[FileRecord], output_dir: Path) -> None:
    hashed = [record for record in records if record.hash_status == "hashed"]
    skipped = [record for record in records if record.hash_status == "skip_huge"]

    with (output_dir / "MANIFEST.sha256.txt").open("w", encoding="utf-8") as f:
        for record in hashed:
            f.write(f"{record.sha256}  {record.rel_path}\n")

    with (output_dir / "HUGE_FILES_SKIPPED_HASHING.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rel_path", "size_bytes", "size_mb", "category"])
        writer.writeheader()
        for record in skipped:
            writer.writerow(
                {
                    "rel_path": record.rel_path,
                    "size_bytes": record.size_bytes,
                    "size_mb": record.size_mb,
                    "category": record.category,
                }
            )


def write_tree(root: Path, output_path: Path, max_depth: int | None = None) -> None:
    lines: list[str] = [f"ROOT: {root}", ""]

    def walk(path: Path, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        try:
            items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            lines.append("  " * depth + "[permission denied]")
            return

        for item in items:
            rel = item.relative_to(root).as_posix()
            if item.is_dir():
                lines.append("  " * depth + f"{item.name}/")
                walk(item, depth + 1)
            else:
                try:
                    size = item.stat().st_size
                except OSError:
                    size = -1
                lines.append("  " * depth + f"{item.name}  ({size:,} bytes)  [{rel}]")

    walk(root, 0)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(root: Path, records: list[FileRecord], hash_threshold_mb: float) -> dict:
    total_bytes = sum(record.size_bytes for record in records)
    by_category = defaultdict(lambda: {"files": 0, "bytes": 0, "mb": 0.0})
    by_ext = defaultdict(lambda: {"files": 0, "bytes": 0, "mb": 0.0})
    for record in records:
        for bucket, key in [(by_category, record.category), (by_ext, record.extension)]:
            bucket[key]["files"] += 1
            bucket[key]["bytes"] += record.size_bytes
            bucket[key]["mb"] = round(bucket[key]["bytes"] / (1024 * 1024), 6)

    largest = sorted(records, key=lambda r: r.size_bytes, reverse=True)[:50]
    return {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "root": str(root),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hash_threshold_mb": hash_threshold_mb,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / (1024**3), 6),
        "hash_status_counts": dict(Counter(record.hash_status for record in records)),
        "category_counts": dict(Counter(record.category for record in records)),
        "by_category": dict(sorted(by_category.items())),
        "by_extension": dict(sorted(by_ext.items())),
        "largest_files": [
            {
                "rel_path": record.rel_path,
                "size_bytes": record.size_bytes,
                "size_mb": record.size_mb,
                "category": record.category,
                "hash_status": record.hash_status,
            }
            for record in largest
        ],
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        message = f"Dataset root does not exist: {root}"
        if args.fail_on_missing:
            raise FileNotFoundError(message)
        print("WARNING:", message)

    hash_threshold_bytes = int(args.hash_threshold_mb * 1024 * 1024)
    chunk_size_bytes = max(1, int(args.hash_chunk_mb * 1024 * 1024))

    print(f"Inventory root: {root}")
    print(f"Output dir:     {output_dir}")
    print(f"Hash threshold: {args.hash_threshold_mb} MB")

    records = collect_records(root, hash_threshold_bytes, chunk_size_bytes)

    write_manifest(records, output_dir / "MANIFEST.csv")
    write_hash_files(records, output_dir)
    write_tree(root, output_dir / "dataset_tree_full.txt", max_depth=None)
    if args.max_tree_depth is not None:
        write_tree(root, output_dir / "dataset_tree_limited.txt", max_depth=args.max_tree_depth)

    summary = build_summary(root, records, args.hash_threshold_mb)
    (output_dir / "inventory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "file_count": summary["file_count"],
        "total_gb": summary["total_gb"],
        "hash_status_counts": summary["hash_status_counts"],
        "output_dir": str(output_dir),
    }, indent=2))
    print("Wrote:")
    print(output_dir / "MANIFEST.csv")
    print(output_dir / "MANIFEST.sha256.txt")
    print(output_dir / "HUGE_FILES_SKIPPED_HASHING.csv")
    print(output_dir / "dataset_tree_full.txt")
    print(output_dir / "inventory_summary.json")


if __name__ == "__main__":
    main()
