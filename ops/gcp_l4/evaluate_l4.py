from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_hard_negative_l4 import (
    HARDWARE_PROFILES,
    RUNS,
    RunConfig,
    command_to_string,
    common_paths,
    image_root_from_args,
)

TEACHER_KEY = "teacher_biovil_t"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate L4 hard-negative runs and the BioViL-T teacher baseline on "
            "full and sampled retrieval pools."
        )
    )
    parser.add_argument("--project-root", default=os.environ.get("GRAD_BIOVIL_ROOT"))
    parser.add_argument("--mimic-root", default=os.environ.get("MIMIC_CXR_ROOT"))
    parser.add_argument("--image-root", default=os.environ.get("MIMIC_CXR_IMAGE_ROOT"))
    parser.add_argument("--work-root", default=os.environ.get("GRAD_BIOVIL_WORK", str(Path.home() / "grad_biovil_runs")))
    parser.add_argument("--run-key", choices=["all", TEACHER_KEY, *RUNS.keys()], default="all")
    parser.add_argument("--include-teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hardware-profile",
        choices=list(HARDWARE_PROFILES),
        default="h100_80gb",
        help="Default evaluation throughput settings. Explicit batch/worker arguments override this profile.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Run suffix used by the training launcher.")
    parser.add_argument("--checkpoint-name", default="best_5k_retrieval.pt")
    parser.add_argument("--candidate-pools", default="32,1000,5000,full")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--similarity-chunk-size", type=int, default=512)
    parser.add_argument("--save-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-topk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile = HARDWARE_PROFILES[args.hardware_profile]
    if args.batch_size is None:
        args.batch_size = profile.epoch_retrieval_batch_size
    if args.num_workers is None:
        args.num_workers = profile.epoch_retrieval_num_workers
    return args


def candidate_pool_values(value: str) -> list[int | None]:
    pools: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        pools.append(None if item == "full" else int(item))
    return pools


def seed_values(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def pool_label(pool: int | None) -> str:
    return "full" if pool is None else str(pool)


def run_dir(work_root: Path, config: RunConfig, epochs: int) -> Path:
    return work_root / "runs_hard_negative_integrated_l4" / f"{config.run_name}_e{epochs}"


def selected_run_keys(run_key: str, include_teacher: bool) -> list[str]:
    if run_key == TEACHER_KEY:
        return [TEACHER_KEY]
    if run_key == "all":
        keys = list(RUNS)
        if include_teacher:
            keys.append(TEACHER_KEY)
        return keys
    return [run_key]


def eval_output_dir(work_root: Path, key: str, pool: int | None, seed: int | None) -> Path:
    if pool is None:
        return work_root / "eval_hard_negative_l4" / key / "pool_full"
    return work_root / "eval_hard_negative_l4" / key / f"pool_{pool}_seed_{seed}"


def student_eval_command(
    args: argparse.Namespace,
    paths: dict[str, Path],
    image_root: Path,
    key: str,
    config: RunConfig,
    pool: int | None,
    seed: int | None,
) -> list[str]:
    source_run_dir = run_dir(paths["work_root"], config, args.epochs)
    checkpoint = source_run_dir / args.checkpoint_name
    output_dir = eval_output_dir(paths["work_root"], key, pool, seed)
    cmd = [
        sys.executable,
        str(paths["track_ab_dir"] / "evaluate_contrastive_retrieval.py"),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--artifacts-dir",
        str(paths["artifacts_dir"]),
        "--splits-dir",
        str(paths["splits_dir"]),
        "--metadata-file",
        "metadata/biovil_t_fixed_metadata.csv",
        "--image-root",
        str(image_root),
        "--split",
        "test",
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--similarity-chunk-size",
        str(args.similarity_chunk_size),
        "--seed",
        str(42 if seed is None else seed),
    ]
    if not args.save_embeddings:
        cmd += ["--no-save-embeddings"]
    if not args.save_topk:
        cmd += ["--no-save-topk"]
    if pool is not None:
        cmd += ["--candidate-pool-size", str(pool)]
    if config.image_student == "mobilevit":
        cmd += ["--mobilevit-checkpoint", str(paths["mobilevit_checkpoint"])]
    else:
        cmd += [
            "--repvit-checkpoint",
            str(paths["repvit_checkpoint"]),
            "--repvit-root",
            str(paths["repvit_root"]),
        ]
    return cmd


def teacher_eval_command(
    args: argparse.Namespace,
    paths: dict[str, Path],
    pool: int | None,
    seed: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(paths["track_ab_dir"] / "evaluate_teacher_retrieval.py"),
        "--artifacts-dir",
        str(paths["artifacts_dir"]),
        "--splits-dir",
        str(paths["splits_dir"]),
        "--output-dir",
        str(eval_output_dir(paths["work_root"], TEACHER_KEY, pool, seed)),
        "--metadata-file",
        "metadata/biovil_t_fixed_metadata.csv",
        "--image-embeddings-file",
        "teacher/biovil_t_fixed_image_embeddings.npy",
        "--text-embeddings-file",
        "teacher/biovil_t_fixed_text_embeddings.npy",
        "--split",
        "test",
        "--similarity-chunk-size",
        str(args.similarity_chunk_size),
        "--seed",
        str(42 if seed is None else seed),
    ]
    if not args.save_embeddings:
        cmd += ["--no-save-embeddings"]
    if not args.save_topk:
        cmd += ["--no-save-topk"]
    if pool is not None:
        cmd += ["--candidate-pool-size", str(pool)]
    return cmd


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("\n" + command_to_string(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def metric_row(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    parent = metrics_path.parent.name
    pool = parent.replace("pool_", "").split("_seed_")[0]
    seed = None
    if "_seed_" in parent:
        seed = int(parent.rsplit("_seed_", 1)[1])
    i2t = metrics["image_to_text"]
    t2i = metrics["text_to_image"]
    return {
        "run": metrics_path.parents[1].name,
        "pool": pool,
        "seed": seed,
        "checkpoint": metrics.get("checkpoint"),
        "checkpoint_epoch": metrics.get("checkpoint_epoch"),
        "candidate_pool_size": metrics.get("candidate_pool_size"),
        "i2t_r1": i2t["R@1"],
        "i2t_r5": i2t["R@5"],
        "i2t_r10": i2t["R@10"],
        "i2t_median_rank": i2t["MedianRank"],
        "i2t_mean_rank": i2t["MeanRank"],
        "t2i_r1": t2i["R@1"],
        "t2i_r5": t2i["R@5"],
        "t2i_r10": t2i["R@10"],
        "t2i_median_rank": t2i["MedianRank"],
        "t2i_mean_rank": t2i["MeanRank"],
        "avg_r1": 0.5 * (float(i2t["R@1"]) + float(t2i["R@1"])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["run"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["run"]), str(row["pool"])), []).append(row)

    metric_names = [
        "i2t_r1",
        "i2t_r5",
        "i2t_r10",
        "i2t_median_rank",
        "t2i_r1",
        "t2i_r5",
        "t2i_r10",
        "t2i_median_rank",
        "avg_r1",
    ]
    output: list[dict[str, Any]] = []
    for (run, pool), group in sorted(grouped.items()):
        record: dict[str, Any] = {
            "run": run,
            "pool": pool,
            "num_evals": len(group),
            "candidate_pool_size": group[0]["candidate_pool_size"],
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(record)
    return output


def summarize(eval_root: Path) -> None:
    rows = [metric_row(path) for path in sorted(eval_root.glob("*/*/retrieval_metrics.json"))]
    raw_path = eval_root / "retrieval_summary_raw.csv"
    agg_path = eval_root / "retrieval_summary_aggregated.csv"
    table13_path = eval_root / "table13_style_summary.csv"

    write_csv(raw_path, rows)
    aggregated = aggregate_rows(rows)
    write_csv(agg_path, aggregated)
    write_csv(
        table13_path,
        aggregated,
        fieldnames=[
            "run",
            "pool",
            "num_evals",
            "candidate_pool_size",
            "i2t_r1_mean",
            "i2t_r1_std",
            "i2t_r5_mean",
            "i2t_r10_mean",
            "i2t_median_rank_mean",
            "t2i_r1_mean",
            "t2i_r1_std",
            "t2i_r5_mean",
            "t2i_r10_mean",
            "t2i_median_rank_mean",
            "avg_r1_mean",
            "avg_r1_std",
        ],
    )
    print(f"Wrote raw summary: {raw_path}")
    print(f"Wrote aggregated summary: {agg_path}")
    print(f"Wrote Table-13-style summary: {table13_path}")


def main() -> None:
    args = parse_args()
    if not args.project_root:
        args.project_root = str(Path(__file__).resolve().parents[3])
    paths = common_paths(args)
    image_root = image_root_from_args(args)
    pools = candidate_pool_values(args.candidate_pools)
    seeds = seed_values(args.seeds)
    if not seeds:
        raise ValueError("--seeds must contain at least one seed.")

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HOME", str(paths["work_root"] / "hf_cache"))
    env["PYTHONPATH"] = f"{paths['track_ab_dir']}{os.pathsep}{env.get('PYTHONPATH', '')}"

    for key in selected_run_keys(args.run_key, args.include_teacher):
        for pool in pools:
            pool_seeds: list[int | None] = [None] if pool is None else seeds
            for seed in pool_seeds:
                if key == TEACHER_KEY:
                    cmd = teacher_eval_command(args, paths, pool, seed)
                else:
                    config = RUNS[key]
                    cmd = student_eval_command(args, paths, image_root, key, config, pool, seed)
                run_command(cmd, env=env, dry_run=args.dry_run)

    if not args.dry_run:
        summarize(paths["work_root"] / "eval_hard_negative_l4")


if __name__ == "__main__":
    main()
