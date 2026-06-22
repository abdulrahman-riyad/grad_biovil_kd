from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from train_stage2_hard_negatives_l4 import HARDWARE_PROFILES, RUNS, command_to_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one L4 model end-to-end: hard-negative training, final retrieval "
            "evaluation, and optional GCS sync."
        )
    )
    parser.add_argument("--run-key", choices=list(RUNS), required=True)
    parser.add_argument("--project-root", default=os.environ.get("GRAD_BIOVIL_ROOT"))
    parser.add_argument("--work-root", default=os.environ.get("GRAD_BIOVIL_WORK", str(Path.home() / "grad-biovil-runs")))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET"))
    parser.add_argument("--gcs-prefix", default="runs/grad-biovil-kd/l4")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--hardware-profile", choices=list(HARDWARE_PROFILES), default="l4_24gb")
    parser.add_argument("--checkpoint-name", default="best_5k_retrieval.pt")
    parser.add_argument("--candidate-pools", default="32,1000,5000,full")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--similarity-chunk-size", type=int, default=1024)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.project_root:
        args.project_root = str(Path(__file__).resolve().parents[3])
    return args


def run_dir(work_root: Path, run_key: str, epochs: int) -> Path:
    config = RUNS[run_key]
    return work_root / "runs_stage2_hard_negative_l4" / f"{config.stage2_run_name}_e{epochs}"


def eval_dir(work_root: Path, run_key: str) -> Path:
    return work_root / "eval_retrieval_l4" / run_key


def log_path(work_root: Path, run_key: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = work_root / "logs_single_model_l4" / f"{run_key}_{stamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_and_log(cmd: list[str], env: dict[str, str], log_file: Path, dry_run: bool) -> None:
    line = "\n" + command_to_string(cmd) + "\n"
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        if dry_run:
            return
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for output_line in process.stdout:
            print(output_line, end="")
            f.write(output_line)
            f.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


def sync_outputs(args: argparse.Namespace, work_root: Path, log_file: Path, env: dict[str, str]) -> None:
    if args.no_sync:
        print("Skipping GCS sync because --no-sync was set.")
        return
    if not args.bucket:
        raise ValueError("Set BUCKET or pass --bucket to sync outputs to GCS.")

    destination = f"gs://{args.bucket}/{args.gcs_prefix.rstrip('/')}/{args.run_key}"
    targets = [
        (run_dir(work_root, args.run_key, args.epochs), f"{destination}/training"),
        (eval_dir(work_root, args.run_key), f"{destination}/evaluation"),
        (log_file, f"{destination}/logs/{log_file.name}"),
    ]
    for source, target in targets:
        if not source.exists():
            print(f"Skipping missing sync source: {source}")
            continue
        if source.is_dir():
            cmd = ["gcloud", "storage", "rsync", "-r", str(source), target]
        else:
            cmd = ["gcloud", "storage", "cp", str(source), target]
        run_and_log(cmd, env=env, log_file=log_file, dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    log_file = log_path(work_root, args.run_key)

    env = os.environ.copy()
    env.setdefault("GRAD_BIOVIL_ROOT", str(project_root))
    env.setdefault("GRAD_BIOVIL_WORK", str(work_root))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.bucket:
        env.setdefault("BUCKET", args.bucket)

    train_cmd = [
        sys.executable,
        str(project_root / "project_repo" / "ops" / "gcp_l4" / "train_stage2_hard_negatives_l4.py"),
        "--run-key",
        args.run_key,
        "--epochs",
        str(args.epochs),
        "--hardware-profile",
        args.hardware_profile,
    ]
    eval_cmd = [
        sys.executable,
        str(project_root / "project_repo" / "ops" / "gcp_l4" / "evaluate_retrieval_l4.py"),
        "--run-key",
        args.run_key,
        "--epochs",
        str(args.epochs),
        "--hardware-profile",
        args.hardware_profile,
        "--checkpoint-name",
        args.checkpoint_name,
        "--candidate-pools",
        args.candidate_pools,
        "--seeds",
        args.seeds,
        "--similarity-chunk-size",
        str(args.similarity_chunk_size),
    ]
    if args.eval_batch_size is not None:
        eval_cmd += ["--batch-size", str(args.eval_batch_size)]
    if args.eval_num_workers is not None:
        eval_cmd += ["--num-workers", str(args.eval_num_workers)]

    run_and_log(train_cmd, env=env, log_file=log_file, dry_run=args.dry_run)
    run_and_log(eval_cmd, env=env, log_file=log_file, dry_run=args.dry_run)
    sync_outputs(args, work_root=work_root, log_file=log_file, env=env)
    print(f"Completed {args.run_key}. Log: {log_file}")


if __name__ == "__main__":
    main()
