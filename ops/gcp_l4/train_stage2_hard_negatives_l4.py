from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_REPO_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_DISTILLATION_DIR = PROJECT_REPO_ROOT / "src" / "biovil_t_retrieval_distillation"
if str(RETRIEVAL_DISTILLATION_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DISTILLATION_DIR))

from configs.final_student_runs import (  # noqa: E402
    FINAL_STUDENT_RUN_ORDER,
    FINAL_STUDENT_RUNS,
    FinalStudentRun as RunConfig,
)


@dataclass(frozen=True)
class HardwareProfile:
    batch_size: int
    num_workers: int
    epoch_retrieval_batch_size: int
    epoch_retrieval_num_workers: int
    amp_dtype: str


HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "l4_24gb": HardwareProfile(
        batch_size=20,
        num_workers=4,
        epoch_retrieval_batch_size=64,
        epoch_retrieval_num_workers=2,
        amp_dtype="bfloat16",
    ),
}


RUNS: dict[str, RunConfig] = FINAL_STUDENT_RUNS


def default_project_root() -> Path:
    # .../structured_grad_biovil_kd/project_repo/ops/gcp_l4/train_stage2_hard_negatives_l4.py
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run integrated hard-negative contrastive fine-tuning on a single NVIDIA GPU."
    )
    parser.add_argument("--project-root", default=os.environ.get("GRAD_BIOVIL_ROOT", str(default_project_root())))
    parser.add_argument("--mimic-root", default=os.environ.get("MIMIC_CXR_ROOT"))
    parser.add_argument("--image-root", default=os.environ.get("MIMIC_CXR_IMAGE_ROOT"))
    parser.add_argument("--work-root", default=os.environ.get("GRAD_BIOVIL_WORK", str(Path.home() / "grad-biovil-runs")))
    parser.add_argument("--run-key", choices=["all", *RUNS.keys()], default="all")
    parser.add_argument(
        "--hardware-profile",
        choices=list(HARDWARE_PROFILES),
        default="l4_24gb",
        help="Default throughput settings. Explicit batch/worker arguments override this profile.",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epoch-retrieval-batch-size", type=int, default=None)
    parser.add_argument("--epoch-retrieval-num-workers", type=int, default=None)
    parser.add_argument("--epoch-retrieval-pools", default="5000")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--hard-negatives-per-sample", type=int, default=8)
    parser.add_argument("--hard-negative-top-k", type=int, default=64)
    parser.add_argument("--false-negative-threshold", type=float, default=0.85)
    parser.add_argument("--min-hard-text-threshold", type=float, default=0.60)
    parser.add_argument("--kd-image-weight", type=float, default=0.05)
    parser.add_argument("--kd-relational-weight", type=float, default=0.10)
    parser.add_argument("--pseudo-label-weight", type=float, default=0.05)
    parser.add_argument("--soft-positive-weight", type=float, default=0.25)
    parser.add_argument("--label-soft-positive-weight", type=float, default=0.15)
    parser.add_argument("--anatomy-soft-positive-weight", type=float, default=0.05)
    parser.add_argument("--longitudinal-weight", type=float, default=0.03)
    parser.add_argument("--uncertainty-weight", type=float, default=0.01)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a short sanity check instead of a real run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument("--force-precompute", action="store_true")
    args = parser.parse_args()
    profile = HARDWARE_PROFILES[args.hardware_profile]
    if args.batch_size is None:
        args.batch_size = profile.batch_size
    if args.num_workers is None:
        args.num_workers = profile.num_workers
    if args.epoch_retrieval_batch_size is None:
        args.epoch_retrieval_batch_size = profile.epoch_retrieval_batch_size
    if args.epoch_retrieval_num_workers is None:
        args.epoch_retrieval_num_workers = profile.epoch_retrieval_num_workers
    if args.amp_dtype is None:
        args.amp_dtype = profile.amp_dtype
    return args


def image_root_from_args(args: argparse.Namespace) -> Path:
    if args.image_root:
        return Path(args.image_root).expanduser().resolve()
    if not args.mimic_root:
        raise ValueError("Pass --mimic-root, --image-root, or set MIMIC_CXR_ROOT/MIMIC_CXR_IMAGE_ROOT.")
    return Path(args.mimic_root).expanduser().resolve() / "official_data_iccv_final" / "files"


def checked_path(path: Path, label: str, dry_run: bool) -> Path:
    if not path.exists() and not dry_run:
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def resolve_stage1_checkpoint(project_root: Path, config: RunConfig, dry_run: bool) -> Path:
    candidates = [project_root / relative for relative in config.stage1_checkpoint_candidates]
    for path in candidates:
        if path.exists():
            return path
    if dry_run:
        return candidates[0]
    formatted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Missing Stage 1 checkpoint for {config.key}:\n{formatted}")


def command_to_string(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("\n" + command_to_string(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def common_paths(args: argparse.Namespace) -> dict[str, Path]:
    project_root = Path(args.project_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    return {
        "project_root": project_root,
        "work_root": work_root,
        "retrieval_distillation_dir": project_root / "project_repo" / "src" / "biovil_t_retrieval_distillation",
        "artifacts_dir": project_root / "data_artifacts",
        "splits_dir": project_root / "data_artifacts" / "splits",
        "mobilevit_checkpoint": project_root / "checkpoints" / "image_students" / "mobilevit_s" / "mobilevit_s_biovil_kd_checkpoint.pt",
        "repvit_checkpoint": project_root / "checkpoints" / "image_students" / "repvit_m1_1" / "best.pt",
        "repvit_root": project_root / "models" / "external_repos" / "RepViT",
        "hard_negative_file": work_root / "hard_negatives" / hard_negative_filename(args),
    }


def threshold_tag(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def hard_negative_filename(args: argparse.Namespace) -> str:
    top_k = int(getattr(args, "hard_negative_top_k", 64))
    false_negative_threshold = float(getattr(args, "false_negative_threshold", 0.85))
    min_hard_text_threshold = float(getattr(args, "min_hard_text_threshold", 0.60))
    return (
        "biovil_teacher_train_"
        f"top{top_k}_"
        f"fn{threshold_tag(false_negative_threshold)}_"
        f"min{threshold_tag(min_hard_text_threshold)}.npz"
    )


def build_env(work_root: Path, retrieval_distillation_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HOME", str(work_root / "hf_cache"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = f"{retrieval_distillation_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def precompute_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        str(paths["retrieval_distillation_dir"] / "mine_hard_negatives.py"),
        "--artifacts-dir",
        str(paths["artifacts_dir"]),
        "--splits-dir",
        str(paths["splits_dir"]),
        "--metadata-file",
        "metadata/biovil_t_fixed_metadata.csv",
        "--image-embeddings-file",
        "teacher/biovil_t_fixed_image_embeddings.npy",
        "--text-embeddings-file",
        "teacher/biovil_t_fixed_text_embeddings.npy",
        "--output-file",
        str(paths["hard_negative_file"]),
        "--split",
        "train",
        "--candidate-split",
        "train",
        "--top-k",
        str(args.hard_negative_top_k),
        "--chunk-size",
        "512",
        "--false-negative-text-threshold",
        str(args.false_negative_threshold),
        "--min-hard-text-threshold",
        str(args.min_hard_text_threshold),
        "--dtype",
        "float16",
    ]


def training_command(
    args: argparse.Namespace,
    paths: dict[str, Path],
    image_root: Path,
    key: str,
    config: RunConfig,
) -> list[str]:
    epochs = 1 if args.smoke else args.epochs
    batch_size = min(args.batch_size, 4) if args.smoke else args.batch_size
    output_suffix = "_smoke" if args.smoke else f"_e{epochs}"
    output_dir = paths["work_root"] / "runs_stage2_hard_negative_l4" / f"{config.stage2_run_name}{output_suffix}"
    cmd = [
        sys.executable,
        str(paths["retrieval_distillation_dir"] / "train_stage2_hard_negative.py"),
        "--artifacts-dir",
        str(paths["artifacts_dir"]),
        "--splits-dir",
        str(paths["splits_dir"]),
        "--metadata-file",
        "metadata/biovil_t_fixed_metadata.csv",
        "--teacher-image-embeddings-file",
        "teacher/biovil_t_fixed_image_embeddings.npy",
        "--teacher-text-embeddings-file",
        "teacher/biovil_t_fixed_text_embeddings.npy",
        "--image-root",
        str(image_root),
        "--output-dir",
        str(output_dir),
        "--hard-negative-file",
        str(paths["hard_negative_file"]),
        "--init-contrastive-checkpoint",
        str(resolve_stage1_checkpoint(paths["project_root"], config, args.dry_run)),
        "--image-student",
        config.image_student,
        "--text-encoder",
        config.text_encoder,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--encoder-lr",
        str(args.encoder_lr),
        "--max-text-length",
        "256",
        "--hard-negatives-per-sample",
        str(args.hard_negatives_per_sample),
        "--hard-negative-weight",
        "1.0",
        "--hard-negative-mode",
        "denominator",
        "--kd-image-weight",
        str(args.kd_image_weight),
        "--kd-text-weight",
        "0.0",
        "--kd-relational-weight",
        str(args.kd_relational_weight),
        "--pseudo-label-weight",
        str(args.pseudo_label_weight),
        "--soft-positive-weight",
        str(args.soft_positive_weight),
        "--soft-positive-threshold",
        str(args.false_negative_threshold),
        "--label-soft-positive-weight",
        str(args.label_soft_positive_weight),
        "--anatomy-soft-positive-weight",
        str(args.anatomy_soft_positive_weight),
        "--longitudinal-weight",
        str(args.longitudinal_weight),
        "--uncertainty-weight",
        str(args.uncertainty_weight),
        "--epoch-retrieval-pool-sizes",
        "" if args.smoke else args.epoch_retrieval_pools,
        "--epoch-retrieval-split",
        "test",
        "--epoch-retrieval-batch-size",
        str(args.epoch_retrieval_batch_size),
        "--epoch-retrieval-num-workers",
        str(args.epoch_retrieval_num_workers),
        "--epoch-retrieval-chunk-size",
        "512",
        "--retrieval-selection-pool",
        "5000",
    ]
    if args.amp:
        cmd += ["--amp", "--amp-dtype", args.amp_dtype]
    if args.smoke:
        cmd += ["--max-train-batches", "20", "--max-val-batches", "5"]
    if config.image_student == "mobilevit":
        cmd += ["--mobilevit-checkpoint", str(paths["mobilevit_checkpoint"])]
    elif config.image_student == "repvit":
        cmd += [
            "--repvit-checkpoint",
            str(paths["repvit_checkpoint"]),
            "--repvit-root",
            str(paths["repvit_root"]),
        ]
    else:
        raise ValueError(f"Unsupported image student for {key}: {config.image_student}")
    return cmd


def main() -> None:
    args = parse_args()
    paths = common_paths(args)
    image_root = image_root_from_args(args)

    for label, path in [
        ("project root", paths["project_root"]),
        ("retrieval distillation source", paths["retrieval_distillation_dir"]),
        ("artifacts", paths["artifacts_dir"]),
        ("splits", paths["splits_dir"]),
        ("image root", image_root),
        ("MobileViT checkpoint", paths["mobilevit_checkpoint"]),
        ("RepViT checkpoint", paths["repvit_checkpoint"]),
        ("RepViT repo", paths["repvit_root"]),
    ]:
        checked_path(path, label, args.dry_run)

    paths["work_root"].mkdir(parents=True, exist_ok=True)
    paths["hard_negative_file"].parent.mkdir(parents=True, exist_ok=True)
    env = build_env(paths["work_root"], paths["retrieval_distillation_dir"])

    if args.force_precompute and paths["hard_negative_file"].exists() and not args.dry_run:
        paths["hard_negative_file"].unlink()

    if not args.skip_precompute and not paths["hard_negative_file"].exists():
        run_command(precompute_command(args, paths), env=env, dry_run=args.dry_run)
    else:
        print(f"Using existing hard negatives: {paths['hard_negative_file']}")

    selected = FINAL_STUDENT_RUN_ORDER if args.run_key == "all" else [args.run_key]
    for key in selected:
        config = RUNS[key]
        resolve_stage1_checkpoint(paths["project_root"], config, args.dry_run)
        run_command(training_command(args, paths, image_root, key, config), env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
