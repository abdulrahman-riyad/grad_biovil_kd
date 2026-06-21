from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    image_student: str
    text_encoder: str
    run_name: str


RUNS: dict[str, RunConfig] = {
    "mobilevit_clinical_distilbert": RunConfig(
        image_student="mobilevit",
        text_encoder="clinical_distilbert",
        run_name="mobilevit_clinical_distilbert_teacher_kd_hn",
    ),
    "repvit_clinical_distilbert": RunConfig(
        image_student="repvit",
        text_encoder="clinical_distilbert",
        run_name="repvit_clinical_distilbert_teacher_kd_hn",
    ),
    "mobilevit_distil_biobert": RunConfig(
        image_student="mobilevit",
        text_encoder="distil_biobert",
        run_name="mobilevit_distil_biobert_teacher_kd_hn",
    ),
    "repvit_distil_biobert": RunConfig(
        image_student="repvit",
        text_encoder="distil_biobert",
        run_name="repvit_distil_biobert_teacher_kd_hn",
    ),
}

RUN_ORDER = [
    "mobilevit_clinical_distilbert",
    "repvit_clinical_distilbert",
    "mobilevit_distil_biobert",
    "repvit_distil_biobert",
]


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teammate-style full-student teacher-KD + hard-negative training on our fixed split."
    )
    parser.add_argument("--project-root", default=os.environ.get("GRAD_BIOVIL_ROOT", str(default_project_root())))
    parser.add_argument("--mimic-root", default=os.environ.get("MIMIC_CXR_ROOT"))
    parser.add_argument("--image-root", default=os.environ.get("MIMIC_CXR_IMAGE_ROOT"))
    parser.add_argument("--work-root", default=os.environ.get("GRAD_BIOVIL_WORK", str(Path.home() / "grad-biovil-runs")))
    parser.add_argument("--run-key", choices=["all", *RUNS.keys()], default="all")
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stage2-lr-multiplier", type=float, default=0.3)
    parser.add_argument("--kd-weight", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hn-pool-size", type=int, default=25000)
    parser.add_argument("--hn-top-k", type=int, default=5)
    parser.add_argument("--hn-refresh-epochs", type=int, default=2)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_to_string(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def image_root_from_args(args: argparse.Namespace) -> Path:
    if args.image_root:
        return Path(args.image_root).expanduser().resolve()
    if not args.mimic_root:
        raise ValueError("Pass --mimic-root, --image-root, or set MIMIC_CXR_ROOT/MIMIC_CXR_IMAGE_ROOT.")
    return Path(args.mimic_root).expanduser().resolve() / "official_data_iccv_final" / "files"


def common_paths(args: argparse.Namespace) -> dict[str, Path]:
    project_root = Path(args.project_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    return {
        "project_root": project_root,
        "work_root": work_root,
        "track_ab_dir": project_root / "project_repo" / "src" / "track_ab",
        "artifacts_dir": project_root / "data_artifacts",
        "splits_dir": project_root / "data_artifacts" / "splits",
        "mobilevit_checkpoint": project_root / "checkpoints" / "image_students" / "mobilevit_s" / "mobilevit_s_biovil_kd_checkpoint.pt",
        "repvit_checkpoint": project_root / "checkpoints" / "image_students" / "repvit_m1_1" / "best.pt",
        "repvit_root": project_root / "models" / "external_repos" / "RepViT",
    }


def checked_path(path: Path, label: str, dry_run: bool) -> Path:
    if not path.exists() and not dry_run:
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def build_env(track_ab_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    prefix = str(track_ab_dir)
    env["PYTHONPATH"] = prefix if not existing else prefix + os.pathsep + existing
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def selected_run_keys(run_key: str) -> list[str]:
    return RUN_ORDER if run_key == "all" else [run_key]


def training_command(args: argparse.Namespace, paths: dict[str, Path], image_root: Path, run_key: str) -> list[str]:
    config = RUNS[run_key]
    output_dir = (
        paths["work_root"]
        / "runs_teacher_kd_hn_l4"
        / f"{config.run_name}_s1{args.stage1_epochs}_s2{args.stage2_epochs}"
    )
    cmd = [
        sys.executable,
        str(paths["track_ab_dir"] / "train_teacher_kd_hn_full_student.py"),
        "--artifacts-dir",
        str(checked_path(paths["artifacts_dir"], "data artifacts", args.dry_run)),
        "--splits-dir",
        str(checked_path(paths["splits_dir"], "split files", args.dry_run)),
        "--image-root",
        str(image_root),
        "--output-dir",
        str(output_dir),
        "--image-student",
        config.image_student,
        "--text-encoder",
        config.text_encoder,
        "--stage1-epochs",
        str(1 if args.smoke else args.stage1_epochs),
        "--stage2-epochs",
        str(1 if args.smoke else args.stage2_epochs),
        "--batch-size",
        str(4 if args.smoke else args.batch_size),
        "--num-workers",
        str(1 if args.smoke else args.num_workers),
        "--lr",
        str(args.lr),
        "--stage2-lr-multiplier",
        str(args.stage2_lr_multiplier),
        "--kd-weight",
        str(args.kd_weight),
        "--temperature",
        str(args.temperature),
        "--hn-pool-size",
        str(128 if args.smoke else args.hn_pool_size),
        "--hn-top-k",
        str(min(args.hn_top_k, 2) if args.smoke else args.hn_top_k),
        "--hn-refresh-epochs",
        str(args.hn_refresh_epochs),
        "--max-text-length",
        str(args.max_text_length),
        "--amp-dtype",
        args.amp_dtype,
    ]
    cmd.append("--amp" if args.amp else "--no-amp")
    if args.smoke:
        cmd += ["--max-train-rows", "128", "--max-val-rows", "64", "--max-train-batches", "4", "--max-val-batches", "2"]
    if config.image_student == "mobilevit":
        cmd += ["--mobilevit-checkpoint", str(checked_path(paths["mobilevit_checkpoint"], "MobileViT checkpoint", args.dry_run))]
    else:
        cmd += [
            "--repvit-checkpoint",
            str(checked_path(paths["repvit_checkpoint"], "RepViT checkpoint", args.dry_run)),
            "--repvit-root",
            str(checked_path(paths["repvit_root"], "RepViT repo", args.dry_run)),
        ]
    return cmd


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("\n" + command_to_string(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    paths = common_paths(args)
    paths["work_root"].mkdir(parents=True, exist_ok=True)
    image_root = image_root_from_args(args)
    checked_path(image_root, "MIMIC-CXR image root", args.dry_run)
    env = build_env(paths["track_ab_dir"])
    for run_key in selected_run_keys(args.run_key):
        print(f"RUN {run_key}", flush=True)
        run_command(training_command(args, paths, image_root, run_key), env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
