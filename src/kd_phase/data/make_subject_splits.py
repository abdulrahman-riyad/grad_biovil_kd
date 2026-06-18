import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def compute_split_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_n = int(round(n * train_ratio))
    val_n = int(round(n * val_ratio))
    train_n = min(max(train_n, 0), n)
    val_n = min(max(val_n, 0), n - train_n)
    test_n = n - train_n - val_n
    return train_n, val_n, test_n


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level train/val/test splits for KD.")
    parser.add_argument("--artifacts-dir", default="weeks output/week1", help="Directory containing teacher artifacts.")
    parser.add_argument("--prefix", default="biovil_t_fixed", help="Artifact filename prefix.")
    parser.add_argument("--output-dir", default="kd_phase/splits", help="Directory for split outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Subject-level train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Subject-level validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Subject-level test ratio.")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    artifacts_dir = Path(args.artifacts_dir)
    metadata_path = artifacts_dir / f"{args.prefix}_metadata.csv"
    scores_path = artifacts_dir / f"{args.prefix}_study_scores.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)

    metadata = pd.read_csv(metadata_path)
    scores = pd.read_csv(scores_path, usecols=["subject_id", "study_id"])

    required_cols = {"subject_id", "study_id"}
    missing = required_cols - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    aligned = (
        metadata[["subject_id", "study_id"]].astype(str).values
        == scores[["subject_id", "study_id"]].astype(str).values
    ).all()
    if not aligned:
        raise ValueError("Metadata and scores are not row-aligned by subject_id/study_id.")

    subjects = metadata["subject_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)

    train_n, val_n, test_n = compute_split_counts(len(subjects), args.train_ratio, args.val_ratio)
    train_subjects = set(subjects[:train_n])
    val_subjects = set(subjects[train_n:train_n + val_n])
    test_subjects = set(subjects[train_n + val_n:])

    split_masks = {
        "train": metadata["subject_id"].isin(train_subjects).to_numpy(),
        "val": metadata["subject_id"].isin(val_subjects).to_numpy(),
        "test": metadata["subject_id"].isin(test_subjects).to_numpy(),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "num_rows": int(len(metadata)),
        "num_subjects": int(len(subjects)),
        "splits": {},
        "leakage_checks": {},
    }

    split_subject_sets = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }

    for name, mask in split_masks.items():
        indices = np.flatnonzero(mask).astype(np.int64)
        split_metadata = metadata.loc[mask].copy()
        np.save(output_dir / f"kd_{name}_indices.npy", indices)
        split_metadata.to_csv(output_dir / f"kd_{name}_metadata.csv", index=False)
        report["splits"][name] = {
            "num_subjects": int(split_metadata["subject_id"].nunique()),
            "num_rows": int(len(split_metadata)),
            "indices_file": str(output_dir / f"kd_{name}_indices.npy"),
            "metadata_file": str(output_dir / f"kd_{name}_metadata.csv"),
        }

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_subject_sets[a] & split_subject_sets[b]
        report["leakage_checks"][f"{a}_{b}_subject_overlap"] = int(len(overlap))
        if overlap:
            raise RuntimeError(f"Subject leakage detected between {a} and {b}: {len(overlap)} subjects")

    report_path = output_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=json_safe), encoding="utf-8")

    print(f"Splits written to {output_dir}")
    print(json.dumps(report["splits"], indent=2, default=json_safe))


if __name__ == "__main__":
    main()

