import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_METADATA_COLUMNS = {
    "subject_id",
    "study_id",
    "report_text",
    "image_paths",
    "num_views_used",
}

REQUIRED_SCORE_COLUMNS = {
    "subject_id",
    "study_id",
    "matched_cosine",
    "random_negative_cosine",
    "cosine_margin_vs_random",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def artifact_paths(artifacts_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "text_embeddings": artifacts_dir / f"{prefix}_text_embeddings.npy",
        "image_embeddings": artifacts_dir / f"{prefix}_image_embeddings.npy",
        "metadata": artifacts_dir / f"{prefix}_metadata.csv",
        "scores": artifacts_dir / f"{prefix}_study_scores.csv",
        "metrics": artifacts_dir / f"{prefix}_metrics.json",
    }


def summarize_norms(array: np.ndarray, sample_size: int) -> dict[str, float]:
    n = min(len(array), sample_size)
    norms = np.linalg.norm(array[:n], axis=1)
    return {
        "sample_size": int(n),
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BioViL-T teacher artifacts for KD.")
    parser.add_argument("--artifacts-dir", default="weeks output/week1", help="Directory containing teacher artifacts.")
    parser.add_argument("--prefix", default="biovil_t_fixed", help="Artifact filename prefix.")
    parser.add_argument("--output", default="kd_phase/outputs/artifact_validation_report.json", help="Validation report path.")
    parser.add_argument("--sample-size", type=int, default=10000, help="Rows used for numeric consistency checks.")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    paths = artifact_paths(artifacts_dir, args.prefix)
    report: dict[str, Any] = {
        "artifacts_dir": str(artifacts_dir),
        "prefix": args.prefix,
        "files": {name: str(path) for name, path in paths.items()},
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    for name, path in paths.items():
        exists = path.exists()
        report["checks"][f"{name}_exists"] = exists
        if not exists:
            report["errors"].append(f"Missing {name}: {path}")

    if report["errors"]:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=json_safe), encoding="utf-8")
        raise SystemExit(f"Validation failed: missing files. See {output_path}")

    text_embeddings = np.load(paths["text_embeddings"], mmap_mode="r")
    image_embeddings = np.load(paths["image_embeddings"], mmap_mode="r")
    metadata = pd.read_csv(paths["metadata"])
    scores = pd.read_csv(paths["scores"])
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))

    report["shapes"] = {
        "text_embeddings": list(text_embeddings.shape),
        "image_embeddings": list(image_embeddings.shape),
        "metadata": list(metadata.shape),
        "scores": list(scores.shape),
    }
    report["dtypes"] = {
        "text_embeddings": str(text_embeddings.dtype),
        "image_embeddings": str(image_embeddings.dtype),
    }
    report["metrics"] = metrics

    n = len(metadata)
    report["checks"]["row_count_match"] = (
        len(text_embeddings) == len(image_embeddings) == len(metadata) == len(scores)
    )
    if not report["checks"]["row_count_match"]:
        report["errors"].append("Row counts do not match across embeddings, metadata, and scores.")

    report["checks"]["embedding_dim_128"] = (
        len(text_embeddings.shape) == 2
        and len(image_embeddings.shape) == 2
        and text_embeddings.shape[1] == 128
        and image_embeddings.shape[1] == 128
    )
    if not report["checks"]["embedding_dim_128"]:
        report["errors"].append("Expected both embedding arrays to have shape (N, 128).")

    report["checks"]["embedding_dtype_float32"] = (
        text_embeddings.dtype == np.float32 and image_embeddings.dtype == np.float32
    )
    if not report["checks"]["embedding_dtype_float32"]:
        report["warnings"].append("Expected float32 embeddings for efficient training.")

    missing_metadata_cols = sorted(REQUIRED_METADATA_COLUMNS - set(metadata.columns))
    missing_score_cols = sorted(REQUIRED_SCORE_COLUMNS - set(scores.columns))
    report["checks"]["metadata_required_columns_present"] = not missing_metadata_cols
    report["checks"]["score_required_columns_present"] = not missing_score_cols
    report["missing_columns"] = {
        "metadata": missing_metadata_cols,
        "scores": missing_score_cols,
    }
    if missing_metadata_cols:
        report["errors"].append(f"Metadata missing columns: {missing_metadata_cols}")
    if missing_score_cols:
        report["errors"].append(f"Scores missing columns: {missing_score_cols}")

    if not missing_metadata_cols and not missing_score_cols:
        id_match = (
            metadata[["subject_id", "study_id"]].astype(str).values
            == scores[["subject_id", "study_id"]].astype(str).values
        ).all()
        report["checks"]["metadata_scores_id_alignment"] = bool(id_match)
        if not id_match:
            report["errors"].append("metadata.csv and study_scores.csv subject/study IDs are not row-aligned.")

    sample_n = min(n, args.sample_size)
    sample_indices = np.linspace(0, max(n - 1, 0), sample_n, dtype=np.int64) if sample_n else np.array([], dtype=np.int64)

    report["norms"] = {
        "text": summarize_norms(text_embeddings, args.sample_size),
        "image": summarize_norms(image_embeddings, args.sample_size),
    }

    if sample_n and not missing_score_cols:
        text_sample = text_embeddings[sample_indices]
        image_sample = image_embeddings[sample_indices]
        recomputed = np.sum(text_sample * image_sample, axis=1)
        saved = scores.loc[sample_indices, "matched_cosine"].to_numpy(dtype=np.float32)
        max_abs_error = float(np.max(np.abs(recomputed - saved)))
        report["matched_cosine_sample_check"] = {
            "sample_size": int(sample_n),
            "max_abs_error": max_abs_error,
            "mean_abs_error": float(np.mean(np.abs(recomputed - saved))),
        }
        report["checks"]["matched_cosine_consistent_sample"] = max_abs_error < 1e-4
        if max_abs_error >= 1e-4:
            report["warnings"].append("Sample recomputed matched cosine differs from saved scores.")

    report["summary"] = {
        "num_rows": int(n),
        "num_subjects": int(metadata["subject_id"].nunique()) if "subject_id" in metadata else None,
        "num_studies": int(metadata["study_id"].nunique()) if "study_id" in metadata else None,
        "mean_matched_cosine": float(scores["matched_cosine"].mean()) if "matched_cosine" in scores else None,
        "median_matched_cosine": float(scores["matched_cosine"].median()) if "matched_cosine" in scores else None,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=json_safe), encoding="utf-8")

    if report["errors"]:
        raise SystemExit(f"Validation failed with {len(report['errors'])} error(s). See {output_path}")

    print(f"Validation passed. Report written to {output_path}")
    print(json.dumps(report["summary"], indent=2, default=json_safe))


if __name__ == "__main__":
    main()

