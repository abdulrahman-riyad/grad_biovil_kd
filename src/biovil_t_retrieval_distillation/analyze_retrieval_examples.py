from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze top-k retrieval successes and failures.")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--topk-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direction", choices=["image_to_text", "text_to_image"], default=None)
    parser.add_argument("--top-k-success", type=int, default=5)
    parser.add_argument("--failure-rank-threshold", type=int, default=100)
    parser.add_argument("--max-examples-per-bucket", type=int, default=50)
    return parser.parse_args()


def clean_text(value: object, max_chars: int = 700) -> str:
    if pd.isna(value):
        return ""
    text = " ".join(str(value).split())
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def load_topk(path: Path, direction: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if direction is not None:
        df = df[df["direction"] == direction].copy()
    if df.empty:
        raise ValueError(f"No top-k rows found in {path} for direction={direction}.")
    return df


def build_query_summary(topk: pd.DataFrame) -> pd.DataFrame:
    return (
        topk.groupby(["direction", "query_position", "query_row_index"], as_index=False)
        .agg(
            target_rank=("target_rank", "first"),
            target_row_index=("target_row_index", "first"),
            top1_row_index=("retrieved_row_index", "first"),
            top1_score=("score", "first"),
            target_in_saved_topk=("is_target", "max"),
        )
        .sort_values(["direction", "target_rank", "query_position"])
    )


def attach_texts(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.reset_index(drop=True)

    def text_for(row_index: int) -> str:
        if row_index < 0 or row_index >= len(metadata):
            return ""
        return clean_text(metadata.iloc[int(row_index)].get("report_text", ""))

    def study_for(row_index: int) -> int | None:
        if row_index < 0 or row_index >= len(metadata):
            return None
        return int(metadata.iloc[int(row_index)].get("study_id"))

    def subject_for(row_index: int) -> int | None:
        if row_index < 0 or row_index >= len(metadata):
            return None
        return int(metadata.iloc[int(row_index)].get("subject_id"))

    df = df.copy()
    for prefix in ("query", "target", "top1"):
        column = f"{prefix}_row_index"
        df[f"{prefix}_study_id"] = df[column].map(study_for)
        df[f"{prefix}_subject_id"] = df[column].map(subject_for)
        df[f"{prefix}_report_text"] = df[column].map(text_for)
    return df


def write_csv(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def bucket_examples(summary: pd.DataFrame, top_k_success: int, failure_rank_threshold: int) -> dict[str, pd.DataFrame]:
    exact_top1 = summary[summary["target_rank"] == 1]
    topk_success = summary[(summary["target_rank"] > 1) & (summary["target_rank"] <= top_k_success)]
    moderate = summary[(summary["target_rank"] > top_k_success) & (summary["target_rank"] <= failure_rank_threshold)]
    failures = summary[summary["target_rank"] > failure_rank_threshold].sort_values("target_rank", ascending=False)
    return {
        "exact_top1": exact_top1,
        f"top{top_k_success}_not_top1": topk_success,
        "moderate_failures": moderate,
        "severe_failures": failures,
    }


def markdown_report(
    output_path: Path,
    query_summary: pd.DataFrame,
    buckets: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# Retrieval Example Analysis")
    lines.append("")
    lines.append(f"Top-k CSV: `{args.topk_csv}`")
    lines.append(f"Direction filter: `{args.direction or 'all'}`")
    lines.append("")
    lines.append("## Query-Level Summary")
    lines.append("")
    total = len(query_summary)
    lines.append(f"- Total queries: {total}")
    lines.append(f"- Exact top-1 matches: {int((query_summary['target_rank'] == 1).sum())}")
    lines.append(f"- Top-{args.top_k_success} matches: {int((query_summary['target_rank'] <= args.top_k_success).sum())}")
    lines.append(f"- Failures with target rank > {args.failure_rank_threshold}: {int((query_summary['target_rank'] > args.failure_rank_threshold).sum())}")
    lines.append("")
    lines.append("## Bucket Counts")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("|---|---:|")
    for name, df in buckets.items():
        lines.append(f"| {name} | {len(df)} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `target_rank` is the rank of the exact paired report/image.")
    lines.append("- `top1_report_text` is the report text of the highest-scoring retrieved candidate.")
    lines.append("- Use severe failures for failure-mode review and explainability case selection.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata_csv)
    topk = load_topk(Path(args.topk_csv), args.direction)
    summary = build_query_summary(topk)
    summary = attach_texts(summary, metadata)
    write_csv(output_dir / "query_retrieval_summary.csv", summary)

    buckets = bucket_examples(summary, args.top_k_success, args.failure_rank_threshold)
    for name, bucket in buckets.items():
        rows = bucket.head(args.max_examples_per_bucket)
        write_csv(output_dir / f"{name}_examples.csv", rows)

    stats: dict[str, Any] = {
        "topk_csv": args.topk_csv,
        "direction": args.direction or "all",
        "num_queries": int(len(summary)),
        "top1_count": int((summary["target_rank"] == 1).sum()),
        f"top{args.top_k_success}_count": int((summary["target_rank"] <= args.top_k_success).sum()),
        "failure_rank_threshold": args.failure_rank_threshold,
        "severe_failure_count": int((summary["target_rank"] > args.failure_rank_threshold).sum()),
        "median_target_rank": float(summary["target_rank"].median()),
        "mean_target_rank": float(summary["target_rank"].mean()),
    }
    (output_dir / "analysis_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    markdown_report(output_dir / "analysis_report.md", summary, buckets, args)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
