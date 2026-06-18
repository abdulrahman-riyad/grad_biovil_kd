from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize retrieval_metrics.json files into one CSV/JSON table.")
    parser.add_argument("--eval-root", required=True, help="Root folder containing evaluation output subfolders.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def row_from_metrics(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": path.parent.name,
        "split": payload.get("split"),
        "image_student": payload.get("image_student"),
        "text_encoder": payload.get("text_encoder"),
        "checkpoint_epoch": payload.get("checkpoint_epoch"),
        "num_rows": payload.get("num_rows"),
        "candidate_pool_size": payload.get("candidate_pool_size"),
        "i2t_r1": payload["image_to_text"]["R@1"],
        "i2t_r5": payload["image_to_text"]["R@5"],
        "i2t_r10": payload["image_to_text"]["R@10"],
        "i2t_median_rank": payload["image_to_text"]["MedianRank"],
        "i2t_mean_rank": payload["image_to_text"]["MeanRank"],
        "t2i_r1": payload["text_to_image"]["R@1"],
        "t2i_r5": payload["text_to_image"]["R@5"],
        "t2i_r10": payload["text_to_image"]["R@10"],
        "t2i_median_rank": payload["text_to_image"]["MedianRank"],
        "t2i_mean_rank": payload["text_to_image"]["MeanRank"],
    }


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    metric_paths = sorted(eval_root.rglob("retrieval_metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"No retrieval_metrics.json files found under {eval_root}")

    rows: list[dict[str, Any]] = []
    for path in metric_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row_from_metrics(path, payload))

    rows.sort(key=lambda row: (str(row["image_student"]), int(row["candidate_pool_size"]), str(row["run"])))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
