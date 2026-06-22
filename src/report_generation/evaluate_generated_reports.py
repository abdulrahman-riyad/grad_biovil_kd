import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    return " ".join(str(text).lower().split())


def tokens(text: object) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def ngrams(items: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(items[i : i + n]) for i in range(max(0, len(items) - n + 1)))


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref = tokens(reference)
    pred = tokens(prediction)
    if not ref or not pred:
        return 0.0

    dp = [[0] * (len(pred) + 1) for _ in range(len(ref) + 1)]
    for i, ref_token in enumerate(ref, start=1):
        for j, pred_token in enumerate(pred, start=1):
            if ref_token == pred_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_n_f1(reference: str, prediction: str, n: int) -> float:
    ref_counts = ngrams(tokens(reference), n)
    pred_counts = ngrams(tokens(prediction), n)
    if not ref_counts or not pred_counts:
        return 0.0

    overlap = sum((ref_counts & pred_counts).values())
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_1(reference: str, prediction: str) -> float:
    ref_tokens = tokens(reference)
    pred_tokens = tokens(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum((ref_counts & pred_counts).values())
    precision = overlap / len(pred_tokens)
    brevity = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens)))
    return brevity * precision


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "std": float(arr.std()) if len(arr) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated radiology text against references.")
    parser.add_argument("--generations-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.generations_csv)
    required = {"reference_text", "generated_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    refs = [normalize_text(value) for value in df["reference_text"]]
    preds = [normalize_text(value) for value in df["generated_text"]]
    rouge_1 = [rouge_n_f1(ref, pred, 1) for ref, pred in zip(refs, preds, strict=True)]
    rouge_2 = [rouge_n_f1(ref, pred, 2) for ref, pred in zip(refs, preds, strict=True)]
    rouge_l = [rouge_l_f1(ref, pred) for ref, pred in zip(refs, preds, strict=True)]
    bleu = [bleu_1(ref, pred) for ref, pred in zip(refs, preds, strict=True)]
    pred_lengths = [len(tokens(pred)) for pred in preds]

    metrics: dict[str, Any] = {
        "num_rows": int(len(df)),
        "empty_generation_rate": float(np.mean([length == 0 for length in pred_lengths])) if pred_lengths else 0.0,
        "generated_token_length": summarize([float(length) for length in pred_lengths]),
        "rouge_1_f1": summarize(rouge_1),
        "rouge_2_f1": summarize(rouge_2),
        "rouge_l_f1": summarize(rouge_l),
        "bleu_1": summarize(bleu),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
