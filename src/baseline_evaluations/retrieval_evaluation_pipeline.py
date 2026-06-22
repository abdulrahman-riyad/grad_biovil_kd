# Auto-exported from project notebook.
# Source notebook: project baseline notebooks/retrival-evaluation-pipeline.ipynb
# Code cells: 11; markdown cells: 12
# Notebook shell commands and magics are preserved as comments.
# ruff: noqa
# pylint: skip-file

# %% [markdown] cell 1
# # Retrieval Evaluation Pipeline (Auto-Discovery Version)
#
# This notebook **auto-discovers** every model/variant embedding pair under an
# `embeddings/` root folder and produces one comparison `DataFrame` of
# image↔text retrieval metrics across all of them.
#
# **Assumed folder layout:**
#
# ```
# embeddings/
# ├── biovil/                     <- model family
# │   ├── teacher/                <- variant
# │   │   ├── image_embeddings.npy
# │   │   └── text_embeddings.npy
# │   └── distilled_v2/
# │       ├── image_embeddings.npy
# │       └── text_embeddings.npy
# ├── mobilevit/
# │   └── student/
# │       ├── img_emb.npy
# │       └── txt_emb.npy
# └── resnet/
#     └── student/
#         ├── ...image...npy
#         └── ...text...npy
# ```
#
# i.e. `embeddings/<model_family>/<variant>/` containing exactly two embedding
# files (one image, one text). Naming inside each variant folder does **not**
# need to be consistent across folders — files are matched by keyword
# (`image_keywords` / `text_keywords` in `CONFIG` below).
#
# Run the cells top to bottom:
# 1. Print the real folder tree (so you can sanity-check the structure).
# 2. Auto-discover every `<model_family>/<variant>` pair and show a paths table.
# 3. Load + L2-normalize embeddings, compute retrieval metrics, and build the
#    final comparison `DataFrame`.
#
# If your files don't have a `pair_id`-style metadata CSV, matching falls back
# to **row order** (row *i* of the image file is assumed to correspond to row
# *i* of the text file). If a variant folder contains a `.csv` file, it's
# picked up automatically and used for ID-based matching instead.

# %% [markdown] cell 2
# ## Environment Setup
# Install the required packages before running this notebook: `torch`, `numpy`, `pandas`, `tqdm`.

# %% code cell 3
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from IPython.display import display

# %% [markdown] cell 4
# ## Configuration

# %% code cell 5
PROJECT_ROOT = Path.cwd()

# Point this at your real "embeddings" folder. You can also set the
# RETRIEVAL_EMBEDDINGS_DIR environment variable instead of editing this line.
EMBEDDINGS_ROOT = "/kaggle/input/datasets/mohamed311ahmed/embeddings/embeddings"

OUTPUT_DIR = Path(
    os.getenv("RETRIEVAL_OUTPUT_DIR", PROJECT_ROOT / "evaluation_outputs")
).expanduser()

CONFIG = {
    "seed": 42,
    "embeddings_root": EMBEDDINGS_ROOT,
    "output_dir": OUTPUT_DIR,
    "top_k": [1, 5, 10],
    "num_failure_examples": 10,

    # Recognized embedding file extensions.
    "embedding_extensions": [".npy", ".pt", ".pth"],

    # Keywords (case-insensitive, matched anywhere in the filename) used to
    # tell the image-embedding file apart from the text-embedding file inside
    # each variant folder. Add to these lists if your files use different
    # naming (e.g. "vision", "caption", "report").
    "image_keywords": ["image", "img", "visual", "vision"],
    "text_keywords": ["text", "txt", "caption", "report", "language"],

    # Metadata column used to match an image row to its text row, used only
    # if a .csv file is found inside a variant folder.
    "matching_column": "pair_id",
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

print(f"Embeddings root: {CONFIG['embeddings_root']}")
print(f"Evaluation outputs: {CONFIG['output_dir']}")

# %% [markdown] cell 6
# ## 1. Explore the folder structure
# Prints the real tree under `embeddings_root` so you can confirm the layout
# matches the assumption above before anything else runs.

# %% code cell 7
def print_tree(root, max_depth=4, _prefix="", _depth=0):
    root = Path(root)
    if _depth == 0:
        print(f"{root}/")
    if _depth >= max_depth:
        return
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except FileNotFoundError:
        print(f"{_prefix}!! folder not found !!")
        return
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(f"{_prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, max_depth=max_depth, _prefix=_prefix + extension, _depth=_depth + 1)

print_tree(CONFIG["embeddings_root"], max_depth=4)

# %% [markdown] cell 8
# ## 2. Auto-discover model variants
# Walks `embeddings_root` **recursively to any depth** and treats every folder
# that directly contains exactly two embedding files as one "model variant"
# leaf. This handles mixed layouts in the same tree, e.g.:
#
# ```
# embeddings/
# ├── BiomedCLIP/                          <- files directly here, no variant subfolder
# │   ├── BiomedCLIP_image_embeddings.pt
# │   └── BiomedCLIP_text_embeddings.pt
# ├── CLIP/
# │   ├── CLIP_ViT_B14/                    <- one variant subfolder per leaf
# │   │   ├── CLIP_ViT_B14_..._image_embeddings.pt
# │   │   └── CLIP_ViT_B14_..._text_embeddings.pt
# │   └── CLIP_ViT_B16/...
# ```
#
# Both `BiomedCLIP` (2 levels deep) and `CLIP/CLIP_ViT_B14` (3 levels deep) are
# discovered the same way — the folder's path relative to `embeddings_root`
# becomes the `ModelID`, the first path component becomes `ModelFamily`, and
# anything after that becomes `Variant` (`"(base)"` if there's no variant
# subfolder).

# %% code cell 9
def classify_embedding_files(files, image_keywords, text_keywords):
    """
    Classify a folder's embedding files into (image_file, text_file) using
    filename keywords. Falls back to alphabetical order if exactly two files
    are present and neither matched a keyword (flagged via 'NamingVerified'
    in the paths table so you can double check it).
    """
    image_candidates = [f for f in files if any(k in f.name.lower() for k in image_keywords)]
    text_candidates = [f for f in files if any(k in f.name.lower() for k in text_keywords)]

    image_file = image_candidates[0] if len(image_candidates) == 1 else None
    text_file = text_candidates[0] if len(text_candidates) == 1 else None
    keyword_matched = image_file is not None and text_file is not None

    if not keyword_matched and len(files) == 2:
        sorted_files = sorted(files, key=lambda f: f.name.lower())
        image_file, text_file = sorted_files[0], sorted_files[1]

    return image_file, text_file, keyword_matched


def discover_embedding_variants(embeddings_root, config):
    """
    Recursively scans every folder under `embeddings_root`. Any folder that
    directly contains exactly two recognized embedding files is registered
    as one model variant, regardless of nesting depth.

    Returns:
        discovered: dict keyed by ModelID (path relative to embeddings_root) -> model_config
        skipped: list of (folder_path, reason) for folders that looked like
                 leaves but couldn't be parsed (wrong file count / ambiguous naming)
    """
    embeddings_root = Path(embeddings_root)
    if not embeddings_root.exists():
        raise FileNotFoundError(f"Embeddings root does not exist: {embeddings_root}")

    extensions = set(config["embedding_extensions"])
    discovered = {}
    skipped = []

    for dirpath, dirnames, filenames in os.walk(embeddings_root):
        dirnames.sort()
        dirpath = Path(dirpath)
        all_files = [dirpath / fn for fn in filenames]
        embedding_files = [f for f in all_files if f.suffix.lower() in extensions]

        if len(embedding_files) == 0:
            # Likely an intermediate folder (e.g. "CLIP/") with no files of
            # its own — nothing to flag, just keep walking its subfolders.
            continue

        if len(embedding_files) != 2:
            skipped.append((dirpath, f"found {len(embedding_files)} embedding file(s), expected exactly 2"))
            continue

        metadata_candidates = [f for f in all_files if f.suffix.lower() == ".csv"]
        image_file, text_file, keyword_matched = classify_embedding_files(
            embedding_files, config["image_keywords"], config["text_keywords"],
        )

        if image_file is None or text_file is None:
            skipped.append((
                dirpath, f"could not tell image vs. text apart among: {[f.name for f in embedding_files]}",
            ))
            continue

        rel_parts = dirpath.relative_to(embeddings_root).parts
        model_id = "/".join(rel_parts) if rel_parts else dirpath.name
        model_family = rel_parts[0] if rel_parts else dirpath.name
        variant = "/".join(rel_parts[1:]) if len(rel_parts) > 1 else "(base)"

        discovered[model_id] = {
            "display_name": model_id,
            "model_family": model_family,
            "variant": variant,
            "image_embeddings": image_file,
            "text_embeddings": text_file,
            "metadata": metadata_candidates[0] if len(metadata_candidates) == 1 else None,
            "naming_verified": keyword_matched,
        }

    return discovered, skipped


discovered_models, skipped_folders = discover_embedding_variants(CONFIG["embeddings_root"], CONFIG)

print(f"Discovered {len(discovered_models)} model variant(s).")
if skipped_folders:
    print(f"\nSkipped {len(skipped_folders)} folder(s):")
    for folder, reason in skipped_folders:
        print(f"  - {folder}: {reason}")

# %% [markdown] cell 10
# ## 3. Paths overview
# This is the "give me all the embedding paths" table — review it before
# running the (heavier) evaluation step. Pay special attention to any row with
# `NamingVerified = False`: those were guessed alphabetically because neither
# filename matched an image/text keyword.

# %% code cell 11
def build_paths_dataframe(discovered_models):
    rows = []
    for model_id, cfg in discovered_models.items():
        rows.append({
            "ModelID": model_id,
            "ModelFamily": cfg["model_family"],
            "Variant": cfg["variant"],
            "ImageEmbeddingsPath": str(cfg["image_embeddings"]),
            "TextEmbeddingsPath": str(cfg["text_embeddings"]),
            "MetadataPath": str(cfg["metadata"]) if cfg["metadata"] else "(none - using row order)",
            "NamingVerified": cfg["naming_verified"],
        })
    return pd.DataFrame(rows).sort_values(["ModelFamily", "Variant"]).reset_index(drop=True)

paths_df = build_paths_dataframe(discovered_models)
display(paths_df)

paths_output_path = CONFIG["output_dir"] / "discovered_embedding_paths.csv"
paths_df.to_csv(paths_output_path, index=False)
print(f"\nSaved discovered paths to: {paths_output_path}")

# %% [markdown] cell 12
# ## 4. Loading & normalizing embeddings

# %% code cell 13
def load_embedding_file(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    elif path.suffix.lower() in {".pt", ".pth"}:
        tensor = torch.load(path, map_location="cpu")
        array = tensor.numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    else:
        raise ValueError(f"Unsupported embedding file type: {path}")
    return array


def normalize_embeddings(embeddings):
    """L2-normalize embeddings."""
    embeddings = torch.tensor(np.asarray(embeddings), dtype=torch.float32)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def load_model_data(model_config):
    image_embeddings = load_embedding_file(model_config["image_embeddings"])
    text_embeddings = load_embedding_file(model_config["text_embeddings"])

    if len(image_embeddings) != len(text_embeddings):
        raise ValueError(
            "The number of image embeddings and text embeddings must match. "
            f"Got {len(image_embeddings)} images and {len(text_embeddings)} texts "
            f"for '{model_config['display_name']}'."
        )

    metadata = None
    if model_config.get("metadata") is not None:
        metadata = pd.read_csv(model_config["metadata"]).copy()
        if len(metadata) != len(image_embeddings):
            raise ValueError(
                "Metadata row count must match the number of embeddings for "
                f"'{model_config['display_name']}'. Got {len(metadata)} metadata rows "
                f"and {len(image_embeddings)} embeddings."
            )

    return {
        "image_embeddings": normalize_embeddings(image_embeddings),
        "text_embeddings": normalize_embeddings(text_embeddings),
        "metadata": metadata,
    }

# %% [markdown] cell 14
# ## 5. Similarity & retrieval metrics
# Bidirectional retrieval is evaluated for every model variant:
# - **Image→Text (I2T)**
# - **Text→Image (T2I)**
#
# Reported per direction: `R@1`, `R@5`, `R@10`, `MedianRank`, `MeanRank`, `MRR`,
# plus `Avg_*` scores averaged across both directions.

# %% code cell 15
def compute_similarity_matrix(image_embeddings, text_embeddings):
    """Cosine similarity matrix, shape [num_images, num_texts]."""
    return image_embeddings @ text_embeddings.T


def resolve_matching_ids(num_rows, metadata, matching_column):
    """
    Resolve the IDs used to pair an image row with its text row.
    Uses `matching_column` from metadata if present, otherwise falls back to
    row order (row i image <-> row i text).
    """
    if metadata is not None and matching_column in metadata.columns:
        ids = metadata[matching_column].astype(str).to_numpy()
        return ids, matching_column

    ids = np.arange(num_rows).astype(str)
    return ids, "__row_order__"


def compute_ranks(similarity_matrix, query_ids, candidate_ids):
    """Rank of the first correct candidate for every query."""
    similarity_matrix = similarity_matrix.cpu()
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    candidate_ids = np.asarray(candidate_ids)

    ranks = []
    for row_index, query_id in enumerate(query_ids):
        ranked_candidate_ids = candidate_ids[sorted_indices[row_index].numpy()]
        matching_positions = np.where(ranked_candidate_ids == query_id)[0]
        if len(matching_positions) == 0:
            raise ValueError(f"No matching candidate was found for query ID '{query_id}'.")
        ranks.append(int(matching_positions[0]) + 1)

    return np.array(ranks)


def compute_recall_at_k(ranks, k):
    return float(np.mean(ranks <= k))


def compute_mrr(ranks):
    return float(np.mean(1.0 / ranks))


def compute_metrics(ranks, top_k=(1, 5, 10)):
    metrics = {}
    for k in top_k:
        metrics[f"R@{k}"] = compute_recall_at_k(ranks, k)
    metrics["MedianRank"] = float(np.median(ranks))
    metrics["MeanRank"] = float(np.mean(ranks))
    metrics["MRR"] = compute_mrr(ranks)
    return metrics


def prefix_metrics(metrics, prefix):
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def evaluate_retrieval_direction(similarity_matrix, query_ids, candidate_ids, top_k, prefix):
    ranks = compute_ranks(similarity_matrix, query_ids=query_ids, candidate_ids=candidate_ids)
    metrics = compute_metrics(ranks, top_k=top_k)
    return {"ranks": ranks, "metrics": metrics, "prefixed_metrics": prefix_metrics(metrics, prefix)}


def compute_average_metrics(i2t_metrics, t2i_metrics):
    average_metrics = {}
    shared_metric_names = sorted(set(i2t_metrics) & set(t2i_metrics))
    for name in shared_metric_names:
        average_metrics[f"Avg_{name}"] = float((i2t_metrics[name] + t2i_metrics[name]) / 2.0)
    return average_metrics

# %% [markdown] cell 16
# ## 6. Evaluate one model variant

# %% code cell 17
def evaluate_model(model_id, model_config, config):
    display_name = model_config.get("display_name", model_id)
    print(f"\nEvaluating: {display_name}")

    data = load_model_data(model_config)
    image_embeddings = data["image_embeddings"]
    text_embeddings = data["text_embeddings"]
    metadata = data["metadata"]

    match_ids, match_source = resolve_matching_ids(
        num_rows=len(image_embeddings),
        metadata=metadata,
        matching_column=config["matching_column"],
    )

    if match_source == "__row_order__":
        print("  matching: row order (no usable metadata column found)")
    else:
        print(f"  matching: metadata column '{match_source}'")

    similarity_matrix = compute_similarity_matrix(image_embeddings, text_embeddings)

    i2t_result = evaluate_retrieval_direction(
        similarity_matrix=similarity_matrix,
        query_ids=match_ids, candidate_ids=match_ids,
        top_k=config["top_k"], prefix="I2T",
    )
    t2i_result = evaluate_retrieval_direction(
        similarity_matrix=similarity_matrix.T,
        query_ids=match_ids, candidate_ids=match_ids,
        top_k=config["top_k"], prefix="T2I",
    )

    average_metrics = compute_average_metrics(i2t_result["metrics"], t2i_result["metrics"])

    metrics = {}
    metrics.update(i2t_result["prefixed_metrics"])
    metrics.update(t2i_result["prefixed_metrics"])
    metrics.update(average_metrics)

    return {
        "model_id": model_id,
        "display_name": display_name,
        "model_family": model_config["model_family"],
        "variant": model_config["variant"],
        "matching_source": match_source,
        "match_ids": match_ids,
        "metrics": metrics,
        "directional_results": {"I2T": i2t_result, "T2I": t2i_result},
        "similarity_matrix": similarity_matrix,
        "num_pairs": len(image_embeddings),
    }

# %% [markdown] cell 18
# ## 7. Compare all discovered models

# %% code cell 19
def compare_models(discovered_models, config):
    results = {}
    for model_id, model_config in tqdm(discovered_models.items(), desc="Evaluating models"):
        results[model_id] = evaluate_model(model_id, model_config, config)

    comparison_rows = []
    for model_id, result in results.items():
        row = {
            "ModelID": model_id,
            "ModelFamily": result["model_family"],
            "Variant": result["variant"],
            "NumPairs": result["num_pairs"],
            "MatchingSource": result["matching_source"],
        }
        row.update(result["metrics"])
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values(
        by=["Avg_MRR", "Avg_R@1"], ascending=[False, False]
    ).reset_index(drop=True)

    return results, comparison_df

# %% [markdown] cell 20
# ## 8. (Optional) Failure analysis
# For each model/direction, surfaces the worst-ranked queries — useful for
# spotting systematic failure patterns per variant.

# %% code cell 21
def build_failure_analysis(results, num_examples=10):
    failure_rows = []
    for result in results.values():
        similarity_matrix = result["similarity_matrix"].cpu()
        match_ids = np.asarray(result["match_ids"])

        for direction_name, direction_result in result["directional_results"].items():
            ranks = direction_result["ranks"]
            score_matrix = similarity_matrix if direction_name == "I2T" else similarity_matrix.T

            failure_indices = np.where(ranks > 1)[0]
            ranked_failures = failure_indices[np.argsort(ranks[failure_indices])[::-1]]
            worst_indices = ranked_failures[:num_examples]

            for query_index in worst_indices:
                ranking = torch.argsort(score_matrix[query_index], descending=True).numpy()
                top_prediction_index = int(ranking[0])
                failure_rows.append({
                    "ModelID": result["model_id"],
                    "Direction": direction_name,
                    "QueryIndex": int(query_index),
                    "QueryID": match_ids[query_index],
                    "CorrectRank": int(ranks[query_index]),
                    "TopPredictionID": match_ids[top_prediction_index],
                    "TopScore": float(score_matrix[query_index, top_prediction_index]),
                })

    failure_df = pd.DataFrame(failure_rows)
    if not failure_df.empty:
        failure_df = failure_df.sort_values(
            by=["CorrectRank", "TopScore"], ascending=[False, False]
        ).reset_index(drop=True)
    return failure_df

# %% [markdown] cell 22
# ## 9. Run everything
# Discovers every variant, evaluates it, builds the comparison `DataFrame`, and
# saves all outputs as CSVs in `CONFIG["output_dir"]`.

# %% code cell 23
if not discovered_models:
    raise RuntimeError(
        "No model variants were discovered. Check CONFIG['embeddings_root'] and the "
        "folder tree printed in step 1, then adjust 'image_keywords'/'text_keywords' "
        "in CONFIG if your filenames use different naming."
    )

results, comparison_df = compare_models(discovered_models, CONFIG)
failure_analysis_df = build_failure_analysis(results, num_examples=CONFIG["num_failure_examples"])

comparison_output_path = CONFIG["output_dir"] / "retrieval_comparison.csv"
failure_analysis_output_path = CONFIG["output_dir"] / "failure_analysis.csv"

comparison_df.to_csv(comparison_output_path, index=False)
failure_analysis_df.to_csv(failure_analysis_output_path, index=False)

print("\nComparison table:")
display(comparison_df)

print("\nFailure analysis preview:")
display(failure_analysis_df.head(CONFIG["num_failure_examples"]))

print(f"\nSaved comparison table to: {comparison_output_path}")
print(f"Saved failure analysis table to: {failure_analysis_output_path}")
