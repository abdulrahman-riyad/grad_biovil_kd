# Auto-exported from project notebook.
# Source notebook: week4_structured_project/Final_ppt/materials/new_teammates_baselines_notebooks/medclip-baseline.ipynb
# Code cells: 9; markdown cells: 9
# Notebook shell commands and magics are preserved as comments.
# ruff: noqa
# pylint: skip-file

# %% [markdown] cell 1
# # MedCLIP test embedding baseline
#
# This notebook is rebuilt for Kaggle/Python 3.12. It extracts MedCLIP embeddings for the test split only, saves the embeddings/metadata, and reports paired cosine, retrieval recall, ranks, and diagnostic statistics.
#
# Run from the first cell after restarting the kernel.

# %% [markdown] cell 2
# ## 1. Install package

# %% code cell 3
import importlib.util
import subprocess
import sys

# Reinstall only the small medclip package so previous notebook patches in the Kaggle runtime do not leak in.
# --no-deps avoids changing Kaggle's Torch / CUDA / RAPIDS stack.
subprocess.check_call([
    sys.executable,
    '-m',
    'pip',
    'install',
    '-q',
    '--force-reinstall',
    '--no-deps',
    'medclip',
])

if importlib.util.find_spec('wget') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'wget'])

print('Install step complete.')

# %% [markdown] cell 4
# ## 2. Imports, compatibility patch, and config

# %% code cell 5
import ast
import importlib.util
import os
import random
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display
from PIL import Image, ImageOps
from tqdm.auto import tqdm

import transformers
from transformers import AutoTokenizer, CLIPImageProcessor


# medclip imports CLIPFeatureExtractor from transformers, but current transformers removed it.
# We only need this so medclip can import cleanly; image preprocessing below is explicit and deterministic.
transformers.CLIPFeatureExtractor = CLIPImageProcessor


def patch_medclip_package() -> None:
    spec = importlib.util.find_spec('medclip')
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError('medclip is not installed. Run the install cell first.')

    package_dir = Path(next(iter(spec.submodule_search_locations)))

    dataset_py = package_dir / 'dataset.py'
    if not dataset_py.exists():
        raise FileNotFoundError(f'Could not find medclip dataset module at: {dataset_py}')
    dataset_original = dataset_py.read_text()
    dataset_patched = dataset_original.replace(
        'from transformers import CLIPFeatureExtractor, CLIPProcessor',
        'from transformers import CLIPImageProcessor as CLIPFeatureExtractor, CLIPProcessor',
    )
    dataset_patched = dataset_patched.replace(
        'splitter = re.compile("[0-9]+\\.+[^0-9]")',
        'splitter = re.compile(r"[0-9]+\\.+[^0-9]")',
    )
    if dataset_patched != dataset_original:
        dataset_py.write_text(dataset_patched)
        print(f'Patched {dataset_py} for current transformers imports.')
    else:
        print('MedCLIP dataset patch already applied or not needed.')

    modeling_py = package_dir / 'modeling_medclip.py'
    if not modeling_py.exists():
        raise FileNotFoundError(f'Could not find medclip modeling module at: {modeling_py}')

    original = modeling_py.read_text()
    patched = original

    # Current torch/transformers checkpoints can include harmless position_ids keys.
    patched = patched.replace('        self.load_state_dict(state_dict)', '        self.load_state_dict(state_dict, strict=False)')

    # The package hard-codes .cuda() in a few places. Make it follow the model device instead.
    cuda_replacements = {
        'input_ids = input_ids.cuda()': 'input_ids = input_ids.to(next(self.parameters()).device)',
        'attention_mask = attention_mask.cuda()': 'attention_mask = attention_mask.to(next(self.parameters()).device)',
        'pixel_values = pixel_values.cuda()': 'pixel_values = pixel_values.to(next(self.parameters()).device)',
        'pixel_values.cuda()': 'pixel_values.to(next(self.parameters()).device)',
    }
    for old, new in cuda_replacements.items():
        patched = patched.replace(old, new)

    if patched != original:
        modeling_py.write_text(patched)
        print(f'Patched {modeling_py} for Kaggle compatibility.')
    else:
        print('MedCLIP modeling patch already applied or not needed.')


patch_medclip_package()

from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip import constants


SEED = 42
VISION_BACKBONE = 'vit'  # 'vit' or 'resnet'
TEXT_BATCH_SIZE = 64
IMAGE_BATCH_SIZE = 64
MAX_ROWS = None  # Set to a small integer, e.g. 100, for a smoke test.

# Update this path to the Kaggle dataset mount containing your pickle.
TEST_DATASET_PATH = Path('/kaggle/input/datasets/shahdammar/distillationdataset-train-val-test/test_df.pkl')

# Add the roots where your image files may live on Kaggle.
IMAGE_ROOT_CANDIDATES = [
    Path('/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files'),
    Path('/kaggle/input/mimic-cxr-dataset/official_data_iccv_final/files'),
    Path('/kaggle/input'),
]

TEXT_COLUMN_CANDIDATES = ('report_text', 'raw_report_text', 'text', 'findings', 'impression')
IMAGE_COLUMN = 'image_paths'
EMBEDDING_TEXT_COLUMN = 'embedding_text'
TEXT_PREPROCESSING_MODE = 'impression_then_findings'  # 'full' or 'impression_then_findings'
IMAGE_AGGREGATION_MODE = 'first'  # 'first' or 'mean'

OUTPUT_DIR = Path('/kaggle/working')
OUTPUT_PREFIX = 'medclip_vit_mimic_cxr_test'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Transformers: {transformers.__version__}')
print(f'Torch: {torch.__version__}')
print(f'MedCLIP constants BERT_TYPE: {constants.BERT_TYPE}')
print(f'MedCLIP image mean/std: {constants.IMG_MEAN} / {constants.IMG_STD}')

# %% [markdown] cell 6
# ## 3. Dataset helpers

# %% code cell 7
def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    major, minor = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    print(f'CUDA device: {gpu_name} sm_{major}{minor}')
    return torch.device('cuda')


def validate_test_dataset_path(path: Path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            'TEST_DATASET_PATH does not exist. Update it in the config cell. Current value:\n'
            f'{path}'
        )


def load_test_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Test pickle must contain a pandas DataFrame. Got: {type(df)}')
    return df.copy()


def select_text_column(df: pd.DataFrame) -> str:
    for column in TEXT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise KeyError(f'No text column found. Expected one of {TEXT_COLUMN_CANDIDATES}. Found: {list(df.columns)}')


def normalize_text(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ''
    text = str(text).replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def extract_report_sections(text: str) -> dict:
    text = normalize_text(text)
    if not text:
        return {}
    pattern = re.compile(
        r'(?is)(impression|findings)\s*:\s*(.*?)(?=(?:impression|findings|conclusion|recommendation|history|indication)\s*:|$)'
    )
    sections = {}
    for match in pattern.finditer(text):
        name = match.group(1).lower()
        value = normalize_text(match.group(2))
        if value:
            sections[name] = value
    return sections


def build_embedding_text(row: pd.Series, selected_text_column: str) -> str:
    candidates = []
    for column in ('raw_report_text', selected_text_column, 'impression', 'findings'):
        if column in row:
            value = normalize_text(row[column])
            if value and value not in candidates:
                candidates.append(value)

    for candidate in candidates:
        if TEXT_PREPROCESSING_MODE == 'impression_then_findings':
            sections = extract_report_sections(candidate)
            if sections.get('impression'):
                return sections['impression']
            if sections.get('findings'):
                return sections['findings']
        if candidate:
            return candidate
    return ''


def clean_image_paths(value) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, Path):
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        posix_matches = re.findall(r'PosixPath\(["\'](.*?)["\']\)', value)
        if posix_matches:
            return posix_matches
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return [value]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed if str(item).strip()]
        return [str(parsed)]
    return [str(value)]


def candidate_suffixes(path_str: str) -> List[Path]:
    raw = Path(path_str)
    suffixes = []
    if not raw.is_absolute():
        suffixes.append(raw)

    parts = raw.parts
    for marker in ('files', 'official_data_iccv_final', 'mimic-cxr-dataset', 'physionet.org'):
        if marker in parts:
            idx = parts.index(marker)
            if marker == 'mimic-cxr-dataset':
                idx += 1
            suffixes.append(Path(*parts[idx:]))

    for tail_size in range(2, min(len(parts), 10) + 1):
        suffixes.append(Path(*parts[-tail_size:]))

    deduped = []
    seen = set()
    for suffix in suffixes:
        key = str(suffix)
        if key and key != '.' and key not in seen:
            deduped.append(suffix)
            seen.add(key)
    return deduped


def resolve_single_image_path(path_str: str):
    raw = Path(path_str)
    if raw.exists():
        return str(raw)
    for root in IMAGE_ROOT_CANDIDATES:
        if not root.exists():
            continue
        for suffix in candidate_suffixes(path_str):
            candidate = root / suffix
            if candidate.exists():
                return str(candidate)
    return None


def resolve_image_path_list(paths: Sequence[str]) -> List[str]:
    resolved = []
    for path_str in paths:
        resolved_path = resolve_single_image_path(path_str)
        if resolved_path is not None:
            resolved.append(resolved_path)
    return resolved


DEVICE = select_device()
print(f'Using device: {DEVICE}')

# %% [markdown] cell 8
# ## 4. Load and prepare test split

# %% code cell 9
validate_test_dataset_path(TEST_DATASET_PATH)
test_df = load_test_dataset(TEST_DATASET_PATH)
TEXT_COLUMN = select_text_column(test_df)

if IMAGE_COLUMN not in test_df.columns:
    raise KeyError(f'Test dataframe is missing required image column: {IMAGE_COLUMN}')

print(f'Loaded test rows: {len(test_df)}')
print(f'Using text column: {TEXT_COLUMN}')
display(test_df.head())

# %% code cell 10
test_df = test_df.copy()
test_df[IMAGE_COLUMN] = test_df[IMAGE_COLUMN].apply(clean_image_paths)
test_df['resolved_image_paths'] = test_df[IMAGE_COLUMN].apply(resolve_image_path_list)
test_df[TEXT_COLUMN] = test_df[TEXT_COLUMN].fillna('').astype(str).str.strip()
test_df[EMBEDDING_TEXT_COLUMN] = test_df.apply(lambda row: build_embedding_text(row, TEXT_COLUMN), axis=1)

missing_text_mask = test_df[EMBEDDING_TEXT_COLUMN].eq('')
missing_images_mask = test_df['resolved_image_paths'].map(len).eq(0)
usable_mask = (~missing_text_mask) & (~missing_images_mask)
ready_test_df = test_df.loc[usable_mask].reset_index(drop=True)

if MAX_ROWS is not None:
    ready_test_df = ready_test_df.head(MAX_ROWS).copy()

prep_summary_df = pd.DataFrame([
    {
        'Split': 'test',
        'InputRows': len(test_df),
        'RowsReady': len(ready_test_df),
        'EmptyTextRows': int(missing_text_mask.sum()),
        'UnresolvedImageRows': int(missing_images_mask.sum()),
        'MaxRowsApplied': MAX_ROWS if MAX_ROWS is not None else 'None',
    }
])
display(prep_summary_df)

if ready_test_df.empty:
    print('Examples with unresolved images:')
    display(test_df.loc[missing_images_mask, [IMAGE_COLUMN]].head(10))
    raise RuntimeError('No usable rows remain. Fix TEST_DATASET_PATH / IMAGE_ROOT_CANDIDATES / text columns before embedding.')

print('Preview of rows used for embedding:')
display(ready_test_df[[EMBEDDING_TEXT_COLUMN, 'resolved_image_paths']].head())

# %% [markdown] cell 11
# ## 5. Load MedCLIP model and define preprocessing

# %% code cell 12
def load_medclip_model(vision_backbone: str = VISION_BACKBONE, device: torch.device = DEVICE):
    if vision_backbone.lower() == 'vit':
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    elif vision_backbone.lower() == 'resnet':
        model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
    else:
        raise ValueError("VISION_BACKBONE must be either 'vit' or 'resnet'.")

    try:
        model.from_pretrained()
    except ModuleNotFoundError as exc:
        if exc.name == 'wget':
            raise RuntimeError("medclip needs the 'wget' package to download pretrained weights. Rerun the install cell.") from exc
        raise
    except Exception as exc:
        raise RuntimeError(
            'Failed to load MedCLIP pretrained weights. On Kaggle, enable Internet or attach/preload the weights.'
        ) from exc

    model = model.to(device)
    model.eval()
    return model


def load_medclip_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
    tokenizer.model_max_length = 77
    return tokenizer


def infer_required_image_channels(model) -> int:
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.weight.shape[1])
    return 3


def scalar_or_sequence(values, channels: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == channels:
        return array
    if array.size == 1:
        return np.repeat(array, channels)
    if channels == 1:
        return np.asarray([float(np.mean(array))], dtype=np.float32)
    return np.resize(array, channels).astype(np.float32)


def pad_square(image: Image.Image, fill=0) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    right = side - width - left
    bottom = side - height - top
    return ImageOps.expand(image, border=(left, top, right, bottom), fill=fill)


def preprocess_medclip_image(path: str, image_channels: int, image_size: int = 224) -> torch.Tensor:
    with Image.open(path) as image:
        if image_channels == 1:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
        image = pad_square(image, fill=0)
        image = image.resize((image_size, image_size), resample=Image.BICUBIC)

    array = np.asarray(image).astype(np.float32)
    if image_channels == 1:
        if array.ndim == 3:
            array = array[..., 0]
        array = array[..., None]
    elif array.ndim == 2:
        array = np.repeat(array[..., None], image_channels, axis=-1)

    if array.max() > 1.0:
        array = array / 255.0

    mean = scalar_or_sequence(constants.IMG_MEAN, array.shape[-1])
    std = scalar_or_sequence(constants.IMG_STD, array.shape[-1])
    array = (array - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    array = np.transpose(array, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(array)


def validate_preprocessing(model, tokenizer, image_channels: int) -> None:
    text_batch = tokenizer(
        ['no acute cardiopulmonary abnormality'],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=77,
    )
    missing = {'input_ids', 'attention_mask'} - set(text_batch.keys())
    if missing:
        raise KeyError(f'Tokenizer output is missing keys: {sorted(missing)}')

    dummy = Image.new('L' if image_channels == 1 else 'RGB', (96, 128), 0)
    dummy_path = OUTPUT_DIR / '_medclip_preprocess_smoke.png'
    dummy.save(dummy_path)
    pixel_values = preprocess_medclip_image(str(dummy_path), image_channels=image_channels).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = model.encode_image(pixel_values)
        text_inputs = {key: value.to(DEVICE) for key, value in text_batch.items()}
        text_features = model.encode_text(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
        )
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError('Smoke test failed: model did not return 2D feature tensors.')
    print(f'Smoke test passed. Image channels: {image_channels}; feature dim: {image_features.shape[-1]}')


model = load_medclip_model(VISION_BACKBONE, DEVICE)
tokenizer = load_medclip_tokenizer()
IMAGE_CHANNELS = infer_required_image_channels(model)
validate_preprocessing(model, tokenizer, IMAGE_CHANNELS)
print(f'Using tokenizer: {constants.BERT_TYPE}')

# %% [markdown] cell 13
# ## 6. Extract text and image embeddings

# %% code cell 14
def encode_texts(texts: List[str], model, tokenizer, batch_size: int = TEXT_BATCH_SIZE) -> torch.Tensor:
    outputs = []
    for start in tqdm(range(0, len(texts), batch_size), desc='Encoding text'):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.no_grad():
            features = model.encode_text(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
            ).cpu()
        outputs.append(features)
    return F.normalize(torch.cat(outputs, dim=0), p=2, dim=1)


def encode_image_tensor_batch(tensors: List[torch.Tensor], model, batch_size: int = IMAGE_BATCH_SIZE) -> torch.Tensor:
    outputs = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start:start + batch_size]).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(batch).cpu()
        outputs.append(features)
    return torch.cat(outputs, dim=0)


def aggregate_image_features(per_image_features: torch.Tensor) -> torch.Tensor:
    per_image_features = F.normalize(per_image_features, p=2, dim=1)
    if IMAGE_AGGREGATION_MODE == 'first':
        return per_image_features[:1]
    if IMAGE_AGGREGATION_MODE == 'mean':
        return F.normalize(per_image_features.mean(dim=0, keepdim=True), p=2, dim=1)
    raise ValueError("IMAGE_AGGREGATION_MODE must be 'first' or 'mean'.")


def encode_images_per_row(image_paths_per_row: List[List[str]], model, image_channels: int, fallback_dim: int) -> torch.Tensor:
    row_features = []
    failed_rows = []
    for row_index, paths in enumerate(tqdm(image_paths_per_row, desc='Encoding images')):
        tensors = []
        for path in paths:
            try:
                tensors.append(preprocess_medclip_image(path, image_channels=image_channels))
            except Exception as exc:
                tqdm.write(f'Row {row_index}: failed to preprocess {path}: {exc}')
        if not tensors:
            failed_rows.append(row_index)
            # Keep alignment. This row will be reported in diagnostics.
            row_features.append(torch.zeros((1, fallback_dim), dtype=torch.float32))
            continue
        per_image_features = encode_image_tensor_batch(tensors, model)
        row_features.append(aggregate_image_features(per_image_features))

    if failed_rows:
        print(f'Warning: {len(failed_rows)} rows had no readable images and received zero vectors. First examples: {failed_rows[:10]}')
    return F.normalize(torch.cat(row_features, dim=0), p=2, dim=1)


TEST_TEXT_EMB_PATH = OUTPUT_DIR / f'{OUTPUT_PREFIX}_text_embeddings.pt'
TEST_IMAGE_EMB_PATH = OUTPUT_DIR / f'{OUTPUT_PREFIX}_image_embeddings.pt'
TEST_METADATA_OUTPUT_PATH = OUTPUT_DIR / f'{OUTPUT_PREFIX}_metadata_used.csv'

texts = ready_test_df[EMBEDDING_TEXT_COLUMN].tolist()
image_paths_per_row = ready_test_df['resolved_image_paths'].tolist()

text_features = encode_texts(texts, model, tokenizer)
image_features = encode_images_per_row(image_paths_per_row, model, IMAGE_CHANNELS, fallback_dim=text_features.shape[-1])

if text_features.shape[0] != image_features.shape[0]:
    raise RuntimeError(f'Feature row mismatch: text={text_features.shape}, image={image_features.shape}')

torch.save(text_features, TEST_TEXT_EMB_PATH)
torch.save(image_features, TEST_IMAGE_EMB_PATH)
ready_test_df.to_csv(TEST_METADATA_OUTPUT_PATH, index=False)

print(f'Text embeddings: {tuple(text_features.shape)} -> {TEST_TEXT_EMB_PATH}')
print(f'Image embeddings: {tuple(image_features.shape)} -> {TEST_IMAGE_EMB_PATH}')
print(f'Metadata saved to: {TEST_METADATA_OUTPUT_PATH}')

# %% [markdown] cell 15
# ## 7. Metrics and diagnostics

# %% code cell 16
def compute_direction_ranks(query_features: torch.Tensor, candidate_features: torch.Tensor, chunk_size: int = 512) -> np.ndarray:
    query_features = F.normalize(query_features.float(), p=2, dim=1)
    candidate_features = F.normalize(candidate_features.float(), p=2, dim=1)
    if query_features.shape[0] != candidate_features.shape[0]:
        raise ValueError('Retrieval metrics require one aligned text/image pair per row.')

    ranks = []
    num_items = query_features.shape[0]
    for start in tqdm(range(0, num_items, chunk_size), desc='Computing ranks'):
        end = min(start + chunk_size, num_items)
        scores = query_features[start:end] @ candidate_features.T
        local = torch.arange(end - start)
        target = torch.arange(start, end)
        target_scores = scores[local, target].unsqueeze(1)
        ranks.append(((scores > target_scores).sum(dim=1) + 1).cpu())
    return torch.cat(ranks).numpy()


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def build_retrieval_metrics(text_features: torch.Tensor, image_features: torch.Tensor) -> pd.DataFrame:
    i2t_ranks = compute_direction_ranks(image_features, text_features)
    t2i_ranks = compute_direction_ranks(text_features, image_features)
    return pd.DataFrame([
        {
            'Direction': 'ImageToText',
            'R@1': recall_at_k(i2t_ranks, 1),
            'R@5': recall_at_k(i2t_ranks, 5),
            'R@10': recall_at_k(i2t_ranks, 10),
            'MeanRank': float(np.mean(i2t_ranks)),
            'MedianRank': float(np.median(i2t_ranks)),
        },
        {
            'Direction': 'TextToImage',
            'R@1': recall_at_k(t2i_ranks, 1),
            'R@5': recall_at_k(t2i_ranks, 5),
            'R@10': recall_at_k(t2i_ranks, 10),
            'MeanRank': float(np.mean(t2i_ranks)),
            'MedianRank': float(np.median(t2i_ranks)),
        },
    ])


def build_similarity_diagnostics(text_features: torch.Tensor, image_features: torch.Tensor, chunk_size: int = 512) -> pd.DataFrame:
    text_features = F.normalize(text_features.float(), p=2, dim=1)
    image_features = F.normalize(image_features.float(), p=2, dim=1)
    num_items = text_features.shape[0]

    paired_scores = []
    random_negative_scores = []
    hardest_negative_scores = []

    for start in tqdm(range(0, num_items, chunk_size), desc='Computing diagnostics'):
        end = min(start + chunk_size, num_items)
        scores = image_features[start:end] @ text_features.T
        local = torch.arange(end - start)
        target = torch.arange(start, end)
        paired_scores.append(scores[local, target].cpu())

        if num_items > 1:
            random_target = (target + 1) % num_items
            random_negative_scores.append(scores[local, random_target].cpu())
            scores[local, target] = -float('inf')
            hardest_negative_scores.append(scores.max(dim=1).values.cpu())

    paired = torch.cat(paired_scores).numpy()
    random_neg = torch.cat(random_negative_scores).numpy() if random_negative_scores else np.full_like(paired, np.nan)
    hardest_neg = torch.cat(hardest_negative_scores).numpy() if hardest_negative_scores else np.full_like(paired, np.nan)

    rows = [
        {'Metric': 'Rows', 'Value': float(num_items)},
        {'Metric': 'PairedMeanCosine', 'Value': float(np.mean(paired))},
        {'Metric': 'PairedMedianCosine', 'Value': float(np.median(paired))},
        {'Metric': 'PairedStdCosine', 'Value': float(np.std(paired))},
        {'Metric': 'PairedMinCosine', 'Value': float(np.min(paired))},
        {'Metric': 'PairedMaxCosine', 'Value': float(np.max(paired))},
        {'Metric': 'PairedP05Cosine', 'Value': float(np.percentile(paired, 5))},
        {'Metric': 'PairedP95Cosine', 'Value': float(np.percentile(paired, 95))},
        {'Metric': 'RandomNegativeMeanCosine', 'Value': float(np.nanmean(random_neg))},
        {'Metric': 'HardestNegativeMeanCosine', 'Value': float(np.nanmean(hardest_neg))},
        {'Metric': 'MeanMarginVsRandomNegative', 'Value': float(np.nanmean(paired - random_neg))},
        {'Metric': 'MeanMarginVsHardestNegative', 'Value': float(np.nanmean(paired - hardest_neg))},
    ]
    return pd.DataFrame(rows), paired


text_features = torch.load(TEST_TEXT_EMB_PATH, map_location='cpu')
image_features = torch.load(TEST_IMAGE_EMB_PATH, map_location='cpu')

retrieval_metrics_df = build_retrieval_metrics(text_features, image_features)
diagnostics_df, paired_similarities = build_similarity_diagnostics(text_features, image_features)

score_output_path = OUTPUT_DIR / f'{OUTPUT_PREFIX}_paired_similarity.csv'
summary_output_path = OUTPUT_DIR / f'{OUTPUT_PREFIX}_similarity_summary.csv'
retrieval_metrics_path = OUTPUT_DIR / f'{OUTPUT_PREFIX}_retrieval_metrics.csv'
diagnostics_output_path = OUTPUT_DIR / f'{OUTPUT_PREFIX}_diagnostics.csv'

id_columns = [column for column in ('subject_id', 'study_id', 'dicom_id') if column in ready_test_df.columns]
score_df = ready_test_df[id_columns + [TEXT_COLUMN, EMBEDDING_TEXT_COLUMN, 'resolved_image_paths']].copy()
score_df['paired_cosine_similarity'] = paired_similarities
score_df.to_csv(score_output_path, index=False)

summary_df = pd.DataFrame([
    {
        'Split': 'test',
        'Rows': len(score_df),
        'MeanPairedCosine': float(np.mean(paired_similarities)),
        'MedianPairedCosine': float(np.median(paired_similarities)),
        'TextEmbeddingFile': str(TEST_TEXT_EMB_PATH),
        'ImageEmbeddingFile': str(TEST_IMAGE_EMB_PATH),
        'SimilarityFile': str(score_output_path),
    }
])
summary_df.to_csv(summary_output_path, index=False)
retrieval_metrics_df.to_csv(retrieval_metrics_path, index=False)
diagnostics_df.to_csv(diagnostics_output_path, index=False)

print('Similarity summary:')
display(summary_df)
print('\nRetrieval metrics:')
display(retrieval_metrics_df)
print('\nDiagnostics:')
display(diagnostics_df)

print('\nSaved files:')
for output_path in [score_output_path, summary_output_path, retrieval_metrics_path, diagnostics_output_path]:
    print(f'  {output_path}')

# %% [markdown] cell 17
# ## 8. Output files

# %% code cell 18
print('Files created in /kaggle/working:')
for output_path in sorted(OUTPUT_DIR.glob(f'{OUTPUT_PREFIX}*')):
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f'  {output_path.name} ({size_mb:.2f} MB)')
