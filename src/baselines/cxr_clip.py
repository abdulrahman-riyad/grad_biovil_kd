# Auto-exported from project notebook.
# Source notebook: week4_structured_project/Final_ppt/materials/new_teammates_baselines_notebooks/cxr-clip.ipynb
# Code cells: 11; markdown cells: 8
# Notebook shell commands and magics are preserved as comments.
# ruff: noqa
# pylint: skip-file

# %% [markdown] cell 1
# # CXR-CLIP style test embedding baseline
#
# This notebook mirrors the MedCLIP baseline pipeline, but is prepared for CXR-CLIP style experiments.
#
# It supports two backends:
#
# - `official_cxr_clip`: use the official CXR-CLIP repo/checkpoint you attach to Kaggle.
# - `biomedclip`: use the Hugging Face/OpenCLIP BiomedCLIP model as a stable CLIP-style medical baseline.
#
# Both backends produce the same outputs: text embeddings, image embeddings, paired cosine, recall@1/5/10, ranks, and diagnostics.

# %% [markdown] cell 2
# ## 1. Configuration and installs

# %% code cell 3
import importlib.util
import subprocess
import sys
from pathlib import Path

# Choose one:
# - 'official_cxr_clip' requires an official CXR-CLIP checkpoint attached to Kaggle.
# - 'biomedclip' is a stable CLIP-like medical baseline available through Hugging Face/OpenCLIP.
MODEL_BACKEND = 'official_cxr_clip'  # 'official_cxr_clip' or 'biomedclip'

# Dataset config. Update these paths on Kaggle.
TEST_DATASET_PATH = Path('/kaggle/input/datasets/shahdammar/distillationdataset-train-val-test/test_df.pkl')
IMAGE_ROOT_CANDIDATES = [
    Path('/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final/files'),
    Path('/kaggle/input/mimic-cxr-dataset/official_data_iccv_final/files'),
    Path('/kaggle/input'),
]

# Official CXR-CLIP config. The repo/checkpoint can be downloaded directly on Kaggle when Internet is enabled.
CXR_CLIP_REPO_URL = 'https://github.com/Soombit-ai/cxr-clip.git'
CXR_CLIP_REPO_DIR = Path('/kaggle/working/cxr-clip')
CXR_CLIP_CHECKPOINT_VARIANT = 'swint_m'  # r50_m, r50_mc, r50_mcc, swint_m, swint_mc, swint_mcc
CXR_CLIP_CHECKPOINT_URLS = {
    'r50_m': 'https://twg.kakaocdn.net/brainrepo/models/cxr-clip/f982386ef38aa3ecd6ce1f8f928e8aa8/r50_m.tar',
    'r50_mc': 'https://twg.kakaocdn.net/brainrepo/models/cxr-clip/f7ebbe4ad815868905d0820dbbde3662/r50_mc.tar',
    'r50_mcc': 'https://twg.kakaocdn.net/brainrepo/models/cxr-clip/de4b5e4ae2047c1fb7960ddcd8d861cb/r50_mcc.tar',
    'swint_m': 'https://twg.kakaocdn.net/brainrepo/models/cxr-clip/a21ef120894e072ae77434daf6b98b72/swint_m.tar',
    'swint_mc': 'https://twg.kakaocdn.net/brainrepo/models/cxr-clip/97cbbdfb347d22ea44e95a70c7b0520a/swint_mc.tar',
    'swint_mcc': 'https://twg.kakaocdn.net/brainrepo/models/cxr-clip/a25ce760064591c899f67c97ed7790de/swint_mcc.tar',
}
CXR_CLIP_CKPT_DIR = Path('/kaggle/working/cxr_clip_checkpoints')
CXR_CLIP_CKPT_PATH = CXR_CLIP_CKPT_DIR / f'{CXR_CLIP_CHECKPOINT_VARIANT}.tar'
CXR_CLIP_CKPT_URL = CXR_CLIP_CHECKPOINT_URLS[CXR_CLIP_CHECKPOINT_VARIANT]

# BiomedCLIP/OpenCLIP config.
BIOMEDCLIP_MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

TEXT_COLUMN_CANDIDATES = ('report_text', 'raw_report_text', 'text', 'findings', 'impression')
IMAGE_COLUMN = 'image_paths'
EMBEDDING_TEXT_COLUMN = 'embedding_text'
TEXT_PREPROCESSING_MODE = 'impression_then_findings'  # 'full' or 'impression_then_findings'
IMAGE_AGGREGATION_MODE = 'first'  # 'first' or 'mean'

SEED = 42
TEXT_BATCH_SIZE = 64
IMAGE_BATCH_SIZE = 32  # T4-safe default. Increase to 64 only if VRAM is comfortable.
MAX_ROWS = None  # Use e.g. 100 for a smoke test.
OUTPUT_DIR = Path('/kaggle/working')
OUTPUT_PREFIX = f'{MODEL_BACKEND}_mimic_cxr_test'
USE_CUDA_AMP = True  # T4 supports float16 tensor cores; set False if you see numerical/runtime issues.


def ensure_package(import_name: str, package_name: str | None = None, extra_args=None):
    package_name = package_name or import_name
    if importlib.util.find_spec(import_name) is not None:
        print(f'{import_name} is already installed.')
        return
    args = [sys.executable, '-m', 'pip', 'install', '-q']
    if extra_args:
        args.extend(extra_args)
    args.append(package_name)
    print(f'Installing {package_name}...')
    subprocess.check_call(args)


ensure_package('numpy')
ensure_package('pandas')
ensure_package('PIL', 'Pillow')
ensure_package('tqdm')
ensure_package('transformers')

if MODEL_BACKEND == 'biomedclip':
    ensure_package('open_clip', 'open_clip_torch')
    ensure_package('timm')
elif MODEL_BACKEND == 'official_cxr_clip':
    ensure_package('omegaconf')
    ensure_package('hydra', 'hydra-core')
    ensure_package('timm')
else:
    raise ValueError("MODEL_BACKEND must be 'official_cxr_clip' or 'biomedclip'.")

print('Configuration ready.')
print(f'MODEL_BACKEND: {MODEL_BACKEND}')
print(f'OUTPUT_PREFIX: {OUTPUT_PREFIX}')
print(f'USE_CUDA_AMP: {USE_CUDA_AMP}')
if MODEL_BACKEND == 'official_cxr_clip':
    print(f'CXR_CLIP_CHECKPOINT_VARIANT: {CXR_CLIP_CHECKPOINT_VARIANT}')
    print(f'CXR_CLIP_CKPT_URL: {CXR_CLIP_CKPT_URL}')

# %% [markdown] cell 4
# ## 2. Imports and shared helpers

# %% code cell 5
import ast
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import AutoTokenizer

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f'CUDA device: {name} sm_{major}{minor}, VRAM={total_gb:.1f} GB')
        return torch.device('cuda')
    return torch.device('cpu')


DEVICE = select_device()
AMP_DTYPE = torch.float16 if DEVICE.type == 'cuda' and USE_CUDA_AMP else None
print(f'Using device: {DEVICE}')
print(f'AMP dtype: {AMP_DTYPE}')


def maybe_empty_cuda_cache() -> None:
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


def autocast_context():
    if AMP_DTYPE is None:
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=True)

# %% [markdown] cell 6
# ## 3. Dataset preparation

# %% code cell 7
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

# %% code cell 8
validate_test_dataset_path(TEST_DATASET_PATH)
test_df = load_test_dataset(TEST_DATASET_PATH)
TEXT_COLUMN = select_text_column(test_df)

if IMAGE_COLUMN not in test_df.columns:
    raise KeyError(f'Test dataframe is missing required image column: {IMAGE_COLUMN}')

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

display(ready_test_df[[EMBEDDING_TEXT_COLUMN, 'resolved_image_paths']].head())

# %% [markdown] cell 9
# ## 4. Backend wrappers

# %% code cell 10
class BaseClipBackend:
    name = 'base'

    def encode_texts(self, texts: List[str], batch_size: int) -> torch.Tensor:
        raise NotImplementedError

    def encode_images_per_row(self, image_paths_per_row: List[List[str]], batch_size: int) -> torch.Tensor:
        raise NotImplementedError


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), p=2, dim=1).cpu()


def aggregate_image_features(per_image_features: torch.Tensor) -> torch.Tensor:
    per_image_features = normalize_features(per_image_features)
    if IMAGE_AGGREGATION_MODE == 'first':
        return per_image_features[:1]
    if IMAGE_AGGREGATION_MODE == 'mean':
        return normalize_features(per_image_features.mean(dim=0, keepdim=True))
    raise ValueError("IMAGE_AGGREGATION_MODE must be 'first' or 'mean'.")


class BiomedClipBackend(BaseClipBackend):
    name = 'biomedclip'

    def __init__(self, model_name: str):
        import open_clip

        self.model_name = model_name
        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(DEVICE).eval()
        print(f'Loaded BiomedCLIP/OpenCLIP model: {model_name}')

    def encode_texts(self, texts: List[str], batch_size: int) -> torch.Tensor:
        outputs = []
        for start in tqdm(range(0, len(texts), batch_size), desc='Encoding text'):
            batch_texts = texts[start:start + batch_size]
            tokens = self.tokenizer(batch_texts).to(DEVICE)
            with torch.inference_mode(), autocast_context():
                features = self.model.encode_text(tokens).cpu()
            outputs.append(features)
        return normalize_features(torch.cat(outputs, dim=0))

    def encode_images_per_row(self, image_paths_per_row: List[List[str]], batch_size: int) -> torch.Tensor:
        row_features = []
        failed_rows = []
        for row_index, paths in enumerate(tqdm(image_paths_per_row, desc='Encoding images')):
            tensors = []
            for path in paths:
                try:
                    with Image.open(path) as image:
                        tensors.append(self.preprocess(image.convert('RGB')))
                except Exception as exc:
                    tqdm.write(f'Row {row_index}: failed to preprocess {path}: {exc}')
            if not tensors:
                failed_rows.append(row_index)
                feature_dim = row_features[-1].shape[-1] if row_features else 512
                row_features.append(torch.zeros((1, feature_dim), dtype=torch.float32))
                continue

            per_image_outputs = []
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[start:start + batch_size]).to(DEVICE)
                with torch.inference_mode(), autocast_context():
                    per_image_outputs.append(self.model.encode_image(batch).cpu())
            row_features.append(aggregate_image_features(torch.cat(per_image_outputs, dim=0)))

        if failed_rows:
            print(f'Warning: {len(failed_rows)} rows had no readable images. First examples: {failed_rows[:10]}')
        return normalize_features(torch.cat(row_features, dim=0))

# %% code cell 11
def call_model_method(method, positional_arg, keyword_args=None):
    keyword_args = keyword_args or {}
    errors = []
    for call in (
        lambda: method(**keyword_args),
        lambda: method(positional_arg),
        lambda: method(keyword_args),
        lambda: method(input_ids=keyword_args.get('input_ids'), attention_mask=keyword_args.get('attention_mask')),
    ):
        try:
            return call()
        except TypeError as exc:
            errors.append(str(exc))
    raise TypeError('All known official CXR-CLIP encode call signatures failed: ' + ' | '.join(errors))


def get_first_tensor_from_dict(output: dict, preferred_keys: Sequence[str]):
    for key in preferred_keys:
        if key in output:
            return output[key]
    for value in output.values():
        if torch.is_tensor(value):
            return value
    raise KeyError(f'Could not find a tensor in model output keys: {list(output.keys())}')


def ensure_official_repo() -> None:
    if CXR_CLIP_REPO_DIR.exists():
        print(f'Using existing CXR-CLIP repo: {CXR_CLIP_REPO_DIR}')
        return
    print(f'Cloning official CXR-CLIP repo to {CXR_CLIP_REPO_DIR}...')
    subprocess.check_call(['git', 'clone', '--depth', '1', CXR_CLIP_REPO_URL, str(CXR_CLIP_REPO_DIR)])


def resolve_official_checkpoint() -> Path:
    if CXR_CLIP_CKPT_PATH.exists() and CXR_CLIP_CKPT_PATH.stat().st_size > 0:
        print(f'Using existing CXR-CLIP checkpoint: {CXR_CLIP_CKPT_PATH}')
        return CXR_CLIP_CKPT_PATH
    if not CXR_CLIP_CKPT_URL:
        raise FileNotFoundError(
            'Official CXR-CLIP checkpoint URL is empty. Set CXR_CLIP_CHECKPOINT_VARIANT or CXR_CLIP_CKPT_URL.'
        )

    CXR_CLIP_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Downloading CXR-CLIP checkpoint variant {CXR_CLIP_CHECKPOINT_VARIANT} to {CXR_CLIP_CKPT_PATH}...')
    try:
        subprocess.check_call(['wget', '-q', '--show-progress', '-O', str(CXR_CLIP_CKPT_PATH), CXR_CLIP_CKPT_URL])
    except Exception:
        print('wget failed; falling back to urllib.request.urlretrieve...')
        urllib.request.urlretrieve(CXR_CLIP_CKPT_URL, CXR_CLIP_CKPT_PATH)

    if not CXR_CLIP_CKPT_PATH.exists() or CXR_CLIP_CKPT_PATH.stat().st_size == 0:
        raise RuntimeError(f'Checkpoint download failed or produced an empty file: {CXR_CLIP_CKPT_PATH}')
    print(f'Downloaded checkpoint: {CXR_CLIP_CKPT_PATH} ({CXR_CLIP_CKPT_PATH.stat().st_size / (1024 ** 2):.1f} MB)')
    return CXR_CLIP_CKPT_PATH


def extract_state_dict(checkpoint):
    for key in ('state_dict', 'model_state_dict', 'model'):
        if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    if isinstance(checkpoint, dict):
        tensor_values = [value for value in checkpoint.values() if torch.is_tensor(value)]
        if tensor_values:
            return checkpoint
    raise KeyError('Could not find a model state_dict in the CXR-CLIP checkpoint.')


def strip_state_dict_prefixes(state_dict):
    prefixes = ('model.', 'module.', 'net.', 'clip.', 'backbone.')
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def find_config(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    candidates = [
        checkpoint.get('config'),
        checkpoint.get('cfg'),
        checkpoint.get('hyper_parameters', {}).get('config') if isinstance(checkpoint.get('hyper_parameters'), dict) else None,
        checkpoint.get('hyper_parameters', {}).get('cfg') if isinstance(checkpoint.get('hyper_parameters'), dict) else None,
        checkpoint.get('hyper_parameters', {}).get('hparams') if isinstance(checkpoint.get('hyper_parameters'), dict) else None,
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


class OfficialCxrClipBackend(BaseClipBackend):
    name = 'official_cxr_clip'

    def __init__(self):
        ensure_official_repo()
        sys.path.insert(0, str(CXR_CLIP_REPO_DIR))
        checkpoint_path = resolve_official_checkpoint()
        # Official checkpoint contains OmegaConf config objects, so PyTorch 2.6+ needs weights_only=False.
        # Only do this for checkpoints downloaded from the trusted official CXR-CLIP URL.
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = find_config(checkpoint)

        try:
            from cxrclip.model import build_model
        except Exception as exc:
            raise ImportError(
                'Could not import cxrclip.model.build_model from the official repo. '
                'Check that CXR_CLIP_REPO_DIR points to the official repository.'
            ) from exc

        tokenizer_name = 'emilyalsentzer/Bio_ClinicalBERT'
        if isinstance(config, dict):
            tokenizer_name = (
                config.get('tokenizer', {}).get('pretrained_model_name_or_path')
                or config.get('text_encoder', {}).get('pretrained_model_name_or_path')
                or config.get('text', {}).get('pretrained_model_name_or_path')
                or tokenizer_name
            )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.model_max_length = 256

        # The official repo has had a few config shapes. Try the common construction first.
        try:
            if isinstance(config, dict):
                self.model = build_model(config.get('model', config), config.get('loss', None), self.tokenizer)
            else:
                self.model = build_model(config.model, getattr(config, 'loss', None), self.tokenizer)
        except Exception as exc:
            raise RuntimeError(
                'Failed to construct the official CXR-CLIP model from checkpoint config. '
                'This loader may need a small adjustment for your exact checkpoint format. '
                'Use MODEL_BACKEND="biomedclip" for a stable CLIP-style baseline while we inspect the checkpoint.'
            ) from exc

        state_dict = strip_state_dict_prefixes(extract_state_dict(checkpoint))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f'Loaded official CXR-CLIP checkpoint: {checkpoint_path}')
        print(f'Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}')
        self.model = self.model.to(DEVICE).eval()

    def encode_texts(self, texts: List[str], batch_size: int) -> torch.Tensor:
        outputs = []
        for start in tqdm(range(0, len(texts), batch_size), desc='Encoding text'):
            batch_texts = texts[start:start + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
            inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
            with torch.inference_mode(), autocast_context():
                if hasattr(self.model, 'encode_text'):
                    features = call_model_method(self.model.encode_text, inputs, inputs)
                elif hasattr(self.model, 'encode_textual'):
                    features = call_model_method(self.model.encode_textual, inputs, inputs)
                else:
                    raise AttributeError('Official CXR-CLIP model has no encode_text/encode_textual method.')
                if isinstance(features, dict):
                    features = get_first_tensor_from_dict(features, ('text_embeds', 'text_features', 'text_projection', 'embeds'))
                outputs.append(features.detach().cpu())
        return normalize_features(torch.cat(outputs, dim=0))

    def image_transform(self, image: Image.Image) -> torch.Tensor:
        image = image.convert('RGB')
        image = ImageOps.pad(image, (224, 224), method=Image.BICUBIC, color=0, centering=(0.5, 0.5))
        array = np.asarray(image).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
        array = np.transpose(array, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(array)

    def encode_images_per_row(self, image_paths_per_row: List[List[str]], batch_size: int) -> torch.Tensor:
        row_features = []
        for row_index, paths in enumerate(tqdm(image_paths_per_row, desc='Encoding images')):
            tensors = []
            for path in paths:
                try:
                    with Image.open(path) as image:
                        tensors.append(self.image_transform(image))
                except Exception as exc:
                    tqdm.write(f'Row {row_index}: failed to preprocess {path}: {exc}')
            if not tensors:
                feature_dim = row_features[-1].shape[-1] if row_features else 512
                row_features.append(torch.zeros((1, feature_dim), dtype=torch.float32))
                continue
            outputs = []
            for start in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[start:start + batch_size]).to(DEVICE)
                with torch.inference_mode(), autocast_context():
                    if hasattr(self.model, 'encode_image'):
                        features = call_model_method(self.model.encode_image, batch)
                    elif hasattr(self.model, 'encode_visual'):
                        features = call_model_method(self.model.encode_visual, batch)
                    else:
                        raise AttributeError('Official CXR-CLIP model has no encode_image/encode_visual method.')
                    if isinstance(features, dict):
                        features = get_first_tensor_from_dict(features, ('image_embeds', 'image_features', 'visual_embeds', 'embeds'))
                    outputs.append(features.detach().cpu())
            row_features.append(aggregate_image_features(torch.cat(outputs, dim=0)))
        return normalize_features(torch.cat(row_features, dim=0))

# %% code cell 12
if MODEL_BACKEND == 'biomedclip':
    backend = BiomedClipBackend(BIOMEDCLIP_MODEL_NAME)
elif MODEL_BACKEND == 'official_cxr_clip':
    backend = OfficialCxrClipBackend()
else:
    raise ValueError(MODEL_BACKEND)

# Smoke test before the full run.
smoke_text = backend.encode_texts(['no acute cardiopulmonary abnormality'], batch_size=1)
smoke_image = backend.encode_images_per_row([ready_test_df['resolved_image_paths'].iloc[0]], batch_size=1)
print('Smoke text shape:', tuple(smoke_text.shape))
print('Smoke image shape:', tuple(smoke_image.shape))
assert smoke_text.ndim == 2 and smoke_image.ndim == 2

# %% [markdown] cell 13
# ## 5. Extract embeddings

# %% code cell 14
TEST_TEXT_EMB_PATH = OUTPUT_DIR / f'{OUTPUT_PREFIX}_text_embeddings.pt'
TEST_IMAGE_EMB_PATH = OUTPUT_DIR / f'{OUTPUT_PREFIX}_image_embeddings.pt'
TEST_METADATA_OUTPUT_PATH = OUTPUT_DIR / f'{OUTPUT_PREFIX}_metadata_used.csv'

texts = ready_test_df[EMBEDDING_TEXT_COLUMN].tolist()
image_paths_per_row = ready_test_df['resolved_image_paths'].tolist()

text_features = backend.encode_texts(texts, batch_size=TEXT_BATCH_SIZE)
image_features = backend.encode_images_per_row(image_paths_per_row, batch_size=IMAGE_BATCH_SIZE)

if text_features.shape[0] != image_features.shape[0]:
    raise RuntimeError(f'Feature row mismatch: text={text_features.shape}, image={image_features.shape}')
if text_features.shape[1] != image_features.shape[1]:
    raise RuntimeError(f'Feature dim mismatch: text={text_features.shape}, image={image_features.shape}')

torch.save(text_features, TEST_TEXT_EMB_PATH)
torch.save(image_features, TEST_IMAGE_EMB_PATH)
ready_test_df.to_csv(TEST_METADATA_OUTPUT_PATH, index=False)

maybe_empty_cuda_cache()
print(f'Text embeddings: {tuple(text_features.shape)} -> {TEST_TEXT_EMB_PATH}')
print(f'Image embeddings: {tuple(image_features.shape)} -> {TEST_IMAGE_EMB_PATH}')
print(f'Metadata saved to: {TEST_METADATA_OUTPUT_PATH}')

# %% [markdown] cell 15
# ## 6. Metrics and diagnostics

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


def build_similarity_diagnostics(text_features: torch.Tensor, image_features: torch.Tensor, chunk_size: int = 512):
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

    diagnostics_df = pd.DataFrame([
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
    ])
    return diagnostics_df, paired

# %% code cell 17
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
        'Backend': MODEL_BACKEND,
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

# %% [markdown] cell 18
# ## 7. Output files

# %% code cell 19
print('Files created in /kaggle/working:')
for output_path in sorted(OUTPUT_DIR.glob(f'{OUTPUT_PREFIX}*')):
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f'  {output_path.name} ({size_mb:.2f} MB)')
