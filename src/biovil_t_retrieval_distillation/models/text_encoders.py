from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import os
from pathlib import Path
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TextEncoderPreset:
    model_id: str
    trust_remote_code: bool
    pooling: str = "auto"
    loader: str = "auto_model"
    use_safetensors: bool | None = None
    local_dir_env: str | None = None


TEXT_ENCODER_PRESETS: dict[str, TextEncoderPreset] = {
    "biovil_t": TextEncoderPreset(
        model_id="microsoft/BiomedVLP-BioViL-T",
        trust_remote_code=True,
        pooling="mean",
    ),
    "cxr_bert": TextEncoderPreset(
        model_id="microsoft/BiomedVLP-CXR-BERT-specialized",
        trust_remote_code=True,
        pooling="mean",
    ),
    "bioclinical_modernbert": TextEncoderPreset(
        model_id="thomas-sounack/BioClinical-ModernBERT-base",
        trust_remote_code=False,
        pooling="mean",
    ),
    "distil_biobert": TextEncoderPreset(
        model_id="nlpie/distil-biobert",
        trust_remote_code=False,
        pooling="cls",
        local_dir_env="DISTIL_BIOBERT_ENCODER_DIR",
    ),
    "clinical_distilbert": TextEncoderPreset(
        model_id="nlpie/clinical-distilbert",
        trust_remote_code=False,
        pooling="cls",
        local_dir_env="CLINICAL_DISTILBERT_ENCODER_DIR",
    ),
    "clinical_mobilebert": TextEncoderPreset(
        model_id="nlpie/clinical-mobilebert",
        trust_remote_code=False,
        pooling="cls",
        loader="auto_model",
        use_safetensors=False,
        local_dir_env="CLINICAL_MOBILEBERT_ENCODER_DIR",
    ),
}


class HFTextEncoder(nn.Module):
    """Hugging Face encoder wrapper with robust pooling for contrastive training."""

    def __init__(
        self,
        model_id: str,
        trust_remote_code: bool = False,
        pooling: str = "auto",
        max_length: int = 256,
        loader: str = "auto_model",
        use_safetensors: bool | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

        if pooling not in {"auto", "cls", "mean", "pooler", "projected"}:
            raise ValueError("pooling must be one of: auto, cls, mean, pooler, projected.")
        if loader not in {"auto_model", "masked_lm_base"}:
            raise ValueError("loader must be one of: auto_model, masked_lm_base.")

        self.model_id = model_id
        self.pooling = pooling
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        is_local_model_dir = Path(model_id).exists()
        if use_safetensors is not None and not is_local_model_dir:
            model_kwargs["use_safetensors"] = use_safetensors
        if loader == "masked_lm_base":
            masked_lm = AutoModelForMaskedLM.from_pretrained(model_id, **model_kwargs)
            self.encoder = masked_lm.base_model
        else:
            self.encoder = AutoModel.from_pretrained(model_id, **model_kwargs)
        self.output_dim = int(getattr(self.encoder.config, "hidden_size", 0))
        if self.output_dim <= 0:
            raise ValueError(f"Could not infer hidden_size for text encoder: {model_id}")

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenize(texts)
        device = next(self.encoder.parameters()).device
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.encoder(**tokens)
        embeddings = self._pool_outputs(outputs, tokens["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)

    def _pool_outputs(self, outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        projected = self._projected_embedding(outputs)
        if projected is not None and self.pooling == "projected":
            return projected

        if self.pooling in {"auto", "pooler"} and hasattr(outputs, "pooler_output"):
            pooler = outputs.pooler_output
            if pooler is not None:
                return pooler

        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        return mean_pool(hidden, attention_mask)

    @staticmethod
    def _projected_embedding(outputs: Any) -> torch.Tensor | None:
        candidate_names = (
            "projected_text_embeddings",
            "text_embeds",
            "sentence_embedding",
        )
        for name in candidate_names:
            if hasattr(outputs, name):
                value = getattr(outputs, name)
                if torch.is_tensor(value):
                    return value
        if isinstance(outputs, dict):
            for name in candidate_names:
                value = outputs.get(name)
                if torch.is_tensor(value):
                    return value
        return None


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def build_text_encoder(name_or_model_id: str, max_length: int = 256) -> HFTextEncoder:
    preset = TEXT_ENCODER_PRESETS.get(name_or_model_id)
    if preset is None:
        return HFTextEncoder(name_or_model_id, trust_remote_code=False, pooling="mean", max_length=max_length)
    model_id = preset.model_id
    if preset.local_dir_env:
        model_id = os.environ.get(preset.local_dir_env, model_id)
    return HFTextEncoder(
        model_id=model_id,
        trust_remote_code=preset.trust_remote_code,
        pooling=preset.pooling,
        max_length=max_length,
        loader=preset.loader,
        use_safetensors=preset.use_safetensors,
    )
