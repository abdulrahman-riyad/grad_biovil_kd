import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.report_generation_dataset import ReportGenerationDataset, collate_report_generation
from models.mobilevit_granite_decoder import MobileViTGranitePrefixDecoder, load_mobilevit_checkpoint


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_image_transform(image_size: int = 224, train: bool = False):
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_loader(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    image_root: str | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    max_views: int,
    target_section: str,
) -> DataLoader:
    dataset = ReportGenerationDataset(
        metadata=metadata,
        indices=indices,
        image_root=image_root,
        transform=build_image_transform(image_size=image_size, train=train),
        max_views=max_views,
        target_section=target_section,
        skip_empty_targets=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        collate_fn=collate_report_generation,
    )


def dtype_from_name(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        rendered = f"User: {prompt}\nAssistant:"

    tokenized = tokenizer(rendered, add_special_tokens=False)
    ids = tokenized["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not all(isinstance(item, int) for item in ids):
        raise TypeError(f"Prompt tokenization produced non-integer ids: {ids[:5]}")
    return list(ids)


def build_lm_batch(
    tokenizer: Any,
    prompts: list[str],
    targets: list[str],
    max_target_tokens: int,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    eos = tokenizer.eos_token or ""

    for prompt, target in zip(prompts, targets, strict=True):
        prompt_ids = encode_prompt(tokenizer, prompt)
        target_ids = tokenizer(
            str(target).strip() + eos,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_tokens,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        input_rows.append(input_ids)
        label_rows.append(labels)

    max_len = max(len(row) for row in input_rows)
    input_ids_tensor = torch.full((len(input_rows), max_len), fill_value=pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(input_rows), max_len), dtype=torch.long)
    labels_tensor = torch.full((len(input_rows), max_len), fill_value=-100, dtype=torch.long)

    for i, (input_ids, labels) in enumerate(zip(input_rows, label_rows, strict=True)):
        length = len(input_ids)
        input_ids_tensor[i, :length] = torch.tensor(input_ids, dtype=torch.long)
        attention_mask[i, :length] = 1
        labels_tensor[i, :length] = torch.tensor(labels, dtype=torch.long)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask,
        "labels": labels_tensor,
    }


def load_granite(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.granite_model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"torch_dtype": dtype_from_name(args.decoder_dtype)}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = "auto"

    decoder = AutoModelForCausalLM.from_pretrained(args.granite_model, **model_kwargs)
    return tokenizer, decoder


def maybe_apply_lora(decoder: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if not args.use_lora:
        return decoder

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.load_in_4bit:
        decoder = prepare_model_for_kbit_training(decoder)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(decoder, config)


def run_epoch(
    model: MobileViTGranitePrefixDecoder,
    tokenizer: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    image_device: torch.device,
    max_target_tokens: int,
    grad_accum_steps: int,
    max_batches: int | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.visual_adapter.train(is_train)
    if any(param.requires_grad for param in model.decoder.parameters()):
        model.decoder.train(is_train)

    total_loss = 0.0
    steps = 0
    optimizer_steps = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress = tqdm(loader, desc="train" if is_train else "val")
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(progress, start=1):
            images = batch["images"].to(image_device, non_blocking=True)
            counts = batch["counts"].to(image_device, non_blocking=True)
            lm_batch = build_lm_batch(tokenizer, batch["prompt_text"], batch["target_text"], max_target_tokens)

            outputs = model(
                images=images,
                counts=counts,
                input_ids=lm_batch["input_ids"],
                attention_mask=lm_batch["attention_mask"],
                labels=lm_batch["labels"],
            )
            loss = outputs.loss

            if is_train:
                (loss / grad_accum_steps).backward()
                if batch_index % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

            total_loss += float(loss.detach().cpu())
            steps += 1
            progress.set_postfix(loss=total_loss / steps)

            if max_batches is not None and steps >= max_batches:
                break

        if is_train and steps % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    if steps == 0:
        raise RuntimeError("No batches were processed.")

    return {"loss": total_loss / steps, "batches": float(steps), "optimizer_steps": float(optimizer_steps)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileViT visual prefix adapter for Granite report generation.")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--mobilevit-checkpoint", required=True)
    parser.add_argument("--granite-model", default="ibm-granite/granite-4.1-3b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-section", choices=["impression", "report"], default="impression")
    parser.add_argument("--image-feature-source", choices=["kd_embedding", "backbone"], default="kd_embedding")
    parser.add_argument("--num-visual-tokens", type=int, default=16)
    parser.add_argument("--adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-views", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--decoder-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata_csv)
    train_indices = np.load(Path(args.splits_dir) / "kd_train_indices.npy")
    val_indices = np.load(Path(args.splits_dir) / "kd_val_indices.npy")
    if args.max_train_rows is not None:
        train_indices = train_indices[: args.max_train_rows]
    if args.max_val_rows is not None:
        val_indices = val_indices[: args.max_val_rows]

    train_loader = make_loader(
        metadata,
        train_indices,
        args.image_root,
        args.image_size,
        args.batch_size,
        args.num_workers,
        train=True,
        max_views=args.max_views,
        target_section=args.target_section,
    )
    val_loader = make_loader(
        metadata,
        val_indices,
        args.image_root,
        args.image_size,
        args.batch_size,
        args.num_workers,
        train=False,
        max_views=args.max_views,
        target_section=args.target_section,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, decoder = load_granite(args)
    decoder = maybe_apply_lora(decoder, args)
    if not args.load_in_4bit:
        decoder.to(device)

    image_encoder = load_mobilevit_checkpoint(args.mobilevit_checkpoint)
    image_encoder.to(device)
    model = MobileViTGranitePrefixDecoder(
        image_encoder=image_encoder,
        decoder=decoder,
        image_feature_source=args.image_feature_source,
        num_visual_tokens=args.num_visual_tokens,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_dropout=args.adapter_dropout,
        freeze_mobilevit=True,
        freeze_granite=not args.use_lora,
    )
    model.visual_adapter.to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args) | {
        "train_rows": int(len(train_loader.dataset)),
        "val_rows": int(len(val_loader.dataset)),
        "device": str(device),
        "trainable_parameters": int(sum(param.numel() for param in trainable_params)),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=json_safe), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            tokenizer,
            train_loader,
            optimizer,
            device,
            args.max_target_tokens,
            args.grad_accum_steps,
            args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            tokenizer,
            val_loader,
            None,
            device,
            args.max_target_tokens,
            args.grad_accum_steps,
            args.max_val_batches,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2, default=json_safe))
        (output_dir / "history.json").write_text(json.dumps(history, indent=2, default=json_safe), encoding="utf-8")

        checkpoint = {
            "epoch": epoch,
            "adapter_state_dict": model.visual_adapter.state_dict(),
            "config": config,
            "val_loss": val_metrics["loss"],
        }
        torch.save(checkpoint, output_dir / "last_adapter.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, output_dir / "best_adapter.pt")

    if args.use_lora:
        model.decoder.save_pretrained(output_dir / "lora_adapter")


if __name__ == "__main__":
    main()
