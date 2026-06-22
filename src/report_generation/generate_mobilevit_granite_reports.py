import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.report_generation_dataset import ReportGenerationDataset, collate_report_generation
from models.mobilevit_granite_decoder import MobileViTGranitePrefixDecoder, load_mobilevit_checkpoint
from train_mobilevit_granite_decoder import build_image_transform, dtype_from_name, encode_prompt


def build_prompt_batch(tokenizer: Any, prompts: list[str]) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")

    rows = [encode_prompt(tokenizer, prompt) for prompt in prompts]
    max_len = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), max_len), fill_value=pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, row in enumerate(rows):
        length = len(row)
        input_ids[i, :length] = torch.tensor(row, dtype=torch.long)
        attention_mask[i, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_granite_for_generation(args: argparse.Namespace):
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
    if args.lora_adapter:
        from peft import PeftModel

        decoder = PeftModel.from_pretrained(decoder, args.lora_adapter)
    return tokenizer, decoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reports with MobileViT visual prefix + Granite decoder.")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--mobilevit-checkpoint", required=True)
    parser.add_argument("--granite-model", default="ibm-granite/granite-4.1-3b")
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--target-section", choices=["impression", "report"], default="impression")
    parser.add_argument("--image-feature-source", choices=["kd_embedding", "backbone"], default="kd_embedding")
    parser.add_argument("--num-visual-tokens", type=int, default=16)
    parser.add_argument("--adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-views", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--decoder-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_csv)
    indices = np.load(Path(args.splits_dir) / f"kd_{args.split}_indices.npy")
    if args.max_rows is not None:
        indices = indices[: args.max_rows]

    dataset = ReportGenerationDataset(
        metadata=metadata,
        indices=indices,
        image_root=args.image_root,
        transform=build_image_transform(args.image_size, train=False),
        max_views=args.max_views,
        target_section=args.target_section,
        skip_empty_targets=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_report_generation,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, decoder = load_granite_for_generation(args)
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
        freeze_granite=True,
    )
    model.visual_adapter.to(device)
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    adapter_state = checkpoint["adapter_state_dict"] if "adapter_state_dict" in checkpoint else checkpoint
    model.visual_adapter.load_state_dict(adapter_state)
    model.eval()

    output_rows: list[dict[str, Any]] = []
    do_sample = args.temperature > 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"generate-{args.split}"):
            images = batch["images"].to(device, non_blocking=True)
            counts = batch["counts"].to(device, non_blocking=True)
            prompt_batch = build_prompt_batch(tokenizer, batch["prompt_text"])
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "top_p": args.top_p,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = args.temperature
            sequences = model.generate(
                images=images,
                counts=counts,
                prompt_input_ids=prompt_batch["input_ids"],
                prompt_attention_mask=prompt_batch["attention_mask"],
                **generation_kwargs,
            )
            generated_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            for i, generated_text in enumerate(generated_texts):
                output_rows.append(
                    {
                        "row_index": int(batch["row_index"][i]),
                        "subject_id": int(batch["subject_id"][i]),
                        "study_id": int(batch["study_id"][i]),
                        "reference_text": batch["target_text"][i],
                        "generated_text": " ".join(generated_text.split()),
                        "image_paths": json.dumps(batch["image_paths"][i]),
                    }
                )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output_path, index=False)
    print(f"Wrote {len(output_rows)} generations to {output_path}")


if __name__ == "__main__":
    main()
