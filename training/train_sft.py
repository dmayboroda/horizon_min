"""SFT training for Duck Hunt VLM — teaches spatial coordinate prediction.

Trains the model to output correct shoot(x, y) coordinates given game frames.
Uses standard cross-entropy loss on completion tokens only.

Usage:
    python train_sft.py --dataset sft_dataset --model LiquidAI/LFM2.5-VL-1.6B
    python train_sft.py --dataset sft_dataset --model LiquidAI/LFM2-VL-3B --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_scheduler
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_sft_dataset(dataset_dir: str) -> list[dict]:
    """Load the SFT dataset from disk."""
    ds_path = Path(dataset_dir)
    meta_path = ds_path / "dataset.json"

    with open(meta_path) as f:
        records = json.load(f)

    logger.info("Loaded %d records from %s", len(records), meta_path)
    return records


def build_training_sample(
    record: dict,
    processor,
    device: torch.device,
) -> dict | None:
    """Build tokenized input/labels for one SFT sample.

    Returns dict with 'input_ids', 'attention_mask', 'labels', 'pixel_values'.
    Labels = -100 for prompt tokens, real IDs for completion tokens.
    """
    # Load images
    images = []
    for img_path in record["image_paths"]:
        images.append(Image.open(img_path).convert("RGB"))

    # Build user content with images
    user_content = []
    for img in images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": record["user_text"]})

    # Full conversation including ground-truth assistant response
    full_messages = [
        {"role": "system", "content": record["system_prompt"]},
        {"role": "user", "content": record["user_fewshot"]},
        {"role": "assistant", "content": record["assistant_fewshot"]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record["completion"]},
    ]

    # Prompt only (without the final assistant message)
    prompt_messages = [
        {"role": "system", "content": record["system_prompt"]},
        {"role": "user", "content": record["user_fewshot"]},
        {"role": "assistant", "content": record["assistant_fewshot"]},
        {"role": "user", "content": user_content},
    ]

    try:
        # Tokenize full conversation
        full_inputs = processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Tokenize prompt only (to find where completion starts)
        prompt_inputs = processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        full_ids = full_inputs["input_ids"][0]
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Labels: -100 for prompt, real IDs for completion
        labels = full_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": full_ids.unsqueeze(0).to(device),
            "attention_mask": full_inputs["attention_mask"].to(device),
            "labels": labels.unsqueeze(0).to(device),
            "prompt_len": prompt_len,
            "comp_len": len(full_ids) - prompt_len,
        }
    except Exception as e:
        logger.warning("Failed to build sample %s: %s", record.get("id", "?"), e)
        return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_sft(
    model_name: str = "LiquidAI/LFM2.5-VL-1.6B",
    dataset_dir: str = "sft_dataset",
    output_dir: str = "outputs/sft",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.05,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_grad_norm: float = 1.0,
    lora_r: int = 16,
    lora_alpha: int = 32,
    save_steps: int = 200,
    wandb_project: str = "duckhunt-sft",
    wandb_run_name: str | None = None,
    seed: int = 42,
) -> None:
    """Run SFT training."""
    random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    records = load_sft_dataset(dataset_dir)
    random.shuffle(records)

    # Load model + processor
    from src.model import load_model_and_processor, apply_lora
    from src.config import ModelConfig, LoRAConfig

    model_config = ModelConfig(
        model_name=model_name,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="auto",
    )

    # Detect LoRA targets based on model
    if "lfm" in model_name.lower() or "liquid" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoRAConfig(
        enabled=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
    )

    model, processor = load_model_and_processor(model_config)
    model = apply_lora(model, lora_config)

    # SFT uses right-padding (unlike GRPO which uses left)
    processor.tokenizer.padding_side = "right"

    device = next(model.parameters()).device

    # Gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Optimizer + scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    total_steps = (len(records) * num_epochs) // (batch_size * grad_accum)
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info("SFT Training:")
    logger.info("  Model: %s", model_name)
    logger.info("  Dataset: %d records", len(records))
    logger.info("  Epochs: %d", num_epochs)
    logger.info("  Total steps: %d (warmup: %d)", total_steps, warmup_steps)
    logger.info("  LR: %s, LoRA r=%d alpha=%d", learning_rate, lora_r, lora_alpha)
    logger.info("  Grad accum: %d, effective batch: %d", grad_accum, batch_size * grad_accum)

    # W&B
    if _WANDB and wandb_project:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"sft-{model_name.split('/')[-1]}",
            config={
                "model": model_name,
                "dataset_size": len(records),
                "epochs": num_epochs,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "grad_accum": grad_accum,
            },
        )

    # Training
    model.train()
    global_step = 0
    running_loss = 0.0
    samples_processed = 0

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        random.shuffle(records)
        epoch_loss = 0.0
        epoch_samples = 0

        for i, record in enumerate(records):
            sample = build_training_sample(record, processor, device)
            if sample is None:
                continue

            # Forward pass
            outputs = model(
                input_ids=sample["input_ids"],
                attention_mask=sample["attention_mask"],
                labels=sample["labels"],
            )
            loss = outputs.loss / grad_accum
            loss.backward()

            running_loss += outputs.loss.item()
            epoch_loss += outputs.loss.item()
            epoch_samples += 1
            samples_processed += 1

            # Optimizer step
            if (i + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                avg_loss = running_loss / grad_accum
                lr = scheduler.get_last_lr()[0]

                if global_step % 10 == 0:
                    logger.info(
                        "epoch=%d  step=%d  loss=%.4f  lr=%.2e  grad=%.4f",
                        epoch + 1, global_step, avg_loss, lr, grad_norm.item(),
                    )

                if _WANDB and wandb.run:
                    wandb.log({
                        "sft/loss": avg_loss,
                        "sft/learning_rate": lr,
                        "sft/gradient_norm": grad_norm.item(),
                        "sft/epoch": epoch + 1,
                        "sft/step": global_step,
                        "sft/samples_processed": samples_processed,
                    })

                running_loss = 0.0

                # Save checkpoint
                if global_step % save_steps == 0:
                    ckpt_dir = out_path / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    model.save_pretrained(str(ckpt_dir))
                    processor.save_pretrained(str(ckpt_dir))
                    logger.info("Saved checkpoint to %s", ckpt_dir)

        avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
        logger.info(
            "Epoch %d complete: avg_loss=%.4f, samples=%d",
            epoch + 1, avg_epoch_loss, epoch_samples,
        )

    # Final save
    final_dir = out_path / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info("Training complete. Final model saved to %s", final_dir)

    if _WANDB and wandb.run:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SFT training for Duck Hunt VLM")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to SFT dataset directory")
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2.5-VL-1.6B",
                        help="Base model name")
    parser.add_argument("--output", type=str, default="outputs/sft",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--wandb-project", type=str, default="duckhunt-sft")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_sft(
        model_name=args.model,
        dataset_dir=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        grad_accum=args.grad_accum,
        save_steps=args.save_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
