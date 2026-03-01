"""Main training script for Duck Hunt GRPO.

Usage::

    # TRL GRPOTrainer (default)
    python train.py --config configs/ministral_config.yaml

    # Custom GRPO loop (fallback)
    python train.py --config configs/ministral_config.yaml --custom

    # Layer configs (base + model + optimizer + forward_mod)
    python train.py --custom \\
        --config configs/base_config.yaml \\
        --config configs/ministral_config.yaml \\
        --config configs/optimizers/muon.yaml \\
        --config configs/forward_mods/action_full.yaml

    # CLI overrides
    python train.py --config configs/ministral_config.yaml \\
        --override training.learning_rate=2e-5 \\
        --override grpo.num_generations=8

    # Resume from checkpoint (custom mode)
    python train.py --config configs/ministral_config.yaml --custom \\
        --resume outputs/duckhunt_grpo/checkpoint-500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make training/src importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import FullConfig
from src.dataset import (
    DuckHuntPromptGenerator,
    make_format_reward_function,
    make_reward_function,
)
from src.model import load_model_and_processor, apply_lora, setup_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================
#  Config helpers
# ===================================================================
def _parse_override(s: str) -> tuple[str, str | float | int | bool]:
    """Parse ``key=value`` into (key, typed_value)."""
    key, _, raw_val = s.partition("=")
    # Try int → float → bool → str
    for converter in (int, float):
        try:
            return key, converter(raw_val)
        except ValueError:
            continue
    if raw_val.lower() in ("true", "false"):
        return key, raw_val.lower() == "true"
    return key, raw_val


def load_config(args: argparse.Namespace) -> FullConfig:
    """Build a ``FullConfig`` from CLI args (yamls + overrides)."""
    if len(args.config) == 1:
        cfg = FullConfig.from_yaml(args.config[0])
    else:
        cfg = FullConfig.from_yamls(*args.config)

    if args.override:
        overrides = dict(_parse_override(o) for o in args.override)
        cfg = FullConfig.with_cli_overrides(cfg, overrides)

    return cfg


# ===================================================================
#  TRL GRPOTrainer path  (Step 7.1)
# ===================================================================
def train_trl(cfg: FullConfig, num_samples: int) -> None:
    """Train using TRL's ``GRPOTrainer``."""
    from peft import LoraConfig as PeftLoraConfig
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    # 1. Generate dataset from environment
    logger.info("Generating %d training samples from environment …", num_samples)
    generator = DuckHuntPromptGenerator(cfg.environment)
    dataset = generator.generate(num_samples)
    logger.info("Dataset ready: %d rows", len(dataset))

    # 2. Load model + processor
    model, processor = load_model_and_processor(cfg.model)

    # 3. LoRA config (let GRPOTrainer apply it via peft_config)
    peft_config = None
    if cfg.lora.enabled:
        peft_config = PeftLoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=cfg.lora.target_modules,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        )

    # 4. Build TRL GRPOConfig (merges our GRPO + Training + Logging fields)
    grpo = cfg.grpo
    train = cfg.training
    log_cfg = cfg.logging

    trl_config = TRLGRPOConfig(
        output_dir=train.output_dir,
        # Training args
        num_train_epochs=train.num_train_epochs,
        max_steps=train.max_steps,
        per_device_train_batch_size=train.per_device_train_batch_size,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        learning_rate=train.learning_rate,
        warmup_ratio=train.warmup_ratio,
        weight_decay=train.weight_decay,
        lr_scheduler_type=train.lr_scheduler_type,
        max_grad_norm=train.max_grad_norm,
        bf16=train.bf16,
        fp16=train.fp16,
        logging_steps=train.logging_steps,
        save_steps=train.save_steps,
        save_total_limit=train.save_total_limit,
        eval_strategy=train.eval_strategy,
        eval_steps=train.eval_steps,
        seed=train.seed,
        dataloader_num_workers=train.dataloader_num_workers,
        gradient_checkpointing=train.gradient_checkpointing,
        # GRPO args
        num_generations=grpo.num_generations,
        max_completion_length=grpo.max_completion_length,
        temperature=grpo.temperature,
        top_p=grpo.top_p,
        top_k=grpo.top_k,
        beta=grpo.beta,
        num_iterations=grpo.num_iterations,
        epsilon=grpo.epsilon,
        loss_type=grpo.loss_type,
        scale_rewards=grpo.scale_rewards,
        use_vllm=grpo.use_vllm,
        # Logging
        report_to=log_cfg.report_to,
        run_name=log_cfg.wandb_run_name,
        log_completions=log_cfg.log_completions,
        # Reward weights: [accuracy, format]
        reward_weights=[cfg.reward.accuracy_weight, cfg.reward.format_weight],
    )

    # 5. Create reward functions
    accuracy_fn = make_reward_function(cfg.reward, cfg.environment.max_horizon)
    format_fn = make_format_reward_function(cfg.environment.max_horizon)

    # 6. Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=trl_config,
        train_dataset=dataset,
        processing_class=processor,
        reward_funcs=[accuracy_fn, format_fn],
        peft_config=peft_config,
    )

    # 7. Train
    logger.info("Starting TRL GRPOTrainer …")
    trainer.train()

    # 8. Save final model
    final_dir = Path(train.output_dir) / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info("Training complete. Model saved to %s", final_dir)

    # 9. Push to HF Hub
    if cfg.hub.push_to_hub and cfg.hub.hub_model_id:
        _push_to_hub(cfg, str(final_dir))


# ===================================================================
#  HF Hub upload (shared by both paths)
# ===================================================================
def _push_to_hub(cfg: FullConfig, checkpoint_dir: str) -> None:
    """Push a checkpoint to Hugging Face Hub."""
    from huggingface_hub import HfApi, ModelCard

    repo_id = cfg.hub.hub_model_id
    logger.info("Pushing %s to HF Hub: %s …", checkpoint_dir, repo_id)

    api = HfApi()
    api.create_repo(repo_id, private=cfg.hub.hub_private, exist_ok=True)

    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        ignore_patterns=["optimizer.pt", "scheduler.pt"],
    )

    card_content = f"""\
---
library_name: peft
base_model: {cfg.model.model_name}
tags:
  - grpo
  - reinforcement-learning
  - duck-hunt
  - vision-language-model
  - tool-calling
---

# {repo_id.split('/')[-1]}

LoRA adapter for [{cfg.model.model_name}](https://huggingface.co/{cfg.model.model_name}),
fine-tuned with GRPO to play Duck Hunt.

## Training

- **Method**: Group Relative Policy Optimization (GRPO)
- **LoRA rank**: {cfg.lora.r}, alpha: {cfg.lora.lora_alpha}
- **Target modules**: {', '.join(cfg.lora.target_modules)}
- **Learning rate**: {cfg.training.learning_rate}

## Usage

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoProcessor

model = AutoPeftModelForCausalLM.from_pretrained("{repo_id}")
processor = AutoProcessor.from_pretrained("{repo_id}")
```
"""
    card = ModelCard(card_content)
    card.push_to_hub(repo_id)

    logger.info("Pushed to https://huggingface.co/%s", repo_id)


# ===================================================================
#  Custom GRPO loop path  (Step 7.2)
# ===================================================================
def train_custom(cfg: FullConfig, resume_from: str | None = None) -> None:
    """Train using the custom ``DuckHuntGRPOTrainer``."""
    from src.environment import DuckHuntEnvWrapper
    from src.trainer import DuckHuntGRPOTrainer

    # 1. Environment
    env = DuckHuntEnvWrapper(cfg.environment)

    # 2. Model + LoRA + forward mods (via setup_model)
    model, processor = setup_model(cfg)

    # 3. Trainer
    trainer = DuckHuntGRPOTrainer(
        model=model,
        processor=processor,
        env=env,
        config=cfg,
    )

    # 4. Train (with optional resume)
    logger.info("Starting custom GRPO training loop …")
    logger.info(
        "Mode: %s | Optimizer: %s | Forward mods: %s",
        cfg.forward_mod.mode,
        cfg.optimizer.name,
        "enabled" if cfg.forward_mod.enabled else "disabled",
    )
    trainer.train(resume_from=resume_from)


# ===================================================================
#  CLI
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Duck Hunt GRPO Training")
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        required=True,
        help="Path to YAML config file (can be repeated to layer configs).",
    )
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Dot-separated override, e.g. training.learning_rate=2e-5",
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use custom GRPO training loop instead of TRL GRPOTrainer.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of prompt samples to generate (TRL mode only).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume training from (custom mode only).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push final model to Hugging Face Hub after training.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="HF Hub repo id, e.g. 'username/duckhunt-ministral-grpo'.",
    )
    args = parser.parse_args()

    cfg = load_config(args)

    # Apply HF Hub CLI overrides
    if args.push_to_hub:
        cfg.hub.push_to_hub = True
    if args.hub_model_id:
        cfg.hub.hub_model_id = args.hub_model_id

    logger.info("Config loaded: model=%s", cfg.model.model_name)

    if args.custom:
        train_custom(cfg, resume_from=args.resume)
    else:
        train_trl(cfg, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
