"""Custom GRPO training loop for Duck Hunt VLM training.

Fallback for when TRL's ``GRPOTrainer`` does not handle the
Mistral-3 multimodal + environment-reward workflow cleanly.

Implements the core GRPO algorithm:

1. For each prompt, sample ``G`` completions.
2. Score each completion with the environment reward.
3. Normalise advantages within the group.
4. Compute the clipped surrogate objective.
5. (Optionally) add KL penalty against a frozen reference model.
"""

from __future__ import annotations

import copy
import json
import logging
import random

from PIL import Image
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_scheduler

from .config import FullConfig
from .dataset import capture_snapshot, simulate_shot
from .environment import DuckHuntEnvWrapper
from .reward import compute_reward
from .utils import build_prompt, parse_tool_call

logger = logging.getLogger(__name__)

try:
    import wandb

    _WANDB = True
except ImportError:
    _WANDB = False


# ===================================================================
#  Helpers
# ===================================================================
def _log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token-level log-probs for *labels* given *logits*.

    ``logits``  shape: (B, T, V)
    ``labels``  shape: (B, T)
    returns     shape: (B, T)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)


# ===================================================================
#  Custom GRPO Trainer
# ===================================================================
class DuckHuntGRPOTrainer:
    """Manual GRPO loop with environment-driven reward."""

    def __init__(
        self,
        model,
        processor,
        env: DuckHuntEnvWrapper,
        config: FullConfig,
    ) -> None:
        self.model = model
        self.processor = processor
        self.env = env
        self.cfg = config

        grpo = config.grpo
        train = config.training

        # Reference model (frozen copy)  — skipped when beta == 0
        self.ref_model = None
        if grpo.beta > 0:
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        # Optimiser
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train.learning_rate,
            weight_decay=train.weight_decay,
        )

        total_steps = train.max_steps if train.max_steps > 0 else 1000
        warmup_steps = int(total_steps * train.warmup_ratio)
        self.scheduler = get_scheduler(
            train.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.device = next(model.parameters()).device

        # Enable gradient checkpointing to save memory
        if train.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Checkpoint tracking
        self.global_step = 0
        self.best_eval_hit_rate = -1.0
        self._reward_baseline = 0.0  # running average for advantage when std=0
        self._saved_checkpoints: list[Path] = []
        self._last_grad_norm = 0.0
        self._total_hits = 0
        self._total_shots = 0

        # Sample outputs buffer for W&B table logging
        self._sample_outputs: list[dict] = []

        # wandb
        self._wandb_active = False
        if _WANDB and config.logging.report_to == "wandb":
            wandb.init(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity,
                name=config.logging.wandb_run_name,
                config=self._build_wandb_config(),
            )
            self._wandb_active = True

    # ------------------------------------------------------------------
    #  Build full config dict for W&B (logged once)
    # ------------------------------------------------------------------
    def _build_wandb_config(self) -> dict:
        """Flatten all config sections into a single dict for W&B."""
        cfg = self.cfg
        return {
            # Model
            "model/name": cfg.model.model_name,
            "model/dtype": cfg.model.torch_dtype,
            "model/attn_implementation": cfg.model.attn_implementation,
            # LoRA
            "lora/enabled": cfg.lora.enabled,
            "lora/r": cfg.lora.r,
            "lora/alpha": cfg.lora.lora_alpha,
            "lora/dropout": cfg.lora.lora_dropout,
            "lora/target_modules": cfg.lora.target_modules,
            "lora/task_type": cfg.lora.task_type,
            # GRPO
            "grpo/num_generations": cfg.grpo.num_generations,
            "grpo/max_completion_length": cfg.grpo.max_completion_length,
            "grpo/temperature": cfg.grpo.temperature,
            "grpo/top_p": cfg.grpo.top_p,
            "grpo/top_k": cfg.grpo.top_k,
            "grpo/beta": cfg.grpo.beta,
            "grpo/epsilon": cfg.grpo.epsilon,
            "grpo/loss_type": cfg.grpo.loss_type,
            # Training
            "training/learning_rate": cfg.training.learning_rate,
            "training/max_steps": cfg.training.max_steps,
            "training/batch_size": cfg.training.per_device_train_batch_size,
            "training/gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "training/effective_batch_size": cfg.training.effective_batch_size,
            "training/warmup_ratio": cfg.training.warmup_ratio,
            "training/weight_decay": cfg.training.weight_decay,
            "training/lr_scheduler": cfg.training.lr_scheduler_type,
            "training/max_grad_norm": cfg.training.max_grad_norm,
            "training/gradient_checkpointing": cfg.training.gradient_checkpointing,
            "training/seed": cfg.training.seed,
            # Reward
            "reward/hit": cfg.reward.hit,
            "reward/double_kill": cfg.reward.double_kill,
            "reward/miss": cfg.reward.miss,
            "reward/lambda_horizon": cfg.reward.lambda_horizon,
            "reward/invalid_action": cfg.reward.invalid_action,
            # Environment
            "env/screen_width": cfg.environment.screen_width,
            "env/screen_height": cfg.environment.screen_height,
            "env/frame_output_size": cfg.environment.frame_output_size,
            "env/max_horizon": cfg.environment.max_horizon,
            "env/fps": cfg.environment.fps,
            "env/latency_options_ms": cfg.environment.latency_options_ms,
            "env/frames_per_observation": cfg.environment.frames_per_observation,
        }

    # ------------------------------------------------------------------
    #  Main training loop
    # ------------------------------------------------------------------
    def train(self, resume_from: str | None = None) -> None:
        cfg = self.cfg
        train = cfg.training
        grpo = cfg.grpo

        total_steps = train.max_steps if train.max_steps > 0 else 1000
        start_step = 0

        # Resume from checkpoint if provided
        if resume_from is not None:
            start_step = self._load_checkpoint(resume_from)
            logger.info("Resumed from step %d", start_step)

        self.model.train()
        self.env.reset()

        for step in range(start_step, total_steps):
            # 1. Collect a batch from the environment
            batch = self._collect_batch()

            # 2. Compute GRPO loss
            loss, metrics = self._compute_grpo_loss(batch)

            # 3. Backprop (with grad accumulation)
            scaled_loss = loss / train.gradient_accumulation_steps
            scaled_loss.backward()

            if (step + 1) % train.gradient_accumulation_steps == 0:
                # Compute gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), train.max_grad_norm
                )
                self._last_grad_norm = grad_norm.item()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Track hit rate
            rewards = batch["rewards"]
            self._total_shots += len(rewards)
            self._total_hits += sum(1 for r in rewards if r > 0)
            metrics["gradient_norm"] = self._last_grad_norm
            metrics["hit_rate"] = self._total_hits / max(self._total_shots, 1)
            metrics["total_hits"] = self._total_hits
            metrics["total_shots"] = self._total_shots

            # 4. Logging
            if (step + 1) % train.logging_steps == 0:
                self._log_metrics(step, metrics)

            # 5. Save checkpoint
            if (step + 1) % train.save_steps == 0:
                self._save_checkpoint(step, metrics)

            # 6. Evaluation
            if (
                train.eval_strategy == "steps"
                and train.eval_steps > 0
                and (step + 1) % train.eval_steps == 0
            ):
                eval_metrics = self._run_eval()
                if eval_metrics is not None:
                    self._log_eval_to_wandb(eval_metrics, step)
                    # Track best checkpoint by hit_rate
                    hit_rate = eval_metrics.get("core", {}).get("hit_rate", 0.0)
                    if hit_rate > self.best_eval_hit_rate:
                        self.best_eval_hit_rate = hit_rate
                        self._save_best_checkpoint(step, eval_metrics)

        # Final save
        self._save_checkpoint(total_steps - 1, {})
        logger.info("Training complete after %d steps.", total_steps)

        # Push to HF Hub
        if cfg.hub.push_to_hub:
            self.push_to_hub()

    # ------------------------------------------------------------------
    #  Collect batch
    # ------------------------------------------------------------------
    def _collect_batch(self) -> dict:
        """Generate a batch of (prompt, completions, rewards, snapshots)."""
        grpo = self.cfg.grpo
        G = grpo.num_generations

        # Find a state with flying ducks (retry up to 20 times)
        for _attempt in range(20):
            if self.env.is_done():
                self.env.reset()
            frames = self.env.get_frames()
            state = self.env.get_state()
            if state.get("ducks_flying", 0) > 0 and len(frames) > 0:
                break
            self.env.advance_frames(random.randint(10, 30))
        else:
            logger.warning("Could not find flying ducks after 20 attempts, using current state")

        # Build prompt
        messages, tools = build_prompt(frames, state)

        # Snapshot for deterministic reward
        snap = capture_snapshot(self.env)

        # Tokenise prompt
        inputs = self.processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # Generate G completions
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=grpo.max_completion_length,
                do_sample=True,
                temperature=grpo.temperature,
                top_p=grpo.top_p,
                top_k=grpo.top_k if grpo.top_k > 0 else None,
                num_return_sequences=G,
            )
        self.model.train()

        # Decode and compute rewards
        completion_ids_list = []
        rewards = []
        completions_text = []

        for i in range(G):
            comp_ids = output_ids[i, prompt_len:]
            completion_ids_list.append(comp_ids)

            decoded = self.processor.decode(comp_ids, skip_special_tokens=False)
            completions_text.append(decoded)
            action = parse_tool_call(decoded, max_horizon=self.cfg.environment.max_horizon)

            if action is not None:
                result = simulate_shot(snap, action)
            else:
                result = {"hit_a": False, "hit_b": False, "had_target": True}

            reward = compute_reward(result, action, self.cfg.reward)
            rewards.append(reward)

            # Log full completion to console
            action_str = f"x={action.x:.3f}, y={action.y:.3f}, h={action.horizon}" if action else "INVALID"
            hit_str = ""
            if action:
                hit_str = f" hit_a={result['hit_a']} hit_b={result['hit_b']}"
            logger.info(
                "  [gen %d] reward=%.2f action=%s%s\n    output: %s",
                i, reward, action_str, hit_str, decoded,
            )

        # Collect parsed actions for W&B visualization
        parsed_actions = []
        for i in range(G):
            decoded = completions_text[i]
            parsed_actions.append(
                parse_tool_call(decoded, max_horizon=self.cfg.environment.max_horizon)
            )

        # Log frames and completions to W&B
        self._log_batch_to_wandb(frames, completions_text, rewards, parsed_actions)

        # Advance the real environment
        self.env.advance_frames(10)

        return {
            "prompt_ids": inputs["input_ids"][0],  # (T_prompt,)
            "completion_ids": completion_ids_list,  # list of G tensors
            "rewards": rewards,  # list of G floats
            "full_ids": output_ids,  # (G, T_prompt + T_comp)
            "prompt_len": prompt_len,
            "attention_mask": inputs.get("attention_mask"),
        }

    # ------------------------------------------------------------------
    #  GRPO loss computation
    # ------------------------------------------------------------------
    def _compute_grpo_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        grpo = self.cfg.grpo
        G = len(batch["rewards"])

        rewards_t = torch.tensor(batch["rewards"], dtype=torch.float32)
        prompt_len = batch["prompt_len"]

        # --- 1. Advantages (group-normalised) ---
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()
        if std_r < 1e-8:
            # All rewards identical — no gradient signal possible.
            # Return zeros to avoid NaN and prevent spurious updates.
            advantages = torch.zeros_like(rewards_t)
        else:
            advantages = (rewards_t - mean_r) / (std_r + 1e-8)
        advantages = advantages.to(self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        total_kl = 0.0
        total_entropy = 0.0
        total_clip = 0

        for i in range(G):
            full_ids = batch["full_ids"][i].unsqueeze(0)  # (1, T)
            comp_ids = batch["completion_ids"][i]
            comp_len = comp_ids.shape[0]

            if comp_len == 0:
                continue

            # Labels: -100 for prompt tokens, real ids for completion
            labels = full_ids.clone()
            labels[0, :prompt_len] = -100

            # --- 2. Policy log-probs ---
            outputs = self.model(input_ids=full_ids, labels=labels)
            logits = outputs.logits[:, prompt_len - 1 : -1, :]  # align with comp
            comp_labels = full_ids[:, prompt_len:]

            log_probs = _log_probs_from_logits(logits, comp_labels)  # (1, comp_len)

            # --- 2b. Token-level entropy (prevents mode collapse) ---
            probs = F.softmax(logits, dim=-1)
            log_probs_full = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs_full).sum(dim=-1).mean()  # scalar
            total_entropy += entropy.item()

            # --- 3. Reference log-probs ---
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_out = self.ref_model(input_ids=full_ids, labels=labels)
                    ref_logits = ref_out.logits[:, prompt_len - 1 : -1, :]
                    ref_log_probs = _log_probs_from_logits(ref_logits, comp_labels)
            else:
                ref_log_probs = log_probs.detach()

            # --- 4. Ratio + clipped surrogate ---
            # For the first iteration (num_iterations=1), old_log_probs == log_probs
            # so ratio ≈ 1; the gradient still flows through log_probs.
            ratio = torch.exp(log_probs - log_probs.detach())
            adv = advantages[i]

            surr1 = ratio * adv
            surr2 = torch.clamp(
                ratio, 1.0 - grpo.epsilon, 1.0 + grpo.epsilon
            ) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # --- 5. KL penalty ---
            kl = (log_probs - ref_log_probs).mean()
            total_kl += kl.item()

            # --- 6. Entropy bonus (subtract to maximize entropy) ---
            loss_i = policy_loss + grpo.beta * kl - grpo.entropy_coeff * entropy
            total_loss = total_loss + loss_i

        total_loss = total_loss / max(G, 1)

        # --- 7. Entropy floor penalty (emergency brake) ---
        mean_entropy = total_entropy / max(G, 1)
        entropy_floor_penalty = 0.0
        if grpo.entropy_floor > 0 and mean_entropy < grpo.entropy_floor:
            entropy_floor_penalty = grpo.entropy_floor_coeff * (grpo.entropy_floor - mean_entropy)
            total_loss = total_loss + entropy_floor_penalty
            logger.warning(
                "Entropy below floor: %.2f < %.2f, adding penalty %.4f",
                mean_entropy, grpo.entropy_floor, entropy_floor_penalty,
            )

        metrics = {
            "loss": total_loss.item(),
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item(),
            "mean_kl": total_kl / max(G, 1),
            "mean_entropy": mean_entropy,
            "entropy_floor_penalty": entropy_floor_penalty,
            "advantages_mean": advantages.mean().item(),
        }

        return total_loss, metrics

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    def _run_eval(self) -> dict | None:
        """Run evaluation and return metrics dict."""
        try:
            # Lazy import to avoid circular dependency
            import importlib
            import sys as _sys

            # evaluate.py is at training/evaluate.py — parent of src/
            eval_dir = str(Path(__file__).resolve().parent.parent)
            if eval_dir not in _sys.path:
                _sys.path.insert(0, eval_dir)

            from evaluate import evaluate as eval_fn

            logger.info("Running evaluation at step %d …", self.global_step)
            metrics = eval_fn(
                model=self.model,
                processor=self.processor,
                env=self.env,
                config=self.cfg,
                num_episodes=2,  # keep eval fast during training
                max_steps_per_episode=50,
            )
            logger.info(
                "Eval hit_rate=%.1f%%  avg_reward=%.3f",
                metrics.get("core", {}).get("hit_rate", 0) * 100,
                metrics.get("core", {}).get("average_reward", 0),
            )
            return metrics
        except Exception as e:
            logger.warning("Evaluation failed: %s", e)
            return None

    # ------------------------------------------------------------------
    #  Logging
    # ------------------------------------------------------------------
    # Crosshair sprite (loaded once, shared across instances)
    _crosshair_sprite: Image.Image | None = None

    @classmethod
    def _get_crosshair_sprite(cls) -> Image.Image:
        """Load the crosshair sprite from assets (cached)."""
        if cls._crosshair_sprite is None:
            from pathlib import Path
            assets = Path(__file__).resolve().parent.parent.parent / "duck_hunt_openenv" / "assets"
            sprite_path = assets / "crosshairs.png"
            cls._crosshair_sprite = Image.open(sprite_path).convert("RGBA")
        return cls._crosshair_sprite

    @staticmethod
    def _draw_crosshair(
        frame: Image.Image,
        x_norm: float,
        y_norm: float,
        color: tuple = (255, 0, 0),
        label: str | None = None,
    ) -> Image.Image:
        """Draw a crosshair sprite on a copy of the frame at normalised (x, y)."""
        from PIL import ImageDraw, ImageChops

        img = frame.copy().convert("RGBA")
        w, h = img.size
        cx = int(x_norm * w)
        cy = int(y_norm * h)

        # Get crosshair sprite and tint it
        raw = DuckHuntGRPOTrainer._get_crosshair_sprite()
        # Scale crosshair to ~8% of frame width
        ch_size = max(24, w // 12)
        sprite = raw.resize((ch_size, ch_size), Image.LANCZOS)

        # Tint the sprite: replace non-transparent pixels with the color
        r_ch, g_ch, b_ch, a_ch = sprite.split()
        tinted = Image.merge("RGBA", (
            r_ch.point(lambda _: color[0]),
            g_ch.point(lambda _: color[1]),
            b_ch.point(lambda _: color[2]),
            a_ch,
        ))

        # Paste centered on (cx, cy)
        paste_x = cx - ch_size // 2
        paste_y = cy - ch_size // 2
        img.paste(tinted, (paste_x, paste_y), tinted)

        # Label text
        if label:
            draw = ImageDraw.Draw(img)
            draw.text(
                (cx + ch_size // 2 + 4, cy - 8),
                label, fill=color + (255,),
            )

        return img.convert("RGB")

    def _log_batch_to_wandb(
        self,
        frames: list,
        completions: list[str],
        rewards: list[float],
        actions: list | None = None,
    ) -> None:
        """Log observation frames (with crosshairs) and completions to W&B."""
        if not self._wandb_active or wandb.run is None:
            return

        log_dict: dict = {}

        # Log raw observation frames
        wandb_images = []
        for i, frame in enumerate(frames):
            wandb_images.append(wandb.Image(frame, caption=f"frame_{i}"))
        if wandb_images:
            log_dict["train/observation_frames"] = wandb_images

        # Log last frame with crosshairs for each valid action
        if actions and frames:
            last_frame = frames[-1]  # most recent frame
            colors = [
                (255, 0, 0),    # red
                (0, 255, 0),    # green
                (0, 100, 255),  # blue
                (255, 255, 0),  # yellow
                (255, 0, 255),  # magenta
                (0, 255, 255),  # cyan
            ]
            shot_images = []
            for i, action in enumerate(actions):
                if action is not None:
                    color = colors[i % len(colors)]
                    hit_str = "hit" if rewards[i] > 0 else "miss"
                    label = f"g{i} h={action.horizon} {hit_str} r={rewards[i]:.2f}"
                    annotated = self._draw_crosshair(
                        last_frame, action.x, action.y,
                        color=color, label=label,
                    )
                    shot_images.append(wandb.Image(
                        annotated,
                        caption=f"gen{i}: ({action.x:.2f},{action.y:.2f}) h={action.horizon} {hit_str}",
                    ))
            if shot_images:
                log_dict["train/shot_predictions"] = shot_images

            # Also log a single combined image with ALL crosshairs
            if any(a is not None for a in actions):
                combined = last_frame.copy().convert("RGB")
                for i, action in enumerate(actions):
                    if action is not None:
                        color = colors[i % len(colors)]
                        hit_str = "hit" if rewards[i] > 0 else "miss"
                        label = f"g{i} h={action.horizon}"
                        combined = self._draw_crosshair(
                            combined, action.x, action.y,
                            color=color, label=label,
                        )
                log_dict["train/all_shots_overlay"] = wandb.Image(
                    combined, caption="All generation predictions",
                )

        # Log completions as a table
        comp_table = wandb.Table(columns=["gen_idx", "completion", "reward"])
        for i, (text, reward) in enumerate(zip(completions, rewards)):
            comp_table.add_data(i, text, reward)
        log_dict["train/completions"] = comp_table

        wandb.log(log_dict)

    def _log_metrics(self, step: int, metrics: dict) -> None:
        lr = self.scheduler.get_last_lr()[0]
        grad_norm = metrics.get("gradient_norm", 0.0)
        hit_rate = metrics.get("hit_rate", 0.0)
        entropy = metrics.get("mean_entropy", 0.0)
        msg = (
            f"step={step + 1}  "
            f"loss={metrics.get('loss', 0):.4f}  "
            f"reward={metrics.get('mean_reward', 0):.3f} +/- {metrics.get('std_reward', 0):.3f}  "
            f"hits={metrics.get('total_hits', 0)}/{metrics.get('total_shots', 0)} ({hit_rate:.1%})  "
            f"entropy={entropy:.2f}  "
            f"grad={grad_norm:.4f}  "
            f"lr={lr:.2e}"
        )
        logger.info(msg)

        if self._wandb_active and wandb.run is not None:
            log_dict = {
                "train/loss": metrics.get("loss", 0),
                "train/mean_reward": metrics.get("mean_reward", 0),
                "train/std_reward": metrics.get("std_reward", 0),
                "train/learning_rate": lr,
                "train/advantages_mean": metrics.get("advantages_mean", 0),
                "train/gradient_norm": grad_norm,
                "train/hit_rate": hit_rate,
                "train/mean_entropy": entropy,
                "train/total_hits": metrics.get("total_hits", 0),
                "train/step": step + 1,
            }
            wandb.log(log_dict)

    def _log_eval_to_wandb(self, eval_metrics: dict, step: int) -> None:
        """Log rich evaluation data to W&B: metrics, histogram, table, chart."""
        if not self._wandb_active or wandb.run is None:
            return

        log_dict: dict = {}

        # ---- All evaluation metrics ----
        core = eval_metrics.get("core", {})
        for k, v in core.items():
            if isinstance(v, (int, float)):
                log_dict[f"eval/{k}"] = v

        horizon = eval_metrics.get("horizon", {})
        for k, v in horizon.items():
            if isinstance(v, (int, float)):
                log_dict[f"eval/horizon_{k}"] = v

        hw = eval_metrics.get("hardware_aware", {})
        for k, v in hw.items():
            if isinstance(v, (int, float)):
                log_dict[f"eval/{k}"] = v

        # ---- Horizon distribution (histogram) ----
        by_latency = eval_metrics.get("by_latency", {})
        all_horizons = []
        for lat_data in by_latency.values():
            avg_h = lat_data.get("average_horizon", 0)
            if avg_h > 0:
                all_horizons.append(avg_h)
        if all_horizons:
            log_dict["eval/horizon_distribution"] = wandb.Histogram(all_horizons)

        # ---- Hit rate by latency (line chart data) ----
        if by_latency:
            lat_table = wandb.Table(
                columns=["latency_ms", "hit_rate", "avg_horizon", "avg_reward"],
            )
            for lat_ms in sorted(by_latency.keys()):
                m = by_latency[lat_ms]
                lat_table.add_data(
                    lat_ms,
                    m.get("hit_rate", 0),
                    m.get("average_horizon", 0),
                    m.get("average_reward", 0),
                )
            log_dict["eval/by_latency"] = lat_table

        # ---- Sample outputs (table) ----
        if self._sample_outputs:
            recent = self._sample_outputs[-self.cfg.logging.num_completions_to_print:]
            sample_table = wandb.Table(
                columns=["step", "gen_idx", "output", "action", "reward", "hit"],
            )
            for s in recent:
                sample_table.add_data(
                    s["step"], s["generation_idx"],
                    s["output"][:200], s["action"],
                    s["reward"], s["hit"],
                )
            log_dict["eval/sample_outputs"] = sample_table

        log_dict["eval/best_hit_rate"] = self.best_eval_hit_rate

        wandb.log(log_dict)

    # ------------------------------------------------------------------
    #  Checkpointing (9.1)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, step: int, metrics: dict) -> None:
        """Save a full checkpoint: model, optimizer, scheduler, trainer state."""
        out_dir = Path(self.cfg.training.output_dir) / f"checkpoint-{step + 1}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. LoRA weights (adapter_model.safetensors + adapter_config.json)
        self.model.save_pretrained(str(out_dir))
        self.processor.save_pretrained(str(out_dir))

        # 2. Optimizer state
        torch.save(self.optimizer.state_dict(), str(out_dir / "optimizer.pt"))

        # 3. Scheduler state
        torch.save(self.scheduler.state_dict(), str(out_dir / "scheduler.pt"))

        # 4. Trainer state (step, config, metrics)
        trainer_state = {
            "global_step": self.global_step,
            "step": step + 1,
            "best_eval_hit_rate": self.best_eval_hit_rate,
            "last_metrics": {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            },
            "config": {
                "model_name": self.cfg.model.model_name,
                "learning_rate": self.cfg.training.learning_rate,
                "lora_r": self.cfg.lora.r,
                "grpo_beta": self.cfg.grpo.beta,
                "num_generations": self.cfg.grpo.num_generations,
                "max_steps": self.cfg.training.max_steps,
                "output_dir": self.cfg.training.output_dir,
            },
        }
        with open(out_dir / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=2)

        logger.info("Saved checkpoint to %s", out_dir)

        # 5. Manage checkpoint rotation (save_total_limit)
        self._saved_checkpoints.append(out_dir)
        limit = self.cfg.training.save_total_limit
        if limit > 0 and len(self._saved_checkpoints) > limit:
            oldest = self._saved_checkpoints.pop(0)
            # Don't delete best_checkpoint
            if oldest.name != "best_checkpoint" and oldest.exists():
                shutil.rmtree(oldest)
                logger.info("Removed old checkpoint %s (save_total_limit=%d)", oldest, limit)

    def _save_best_checkpoint(self, step: int, eval_metrics: dict) -> None:
        """Save the best checkpoint (by eval hit_rate)."""
        best_dir = Path(self.cfg.training.output_dir) / "best_checkpoint"
        best_dir.mkdir(parents=True, exist_ok=True)

        # Model + processor
        self.model.save_pretrained(str(best_dir))
        self.processor.save_pretrained(str(best_dir))

        # Optimizer + scheduler
        torch.save(self.optimizer.state_dict(), str(best_dir / "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), str(best_dir / "scheduler.pt"))

        # Trainer state with eval metrics
        trainer_state = {
            "global_step": self.global_step,
            "step": step + 1,
            "best_eval_hit_rate": self.best_eval_hit_rate,
            "eval_metrics": eval_metrics,
            "config": {
                "model_name": self.cfg.model.model_name,
                "learning_rate": self.cfg.training.learning_rate,
                "lora_r": self.cfg.lora.r,
                "grpo_beta": self.cfg.grpo.beta,
                "num_generations": self.cfg.grpo.num_generations,
            },
        }
        with open(best_dir / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=2, default=str)

        logger.info(
            "New best checkpoint saved (hit_rate=%.1f%%) at step %d",
            self.best_eval_hit_rate * 100, step + 1,
        )

    def _load_checkpoint(self, checkpoint_dir: str) -> int:
        """Resume training from a saved checkpoint.

        Returns the step number to resume from.
        """
        ckpt = Path(checkpoint_dir)

        # 1. Load model weights (LoRA adapter)
        from peft import PeftModel

        if hasattr(self.model, "load_adapter"):
            self.model.load_adapter(str(ckpt), adapter_name="default")
        else:
            logger.warning("Model does not support load_adapter; skipping weight load.")

        # 2. Optimizer state
        opt_path = ckpt / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(torch.load(str(opt_path), map_location=self.device))
            logger.info("Loaded optimizer state from %s", opt_path)

        # 3. Scheduler state
        sched_path = ckpt / "scheduler.pt"
        if sched_path.exists():
            self.scheduler.load_state_dict(torch.load(str(sched_path), map_location=self.device))
            logger.info("Loaded scheduler state from %s", sched_path)

        # 4. Trainer state
        state_path = ckpt / "trainer_state.json"
        resume_step = 0
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            resume_step = state.get("step", 0)
            self.global_step = state.get("global_step", 0)
            self.best_eval_hit_rate = state.get("best_eval_hit_rate", -1.0)
            logger.info(
                "Loaded trainer state: step=%d, global_step=%d, best_hit_rate=%.1f%%",
                resume_step, self.global_step, self.best_eval_hit_rate * 100,
            )

        return resume_step

    # ------------------------------------------------------------------
    #  Hugging Face Hub upload
    # ------------------------------------------------------------------
    def push_to_hub(self, checkpoint_dir: str | None = None) -> None:
        """Push model checkpoint to Hugging Face Hub.

        Parameters
        ----------
        checkpoint_dir : str, optional
            Path to the checkpoint to upload.  Defaults to
            ``best_checkpoint`` if it exists, otherwise the latest
            checkpoint in the output directory.
        """
        from huggingface_hub import HfApi, ModelCard, ModelCardData

        hub = self.cfg.hub
        repo_id = hub.hub_model_id
        if not repo_id:
            logger.warning("hub.hub_model_id is not set — skipping push_to_hub.")
            return

        # Resolve which checkpoint to upload
        if checkpoint_dir is not None:
            upload_dir = Path(checkpoint_dir)
        else:
            best = Path(self.cfg.training.output_dir) / "best_checkpoint"
            if best.exists():
                upload_dir = best
            else:
                # Fall back to latest checkpoint-N/
                out = Path(self.cfg.training.output_dir)
                ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
                if not ckpts:
                    logger.warning("No checkpoints found to push.")
                    return
                upload_dir = ckpts[-1]

        logger.info("Pushing %s to HF Hub: %s …", upload_dir, repo_id)

        api = HfApi()

        # Create repo if it doesn't exist
        api.create_repo(repo_id, private=hub.hub_private, exist_ok=True)

        # Upload adapter files
        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            ignore_patterns=["optimizer.pt", "scheduler.pt"],
        )

        # Create a model card
        trainer_state_path = upload_dir / "trainer_state.json"
        eval_info = ""
        if trainer_state_path.exists():
            with open(trainer_state_path) as f:
                state = json.load(f)
            hit_rate = state.get("best_eval_hit_rate", state.get("eval_metrics", {}).get("core", {}).get("hit_rate"))
            if hit_rate is not None:
                eval_info = f"- **Best eval hit rate**: {hit_rate:.1%}\n"

        card_content = f"""\
---
library_name: peft
base_model: {self.cfg.model.model_name}
tags:
  - grpo
  - reinforcement-learning
  - duck-hunt
  - vision-language-model
  - tool-calling
---

# {repo_id.split('/')[-1]}

LoRA adapter for [{self.cfg.model.model_name}](https://huggingface.co/{self.cfg.model.model_name}),
fine-tuned with GRPO to play Duck Hunt.

## Training

- **Method**: Group Relative Policy Optimization (GRPO)
- **LoRA rank**: {self.cfg.lora.r}, alpha: {self.cfg.lora.lora_alpha}
- **Target modules**: {', '.join(self.cfg.lora.target_modules)}
- **Learning rate**: {self.cfg.training.learning_rate}
- **Generations per prompt**: {self.cfg.grpo.num_generations}
{eval_info}
## Usage

The adapter produces Mistral native tool calls (`[TOOL_CALLS]`), compatible with
OpenAI SDK via vLLM or TGI.

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
