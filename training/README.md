# Duck Hunt GRPO Training

Train [Ministral-3-8B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) to play Duck Hunt using **Group Relative Policy Optimization (GRPO)** with the [Duck Hunt OpenEnv](../duck_hunt_openenv/).

The model observes sequences of game frames through its Pixtral vision encoder and learns to predict where ducks will be after a variable number of future frames, outputting `shoot(x, y, horizon)` tool calls compatible with the OpenAI SDK.

## Architecture

```
                   Game Frames (512x512)
                          |
                   Pixtral Vision Encoder (0.4B)
                          |
                   Ministral LLM (8.4B)
                          |
               [TOOL_CALLS] shoot(x, y, horizon)
                          |
              Environment simulates shot -> reward
                          |
                     GRPO update
```

**Model**: 8.4B LLM + 0.4B vision encoder = ~9B params (BF16). Fits in 24GB VRAM, <12GB quantised.

**Output format**: Mistral native tool calling tokens (`[TOOL_CALLS]`), servable via vLLM/TGI with standard OpenAI `tools`/`tool_choice` parameters.

## Quick Start

```bash
cd training

# Run with defaults (TRL GRPOTrainer, W&B logging)
./run_training.sh

# Or without W&B
./run_training.sh --no-wandb

# Custom GRPO loop (fallback)
./run_training.sh --custom
```

## Requirements

- Python 3.10+
- CUDA GPU with 24GB+ VRAM (BF16) or 12GB+ (quantised)
- The Duck Hunt OpenEnv game engine (included in this repo)

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.4`, `transformers>=4.45`, `trl>=0.15`, `peft>=0.13`, `flash-attn>=2.6`, `wandb>=0.18`.

## Project Structure

```
training/
├── run_training.sh              # Launch script
├── train.py                     # Main entry point (TRL + custom modes)
├── evaluate.py                  # Evaluation with baselines
├── requirements.txt
├── configs/
│   ├── base_config.yaml         # Default hyperparameters
│   └── ministral_config.yaml    # Ministral-specific overrides
└── src/
    ├── config.py                # Configuration dataclasses + YAML loading
    ├── model.py                 # Model/processor loading, LoRA setup
    ├── environment.py           # DuckHuntEnvWrapper (direct, no HTTP)
    ├── utils.py                 # Tool schema, prompt builder, output parser
    ├── reward.py                # Reward computation
    ├── dataset.py               # Dataset generation, snapshot system, TRL reward fns
    └── trainer.py               # Custom GRPO trainer with checkpointing + W&B
```

## Training Modes

### TRL GRPOTrainer (default)

Uses HuggingFace TRL's `GRPOTrainer`. Generates an offline dataset of `(prompt, images, snapshot)` tuples, then trains with two reward functions:

```bash
python train.py --config configs/ministral_config.yaml --num-samples 1000
```

### Custom GRPO Loop

Fallback for when TRL doesn't handle the multimodal + environment-reward workflow cleanly. Runs the full GRPO algorithm manually with online environment interaction:

```bash
python train.py --config configs/ministral_config.yaml --custom
```

Supports resuming from checkpoints:

```bash
python train.py --config configs/ministral_config.yaml --custom \
    --resume outputs/ministral_duckhunt_grpo/checkpoint-500
```

## How It Works

### Observation

The model receives 4 consecutive 512x512 game frames (oldest to newest) plus game state metadata (round, bullets remaining, latency). This lets it estimate duck velocity from frame-to-frame displacement.

### Action Space

A single tool call with three parameters:

| Parameter | Type    | Range   | Description                              |
|-----------|---------|---------|------------------------------------------|
| `x`       | float   | 0.0-1.0 | Horizontal position (0=left, 1=right)    |
| `y`       | float   | 0.0-1.0 | Vertical position (0=top, 1=bottom)      |
| `horizon` | integer | 0-30    | Extra frames to wait before shooting     |

The total prediction distance is `processing_latency_frames + horizon`. The model must learn to lead its shots based on estimated duck velocity and the combined latency.

### Reward Function

| Outcome         | Reward |
|-----------------|--------|
| Hit one duck    | +1.0   |
| Double kill     | +2.5   |
| Miss            | -0.3   |
| No target       | -0.5   |
| Invalid output  | -1.0   |

Hits incur a horizon penalty: `-0.1 * (horizon / 30)` to encourage faster shots when possible.

Two reward signals are combined during training:
- **Accuracy reward** (weight 1.0): Environment simulation determines if the shot hits
- **Format reward** (weight 0.3): Bonus for producing a valid tool call structure

### GRPO Algorithm

For each game state:

1. Sample `G=4` completions from the current policy
2. Score each completion against the environment (deterministic replay via snapshot)
3. Normalise advantages within the group
4. Compute the clipped surrogate objective (PPO-style, epsilon=0.2)
5. Optionally add KL penalty against a frozen reference model (beta=0.0 by default)

### Deterministic Replay

To compute rewards for the TRL path (offline dataset), the system captures a **snapshot** of the game state at each training sample — duck positions, velocities, RNG state. During reward evaluation, it restores this snapshot and deterministically simulates the shot, ensuring reproducible outcomes regardless of when the reward function is called.

## Configuration

Configs use a layered YAML system. The base config sets defaults; model-specific configs override them:

```bash
# Single config
python train.py --config configs/ministral_config.yaml

# Layered (base + overrides)
python train.py \
    --config configs/base_config.yaml \
    --config configs/ministral_config.yaml

# CLI overrides (dot-separated)
python train.py --config configs/ministral_config.yaml \
    --override training.learning_rate=2e-6 \
    --override grpo.num_generations=8
```

### Key Hyperparameters

| Parameter | Default (Ministral) | Description |
|-----------|---------------------|-------------|
| `training.learning_rate` | 5e-6 | Peak learning rate |
| `training.per_device_train_batch_size` | 1 | Batch size per GPU |
| `training.gradient_accumulation_steps` | 8 | Effective batch = 8 |
| `grpo.num_generations` | 4 | Completions per prompt |
| `grpo.temperature` | 0.7 | Sampling temperature |
| `grpo.beta` | 0.0 | KL penalty (0 = no ref model) |
| `grpo.epsilon` | 0.2 | Clipping range |
| `lora.r` | 16 | LoRA rank |
| `lora.lora_alpha` | 32 | LoRA scaling |
| `training.save_steps` | 100 | Checkpoint interval |
| `training.eval_steps` | 100 | Evaluation interval |

### LoRA

Fine-tuning uses LoRA on the attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with rank 16. Only ~0.2% of parameters are trainable.

## Checkpointing

The custom trainer saves full checkpoints at every `save_steps`:

```
outputs/ministral_duckhunt_grpo/
├── checkpoint-100/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── trainer_state.json
├── checkpoint-200/
├── best_checkpoint/          # Highest eval hit_rate
└── final/
```

- Old checkpoints rotate out based on `save_total_limit` (default: 3)
- `best_checkpoint/` is preserved regardless of rotation
- Resume with `--resume outputs/.../checkpoint-N`

## Logging (W&B)

When `report_to: wandb` is set (default):

**Per step**: loss, learning_rate, gradient_norm, mean_reward, std_reward, KL divergence

**Per eval**: hit_rate, double_kill_rate, miss_rate, average_reward, horizon distribution histogram, sample outputs table, hit rate by latency chart

**Once**: full config (model, LoRA, GRPO, training, reward, environment parameters)

Disable with `--no-wandb` or `--override logging.report_to=none`.

## Pushing to Hugging Face Hub

Upload trained checkpoints to HF Hub after training:

```bash
# Via CLI flags
./run_training.sh --push-to-hub --hub-model-id username/duckhunt-ministral-grpo

# Via config override
python train.py --config configs/ministral_config.yaml \
    --override hub.push_to_hub=true \
    --override hub.hub_model_id=username/duckhunt-ministral-grpo
```

Or set it in your YAML config:

```yaml
hub:
  push_to_hub: true
  hub_model_id: "username/duckhunt-ministral-grpo"
  hub_private: true  # default
```

What gets uploaded:
- LoRA adapter weights (`adapter_model.safetensors`, `adapter_config.json`)
- Processor/tokenizer files
- Auto-generated model card with training details
- Optimizer and scheduler states are excluded to keep the repo lightweight

The best checkpoint (by eval hit_rate) is uploaded by default. If no evaluation was run, the latest checkpoint is used instead.

Requires authentication: `huggingface-cli login` before training.

## Evaluation

```bash
# Evaluate a checkpoint
python evaluate.py --config configs/ministral_config.yaml \
    --checkpoint outputs/ministral_duckhunt_grpo/best_checkpoint \
    --num-episodes 5

# With baselines comparison
python evaluate.py --config configs/ministral_config.yaml \
    --checkpoint outputs/ministral_duckhunt_grpo/best_checkpoint \
    --baselines --output results.json
```

Evaluation runs across all latency buckets (100-600ms) and reports:

- **Core**: hit_rate, double_kill_rate, miss_rate, invalid_action_rate, average_reward
- **Horizon**: average, std, min, max
- **Per-latency**: hit_rate and average_horizon broken down by latency bucket
- **Hardware-aware**: generalization_gap (best - worst latency), adaptation_score (correlation between latency and horizon)

Baselines:
- **Random**: uniform random x, y (upper half), horizon
- **Fixed horizon**: centre shot (0.5, 0.25) with horizon=10

## Latency Simulation

The environment simulates real-world inference latency. During training, `processing_latency_frames` is randomly sampled from `[100, 200, 300, 400, 500, 600]` ms (converted to frames at 30 FPS = `[3, 6, 9, 12, 15, 18]` frames). The model sees the latency value in its prompt and must learn to adjust its `horizon` parameter accordingly — higher latency requires more aggressive prediction.

## Serving the Trained Model

The trained LoRA adapter produces standard Mistral tool calls, compatible with any OpenAI-SDK-compatible server:

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Ministral-3-8B-Instruct-2512-BF16 \
    --enable-lora \
    --lora-modules duckhunt=outputs/ministral_duckhunt_grpo/best_checkpoint

# Client (OpenAI SDK)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="duckhunt",
    messages=[...],
    tools=[{
        "type": "function",
        "function": {
            "name": "shoot",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "horizon": {"type": "integer"}
                },
                "required": ["x", "y", "horizon"]
            }
        }
    }],
    tool_choice="required",
)
```
