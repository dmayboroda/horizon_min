# Teaching a Vision Model to Play Duck Hunt with Reinforcement Learning

![Duck Hunt](duckhunt.webp)

This project fine-tunes a vision-language model to play the classic NES Duck Hunt from raw pixels. The model — [Ministral-3B](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) (8.4B LLM + 0.4B Pixtral vision encoder) — observes sequences of game frames, estimates duck velocity, predicts where ducks will be in the future, and fires by outputting tool calls: `shoot(x, y, horizon)`.

**After training, the model achieved a 60.9% hit rate** — up from near-zero for the base model and well above the ~5% random baseline.

> **[Live Demo](https://huggingface.co/spaces/dmayboroda/duck_hunt)** · **[Trained Model](https://huggingface.co/dmayboroda/dh_ministal_gpro)**

## The Challenge

Duck Hunt is deceptively hard for an AI. Ducks move fast, bounce unpredictably off screen edges, and the model must account for its own processing latency — the time between seeing frames and the shot actually landing. At 300ms latency (9 frames at 30 FPS), a duck traveling at 6 pixels/frame has moved 54 pixels by the time the bullet arrives. The model has to lead its shots.

## How It Works

```
Game Frames (512x512) -> Pixtral Vision Encoder -> Ministral LLM -> shoot(x, y, horizon)
                                                                          |
                                                      Environment simulates shot -> reward
                                                                          |
                                                                     GRPO update
```

The training pipeline uses **Group Relative Policy Optimization (GRPO)** — a reinforcement learning algorithm that samples multiple shot predictions for each game state, scores them against the environment, and updates the model toward better-rewarded outputs.

The action space is a single function call with three parameters:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `x` | float | 0.0–1.0 | Horizontal position (0=left, 1=right) |
| `y` | float | 0.0–1.0 | Vertical position (0=top, 1=bottom) |
| `horizon` | integer | 0–30 | Extra frames to wait before firing |

The total prediction distance is `processing_latency_frames + horizon`. The model must learn to lead its shots based on estimated duck velocity and the combined latency.

### Latency-Aware Training

The model is trained across six latency buckets (100–600ms), forcing it to generalize rather than memorize a single timing. A horizon penalty (`-0.1 * horizon/30`) on successful hits encourages the model to shoot quickly when it can.

### Reward Function

| Outcome | Reward |
|---------|--------|
| Hit one duck | +1.0 |
| Double kill | +2.5 |
| Miss | -0.3 |
| No target | -0.5 |
| Invalid output | -1.0 |
| Horizon penalty | -0.1 x (horizon / 30) on hits |

Two reward signals are combined: **accuracy** (did the shot hit?) and **format** (is the output a valid tool call?).

### Training Setup

- **Base model**: [Ministral-3-8B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16)
- **Fine-tuning**: LoRA (rank 16) on attention projections — ~0.2% of parameters trainable
- **Input**: 4 sequential 512x512 game frames + latency metadata
- **Output**: Mistral native tool calls (`[TOOL_CALLS]`), directly servable via vLLM/TGI with the OpenAI SDK
- **GRPO**: G=4 completions per state, clipped surrogate objective (epsilon=0.2)
- **Deterministic replay**: Game snapshots capture duck state + RNG seeds for reproducible reward computation

## Quick Start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
cd horizon_min
uv sync --extra training
```

### Train

```bash
cd training

# Run with defaults (Ministral-3-8B, TRL GRPOTrainer, W&B logging)
./run_training.sh

# Custom GRPO loop
./run_training.sh --custom

# Without W&B
./run_training.sh --no-wandb
```

### Evaluate

```bash
python evaluate.py --config configs/ministral_config.yaml \
    --checkpoint outputs/ministral_duckhunt_grpo/best_checkpoint \
    --baselines
```

### Publish to Hugging Face Hub

```bash
./run_training.sh --push-to-hub --hub-model-id username/duckhunt-ministral-grpo
```

See [training/README.md](training/README.md) for full training documentation.

## Game Environment

A headless Duck Hunt implementation with no display required — pure Python API with PIL rendering.

| Parameter | Value |
|-----------|-------|
| Screen | 800 x 500 px |
| Model input | 512 x 512 px |
| FPS | 30 |
| Match duration | 30s |
| Ducks per match | 2 |
| Bullets per match | 3 |
| Game over | 4 misses |
| Coordinates | 0.0–1.0 normalized |

### Running the Server

```bash
uv run uvicorn duck_hunt_openenv.server.app:app --reload --port 8000
```

### Client Usage

```python
from duck_hunt_openenv.duck_hunt_env.client import DuckHuntEnv
from duck_hunt_openenv.duck_hunt_env.models import ShootAction

env = DuckHuntEnv.from_local(host="localhost", port=8000)
obs = env.reset()

while not obs.done:
    action = ShootAction(x=400, y=250, horizon=5)
    obs = env.step(action)
    print(f"Result: {obs.last_action_result}, Reward: {obs.reward}")

env.close()
```

## Serving the Trained Model

The trained LoRA adapter produces standard Mistral tool calls, compatible with any OpenAI-SDK-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Ministral-3-8B-Instruct-2512-BF16 \
    --enable-lora \
    --lora-modules duckhunt=outputs/ministral_duckhunt_grpo/best_checkpoint
```

## Project Structure

```
horizon_min/
├── duck_hunt_openenv/          # Game environment
│   ├── duck_hunt_env/          # Python client
│   ├── server/                 # Game engine + PIL renderer
│   ├── experiments/            # VLM agent experiments
│   ├── tests/                  # Unit tests
│   └── assets/                 # Sprites, background, font
├── training/                   # GRPO training pipeline
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation with baselines
│   ├── configs/                # YAML configs
│   └── src/                    # Model, environment, rewards, prompts
├── demo/                       # HuggingFace Spaces demo
│   ├── app.py                  # Gradio UI + episode runner
│   └── assets/                 # Game assets
└── GAME_PARAMETERS.md          # Game mechanics reference
```

## Why This Matters

This demonstrates that small vision-language models can learn reactive, spatiotemporal reasoning from reinforcement learning alone — no human demonstrations, no reward shaping beyond hit/miss. The latency-aware design mirrors real-world deployment constraints where models must predict ahead to compensate for inference time.

## License

MIT
