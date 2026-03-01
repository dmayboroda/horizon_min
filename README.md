# Horizon Min - Duck Hunt VLM Environment

![Duck Hunt](duckhunt.webp)

A headless Duck Hunt game environment designed for training and evaluating Vision Language Models (VLMs). This project provides an API-first implementation of the classic Nintendo Duck Hunt game as a research platform for AI agents, paired with a full GRPO training pipeline for fine-tuning multimodal models to play the game.

## Project Structure

```
horizon_min/
├── duck_hunt_openenv/          # OpenEnv game environment
│   ├── duck_hunt_env/          # Python client package
│   ├── server/                 # FastAPI backend + game engine
│   ├── experiments/            # VLM agent experiments (Weave)
│   ├── tests/                  # Unit and integration tests
│   └── assets/                 # Game sprites and backgrounds
├── training/                   # GRPO training pipeline
│   ├── train.py                # Main training entry point
│   ├── evaluate.py             # Evaluation with baselines
│   ├── run_training.sh         # Launch script
│   ├── configs/                # YAML training configs
│   └── src/                    # Training modules
│       ├── config.py           # Configuration dataclasses
│       ├── model.py            # Model loading + LoRA
│       ├── environment.py      # Direct env wrapper (no HTTP)
│       ├── utils.py            # Prompts, tool schema, parsing
│       ├── reward.py           # Reward computation
│       ├── dataset.py          # Dataset generation + snapshot system
│       └── trainer.py          # Custom GRPO trainer
├── duckhunt/                   # Original pygame-based game (reference)
└── GAME_PARAMETERS.md          # Game mechanics documentation
```

## Features

- **Headless Design**: No display required, pure Python API
- **VLM-Ready**: Base64 frame encoding, OpenAI function calling support
- **GRPO Training**: Full pipeline for fine-tuning VLMs with reinforcement learning
- **OpenAI SDK Compatible**: Trained models serve via vLLM/TGI with standard `tools`/`tool_choice`
- **Latency-Aware**: Simulates real-world inference latency (100-600ms) during training
- **Experiment Tracking**: W&B integration for training metrics and Weave for experiments
- **Configurable**: Layered YAML configs with CLI overrides

## Quick Start

### Environment Setup

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
cd horizon_min
uv sync

# With training dependencies
uv sync --extra training
```

### Running the Server

```bash
# With uv
uv run uvicorn duck_hunt_openenv.server.app:app --reload --port 8000

# Or directly
cd duck_hunt_openenv/server
uvicorn app:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### Training a Model

```bash
cd training

# Run with defaults (Ministral-3-8B, TRL GRPOTrainer, W&B logging)
./run_training.sh

# Custom GRPO loop
./run_training.sh --custom

# Without W&B
./run_training.sh --no-wandb
```

See [training/README.md](training/README.md) for full training documentation.

### Publishing to Hugging Face Hub

```bash
cd training

# Push after training
./run_training.sh --push-to-hub --hub-model-id username/duckhunt-ministral-grpo
```

### Evaluating a Checkpoint

```bash
cd training

python evaluate.py --config configs/ministral_config.yaml \
    --checkpoint outputs/ministral_duckhunt_grpo/best_checkpoint \
    --baselines
```

## Training Pipeline

The training pipeline uses **Group Relative Policy Optimization (GRPO)** to fine-tune [Ministral-3-8B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) (8.4B LLM + 0.4B Pixtral vision encoder).

```
Game Frames (512x512) -> Pixtral Vision Encoder -> Ministral LLM -> shoot(x, y, horizon)
                                                                          |
                                                      Environment simulates shot -> reward
                                                                          |
                                                                     GRPO update
```

The model observes consecutive game frames, estimates duck velocity, and predicts where ducks will be after `processing_latency + horizon` frames. It outputs Mistral native tool calls, making the trained adapter directly servable via OpenAI-compatible APIs.

**Key design choices:**
- **LoRA fine-tuning** (rank 16, ~0.2% trainable params) on attention projections
- **Dual reward**: accuracy (environment simulation) + format (valid tool call structure)
- **Deterministic replay**: Snapshots capture duck state + RNG for reproducible reward computation
- **Latency curriculum**: Training across 6 latency buckets (100-600ms) for robust generalization

## Game Mechanics

| Parameter | Value | Description |
|-----------|-------|-------------|
| Screen Size | 800 x 500 | Game viewport |
| Frame Output | 512 x 512 | Resized for model input |
| FPS | 30 | Frames per second |
| Match Duration | 30s | Time limit per match |
| Ducks per Match | 2 | Simultaneous ducks |
| Bullets per Match | 3 | Shots available |
| Max Misses | 4 | Game over threshold |
| Coordinates | 0.0-1.0 | Normalised (x: left-right, y: top-bottom) |

### Rewards

| Event | Reward |
|-------|--------|
| Hit | +1.0 |
| Double Kill | +2.5 |
| Miss | -0.3 |
| No Target | -0.5 |
| Invalid Output | -1.0 |
| Horizon Penalty | -0.1 x (horizon / 30) on hits |

## Client Usage

```python
from duck_hunt_openenv.duck_hunt_env.client import DuckHuntEnv
from duck_hunt_openenv.duck_hunt_env.models import ShootAction

# Connect to server
env = DuckHuntEnv.from_local(host="localhost", port=8000)

# Reset environment
obs = env.reset()

# Game loop
while not obs.done:
    action = ShootAction(x=400, y=250, horizon=5)
    obs = env.step(action)
    print(f"Result: {obs.last_action_result}, Reward: {obs.reward}")

env.close()
```

## VLM Agent Experiments

```bash
# Install with experiment dependencies
uv sync --extra experiments

# Configure API keys
export OPENAI_API_KEY="your-key"
wandb login

# Single episode with GPT-4o
uv run python -m duck_hunt_openenv.experiments.run --mode single --model gpt-4o

# Batch evaluation
uv run python -m duck_hunt_openenv.experiments.run --mode batch --episodes 10 --model gpt-4o-mini
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment |
| `/step` | POST | Execute action |

### Action Format

```json
{
  "x": 400,
  "y": 250,
  "horizon": 5,
  "confidence": "high"
}
```

## Running Tests

```bash
# Unit tests
uv run pytest duck_hunt_openenv/tests/test_game.py -v

# Integration tests (random agent)
uv run python duck_hunt_openenv/tests/test_random_agent.py --episodes 100
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run all tests
uv run pytest

# Format code
uv run ruff format .

# Lint
uv run ruff check .
```

## License

MIT
