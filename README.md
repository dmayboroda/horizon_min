# Horizon Min - Duck Hunt VLM Environment

![Duck Hunt](duckhunt.webp)

A headless Duck Hunt game environment designed for training and evaluating Vision Language Models (VLMs). This project provides an API-first implementation of the classic Nintendo Duck Hunt game as a research platform for AI agents.

## Project Structure

```
horizon_min/
├── duck_hunt_openenv/          # Main OpenEnv implementation
│   ├── duck_hunt_env/          # Python client package
│   ├── server/                 # FastAPI backend
│   ├── experiments/            # VLM agent experiments (Weave)
│   ├── tests/                  # Unit and integration tests
│   └── assets/                 # Game sprites and backgrounds
├── duckhunt/                   # Original pygame-based game (reference)
└── GAME_PARAMETERS.md          # Game mechanics documentation
```

## Features

- **Headless Design**: No display required, pure Python API
- **VLM-Ready**: Base64 frame encoding, OpenAI function calling support
- **Experiment Tracking**: W&B Weave integration for reproducible research
- **Configurable**: All game parameters in a single config file
- **Well-Tested**: Unit and integration test coverage

## Quick Start

### Installation with uv

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
cd horizon_min
uv sync
```

### Installation with pip

```bash
cd horizon_min
pip install -e .
```

## Running the Server

```bash
# With uv
uv run uvicorn duck_hunt_openenv.server.app:app --reload --port 8000

# Or directly
cd duck_hunt_openenv/server
uvicorn app:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

## Running Tests

### Unit Tests

```bash
# With uv
uv run pytest duck_hunt_openenv/tests/test_game.py -v

# Or directly
pytest duck_hunt_openenv/tests/test_game.py -v
```

### Integration Tests (Random Agent)

```bash
# Run 100 episodes
uv run python duck_hunt_openenv/tests/test_random_agent.py --episodes 100

# Run single detailed episode
uv run python duck_hunt_openenv/tests/test_random_agent.py --detailed
```

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

### Setup

```bash
# Install with experiment dependencies
uv sync --extra experiments

# Configure API keys
export OPENAI_API_KEY="your-key"
wandb login
```

### Run Experiments

```bash
# Single episode with GPT-4o
uv run python -m duck_hunt_openenv.experiments.run --mode single --model gpt-4o

# Batch evaluation
uv run python -m duck_hunt_openenv.experiments.run --mode batch --episodes 10 --model gpt-4o-mini
```

## Game Mechanics

| Parameter | Value | Description |
|-----------|-------|-------------|
| Screen Size | 800 × 500 | Game viewport |
| FPS | 30 | Frames per second |
| Match Duration | 30s | Time limit per match |
| Ducks per Match | 2 | Simultaneous ducks |
| Bullets per Match | 3 | Shots available |
| Max Misses | 4 | Game over threshold |

### Rewards

| Event | Reward |
|-------|--------|
| Hit | +1.0 |
| Double Kill | +2.5 |
| Miss | -0.3 |
| No Target | -0.5 |

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
