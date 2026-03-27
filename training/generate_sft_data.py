"""Generate SFT dataset for Duck Hunt VLM training.

For each game state, captures 2 observation frames and computes ground-truth
shot coordinates for ALL configured latency values. The ground truth is a
random point inside the duck's hitbox at the future position — because any
coordinate within the hitbox is a valid hit.

Usage:
    python generate_sft_data.py --num-samples 8000 --output sft_dataset
    python generate_sft_data.py --num-samples 100 --frame-skip 3 --output sft_test
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from PIL import Image

# Make server package importable
_SERVER_DIR = Path(__file__).resolve().parent.parent / "duck_hunt_openenv" / "server"
sys.path.insert(0, str(_SERVER_DIR))

from game_engine import Duck, DuckState, Match, Round  # noqa: E402
from config import (  # noqa: E402
    SCREEN_WIDTH, SCREEN_HEIGHT,
    SPRITE_WIDTH, SPRITE_HEIGHT,
    HITBOX_WIDTH, HITBOX_HEIGHT,
    SPEED_BASE, SPEED_VARIANCE,
    FPS,
)

from src.environment import DuckHuntEnvWrapper  # noqa: E402
from src.config import EnvironmentConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot / restore (same logic as dataset.py but self-contained)
# ---------------------------------------------------------------------------
def _snapshot_duck(duck: Duck) -> dict:
    return {
        "x": duck.x, "y": duck.y,
        "dx": duck.dx, "dy": duck.dy,
        "state": duck.state.value,
        "sprite_dir": duck.sprite_dir,
    }


def _restore_duck(data: dict, round_number: int) -> Duck:
    duck = object.__new__(Duck)
    duck.x = data["x"]
    duck.y = data["y"]
    duck.dx = data["dx"]
    duck.dy = data["dy"]
    duck.state = DuckState(data["state"])
    duck.sprite_dir = data["sprite_dir"]
    return duck


def _snapshot_match(match: Match, round_number: int) -> dict:
    snap = {
        "duck_a": _snapshot_duck(match.duck_a),
        "round_number": round_number,
        "rng_state": random.getstate(),
    }
    if match.duck_b is not None:
        snap["duck_b"] = _snapshot_duck(match.duck_b)
    return snap


# ---------------------------------------------------------------------------
# Compute ground-truth: random point inside hitbox at future position
# ---------------------------------------------------------------------------
def _is_duck_visible(duck_data: dict) -> bool:
    """Check if a duck is visible on screen (not off-screen or spawning)."""
    x, y = duck_data["x"], duck_data["y"]
    if duck_data["state"] != "flying":
        return False
    # Duck sprite must be at least partially on screen
    if x + SPRITE_WIDTH < 0 or x > SCREEN_WIDTH:
        return False
    if y + SPRITE_HEIGHT < 0 or y > SCREEN_HEIGHT:
        return False
    return True


def _is_duck_on_screen(duck: Duck) -> bool:
    """Check if a simulated duck is visible on screen."""
    if duck.state != DuckState.FLYING:
        return False
    if duck.x + SPRITE_WIDTH < 0 or duck.x > SCREEN_WIDTH:
        return False
    if duck.y + SPRITE_HEIGHT < 0 or duck.y > SCREEN_HEIGHT:
        return False
    return True


def compute_ground_truth(
    snapshot: dict,
    target_duck_key: str,
    latency_frames: int,
) -> dict | None:
    """Simulate duck forward by latency_frames and return hitbox coords + future game state.

    Returns dict with:
    - x1, y1, x2, y2: hitbox in normalized coords
    - future_game_state: dict for rendering the future frame

    Returns None if duck escapes/falls/off-screen.
    """
    round_number = snapshot["round_number"]

    # Restore RNG state so bounces are deterministic
    random.setstate(snapshot["rng_state"])

    # Restore both ducks — must update both to consume RNG in correct order
    duck_a = _restore_duck(snapshot["duck_a"], round_number)
    has_b = "duck_b" in snapshot
    duck_b = _restore_duck(snapshot["duck_b"], round_number) if has_b else None

    # Advance both ducks together (RNG order must match game engine)
    for _ in range(latency_frames):
        duck_a.update(round_number)
        if duck_b is not None:
            duck_b.update(round_number)

    # Get the target duck after advancement
    target = duck_a if target_duck_key == "duck_a" else duck_b
    if target is None or target.state != DuckState.FLYING:
        return None

    if not _is_duck_on_screen(target):
        return None

    # Hitbox rectangle (centered on sprite)
    hx = target.x + (SPRITE_WIDTH - HITBOX_WIDTH) / 2
    hy = target.y + (SPRITE_HEIGHT - HITBOX_HEIGHT) / 2

    if hx < 0 or hx + HITBOX_WIDTH > SCREEN_WIDTH:
        return None
    if hy < 0 or hy + HITBOX_HEIGHT > SCREEN_HEIGHT:
        return None

    x1 = round(hx / SCREEN_WIDTH, 3)
    y1 = round(hy / SCREEN_HEIGHT, 3)
    x2 = round((hx + HITBOX_WIDTH) / SCREEN_WIDTH, 3)
    y2 = round((hy + HITBOX_HEIGHT) / SCREEN_HEIGHT, 3)

    # Build future game state for rendering
    future_state = {
        "duck_a": {
            "x": duck_a.x, "y": duck_a.y,
            "state": duck_a.state.value,
            "sprite_dir": duck_a.sprite_dir,
        },
    }
    if duck_b is not None:
        future_state["duck_b"] = {
            "x": duck_b.x, "y": duck_b.y,
            "state": duck_b.state.value,
            "sprite_dir": duck_b.sprite_dir,
        }

    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "future_game_state": future_state,
    }


# ---------------------------------------------------------------------------
# Build SFT example
# ---------------------------------------------------------------------------
def build_sft_example(
    frames: list[Image.Image],
    latency_frames: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    ducks_flying: int,
    num_frames: int = 2,
) -> dict:
    """Build a single SFT training example.

    The model learns to predict the duck's hitbox after latency frames.
    Output: locate(x1, y1, x2, y2) — top-left and bottom-right of hitbox.
    """
    system_prompt = (
        "You are a Duck Hunt AI. Locate flying ducks by calling the locate tool.\n\n"
        f"You see {num_frames} frames. Latency: {latency_frames} frames.\n"
        "Coordinates: x (0=left, 1=right), y (0=top, 1=bottom).\n"
        "Predict the hitbox of the duck after latency frames.\n"
        "x1,y1 = top-left corner. x2,y2 = bottom-right corner.\n\n"
        "IMPORTANT: Respond ONLY with the tool call. Do NOT explain your reasoning."
    )

    # Few-shot with random values
    fs_x1 = round(random.uniform(0.1, 0.7), 2)
    fs_y1 = round(random.uniform(0.1, 0.5), 2)
    fs_x2 = round(fs_x1 + 0.10, 2)
    fs_y2 = round(fs_y1 + 0.14, 2)

    user_content = []
    for img in frames:
        user_content.append({"type": "image", "image": img})
    user_content.append({
        "type": "text",
        "text": f"{num_frames} frames, {ducks_flying} ducks flying, latency {latency_frames} frames. Locate the duck.",
    })

    # Ground truth: hitbox coordinates
    completion = f"<|tool_call_start|>[locate(x1={x1}, y1={y1}, x2={x2}, y2={y2})]<|tool_call_end|>"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Frame sequence: {num_frames} frames. Ducks flying: 2. Latency: {latency_frames} frames. Locate the duck."},
        {"role": "assistant", "content": f"<|tool_call_start|>[locate(x1={fs_x1}, y1={fs_y1}, x2={fs_x2}, y2={fs_y2})]<|tool_call_end|>"},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": completion},
    ]

    return {
        "messages": messages,
        "images": list(frames),
        "completion": completion,
        "latency_frames": latency_frames,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate_dataset(
    num_observations: int = 2000,
    latency_options_ms: list[int] | None = None,
    frame_skip: int = 6,
    fps: int = 30,
    output_dir: str = "sft_dataset",
) -> None:
    """Generate SFT dataset.

    For each observation state, generates one example per latency value.
    Total examples = num_observations × len(latency_options_ms).
    """
    if latency_options_ms is None:
        latency_options_ms = [100, 150, 200, 250, 300]

    latency_frames_list = [int(ms / 1000 * fps) for ms in latency_options_ms]

    # Environment config
    env_config = EnvironmentConfig(
        frames_per_observation=2,
        frame_skip=frame_skip,
        latency_options_ms=latency_options_ms,
    )
    env = DuckHuntEnvWrapper(env_config)
    env.reset()

    all_examples = []
    skipped = 0
    obs_count = 0

    logger.info(
        "Generating SFT data: %d observations × %d latencies = %d target examples",
        num_observations, len(latency_options_ms),
        num_observations * len(latency_options_ms),
    )
    logger.info("Latency options: %s ms → %s frames", latency_options_ms, latency_frames_list)
    logger.info("Frame skip: %d", frame_skip)

    while obs_count < num_observations:
        # New match with fresh ducks
        env.auto_advance_to_next_match()
        env.advance_frames(random.randint(8, 25))

        # Check for flying ducks
        frames = env.get_frames()
        state = env.get_state()
        flying = state.get("ducks_flying", 0)

        if flying < 1 or len(frames) < 2:
            if env.is_done():
                env.reset()
            env.advance_frames(random.randint(5, 15))
            continue

        # Snapshot current state
        inner = env._env
        match = inner.round.current_match
        snapshot = _snapshot_match(match, inner.round_number)

        # Pick random flying duck that is VISIBLE on screen
        flying_ducks = []
        if _is_duck_visible(snapshot["duck_a"]):
            flying_ducks.append("duck_a")
        if "duck_b" in snapshot and _is_duck_visible(snapshot["duck_b"]):
            flying_ducks.append("duck_b")

        if not flying_ducks:
            skipped += 1
            continue

        target_key = random.choice(flying_ducks)

        # Compute CURRENT hitbox position (for visualization)
        duck_now = snapshot[target_key]
        cur_hx = duck_now["x"] + (SPRITE_WIDTH - HITBOX_WIDTH) / 2
        cur_hy = duck_now["y"] + (SPRITE_HEIGHT - HITBOX_HEIGHT) / 2
        cur_x1 = round(cur_hx / SCREEN_WIDTH, 3)
        cur_y1 = round(cur_hy / SCREEN_HEIGHT, 3)
        cur_x2 = round((cur_hx + HITBOX_WIDTH) / SCREEN_WIDTH, 3)
        cur_y2 = round((cur_hy + HITBOX_HEIGHT) / SCREEN_HEIGHT, 3)

        # Convert frames from base64 to PIL
        import base64
        from io import BytesIO
        pil_frames = []
        for b64 in inner.frame_buffer[:2]:
            raw = base64.b64decode(b64)
            pil_frames.append(Image.open(BytesIO(raw)).convert("RGB"))

        # Get renderer for future frames
        renderer = inner.renderer

        # Generate one example per latency value
        generated_any = False
        for lat_ms, lat_frames in zip(latency_options_ms, latency_frames_list):
            result = compute_ground_truth(snapshot, target_key, lat_frames)
            if result is None:
                skipped += 1
                continue

            x1, y1, x2, y2 = result["x1"], result["y1"], result["x2"], result["y2"]

            # Render future frame (what the game looks like when shot lands)
            future_frame = renderer.render_and_resize(
                result["future_game_state"], lat_frames,
            )
            future_frame_pil = future_frame.convert("RGB")

            example = build_sft_example(
                frames=pil_frames,
                latency_frames=lat_frames,
                x1=x1, y1=y1, x2=x2, y2=y2,
                ducks_flying=flying,
                num_frames=2,
            )
            # Store current position and future frame for visualization
            example["cur_x1"] = cur_x1
            example["cur_y1"] = cur_y1
            example["cur_x2"] = cur_x2
            example["cur_y2"] = cur_y2
            example["future_frame"] = future_frame_pil
            all_examples.append(example)
            generated_any = True

        if generated_any:
            obs_count += 1

        if obs_count % 200 == 0 and obs_count > 0:
            logger.info(
                "Progress: %d/%d observations, %d examples generated, %d skipped",
                obs_count, num_observations, len(all_examples), skipped,
            )

    logger.info(
        "Done: %d observations → %d examples (%d skipped)",
        obs_count, len(all_examples), skipped,
    )

    # Save dataset
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save images separately, messages as JSON
    images_dir = out_path / "images"
    images_dir.mkdir(exist_ok=True)

    dataset_records = []
    for i, ex in enumerate(all_examples):
        # Save observation images
        img_paths = []
        for j, img in enumerate(ex["images"]):
            img_path = images_dir / f"{i:06d}_frame{j}.png"
            img.save(str(img_path))
            img_paths.append(str(img_path))

        # Save future frame
        future_path = images_dir / f"{i:06d}_future.png"
        ex["future_frame"].save(str(future_path))

        # Build record (without PIL objects)
        record = {
            "id": i,
            "image_paths": img_paths,
            "completion": ex["completion"],
            "latency_frames": ex["latency_frames"],
            "x1": ex["x1"], "y1": ex["y1"],
            "x2": ex["x2"], "y2": ex["y2"],
            "cur_x1": ex["cur_x1"], "cur_y1": ex["cur_y1"],
            "cur_x2": ex["cur_x2"], "cur_y2": ex["cur_y2"],
            "future_frame_path": str(future_path),
            "system_prompt": ex["messages"][0]["content"],
            "user_fewshot": ex["messages"][1]["content"],
            "assistant_fewshot": ex["messages"][2]["content"],
            "user_text": ex["messages"][3]["content"][-1]["text"],  # text portion
            "num_images": len(img_paths),
        }
        dataset_records.append(record)

    # Save metadata
    meta_path = out_path / "dataset.json"
    with open(meta_path, "w") as f:
        json.dump(dataset_records, f, indent=2)

    logger.info("Dataset saved to %s (%d records)", out_path, len(dataset_records))
    logger.info("  Images: %s", images_dir)
    logger.info("  Metadata: %s", meta_path)

    # Print sample
    if dataset_records:
        sample = dataset_records[0]
        logger.info("Sample record:")
        logger.info("  completion: %s", sample["completion"])
        logger.info("  latency: %d frames", sample["latency_frames"])
        logger.info("  hitbox: (%.3f, %.3f) → (%.3f, %.3f)",
                     sample["x1"], sample["y1"], sample["x2"], sample["y2"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset for Duck Hunt")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="Number of observation states to generate (total examples = this × num latencies)")
    parser.add_argument("--output", type=str, default="sft_dataset",
                        help="Output directory for the dataset")
    parser.add_argument("--frame-skip", type=int, default=6,
                        help="Game frames to skip between observation frames")
    parser.add_argument("--latencies", type=str, default="100,150,200,250,300",
                        help="Comma-separated latency values in ms")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    latency_options = [int(x) for x in args.latencies.split(",")]

    generate_dataset(
        num_observations=args.num_samples,
        latency_options_ms=latency_options,
        frame_skip=args.frame_skip,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
