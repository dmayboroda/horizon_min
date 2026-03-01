"""Reward functions for Duck Hunt GRPO training."""

from __future__ import annotations

import logging
import math

from .config import RewardConfig
from .utils import Action

logger = logging.getLogger(__name__)


def compute_reward(
    result: dict,
    action: Action | None,
    config: RewardConfig,
) -> float:
    """Compute the scalar reward for a single step.

    Parameters
    ----------
    result : dict
        Shot outcome from ``DuckHuntEnvWrapper.process_shot()`` or the
        observation returned by ``step()``.  Expected keys:

        * ``hit_a``  (bool) – duck A was hit
        * ``hit_b``  (bool) – duck B was hit
        * ``had_target`` (bool) – at least one duck was flying
        * ``duck_a_pos`` (tuple[float, float], optional) – normalised (x, y)
        * ``duck_b_pos`` (tuple[float, float], optional) – normalised (x, y)
    action : Action | None
        The parsed action, or ``None`` if the model's output could not be
        parsed into a valid tool call.
    config : RewardConfig
        Reward hyper-parameters.

    Returns
    -------
    float
        The total reward (base + horizon penalty + distance shaping).
    """
    # ---- unparseable output ----
    if action is None:
        return config.invalid_action

    hit_a: bool = result.get("hit_a", False)
    hit_b: bool = result.get("hit_b", False)
    had_target: bool = result.get("had_target", False)

    # ---- no ducks were flying ----
    if not had_target:
        return config.shoot_nothing

    # ---- base reward by outcome ----
    if hit_a and hit_b:
        base = config.double_kill
    elif hit_a or hit_b:
        base = config.hit
    else:
        base = config.miss

    # ---- horizon penalty (only on hits) ----
    if hit_a or hit_b:
        penalty = config.lambda_horizon * (
            action.horizon / max(config.max_horizon, 1)
        )
    else:
        # no penalty on misses — same as the OpenEnv env
        penalty = 0.0

    reward = base - penalty

    # ---- distance-based reward shaping (for misses in action mode) ----
    if config.distance_reward_weight > 0 and not (hit_a or hit_b) and had_target:
        distance_bonus = _compute_distance_reward(result, action)
        reward += config.distance_reward_weight * distance_bonus

    return reward


def _compute_distance_reward(result: dict, action: Action) -> float:
    """Compute a distance-based shaping reward for near misses.

    Returns a value in [0, 1] — closer to a duck is better.
    Uses the minimum distance to either duck (normalised).
    """
    min_dist = float("inf")

    for key in ["duck_a_pos", "duck_b_pos"]:
        pos = result.get(key)
        if pos is None:
            continue
        dx = action.x - pos[0]
        dy = action.y - pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        min_dist = min(min_dist, dist)

    if min_dist == float("inf"):
        return 0.0

    # Max possible distance in normalised space is sqrt(2) ≈ 1.414
    # Convert to [0, 1] reward (closer = higher)
    max_dist = math.sqrt(2.0)
    return max(0.0, 1.0 - min_dist / max_dist)
