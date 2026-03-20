"""Reward functions for Duck Hunt GRPO training."""

from __future__ import annotations

import logging
import math

from .config import RewardConfig
from .utils import Action

logger = logging.getLogger(__name__)


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_reward(
    result: dict,
    action: Action | None,
    config: RewardConfig,
) -> float:
    """Compute the scalar reward for a single step.

    Parameters
    ----------
    result : dict
        Shot outcome from ``simulate_shot()``.  Expected keys:

        * ``hit_a``  (bool) – duck A was hit
        * ``hit_b``  (bool) – duck B was hit
        * ``had_target`` (bool) – at least one duck was flying at observation time
        * ``had_target_at_shot`` (bool) – at least one duck still flying when shot lands
        * ``duck_a_pos`` (tuple) – normalised (x, y) of duck A after advancement
        * ``duck_b_pos`` (tuple) – normalised (x, y) of duck B after advancement
        * ``duck_a_state`` (str) – state of duck A after advancement
        * ``duck_b_state`` (str) – state of duck B after advancement
        * ``shot_pos`` (tuple) – normalised (x, y) of the shot
    action : Action | None
        The parsed action, or ``None`` if the model's output could not be
        parsed into a valid tool call.
    config : RewardConfig
        Reward hyper-parameters.

    Returns
    -------
    float
        The total reward.
    """
    # ---- unparseable output ----
    if action is None:
        return config.invalid_action

    hit_a: bool = result.get("hit_a", False)
    hit_b: bool = result.get("hit_b", False)
    had_target: bool = result.get("had_target", False)
    had_target_at_shot: bool = result.get("had_target_at_shot", had_target)

    # ---- no ducks were flying at observation time ----
    if not had_target:
        return config.shoot_nothing

    # ---- ducks escaped/fell during latency+horizon (shot too late) ----
    if not had_target_at_shot and not (hit_a or hit_b):
        return config.shoot_dead_duck

    # ---- base reward by outcome ----
    if hit_a and hit_b:
        base = config.double_kill
    elif hit_a or hit_b:
        base = config.hit
    else:
        base = config.miss

    # ---- horizon penalty (only on hits — reward minimizing horizon) ----
    if hit_a or hit_b:
        penalty = config.lambda_horizon * (
            action.horizon / max(config.max_horizon, 1)
        )
    else:
        penalty = 0.0

    # ---- proximity bonus (on misses — gives gradient signal) ----
    # Only measure distance to FLYING ducks — reward aiming at alive targets
    proximity = 0.0
    if not (hit_a or hit_b) and config.proximity_bonus > 0:
        shot_pos = result.get("shot_pos")
        duck_a_pos = result.get("duck_a_pos")
        duck_b_pos = result.get("duck_b_pos")
        duck_a_state = result.get("duck_a_state", "flying")
        duck_b_state = result.get("duck_b_state", "flying")

        if shot_pos:
            distances = []
            if duck_a_pos and duck_a_state == "flying":
                distances.append(_distance(shot_pos, duck_a_pos))
            if duck_b_pos and duck_b_state == "flying":
                distances.append(_distance(shot_pos, duck_b_pos))

            if distances:
                min_dist = min(distances)
                proximity = config.proximity_bonus * math.exp(
                    -config.proximity_decay * min_dist
                )

    return base - penalty + proximity
