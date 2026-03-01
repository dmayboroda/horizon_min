"""Reward functions for Duck Hunt GRPO training."""

from __future__ import annotations

import logging

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
    action : Action | None
        The parsed action, or ``None`` if the model's output could not be
        parsed into a valid tool call.
    config : RewardConfig
        Reward hyper-parameters.

    Returns
    -------
    float
        The total reward (base + horizon penalty).
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

    return base - penalty
