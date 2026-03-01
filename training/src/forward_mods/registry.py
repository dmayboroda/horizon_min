"""Registry for forward modification modules."""

from __future__ import annotations

import logging

import torch.nn as nn

from ..config import ForwardModConfig
from .base import ForwardMod

logger = logging.getLogger(__name__)

# Maps mod name → factory function
_REGISTRY: dict[str, type[ForwardMod]] = {}


def register_mod(name: str):
    """Decorator to register a ForwardMod subclass."""

    def decorator(cls: type[ForwardMod]):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_forward_mod(name: str, config: ForwardModConfig) -> ForwardMod:
    """Instantiate a registered ForwardMod by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown forward mod '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](config)


def apply_forward_mods(
    model: nn.Module,
    config: ForwardModConfig,
) -> tuple[nn.Module, list[ForwardMod]]:
    """Apply all enabled forward mods to *model*.

    Returns the modified model and the list of applied mods (for
    collecting extra parameters).
    """
    if not config.enabled:
        return model, []

    # Import submodules to trigger @register_mod decorators
    from . import temporal_position  # noqa: F401
    from . import cross_frame_attention  # noqa: F401
    from . import spatial_decoder  # noqa: F401

    applied: list[ForwardMod] = []

    if config.temporal_position:
        mod = get_forward_mod("temporal_position", config)
        model = mod.apply(model)
        applied.append(mod)
        logger.info("Applied forward mod: temporal_position")

    if config.cross_frame_attention:
        mod = get_forward_mod("cross_frame_attention", config)
        model = mod.apply(model)
        applied.append(mod)
        logger.info("Applied forward mod: cross_frame_attention")

    if config.spatial_decoder:
        mod = get_forward_mod("spatial_decoder", config)
        model = mod.apply(model)
        applied.append(mod)
        logger.info("Applied forward mod: spatial_decoder")

    return model, applied
