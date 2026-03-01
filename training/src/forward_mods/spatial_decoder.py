"""Spatial decoder and action head for VLA (continuous action output).

Replaces the last N LM layers (default: layers 30-33) and the LM head
with:
1. SpatialDecoderLayers — modified transformer layers that preserve
   spatial structure rather than collapsing to token probabilities
2. ActionHead — Gaussian policy head that outputs mean + log_std for
   continuous (x, y, horizon) actions

Architecture reference (from verify_model.py):
    - LM: 34 layers (0..33), hidden_dim=4096, intermediate=14336
    - Keep layers 0..29, replace 30..33 with SpatialDecoderLayers
    - Replace LM head Linear(4096, 131072) with ActionHead
    - num_heads=32, num_kv_heads=8, head_dim=128
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from ..config import ForwardModConfig
from .base import ForwardMod
from .registry import register_mod

logger = logging.getLogger(__name__)


class SpatialDecoderLayer(nn.Module):
    """A simplified transformer decoder layer for spatial features.

    Uses standard multi-head self-attention + FFN but without causal
    masking, since we want bidirectional attention over all tokens
    to aggregate spatial information.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        intermediate_dim: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass with bidirectional self-attention.

        Returns a tuple to match the interface of the original LM layers.
        """
        # Self-attention (no causal mask)
        normed = self.norm1(hidden_states)
        attn_out, _ = self.self_attn(normed, normed, normed)
        hidden_states = hidden_states + attn_out

        # FFN
        normed = self.norm2(hidden_states)
        hidden_states = hidden_states + self.ffn(normed)

        return (hidden_states,)


class ActionHead(nn.Module):
    """Gaussian policy head for continuous actions.

    Pools the sequence into a single vector, then predicts mean and
    log_std for each action dimension.

    Output: (mean, log_std) each of shape (B, action_dim).
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        action_dim: int = 3,
        action_hidden_dim: int = 256,
        log_std_init: float = -1.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.pool_proj = nn.Linear(hidden_dim, action_hidden_dim)
        self.mean_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(action_hidden_dim, action_dim),
            nn.Sigmoid(),  # Clamp to [0, 1] for normalised coords
        )
        self.log_std_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(action_hidden_dim, action_dim),
        )

        # Initialize log_std bias
        nn.init.constant_(self.log_std_head[-1].bias, log_std_init)
        # Small init for mean head to start near center
        nn.init.normal_(self.mean_head[-2].weight, std=0.01)
        nn.init.constant_(self.mean_head[-2].bias, 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict action distribution parameters.

        Parameters
        ----------
        hidden_states : (B, T, D)
            Sequence output from spatial decoder layers.
        pool_mask : (B, T), optional
            Mask for mean-pooling (True = include). If None, uses
            mean over all positions.

        Returns
        -------
        mean : (B, action_dim)
            Action means in [0, 1] (Sigmoid applied).
        log_std : (B, action_dim)
            Clamped log standard deviations.
        """
        # Mean-pool across sequence
        if pool_mask is not None:
            mask = pool_mask.unsqueeze(-1).float()  # (B, T, 1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)  # (B, D)

        projected = self.pool_proj(pooled)  # (B, action_hidden_dim)

        mean = self.mean_head(projected)  # (B, action_dim)
        log_std = self.log_std_head(projected)  # (B, action_dim)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)

        return mean, log_std


@register_mod("spatial_decoder")
class SpatialDecoderMod(ForwardMod):
    """Replaces last N LM layers + LM head with spatial decoder + action head."""

    def __init__(self, config: ForwardModConfig) -> None:
        super().__init__()
        self.config = config

        # Will be populated during apply()
        self.decoder_layers = nn.ModuleList()
        self.action_head: ActionHead | None = None
        self._replaced_layers: list[int] = []

    def apply(self, model: nn.Module) -> nn.Module:
        """Replace LM layers [decoder_start_layer:] and lm_head."""
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)

        # Find language model
        lm = None
        for attr in ["language_model", "model"]:
            candidate = getattr(base, attr, None)
            if candidate is not None:
                lm = candidate
                break

        if lm is None:
            logger.warning("Could not find language model — spatial_decoder not applied.")
            return model

        # Find layers
        layers_container = lm
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            layers_container = lm.model

        if not hasattr(layers_container, "layers"):
            logger.warning("Could not find transformer layers — spatial_decoder not applied.")
            return model

        layers = layers_container.layers
        num_layers = len(layers)
        start = self.config.decoder_start_layer

        if start >= num_layers:
            logger.warning(
                "decoder_start_layer=%d >= num_layers=%d — spatial_decoder not applied.",
                start, num_layers,
            )
            return model

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Create spatial decoder layers to replace layers[start:]
        num_replaced = num_layers - start
        for i in range(num_replaced):
            decoder_layer = SpatialDecoderLayer(
                hidden_dim=4096,
                num_heads=32,
                intermediate_dim=4096,
            ).to(device=device, dtype=dtype)
            self.decoder_layers.append(decoder_layer)

        # Replace the layers
        for i, layer_idx in enumerate(range(start, num_layers)):
            layers[layer_idx] = self.decoder_layers[i]
            self._replaced_layers.append(layer_idx)
            logger.info(
                "Replaced LM layer %d with SpatialDecoderLayer %d", layer_idx, i
            )

        # Create action head
        self.action_head = ActionHead(
            hidden_dim=4096,
            action_dim=self.config.action_dim,
            action_hidden_dim=self.config.action_hidden_dim,
            log_std_init=self.config.log_std_init,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
        ).to(device=device, dtype=dtype)

        # Store action_head on the model for easy access
        model._action_head = self.action_head
        model._spatial_decoder_mod = self

        logger.info(
            "Spatial decoder applied: replaced layers %d-%d, "
            "added ActionHead(dim=%d, hidden=%d)",
            start, num_layers - 1,
            self.config.action_dim,
            self.config.action_hidden_dim,
        )

        return model

    def get_action_head(self) -> ActionHead:
        """Return the ActionHead module."""
        if self.action_head is None:
            raise RuntimeError("ActionHead not initialized — call apply() first.")
        return self.action_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Not used directly."""
        return x
