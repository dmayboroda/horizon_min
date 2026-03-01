"""Cross-frame attention for multi-frame temporal reasoning.

Injects lightweight cross-attention layers at specified positions in
the transformer stack.  Each cross-frame attention layer attends
across vision tokens from different frames, enabling the model to
compute inter-frame motion features.

Architecture reference:
    - LM has 34 layers (0..33), hidden_dim=4096
    - Default injection points: layers 0, 1, 2, 3 (early layers)
    - Each 512x512 frame → 324 vision tokens
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..config import ForwardModConfig
from .base import ForwardMod
from .registry import register_mod

logger = logging.getLogger(__name__)

_TOKENS_PER_FRAME = 324


class CrossFrameAttentionLayer(nn.Module):
    """Single cross-frame attention block.

    Pools each frame's vision tokens into a summary, then uses
    cross-attention to let each frame attend to all frame summaries.
    This is lighter than full cross-frame token-level attention.
    """

    def __init__(self, hidden_dim: int, num_frames: int, num_heads: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.num_heads = num_heads

        # Frame-level pooling: mean-pool + project
        self.frame_proj = nn.Linear(hidden_dim, hidden_dim)

        # Cross-attention: Q from current frame, K/V from all frame summaries
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Parameter(torch.zeros(1))  # Start at 0 — no effect initially

    def forward(
        self,
        hidden_states: torch.Tensor,
        vision_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply cross-frame attention.

        Parameters
        ----------
        hidden_states : (B, T, D)
            Full sequence hidden states.
        vision_token_mask : (B, T), optional
            Boolean mask where True = vision token.

        Returns
        -------
        (B, T, D) with cross-frame attention added to vision tokens.
        """
        B, T, D = hidden_states.shape

        if vision_token_mask is None:
            # Assume first (num_frames * tokens_per_frame) tokens are vision
            n_vision = self.num_frames * _TOKENS_PER_FRAME
            if T < n_vision:
                return hidden_states
            vision_tokens = hidden_states[:, :n_vision, :]  # (B, V, D)
        else:
            # Extract vision tokens — for now just use first n_vision
            n_vision = self.num_frames * _TOKENS_PER_FRAME
            vision_tokens = hidden_states[:, :n_vision, :]

        # Reshape into frames: (B, num_frames, tokens_per_frame, D)
        actual_frames = n_vision // _TOKENS_PER_FRAME
        if actual_frames == 0:
            return hidden_states

        frames = vision_tokens[:, : actual_frames * _TOKENS_PER_FRAME, :].reshape(
            B, actual_frames, _TOKENS_PER_FRAME, D
        )

        # Pool each frame: mean → project → (B, num_frames, D)
        frame_summaries = frames.mean(dim=2)
        frame_summaries = self.frame_proj(frame_summaries)

        # Cross-attend: each token queries all frame summaries
        # Q: vision_tokens (B, V, D), K/V: frame_summaries (B, F, D)
        normed = self.norm(vision_tokens[:, : actual_frames * _TOKENS_PER_FRAME, :])
        cross_out, _ = self.cross_attn(
            normed, frame_summaries, frame_summaries
        )

        # Gated residual — gate starts at 0
        result = hidden_states.clone()
        n = actual_frames * _TOKENS_PER_FRAME
        result[:, :n, :] = result[:, :n, :] + self.gate * cross_out

        return result


@register_mod("cross_frame_attention")
class CrossFrameAttentionMod(ForwardMod):
    """Injects CrossFrameAttentionLayers at specified LM layers."""

    def __init__(self, config: ForwardModConfig) -> None:
        super().__init__()
        self.config = config
        self.cross_layers: nn.ModuleList = nn.ModuleList()

    def apply(self, model: nn.Module) -> nn.Module:
        """Wrap specified LM layers with cross-frame attention."""
        # Find the transformer layers
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)

        lm = None
        for attr in ["language_model", "model"]:
            candidate = getattr(base, attr, None)
            if candidate is not None:
                lm = candidate
                break

        if lm is None:
            logger.warning("Could not find language model — cross_frame_attention not applied.")
            return model

        layers = None
        if hasattr(lm, "layers"):
            layers = lm.layers
        elif hasattr(lm, "model") and hasattr(lm.model, "layers"):
            layers = lm.model.layers

        if layers is None:
            logger.warning("Could not find transformer layers — cross_frame_attention not applied.")
            return model

        device = next(model.parameters()).device
        target_layers = self.config.cross_frame_layers

        for layer_idx in target_layers:
            if layer_idx >= len(layers):
                logger.warning("Layer %d out of range (model has %d layers)", layer_idx, len(layers))
                continue

            cross_layer = CrossFrameAttentionLayer(
                hidden_dim=4096,
                num_frames=self.config.num_frames,
            ).to(device)
            self.cross_layers.append(cross_layer)

            # Wrap the original layer's forward
            original_layer = layers[layer_idx]
            original_forward = original_layer.forward
            cross_ref = cross_layer  # closure capture

            def _make_wrapped_forward(orig_fwd, cross_attn_layer):
                def wrapped_forward(*args, **kwargs):
                    output = orig_fwd(*args, **kwargs)
                    # output is typically a tuple (hidden_states, ...)
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        hidden_states = cross_attn_layer(hidden_states)
                        return (hidden_states,) + output[1:]
                    else:
                        return cross_attn_layer(output)
                return wrapped_forward

            layers[layer_idx].forward = _make_wrapped_forward(
                original_forward, cross_layer
            )
            logger.info("Injected cross-frame attention at layer %d", layer_idx)

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Not used directly — logic is in the wrapped layer forwards."""
        return x
