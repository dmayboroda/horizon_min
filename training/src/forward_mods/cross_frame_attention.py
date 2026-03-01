"""Cross-frame attention for multi-frame temporal reasoning.

Hooks into the vision tower output (dim=1024) to perform cross-frame
attention BEFORE the patch merger and projector.  This lets the model
compare patches at the same spatial position across frames — crucial
for detecting duck motion between frames.

Architecture (verified against Ministral 3 8B):
    - Vision encoder: 24-layer PixTral ViT, output dim = 1024
    - 256×256 frame → 18×18 = 324 raw patches at dim 1024
    - After 2×2 merge: 9×9 = 81 patches at dim 4096
    - We hook BEFORE merge at dim 1024
    - patches_per_frame is computed dynamically from the actual output
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..config import ForwardModConfig
from .base import ForwardMod
from .registry import register_mod

logger = logging.getLogger(__name__)

# Vision encoder output dim (PixTral ViT)
_VISION_DIM = 1024


class CrossFrameAttentionLayer(nn.Module):
    """Single cross-frame attention block at vision encoder dim.

    Pools each frame's vision tokens into a summary, then uses
    cross-attention to let each frame attend to all frame summaries.
    Operates at dim=1024 (before patch merger).
    """

    def __init__(
        self,
        hidden_dim: int = _VISION_DIM,
        num_frames: int = 4,
        num_heads: int = 8,
    ) -> None:
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
    ) -> torch.Tensor:
        """Apply cross-frame attention.

        Parameters
        ----------
        hidden_states : (B, total_patches, D)
            Vision tokens from all frames concatenated.

        Returns
        -------
        (B, total_patches, D) with cross-frame attention added.
        """
        B, T, D = hidden_states.shape

        # Dynamically compute patches per frame
        patches_per_frame = T // self.num_frames
        if patches_per_frame == 0 or T % self.num_frames != 0:
            return hidden_states

        actual_frames = self.num_frames
        n_vision = actual_frames * patches_per_frame

        # Reshape into frames: (B, num_frames, patches_per_frame, D)
        vision_tokens = hidden_states[:, :n_vision, :]
        frames = vision_tokens.reshape(B, actual_frames, patches_per_frame, D)

        # Pool each frame: mean → project → (B, num_frames, D)
        frame_summaries = frames.mean(dim=2)
        frame_summaries = self.frame_proj(frame_summaries)

        # Cross-attend: each vision token queries all frame summaries
        # Q: vision_tokens (B, V, D), K/V: frame_summaries (B, F, D)
        normed = self.norm(vision_tokens)
        cross_out, _ = self.cross_attn(
            normed, frame_summaries, frame_summaries
        )

        # Gated residual — gate starts at 0
        result = hidden_states.clone()
        result[:, :n_vision, :] = result[:, :n_vision, :] + self.gate * cross_out

        return result


@register_mod("cross_frame_attention")
class CrossFrameMotionAttention(ForwardMod):
    """Hooks cross-frame attention into the vision tower output.

    Unlike the previous version that wrapped LM layers, this hooks
    directly onto the vision tower output at dim=1024, before the
    patch merger and projector.
    """

    def __init__(self, config: ForwardModConfig) -> None:
        super().__init__()
        self.config = config
        self.cross_layer = CrossFrameAttentionLayer(
            hidden_dim=_VISION_DIM,
            num_frames=config.num_frames,
            num_heads=8,
        )
        self._hook_handle = None

    def apply(self, model: nn.Module) -> nn.Module:
        """Register a forward hook on the vision tower."""
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)

        # Find the vision tower
        vision_tower = None
        for attr in ["vision_tower", "vision_model", "visual"]:
            candidate = getattr(base, attr, None)
            if candidate is not None:
                vision_tower = candidate
                break

        if vision_tower is None:
            logger.warning(
                "Could not find vision tower — "
                "cross_frame_attention not applied."
            )
            return model

        # Move to same device/dtype
        device = next(vision_tower.parameters()).device
        dtype = next(vision_tower.parameters()).dtype
        self.cross_layer = self.cross_layer.to(device=device, dtype=dtype)

        def _cross_frame_hook(module, input, output):
            return self._apply_cross_frame(output)

        self._hook_handle = vision_tower.register_forward_hook(
            _cross_frame_hook
        )
        logger.info(
            "Registered cross-frame attention hook on %s (dim=%d)",
            vision_tower.__class__.__name__,
            _VISION_DIM,
        )

        return model

    def _apply_cross_frame(self, vit_output):
        """Apply cross-frame attention to vision tower output."""
        # Extract hidden states (same logic as temporal_position)
        if isinstance(vit_output, torch.Tensor):
            hidden = vit_output
            modified = self.cross_layer(hidden)
            return modified
        elif isinstance(vit_output, (tuple, list)):
            hidden = vit_output[0]
            modified = self.cross_layer(hidden)
            if isinstance(vit_output, tuple):
                return (modified,) + vit_output[1:]
            else:
                return [modified] + list(vit_output[1:])
        elif hasattr(vit_output, "last_hidden_state"):
            hidden = vit_output.last_hidden_state
            modified = self.cross_layer(hidden)
            vit_output.last_hidden_state = modified
            return vit_output
        else:
            return vit_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Not used directly — logic is in the hook."""
        return x
