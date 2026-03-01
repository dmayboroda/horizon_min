"""Temporal position encoding for multi-frame observations.

Adds a learned per-frame embedding to vision tokens so the model can
distinguish frame ordering (oldest → newest).  The embedding is added
*after* the vision projector, before the tokens enter the LM layers.

Architecture (from verify_model.py):
    - Each 512x512 frame → 324 vision tokens (after spatial merge 2x2)
    - Vision hidden dim = 4096 (after projector)
    - 4 frames → 1296 total vision tokens
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..config import ForwardModConfig
from .base import ForwardMod
from .registry import register_mod

logger = logging.getLogger(__name__)

# Tokens per frame after Pixtral spatial merge (2x2 on 512x512, patch_size=14)
# = (512/14)^2 / (2*2) = 36.57^2 / 4 ≈ 1369/4 ≈ 324 (ceil)
_TOKENS_PER_FRAME = 324


@register_mod("temporal_position")
class TemporalPositionEncoding(ForwardMod):
    """Adds learned temporal embeddings to per-frame vision tokens."""

    def __init__(self, config: ForwardModConfig) -> None:
        super().__init__()
        self.num_frames = config.num_frames
        self.hidden_dim = 4096  # Ministral LM hidden dim (post-projector)

        # Learned embeddings: one per frame
        self.temporal_embeddings = nn.Embedding(
            self.num_frames, self.hidden_dim
        )
        # Initialize small so we don't disrupt pre-trained representations
        nn.init.normal_(self.temporal_embeddings.weight, std=0.02)

        self._hook_handle = None

    def apply(self, model: nn.Module) -> nn.Module:
        """Register a forward hook on the multimodal projector."""
        # Find the projector
        projector = None
        for attr in ["multi_modal_projector", "mm_projector", "projector"]:
            if hasattr(model, attr):
                projector = getattr(model, attr)
                break

        if projector is None:
            # Try through base_model (PeftModel wrapping)
            base = getattr(model, "base_model", model)
            base = getattr(base, "model", base)
            for attr in ["multi_modal_projector", "mm_projector", "projector"]:
                if hasattr(base, attr):
                    projector = getattr(base, attr)
                    break

        if projector is None:
            logger.warning(
                "Could not find multimodal projector — "
                "temporal_position mod not applied."
            )
            return model

        # Move embeddings to same device
        device = next(projector.parameters()).device
        self.temporal_embeddings = self.temporal_embeddings.to(device)

        # Register hook that adds temporal embeddings after projection
        def _add_temporal_hook(module, input, output):
            return self._add_temporal_embeddings(output)

        self._hook_handle = projector.register_forward_hook(_add_temporal_hook)
        logger.info(
            "Registered temporal position hook on %s "
            "(num_frames=%d, tokens_per_frame=%d)",
            projector.__class__.__name__,
            self.num_frames,
            _TOKENS_PER_FRAME,
        )

        return model

    def _add_temporal_embeddings(self, projected: torch.Tensor) -> torch.Tensor:
        """Add frame-index embeddings to projected vision tokens.

        ``projected`` shape: (batch, num_vision_tokens, hidden_dim)
        where num_vision_tokens = num_frames * tokens_per_frame.
        """
        if projected.dim() != 3:
            return projected

        total_tokens = projected.shape[1]
        expected = self.num_frames * _TOKENS_PER_FRAME

        if total_tokens != expected:
            # Mismatch — could be different number of frames or image size.
            # Try to infer num_frames from total.
            if total_tokens % _TOKENS_PER_FRAME == 0:
                actual_frames = total_tokens // _TOKENS_PER_FRAME
                if actual_frames > self.num_frames:
                    actual_frames = self.num_frames
            else:
                # Can't cleanly split — skip
                return projected

            frame_indices = torch.arange(
                actual_frames, device=projected.device
            )
        else:
            frame_indices = torch.arange(
                self.num_frames, device=projected.device
            )

        num_frames = frame_indices.shape[0]
        # (num_frames, hidden_dim)
        embeddings = self.temporal_embeddings(frame_indices)
        # Expand to (num_frames * tokens_per_frame, hidden_dim)
        embeddings = embeddings.unsqueeze(1).expand(
            -1, _TOKENS_PER_FRAME, -1
        ).reshape(num_frames * _TOKENS_PER_FRAME, -1)

        # Add to the first (num_frames * tokens_per_frame) tokens
        n = min(embeddings.shape[0], projected.shape[1])
        result = projected.clone()
        result[:, :n, :] = result[:, :n, :] + embeddings[:n].unsqueeze(0)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Not used directly — logic is in the hook."""
        return x
