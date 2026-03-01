"""Temporal position encoding for multi-frame observations.

Adds a learned per-frame embedding to vision tokens so the model can
distinguish frame ordering (oldest → newest).  The embedding is added
*before* the patch merger and projector, at the vision encoder output
(dim=1024).

Architecture (verified against Ministral 3 8B):
    - Vision encoder: 24-layer PixTral ViT, output dim = 1024
    - Patch size 14, spatial_merge_size = 2 (2×2 merge: 1024 → 4096)
    - 256×256 frame → 18×18 = 324 raw patches at dim 1024
    - After merge: 9×9 = 81 patches at dim 4096
    - We hook BEFORE merge at dim 1024, 324 patches/frame
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


@register_mod("temporal_position")
class TemporalPositionEncoding(ForwardMod):
    """Adds learned temporal embeddings to per-frame vision tokens.

    Hooks into the vision tower output (dim=1024) BEFORE the patch
    merger and projector.  This tags each frame's patches with a
    temporal index before spatial positions get scrambled by the 2×2
    merge.
    """

    def __init__(self, config: ForwardModConfig) -> None:
        super().__init__()
        self.num_frames = config.num_frames

        # Learned embeddings: one per frame, at vision encoder dim
        self.temporal_embeddings = nn.Embedding(
            self.num_frames, _VISION_DIM
        )
        # Initialize small so we don't disrupt pre-trained representations
        nn.init.normal_(self.temporal_embeddings.weight, std=0.02)

        self._hook_handle = None

    def apply(self, model: nn.Module) -> nn.Module:
        """Register a forward hook on the vision tower (before merger)."""
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
                "temporal_position mod not applied."
            )
            return model

        # Move embeddings to same device/dtype
        device = next(vision_tower.parameters()).device
        dtype = next(vision_tower.parameters()).dtype
        self.temporal_embeddings = self.temporal_embeddings.to(
            device=device, dtype=dtype
        )

        # Register hook on vision tower output
        def _add_temporal_hook(module, input, output):
            return self._add_temporal_embeddings(output)

        self._hook_handle = vision_tower.register_forward_hook(
            _add_temporal_hook
        )
        logger.info(
            "Registered temporal position hook on %s "
            "(num_frames=%d, dim=%d)",
            vision_tower.__class__.__name__,
            self.num_frames,
            _VISION_DIM,
        )

        return model

    def _add_temporal_embeddings(self, vit_output):
        """Add frame-index embeddings to vision tower output.

        Vision tower output can be:
        - A tensor of shape (batch, total_patches, 1024)
        - A tuple/list where first element is the tensor
        - A BaseModelOutput-like object with .last_hidden_state

        We need to handle all cases and return the same type.
        """
        # Extract the hidden states tensor
        if isinstance(vit_output, torch.Tensor):
            hidden = vit_output
            return_tensor = True
        elif isinstance(vit_output, (tuple, list)):
            hidden = vit_output[0]
            return_tensor = False
        elif hasattr(vit_output, "last_hidden_state"):
            hidden = vit_output.last_hidden_state
            return_tensor = False
        else:
            return vit_output

        if hidden.dim() != 3:
            return vit_output

        B, total_patches, D = hidden.shape

        # Infer patches per frame
        if total_patches == 0:
            return vit_output

        # Try to split evenly across frames
        num_frames = self.num_frames
        if total_patches % num_frames != 0:
            # Variable number of patches — try to use what we can
            patches_per_frame = total_patches // num_frames
            if patches_per_frame == 0:
                return vit_output
            usable = patches_per_frame * num_frames
        else:
            patches_per_frame = total_patches // num_frames
            usable = total_patches

        # Build frame indices: [0,0,...,0, 1,1,...,1, 2,2,...,2, 3,3,...,3]
        frame_ids = torch.arange(
            num_frames, device=hidden.device
        ).repeat_interleave(patches_per_frame)  # (usable,)

        # Get embeddings: (usable, D)
        embeddings = self.temporal_embeddings(frame_ids)

        # Add to hidden states
        result = hidden.clone()
        result[:, :usable, :] = result[:, :usable, :] + embeddings.unsqueeze(0)

        # Return in same format as input
        if return_tensor:
            return result
        elif isinstance(vit_output, tuple):
            return (result,) + vit_output[1:]
        elif isinstance(vit_output, list):
            return [result] + list(vit_output[1:])
        else:
            # BaseModelOutput-like
            vit_output.last_hidden_state = result
            return vit_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Not used directly — logic is in the hook."""
        return x
