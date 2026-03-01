"""Base class for forward modifications."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ForwardMod(ABC, nn.Module):
    """Base class for composable architecture modifications.

    Each ForwardMod is applied to the model after loading but before
    training.  Subclasses implement ``apply`` to modify the model's
    forward graph (e.g. inject new layers, replace modules, add hooks).
    """

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """Modify *model* in-place and return it.

        Parameters
        ----------
        model : nn.Module
            The full ``Mistral3ForConditionalGeneration`` model (or its
            PeftModel wrapper).

        Returns
        -------
        nn.Module
            The (possibly modified) model.
        """
        ...

    def extra_parameters(self) -> list[nn.Parameter]:
        """Return any new trainable parameters introduced by this mod.

        These will be added to the optimizer's param groups alongside
        the LoRA parameters.
        """
        return list(self.parameters())
