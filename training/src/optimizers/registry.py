"""Optimizer registry — create optimizer by name from config."""

from __future__ import annotations

import logging
from typing import Iterator

import torch
import torch.nn as nn

from ..config import OptimizerConfig, TrainingConfig

logger = logging.getLogger(__name__)


def create_optimizer(
    params: Iterator[nn.Parameter] | list[dict],
    opt_config: OptimizerConfig,
    train_config: TrainingConfig,
) -> torch.optim.Optimizer:
    """Create an optimizer by name.

    Parameters
    ----------
    params
        Model parameters (or param groups).
    opt_config
        Optimizer-specific configuration.
    train_config
        Training config (for lr, weight_decay).

    Returns
    -------
    torch.optim.Optimizer
    """
    name = opt_config.name.lower()
    lr = train_config.learning_rate
    wd = train_config.weight_decay

    if name == "adamw":
        return _create_adamw(params, lr, wd, opt_config)
    elif name == "adamw_8bit":
        return _create_adamw_8bit(params, lr, wd, opt_config)
    elif name == "muon":
        return _create_muon(params, lr, wd, opt_config)
    elif name == "soap":
        return _create_soap(params, lr, wd, opt_config)
    elif name == "shampoo":
        return _create_shampoo(params, lr, wd, opt_config)
    elif name == "prodigy":
        return _create_prodigy(params, lr, wd, opt_config)
    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Available: adamw, adamw_8bit, muon, soap, shampoo, prodigy"
        )


# ---------------------------------------------------------------------------
#  AdamW (torch native)
# ---------------------------------------------------------------------------
def _create_adamw(params, lr, wd, opt_config):
    logger.info("Creating AdamW optimizer (lr=%.2e, wd=%.2e)", lr, wd)
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=opt_config.betas,
        eps=opt_config.eps,
        weight_decay=wd,
    )


# ---------------------------------------------------------------------------
#  AdamW 8-bit (bitsandbytes)
# ---------------------------------------------------------------------------
def _create_adamw_8bit(params, lr, wd, opt_config):
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes required for adamw_8bit. "
            "Install with: pip install bitsandbytes"
        )

    logger.info("Creating AdamW 8-bit optimizer (lr=%.2e, wd=%.2e)", lr, wd)
    return bnb.optim.AdamW8bit(
        params,
        lr=lr,
        betas=opt_config.betas,
        eps=opt_config.eps,
        weight_decay=wd,
    )


# ---------------------------------------------------------------------------
#  Muon
# ---------------------------------------------------------------------------
def _create_muon(params, lr, wd, opt_config):
    try:
        from muon import Muon
    except ImportError:
        raise ImportError(
            "muon required. Install with: pip install muon"
        )

    logger.info(
        "Creating Muon optimizer (lr=%.2e, momentum=%.2f, nesterov=%s)",
        lr, opt_config.muon_momentum, opt_config.muon_nesterov,
    )
    return Muon(
        params,
        lr=lr,
        momentum=opt_config.muon_momentum,
        nesterov=opt_config.muon_nesterov,
    )


# ---------------------------------------------------------------------------
#  SOAP
# ---------------------------------------------------------------------------
def _create_soap(params, lr, wd, opt_config):
    try:
        from soap import SOAP
    except ImportError:
        raise ImportError(
            "SOAP required. Install with: pip install soap-optimizer"
        )

    logger.info(
        "Creating SOAP optimizer (lr=%.2e, shampoo_beta=%.2f, precond_freq=%d)",
        lr, opt_config.soap_shampoo_beta, opt_config.soap_precondition_frequency,
    )
    return SOAP(
        params,
        lr=lr,
        betas=opt_config.betas,
        weight_decay=wd,
        shampoo_beta=opt_config.soap_shampoo_beta,
        precondition_frequency=opt_config.soap_precondition_frequency,
    )


# ---------------------------------------------------------------------------
#  Shampoo (distributed_shampoo from PyTorch)
# ---------------------------------------------------------------------------
def _create_shampoo(params, lr, wd, opt_config):
    try:
        from distributed_shampoo import DistributedShampoo
    except ImportError:
        try:
            from optimizers.shampoo import Shampoo as DistributedShampoo
        except ImportError:
            raise ImportError(
                "Shampoo required. Install with: "
                "pip install distributed-shampoo or provide a local implementation."
            )

    logger.info(
        "Creating Shampoo optimizer (lr=%.2e, update_freq=%d)",
        lr, opt_config.shampoo_update_freq,
    )
    return DistributedShampoo(
        params,
        lr=lr,
        weight_decay=wd,
        precondition_frequency=opt_config.shampoo_update_freq,
    )


# ---------------------------------------------------------------------------
#  Prodigy
# ---------------------------------------------------------------------------
def _create_prodigy(params, lr, wd, opt_config):
    try:
        from prodigyopt import Prodigy
    except ImportError:
        raise ImportError(
            "prodigyopt required. Install with: pip install prodigyopt"
        )

    logger.info(
        "Creating Prodigy optimizer (lr=%.2e, d_coef=%.2f)",
        lr, opt_config.prodigy_d_coef,
    )
    return Prodigy(
        params,
        lr=lr,
        weight_decay=wd,
        d_coef=opt_config.prodigy_d_coef,
        growth_rate=opt_config.prodigy_growth_rate,
    )
