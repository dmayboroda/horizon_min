"""Optimizer registry — create optimizer by name from config.

Supports: adamw, adamw_8bit, muon, soap, shampoo, prodigy.

Install commands:
    pip install bitsandbytes          # adamw_8bit
    pip install muon                  # muon (MuonWithAuxAdam)
    pip install distributed_shampoo   # shampoo
    pip install prodigyopt            # prodigy
    # SOAP is bundled as soap_impl.py (no install needed)
"""

from __future__ import annotations

import logging
from typing import Iterator

import torch
import torch.nn as nn

from ..config import OptimizerConfig, TrainingConfig

logger = logging.getLogger(__name__)


def get_optimizer(
    name: str,
    params: Iterator[nn.Parameter] | list[dict],
    lr: float = 1e-5,
    weight_decay: float = 0.01,
    **kwargs,
) -> torch.optim.Optimizer:
    """Simple API: create an optimizer by name with explicit lr.

    Parameters
    ----------
    name
        Optimizer name (adamw, adamw_8bit, muon, soap, shampoo, prodigy).
    params
        Model parameters (or param groups).
    lr
        Learning rate.
    weight_decay
        Weight decay.

    Returns
    -------
    torch.optim.Optimizer
    """
    opt_config = OptimizerConfig(name=name)
    train_config = TrainingConfig(learning_rate=lr, weight_decay=weight_decay)
    return create_optimizer(params, opt_config, train_config)


def create_optimizer(
    params: Iterator[nn.Parameter] | list[dict],
    opt_config: OptimizerConfig,
    train_config: TrainingConfig,
    *,
    named_params: list[tuple[str, nn.Parameter]] | None = None,
) -> torch.optim.Optimizer:
    """Create an optimizer by name.

    Parameters
    ----------
    params
        Model parameters (or param groups).  Used by all optimizers
        except muon (which needs named_params for ndim split).
    opt_config
        Optimizer-specific configuration.
    train_config
        Training config (for lr, weight_decay).
    named_params
        Optional list of (name, param) tuples.  Required for Muon
        to split params by ndim (>=2D -> Muon, 1D -> AdamW).

    Returns
    -------
    torch.optim.Optimizer
    """
    name = opt_config.name.lower()
    # Prefer optimizer-specific LR, fall back to training.learning_rate
    lr = opt_config.lr if opt_config.lr is not None else train_config.learning_rate
    wd = train_config.weight_decay

    if name == "adamw":
        return _create_adamw(params, lr, wd, opt_config)
    elif name == "adamw_8bit":
        return _create_adamw_8bit(params, lr, wd, opt_config)
    elif name == "muon":
        return _create_muon(params, lr, wd, opt_config, named_params=named_params)
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
#  Muon (MuonWithAuxAdam — splits params by ndim)
# ---------------------------------------------------------------------------
def _create_muon(params, lr, wd, opt_config, *, named_params=None):
    """Create Muon optimizer with auxiliary AdamW for 1D params.

    Muon only works on >=2D weight matrices (uses Newton-Schulz
    orthogonalisation).  1D params (biases, norms, embeddings) must
    be handled by a separate AdamW.  ``MuonWithAuxAdam`` does this
    split internally.

    If ``named_params`` is provided, we split by ``param.ndim >= 2``.
    Otherwise, we fall back to splitting from the flat param list.
    """
    try:
        from muon import MuonWithAuxAdam
    except ImportError:
        raise ImportError(
            "muon required. Install with: pip install muon\n"
            "Requires MuonWithAuxAdam for proper param splitting."
        )

    # Split params by ndim
    muon_params = []
    adam_params = []

    if named_params is not None:
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                adam_params.append(param)
    else:
        # Fallback: split from flat param list
        param_list = list(params) if not isinstance(params, list) else params
        for p in param_list:
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adam_params.append(p)

    logger.info(
        "Creating MuonWithAuxAdam optimizer "
        "(muon_lr=%.2e, %d muon params [>=2D], %d adam params [1D], "
        "momentum=%.2f, nesterov=%s)",
        lr, len(muon_params), len(adam_params),
        opt_config.muon_momentum, opt_config.muon_nesterov,
    )

    return MuonWithAuxAdam(
        muon_params=muon_params,
        lr=lr,
        momentum=opt_config.muon_momentum,
        nesterov=opt_config.muon_nesterov,
        ns_steps=5,
        adamw_params=adam_params,
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.95),
        adamw_wd=wd,
    )


# ---------------------------------------------------------------------------
#  SOAP (bundled from nikhilvyas/SOAP)
# ---------------------------------------------------------------------------
def _create_soap(params, lr, wd, opt_config):
    try:
        from .soap_impl import SOAP
    except ImportError:
        raise ImportError(
            "SOAP implementation not found. "
            "Expected soap_impl.py in training/src/optimizers/. "
            "Download from: https://github.com/nikhilvyas/SOAP"
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
#  Shampoo (distributed_shampoo — single-GPU mode)
# ---------------------------------------------------------------------------
def _create_shampoo(params, lr, wd, opt_config):
    try:
        from distributed_shampoo.distributed_shampoo import DistributedShampoo
        from distributed_shampoo.shampoo_types import AdamGraftingConfig
    except ImportError:
        raise ImportError(
            "distributed_shampoo required. "
            "Install with: pip install distributed-shampoo"
        )

    logger.info(
        "Creating Shampoo optimizer (lr=%.2e, update_freq=%d, single-GPU mode)",
        lr, opt_config.shampoo_update_freq,
    )

    # Single-GPU config: no distributed_config, use AdamGraftingConfig
    return DistributedShampoo(
        params,
        lr=lr,
        betas=opt_config.betas,
        epsilon=opt_config.eps,
        weight_decay=wd,
        precondition_frequency=opt_config.shampoo_update_freq,
        grafting_config=AdamGraftingConfig(
            beta2=opt_config.betas[1],
            epsilon=opt_config.eps,
        ),
    )


# ---------------------------------------------------------------------------
#  Prodigy (learning-rate-free)
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
