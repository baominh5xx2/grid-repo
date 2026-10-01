"""Unified optimizer update with gradient unscaling, clipping, and overflow detection."""
from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn as nn


def optimizer_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    *,
    max_grad_norm: float = 1.0,
) -> bool:
    """Executes gradient unscaling, norm clipping, optimizer step, and optional scheduler step.

    Returns True if optimizer successfully stepped (no scaler overflow), False if skipped.
    """
    if scaler is not None and hasattr(scaler, "unscale_") and hasattr(scaler, "step"):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        stepped = scale_before <= scale_after
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        stepped = True

    optimizer.zero_grad(set_to_none=True)
    if stepped and scheduler is not None:
        scheduler.step()
    return stepped
