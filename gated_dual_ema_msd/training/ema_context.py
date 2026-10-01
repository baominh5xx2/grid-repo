"""Context manager for temporary EMA shadow weights application and restoration."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional
import torch.nn as nn


@contextmanager
def ema_weights(model: nn.Module, ema: Optional[Any]) -> Iterator[bool]:
    """Temporarily applies EMA shadow weights to model and guarantees restoration in finally."""
    applied = bool(ema is not None and ema.apply_shadow(model))
    try:
        yield applied
    finally:
        if applied:
            ema.restore(model)
