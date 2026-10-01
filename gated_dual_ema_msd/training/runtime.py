"""Runtime initialization and seed management."""
from __future__ import annotations

import random
from typing import Callable, TypeVar

import numpy as np
import torch
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)


def set_seed(seed: int) -> None:
    """Set global RNG seeds across random, numpy, and torch with deterministic flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_model(seed: int, factory: Callable[[], M]) -> M:
    """Initialize a model strictly after setting the target seed."""
    set_seed(seed)
    return factory()
