"""Exponential Moving Average (EMA) of model parameters for smooth optimization."""
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn


class ModelEMA:
    """Moving average of trainable parameters, updated after optimizer steps."""

    def __init__(self, model: nn.Module, decay: float = 0.992):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.num_updates = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @property
    def ready(self) -> bool:
        return self.num_updates > 0

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with exponential moving average."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )
        self.num_updates += 1

    def apply_shadow(self, model: nn.Module) -> bool:
        """Swap model weights with EMA shadow weights for evaluation."""
        if not self.ready:
            return False
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])
        return True

    def restore(self, model: nn.Module) -> None:
        """Restore original parameters after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, Any]:
        """Returns EMA state dictionary for checkpointing."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads EMA state from state dictionary."""
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict["num_updates"])
        self.shadow = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in state_dict["shadow"].items()
        }
