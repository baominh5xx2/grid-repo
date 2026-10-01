"""Modular Loss Functions & Criteria for NLI Training."""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for hard example weighting in adversarial/imbalanced datasets."""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class NLILossCriteria(nn.Module):
    """Unified loss orchestrator for NLI models supporting CE, Label Smoothing, and Focal Loss."""
    def __init__(
        self,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone().detach())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        if self.focal_gamma > 0.0:
            crit = FocalLoss(gamma=self.focal_gamma, weight=weights, label_smoothing=self.label_smoothing)
            return crit(logits, labels)
        else:
            return F.cross_entropy(
                logits,
                labels,
                weight=weights,
                label_smoothing=self.label_smoothing,
            )
