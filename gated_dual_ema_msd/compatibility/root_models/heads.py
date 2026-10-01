"""Modular Classification Heads (Standard, MSD, MLP)."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from gated_dual_ema_msd.compatibility.root_models.loss import NLILossCriteria


class BaseClassifierHead(nn.Module):
    """Base class for classification heads."""
    def __init__(self, label_smoothing: float = 0.0, focal_gamma: float = 0.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss_criteria = NLILossCriteria(
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_criteria(logits, labels)


class StandardClassifierHead(BaseClassifierHead):
    """Standard Linear Head: Dropout -> Linear(Hidden -> NumLabels)."""
    def __init__(
        self,
        in_features: int,
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__(label_smoothing=label_smoothing, focal_gamma=focal_gamma, class_weights=class_weights)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_labels)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.classifier(self.dropout(x))
        loss = self.compute_loss(logits, labels) if labels is not None else None
        return logits, loss


class MultiSampleDropoutHead(BaseClassifierHead):
    """Multi-Sample Dropout Head (MSD): K parallel dropouts -> Mean Logits & Loss."""
    def __init__(
        self,
        in_features: int,
        num_labels: int = 3,
        dropout_rates: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__(label_smoothing=label_smoothing, focal_gamma=focal_gamma, class_weights=class_weights)
        rates = dropout_rates or [0.1, 0.2, 0.3, 0.4, 0.5]
        self.msd_layers = nn.ModuleList([nn.Dropout(p) for p in rates])
        self.classifier = nn.Linear(in_features, num_labels)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.training:
            msd_logits = [self.classifier(drop(x)) for drop in self.msd_layers]
            logits = torch.stack(msd_logits, dim=0).mean(dim=0)
            loss = None
            if labels is not None:
                losses = [self.compute_loss(l_i, labels) for l_i in msd_logits]
                loss = torch.stack(losses).mean()
            return logits, loss
        else:
            logits = self.classifier(x)
            loss = self.compute_loss(logits, labels) if labels is not None else None
            return logits, loss
