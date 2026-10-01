"""Abstract Base Class for all NLI Models in the framework."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

LABELS = ["E", "C", "N"]
LABEL2ID = {"E": 0, "C": 1, "N": 2}
ID2LABEL = {0: "E", 1: "C", 2: "N"}


class BaseNLIModel(nn.Module, ABC):
    """Abstract Base Class ensuring a uniform interface for all Proposed & Baseline NLI models.

    Any baseline or novel method MUST implement forward() returning:
      - 'logits': [BatchSize, NumLabels]
      - 'loss': Scalar Loss Tensor (when labels are provided, else None)
      - 'hidden': [BatchSize, HiddenDim] representation (optional)
    """
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Human-readable identifier of the architecture (e.g. 'gated_dual_msd', 'vanilla_cls')."""
        raise NotImplementedError
