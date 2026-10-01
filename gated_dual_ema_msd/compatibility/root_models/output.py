"""Typed dataclass output for NLI models with backward-compatible dict access."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import torch


@dataclass
class NLIModelOutput:
    """Standardized output container for NLI models.

    Provides typed attributes while maintaining full dictionary-style subscripting
    (e.g., out['logits'], out['loss']) for 100% backward compatibility.
    """
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    gate_weights: Optional[torch.Tensor] = None
    relation_vectors: Optional[torch.Tensor] = None

    def __getitem__(self, key: str) -> Any:
        if key == "logits":
            return self.logits
        elif key == "loss":
            return self.loss
        elif key in ("hidden", "hidden_states"):
            return self.hidden_states
        elif key in ("gate", "gate_weights"):
            return self.gate_weights
        elif key in ("relation", "relation_vectors"):
            return self.relation_vectors
        raise KeyError(f"Invalid key '{key}' for NLIModelOutput")

    def __contains__(self, key: str) -> bool:
        return key in ("logits", "loss", "hidden", "hidden_states", "gate", "gate_weights", "relation", "relation_vectors")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict:
        return {
            "logits": self.logits,
            "loss": self.loss,
            "hidden": self.hidden_states,
            "gate": self.gate_weights,
            "relation": self.relation_vectors,
        }
