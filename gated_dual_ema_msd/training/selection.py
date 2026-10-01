"""Model selection and early stopping tracking on development evaluations."""
from __future__ import annotations

import math
from typing import Any, Dict


class DevSelection:
    """Tracks best development metric, source weights (current vs EMA), and patience."""

    def __init__(self, patience: int = 5, mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        self.patience = int(patience)
        self.mode = mode
        self.best_metric: float = float("-inf") if mode == "max" else float("inf")
        self.best_step: int = 0
        self.best_epoch: int = 0
        self.weight_source: str = "current"
        self.no_improve: int = 0

    @property
    def should_stop(self) -> bool:
        return bool(self.patience > 0 and self.no_improve >= self.patience)

    def observe(
        self,
        metric: float,
        step: int,
        source: str = "current",
        epoch: int = 0,
    ) -> bool:
        """Records an evaluation metric at step and returns True if strictly improved."""
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError(f"Non-finite evaluation metric: {metric}")

        is_better = (
            metric > self.best_metric if self.mode == "max" else metric < self.best_metric
        )
        if is_better:
            self.best_metric = metric
            self.best_step = int(step)
            self.weight_source = str(source)
            self.best_epoch = int(epoch)
            self.no_improve = 0
            return True
        else:
            self.no_improve += 1
            return False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "mode": self.mode,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "best_epoch": self.best_epoch,
            "weight_source": self.weight_source,
            "no_improve": self.no_improve,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.patience = int(state["patience"])
        self.mode = str(state["mode"])
        self.best_metric = float(state["best_metric"])
        self.best_step = int(state["best_step"])
        self.best_epoch = int(state["best_epoch"])
        self.weight_source = str(state["weight_source"])
        self.no_improve = int(state["no_improve"])
