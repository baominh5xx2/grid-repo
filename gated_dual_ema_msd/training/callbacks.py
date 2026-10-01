"""Observer Pattern Callback Architecture for Training Lifecycles."""
from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from gated_dual_ema_msd.training.ema import ModelEMA


class BaseCallback:
    """Base callback interface with lifecycle hook methods."""

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_batch_begin(self, trainer: Any, batch_idx: int) -> None:
        pass

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        pass

    def on_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_validation_end(self, trainer: Any, val_metrics: Dict[str, float]) -> None:
        pass


class EarlyStoppingCallback(BaseCallback):
    """Monitors a metric (e.g. dev_macro_f1) and triggers early stopping."""

    def __init__(self, monitor: str = "macro_f1", patience: int = 3, mode: str = "max"):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_score: Optional[float] = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_validation_end(self, trainer: Any, val_metrics: Dict[str, float]) -> None:
        score = val_metrics.get(self.monitor)
        if score is None:
            return

        is_better = False
        if self.best_score is None:
            is_better = True
        elif self.mode == "max" and score > self.best_score:
            is_better = True
        elif self.mode == "min" and score < self.best_score:
            is_better = True

        if is_better:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.should_stop = True
                print(f"[EarlyStopping] Triggered at epoch {trainer.current_epoch} (patience={self.patience})")


class ModelCheckpointCallback(BaseCallback):
    """Saves best model weights locally and handles checkpointing."""

    def __init__(self, output_dir: pathlib.Path, monitor: str = "macro_f1", mode: str = "max"):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score: Optional[float] = None

    def on_validation_end(self, trainer: Any, val_metrics: Dict[str, float]) -> None:
        score = val_metrics.get(self.monitor)
        if score is None:
            return

        is_better = False
        if self.best_score is None:
            is_better = True
        elif self.mode == "max" and score > self.best_score:
            is_better = True
        elif self.mode == "min" and score < self.best_score:
            is_better = True

        if is_better:
            self.best_score = score
            save_path = self.output_dir / "best_model.pt"
            torch.save(trainer.model.state_dict(), save_path)
            print(f"[Checkpoint] Saved new best checkpoint ({self.monitor}={score:.4f}) to {save_path.name}")


class EMACallback(BaseCallback):
    """Maintains Exponential Moving Average of weights during training."""

    def __init__(self, decay: float = 0.992):
        self.decay = decay
        self.ema: Optional[ModelEMA] = None

    def on_train_begin(self, trainer: Any) -> None:
        self.ema = ModelEMA(trainer.model, decay=self.decay)
        trainer.ema = self.ema

    def on_optimizer_step(self, trainer: Any) -> None:
        if self.ema is not None:
            self.ema.update(trainer.model)
