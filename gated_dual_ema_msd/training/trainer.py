"""Object-Oriented BaseTrainer with Template Method Pattern and Callback Architecture."""
from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gated_dual_ema_msd.data.datamodule import NLIDataModule
from gated_dual_ema_msd.evaluation.evaluator import NLIEvaluator
from gated_dual_ema_msd.training.callbacks import BaseCallback
from gated_dual_ema_msd.training.ema_context import ema_weights


class BaseTrainer:
    """Template Method BaseTrainer for PyTorch NLI models.

    Decouples training loop execution from cross-cutting concerns (W&B, Checkpoints, EMA)
    via an extensible Callback architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_module: NLIDataModule,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        device: Optional[torch.device] = None,
        fp16: bool = True,
        grad_accum_steps: int = 4,
        max_epochs: int = 5,
        eval_every_steps: Optional[int] = None,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_module = data_module
        self.callbacks = callbacks or []
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.fp16 = fp16 and torch.cuda.is_available()
        self.grad_accum_steps = grad_accum_steps
        self.max_epochs = max_epochs
        self.eval_every_steps = eval_every_steps
        self.max_grad_norm = max_grad_norm

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.evaluator = NLIEvaluator(model=self.model, device=self.device)

        # State attributes
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        self.ema: Optional[Any] = None

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> float:
        """Executes a single forward + backward step with AMP and Gradient Accumulation."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(self.device)

        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=self.fp16)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
            else torch.cuda.amp.autocast(enabled=self.fp16)
        )

        with autocast_ctx:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = output["loss"] if isinstance(output, dict) else output.loss
            scaled_loss = loss / self.grad_accum_steps

        self.scaler.scale(scaled_loss).backward()
        return loss.item()

    def optimizer_step(self) -> None:
        """Unscales gradients, clips grad norm, updates weights and scheduler."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1
        for cb in self.callbacks:
            cb.on_optimizer_step(self)

    def validate(self) -> Dict[str, Any]:
        """Runs validation using either standard or EMA shadow weights."""
        dataloader = self.data_module.val_dataloader()
        with ema_weights(self.model, self.ema):
            metrics = self.evaluator.evaluate(dataloader)
        return metrics

    def fit(self) -> Dict[str, Any]:
        """Main training lifecycle execution."""
        self.model.to(self.device)
        for cb in self.callbacks:
            cb.on_train_begin(self)

        train_loader = self.data_module.train_dataloader()
        last_val_metrics: Dict[str, Any] = {}

        for epoch in range(1, self.max_epochs + 1):
            if self.should_stop:
                break
            self.current_epoch = epoch
            self.model.train()
            self.optimizer.zero_grad()

            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader, start=1):
                for cb in self.callbacks:
                    cb.on_batch_begin(self, batch_idx)

                loss_val = self.training_step(batch, batch_idx)
                epoch_loss += loss_val
                num_batches += 1

                for cb in self.callbacks:
                    cb.on_batch_end(self, batch_idx, loss_val)

                # Optimizer step on accumulated batches
                if batch_idx % self.grad_accum_steps == 0 or batch_idx == len(train_loader):
                    self.optimizer_step()

                    # Periodic step evaluation
                    if self.eval_every_steps and self.global_step % self.eval_every_steps == 0:
                        last_val_metrics = self.validate()
                        for cb in self.callbacks:
                            cb.on_validation_end(self, last_val_metrics)
                        self.model.train()
                        if self.should_stop:
                            break

            # End of epoch evaluation if not evaluated per step
            if not self.eval_every_steps:
                last_val_metrics = self.validate()
                for cb in self.callbacks:
                    cb.on_validation_end(self, last_val_metrics)

            epoch_metrics = {
                "train_loss": epoch_loss / max(1, num_batches),
                "val_macro_f1": last_val_metrics.get("macro_f1", 0.0),
                "val_accuracy": last_val_metrics.get("accuracy", 0.0),
            }
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, epoch_metrics)

        for cb in self.callbacks:
            cb.on_train_end(self)

        return last_val_metrics
