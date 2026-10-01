"""Object-Oriented Evaluator for NLI Models enforcing FP32 Precision."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gated_dual_ema_msd.evaluation.metrics import compute_confusion_matrix, compute_metrics


class BaseEvaluator:
    """Abstract Evaluator Interface."""
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        raise NotImplementedError


class NLIEvaluator(BaseEvaluator):
    """Evaluates NLI models strictly in FP32 precision to prevent argmax flipping."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        self.model.to(self.device)

        all_preds = []
        all_labels = []
        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(self.device)

            # Strict FP32 evaluation (autocast disabled)
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            logits = output["logits"] if isinstance(output, dict) else output.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

            if labels is not None:
                all_labels.extend(labels.cpu().numpy())
                loss = output["loss"] if isinstance(output, dict) else output.loss
                if loss is not None:
                    total_loss += loss.item()
                    total_batches += 1

        results = compute_metrics(all_preds, all_labels)
        if total_batches > 0:
            results["loss"] = total_loss / total_batches

        results["predictions"] = all_preds
        results["references"] = all_labels
        results["confusion_matrix"] = compute_confusion_matrix(all_preds, all_labels).tolist()

        return results
