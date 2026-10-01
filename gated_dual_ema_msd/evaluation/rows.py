"""Shared FP32 Evaluation across dataset rows."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gated_dual_ema_msd.config.contracts import (
    ID2LABEL,
    LABEL2ID,
    parse_label,
    require_max_length,
)
from gated_dual_ema_msd.evaluation.metrics import compute_metrics
from gated_dual_ema_msd.evaluation.predictions import validate_predictions


def evaluate_rows(
    model: torch.nn.Module,
    rows: Sequence[dict],
    tokenizer: Any,
    *,
    dataset: str,
    max_length: int,
    device: Any,
    batch_size: int = 32,
    compute_loss: bool = True,
    loss_reduction: str = "mean_sample",
) -> Tuple[Dict[str, float], Optional[float], pd.DataFrame]:
    """Strict FP32 evaluation over dataset rows, restoring training state on completion."""
    require_max_length(dataset, max_length)

    was_training = model.training
    dev_type = device.type if hasattr(device, "type") else "cpu"

    try:
        model.eval()
        all_logits: List[np.ndarray] = []
        all_sample_ids: List[str] = []
        all_gold_labels: List[str] = []
        all_gold_ids: List[int] = []
        batch_losses: List[float] = []
        sample_losses: List[float] = []

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none") if compute_loss else None

        with torch.no_grad(), torch.autocast(device_type=dev_type, enabled=False):
            for i in range(0, len(rows), batch_size):
                batch_rows = rows[i : i + batch_size]
                sample_ids = [str(r.get("id", f"sample_{i+j}")) for j, r in enumerate(batch_rows)]
                golds = [
                    parse_label(r["label"], sample_ids[j])
                    for j, r in enumerate(batch_rows)
                ]
                gold_names = [r.get("label", ID2LABEL[g]) if isinstance(r.get("label"), str) else ID2LABEL[g] for j, (r, g) in enumerate(zip(batch_rows, golds))]

                enc = tokenizer(
                    [str(r["premise"]) for r in batch_rows],
                    [str(r["hypothesis"]) for r in batch_rows],
                    truncation=True,
                    max_length=max_length,
                    padding="longest",
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in enc.items()}

                output = model(**inputs)
                if isinstance(output, dict):
                    logits = output["logits"].float()
                elif hasattr(output, "logits"):
                    logits = output.logits.float()
                else:
                    logits = output[0].float()

                if loss_fn is not None:
                    target_tensor = torch.tensor(golds, dtype=torch.long, device=device)
                    losses = loss_fn(logits, target_tensor)
                    sample_losses.extend(losses.cpu().tolist())
                    batch_losses.append(float(losses.mean().item()))

                all_logits.append(logits.cpu().numpy())
                all_sample_ids.extend(sample_ids)
                all_gold_labels.extend(gold_names)
                all_gold_ids.extend(golds)

        logits_matrix = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 3))
        pred_ids = logits_matrix.argmax(axis=1) if len(logits_matrix) > 0 else np.array([])
        pred_labels = [ID2LABEL[p] for p in pred_ids]

        probs = F.softmax(torch.from_numpy(logits_matrix), dim=-1).numpy() if len(logits_matrix) > 0 else np.zeros((0, 3))

        metrics = compute_metrics(pred_ids, all_gold_ids) if len(pred_ids) > 0 else {}

        mean_loss = None
        if compute_loss and sample_losses:
            if loss_reduction == "mean_batch":
                mean_loss = float(np.mean(batch_losses))
            else:
                mean_loss = float(np.mean(sample_losses))
            metrics["loss"] = mean_loss

        frame = pd.DataFrame(
            {
                "sample_id": all_sample_ids,
                "gold_label": all_gold_labels,
                "pred_label": pred_labels,
                "logit_E": logits_matrix[:, 0] if len(logits_matrix) > 0 else [],
                "logit_C": logits_matrix[:, 1] if len(logits_matrix) > 0 else [],
                "logit_N": logits_matrix[:, 2] if len(logits_matrix) > 0 else [],
                "prob_E": probs[:, 0] if len(probs) > 0 else [],
                "prob_C": probs[:, 1] if len(probs) > 0 else [],
                "prob_N": probs[:, 2] if len(probs) > 0 else [],
            }
        )

        validate_predictions(frame, all_sample_ids)
        return metrics, mean_loss, frame

    finally:
        model.train(was_training)
