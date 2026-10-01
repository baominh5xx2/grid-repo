"""Shared forward-pass/eval loop — used by both the two-stage trainer (dev-set model
selection, with loss) and scripts/evaluate_multisource.py (final test-set inference,
logits only). One implementation, so the numbers in both places are guaranteed
consistent (same batching/padding/pooling path)."""
from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from gated_dual_ema_msd.data.dataset import LABEL2ID, LABELS, collate_rows, eval_batches
from gated_dual_ema_msd.evaluation.metrics import compute_metrics


@torch.no_grad()
def run_inference(model, rows: Sequence[dict], tokenizer, max_length: int, device,
                   batch_size: int, compute_loss: bool = False):
    """Returns (metrics: dict, avg_loss: float|None, df: pd.DataFrame).
    df columns: sample_id, domain, gold_label, logit_E, logit_C, logit_N, pred_label.
    metrics is compute_metrics() over (gold, pred) — empty/zeroed if rows is empty."""
    model.eval()
    all_logits, all_gold, all_ids, all_domain = [], [], [], []
    total_loss, n_batches = 0.0, 0

    for batch_rows in eval_batches(rows, batch_size):
        batch = collate_rows(batch_rows, tokenizer, max_length)
        tens = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        labels = batch["labels"].to(device) if compute_loss else None
        # Evaluation stays in float32 on every device. Autocast can flip a
        # borderline argmax (observed as one ViNLI sample on L4), which breaks
        # immutable source-semantic verification and cross-device reproducibility.
        with torch.autocast(device_type=device.type, enabled=False):
            out = model(**tens, labels=labels)
        if compute_loss and out["loss"] is not None:
            total_loss += out["loss"].item()
            n_batches += 1
        all_logits.append(out["logits"].float().cpu().numpy())
        all_gold.extend(batch["gold_label"])
        all_ids.extend(batch["sample_id"])
        all_domain.extend(batch["domain"])

    if not all_logits:
        return {"accuracy": 0.0, "macro_f1": 0.0, "f1_E": 0.0, "f1_C": 0.0, "f1_N": 0.0}, None, pd.DataFrame(
            columns=["sample_id", "domain", "gold_label", "logit_E", "logit_C", "logit_N", "pred_label"])

    all_logits = np.concatenate(all_logits, axis=0)
    pred = all_logits.argmax(axis=1)
    y_true = [LABEL2ID[g] for g in all_gold]
    metrics = compute_metrics(pred, y_true)
    avg_loss = total_loss / n_batches if n_batches else None

    df = pd.DataFrame({
        "sample_id": all_ids, "domain": all_domain, "gold_label": all_gold,
        "logit_E": all_logits[:, 0], "logit_C": all_logits[:, 1], "logit_N": all_logits[:, 2],
        "pred_label": [LABELS[p] for p in pred],
    })
    return metrics, avg_loss, df
