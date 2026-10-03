"""Aggregate detached architecture diagnostics without additional forward passes.

Models expose ``last_architecture_diagnostics`` as a mapping of names to
``Tensor[batch]``. Entropies have already been masked and reduced within each
sample by the model. The evaluator keeps per-example values in prediction CSVs
and computes population statistics over samples. ``last_gate[batch, channel]``
also supplies per-channel profiles, whose sample axis must not be confused with
the within-sample channel standard deviation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch


DEFINITIONS = {
    "gate_channel_std": "Population std over gate channels within each sample; summary mean/std over samples.",
    "gate_fraction_below_005": "Fraction of gate channels below 0.05 within each sample; summary over samples.",
    "gate_fraction_above_095": "Fraction of gate channels above 0.95 within each sample; summary over samples.",
    "relation_to_cls_norm_ratio": "L2 norm(g * LN(relation)) / max(L2 norm(CLS), 1e-8) within each sample; summary over samples.",
    "premise_attention_entropy": "Premise segment attention entropy normalized by log(valid premise token count); empty/singleton segments are zero; summary over samples.",
    "hypothesis_attention_entropy": "Hypothesis segment attention entropy normalized by log(valid hypothesis token count); empty/singleton segments are zero; summary over samples.",
    "premise_alignment_entropy": "Premise-to-hypothesis entropy normalized by log(valid hypothesis target count), then mean over valid query premise tokens; empty/singleton targets are zero; summary over samples.",
    "hypothesis_alignment_entropy": "Hypothesis-to-premise entropy normalized by log(valid premise target count), then mean over valid query hypothesis tokens; empty/singleton targets are zero; summary over samples.",
}


class ArchitectureDiagnostics:
    """Streaming batch collector with equal weight per sample, not per batch."""

    def __init__(self):
        self.batches = []
        self.gate_sum: Optional[np.ndarray] = None
        self.gate_square_sum: Optional[np.ndarray] = None
        self.gate_samples = 0

    def add_batch(self, model: Any, size: int) -> None:
        diagnostics = getattr(model, "last_architecture_diagnostics", None) or {}
        scalars = {}
        for name, tensor in diagnostics.items():
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (size,):
                raise ValueError(f"Architecture diagnostic {name!r} must be a Tensor[{size}]")
            scalars[f"diag_{name}"] = tensor.detach().float().cpu().numpy()
        self.batches.append((size, scalars))
        gate = getattr(model, "last_gate", None)
        if isinstance(gate, torch.Tensor) and gate.ndim == 2 and gate.shape[0] == size:
            values = gate.detach().float().cpu().numpy().astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Architecture gate diagnostics contain non-finite values")
            if self.gate_sum is None:
                self.gate_sum = np.zeros(values.shape[1], dtype=np.float64)
                self.gate_square_sum = np.zeros(values.shape[1], dtype=np.float64)
            if self.gate_sum.shape != (values.shape[1],):
                raise ValueError("Architecture gate channel count changed during evaluation")
            self.gate_sum += values.sum(axis=0)
            self.gate_square_sum += np.square(values).sum(axis=0)
            self.gate_samples += size

    def attach(self, frame: pd.DataFrame) -> None:
        names = sorted({name for _, values in self.batches for name in values})
        for name in names:
            frame[name] = np.concatenate([
                values.get(name, np.full(size, np.nan)) for size, values in self.batches
            ])
        if self.gate_samples:
            mean = self.gate_sum / self.gate_samples
            variance = np.maximum(self.gate_square_sum / self.gate_samples - np.square(mean), 0.0)
            frame.attrs["gate_channels"] = {
                "sample_count": self.gate_samples,
                "channel_count": len(mean),
                "mean_across_samples": mean.tolist(),
                "std_across_samples": np.sqrt(variance).tolist(),
                "reduction_axes": "sample axis for each channel; population std (ddof=0)",
            }


def architecture_summary(frame: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Summarize each diagnostic over samples and retain channel profiles."""
    statistics = {}
    definitions = {}
    for column in sorted(frame.columns):
        if not column.startswith("diag_"):
            continue
        name = column[len("diag_"):]
        values = frame[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        statistics[name] = {
            "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std(ddof=0)) if len(finite) else None,
            "count": len(finite),
            "missing_count": int(len(values) - len(finite)),
            "reduction_axes": "sample axis after the model's within-sample reduction; population std (ddof=0)",
        }
        definitions[name] = DEFINITIONS.get(name, "Per-example model diagnostic; summary over samples.")
    gate_channels = frame.attrs.get("gate_channels")
    if not statistics and gate_channels is None:
        return None
    return {
        "schema_version": 1,
        "statistics": statistics,
        "definitions": definitions,
        "gate_channels": gate_channels,
    }
