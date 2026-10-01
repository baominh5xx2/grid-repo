"""Checkpoint saving, loading, and metadata management."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Optional
import torch

from gated_dual_ema_msd.compatibility.checkpoints import checkpoint_schema


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint_dir: pathlib.Path,
    cfg: dict,
    stage_name: str,
    metric: float,
    optimizer_step: int,
    profile: Optional[str] = None,
) -> None:
    """Save model weights, tokenizer, config, and schema metadata to checkpoint_dir."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")

    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(checkpoint_dir)

    if hasattr(model, "backbone") and hasattr(model.backbone, "config") and hasattr(model.backbone.config, "save_pretrained"):
        model.backbone.config.save_pretrained(checkpoint_dir)

    schema = checkpoint_schema(profile) if profile is not None else "root-flat-v1"
    model_cfg = dict(cfg.get("model", {}))
    identity = {
        "stage": stage_name,
        "base_model": model_cfg.get("name", "uitnlp/CafeBERT"),
        "base_revision": model_cfg.get("revision", "af76fcf2a04096b2b54b348a3e4eb48253c93c5d"),
        "label_order": ["E", "C", "N"],
        "selection_metric": "macro_f1",
        "best_dev_macro_f1": float(metric),
        "optimizer_step": int(optimizer_step),
        "model_config": model_cfg,
        "checkpoint_schema": schema,
    }

    metadata_bytes = json.dumps(identity, indent=2).encode("utf-8")
    (checkpoint_dir / "checkpoint_metadata.json").write_bytes(metadata_bytes)
    (checkpoint_dir / "metadata.json").write_bytes(metadata_bytes)


def load_best(
    model: torch.nn.Module,
    checkpoint_dir: pathlib.Path,
    device: Any,
) -> None:
    """Load the best checkpoint weights into model strictly."""
    weights_path = checkpoint_dir / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
