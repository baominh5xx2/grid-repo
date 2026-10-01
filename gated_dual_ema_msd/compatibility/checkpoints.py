"""Bijective checkpoint key renaming and strict versioned loading."""
from __future__ import annotations

from typing import Mapping, Optional
import torch
from torch import Tensor


def rename_state_keys(
    state: Mapping[str, Tensor],
    key_map: Mapping[str, str],
) -> dict[str, Tensor]:
    """Rename state dict keys with collision detection."""
    output: dict[str, Tensor] = {}
    for key, value in state.items():
        target = key_map.get(key, key)
        if target in output:
            raise ValueError(f"checkpoint key collision: {target}")
        output[target] = value
    return output


def load_compatible_state(
    model: torch.nn.Module,
    state: Mapping[str, Tensor],
    *,
    key_map: Optional[Mapping[str, str]] = None,
) -> None:
    """Load state dict strictly into model after applying key mapping."""
    mapped = rename_state_keys(state, key_map or {})
    missing_keys, unexpected_keys = model.load_state_dict(mapped, strict=False)
    if missing_keys:
        raise RuntimeError(f"Missing key(s) in state_dict: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected key(s) in state_dict: {unexpected_keys}")


def checkpoint_schema(profile: str) -> str:
    """Return the versioned checkpoint schema identifier for a profile."""
    schemas = {
        "root-flat": "root-flat-v1",
        "root-oop": "root-oop-v1",
        "matrix": "matrix-v1",
        "standalone": "standalone-v1",
    }
    if profile not in schemas:
        raise ValueError(
            f"Unknown profile '{profile}'. Choose from {list(schemas.keys())}"
        )
    return schemas[profile]
