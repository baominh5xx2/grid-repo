"""Configuration Validation Engine using Shared Contracts."""
from __future__ import annotations

from typing import Any, Mapping
from gated_dual_ema_msd.config.contracts import require_max_length


def validate_run_config(cfg: Mapping[str, Any], *, protocol: str = "r2") -> None:
    """Validate runtime configuration dictionary against Rule 0b and protocol rules."""
    # Check search splits lock
    search_splits = cfg.get("search_splits")
    if search_splits is not None:
        if "test" in [s.lower() for s in search_splits]:
            raise ValueError(
                "Violation: 'test' split cannot be included in search_splits! ViANLI test is LOCKED during search."
            )

    # Check max_length against dataset
    dataset = cfg.get("dataset")
    if dataset is not None and isinstance(dataset, str):
        requested_max_len = cfg.get("max_length")
        require_max_length(dataset, requested_max_len)

    # Multi-dataset check (e.g. data dict with multiple sources)
    datasets_cfg = cfg.get("datasets")
    if isinstance(datasets_cfg, dict):
        for dname, dinfo in datasets_cfg.items():
            req_len = dinfo.get("max_length") if isinstance(dinfo, dict) else None
            require_max_length(dname, req_len)
