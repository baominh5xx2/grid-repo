"""Test split access policy and final evaluation authorization."""
from __future__ import annotations

from typing import Any, Mapping


def allow_final_test(*, frozen_final: bool = False, no_test: bool = False) -> bool:
    """Determine whether evaluation on test split is authorized.

    Search/development evaluation on the test split is strictly locked.
    Only explicit frozen_final authorization without no_test override may open it.
    """
    return bool(frozen_final and not no_test)


def r2_final_enabled(cfg: Mapping[str, Any]) -> bool:
    """Check whether R2 experiment config authorizes final test evaluation.

    Requires the approved validated combination:
    - method_frozen: True
    - target_test_enabled: True
    - target_test_locked: False
    Missing evaluation keys default to False.
    """
    project_cfg = cfg.get("project", {})
    eval_cfg = cfg.get("evaluation", {})

    target_test_locked = project_cfg.get("target_test_locked", True)
    target_test_enabled = eval_cfg.get("target_test_enabled", False)
    method_frozen = eval_cfg.get("method_frozen", False)

    return bool(method_frozen and target_test_enabled and not target_test_locked)
