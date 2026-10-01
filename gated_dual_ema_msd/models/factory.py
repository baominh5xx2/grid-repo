"""Model Factory & Registry for easy baseline plug-and-play."""
from __future__ import annotations

import inspect
from typing import Any, Dict, Type

from gated_dual_ema_msd.models.ablations import (
    GatedDualMeanPoolCafeBERT,
    GatedDualNoBnCafeBERT,
    GatedDualNoGateCafeBERT,
    GatedDualNoInteractionCafeBERT,
)
from gated_dual_ema_msd.models.base import BaseNLIModel
from gated_dual_ema_msd.models.baselines import (
    MeanPoolCafeBERT,
    ParameterMatchedCLSCafeBERT,
    SimpleRelationCafeBERT,
    VanillaCafeBERT,
    VanillaMSDCafeBERT,
)
from gated_dual_ema_msd.models.gated_dual import GatedDualCafeBERT

MODEL_REGISTRY: Dict[str, Type[BaseNLIModel]] = {
    # Proposed SOTA Method (P0)
    "gated_dual": GatedDualCafeBERT,
    "grand_master": GatedDualCafeBERT,
    # Baselines (B1, B2, B3, B4)
    "vanilla": VanillaCafeBERT,
    "vanilla_cls": VanillaCafeBERT,
    "mean_pool": MeanPoolCafeBERT,
    "parameter_matched_cls": ParameterMatchedCLSCafeBERT,
    "simple_relation": SimpleRelationCafeBERT,
    "vanilla_msd": VanillaMSDCafeBERT,
    "cls_msd_ema": VanillaMSDCafeBERT,
    # Ablations (A3, A4)
    "gated_dual_mean_pool": GatedDualMeanPoolCafeBERT,
    "gated_dual_no_interaction": GatedDualNoInteractionCafeBERT,
    "gated_dual_no_gate": GatedDualNoGateCafeBERT,
    "gated_dual_no_bottleneck": GatedDualNoBnCafeBERT,
}


def register_model(name: str, model_cls: Type[BaseNLIModel]) -> None:
    """Register a new baseline model cleanly without modifying existing codebase."""
    MODEL_REGISTRY[name.lower()] = model_cls


def create_nli_model(
    model_type: str = "gated_dual",
    *,
    source_profile: str = "matrix",
    **kwargs: Any,
) -> torch.nn.Module:
    """Factory method to instantiate any proposed or baseline NLI model by type and profile."""
    if source_profile == "root-oop":
        from gated_dual_ema_msd.compatibility.root_models.factory import (
            create_nli_model as root_create,
        )
        return root_create(model_type, **kwargs)
    elif source_profile == "root-flat":
        from gated_dual_ema_msd.models.flat_cafebert import FlatCafeBERT
        kwargs.setdefault("model_name", "uitnlp/CafeBERT")
        return FlatCafeBERT(**kwargs)

    elif source_profile == "standalone":
        from gated_dual_ema_msd.compatibility.standalone_model import (
            GatedDualCafeBERT as StandaloneModel,
        )
        return StandaloneModel(**kwargs)
    elif source_profile == "matrix":
        key = model_type.lower()
        if key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type '{model_type}'. Available registered models: {list(MODEL_REGISTRY.keys())}"
            )
        cls = MODEL_REGISTRY[key]
        sig = inspect.signature(cls.__init__)
        valid_params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in valid_params.values()):
            filtered = kwargs
        else:
            filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        return cls(**filtered)
    else:
        raise ValueError(
            f"Unknown source_profile: '{source_profile}'. Choose from ['matrix', 'root-oop', 'root-flat', 'standalone']"
        )
