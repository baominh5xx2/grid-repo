"""Model Factory & Registry for NLI Models."""
from __future__ import annotations

from typing import Any, Callable, Dict, Type

from gated_dual_ema_msd.compatibility.root_models.ablations import (
    GatedDualMeanPoolCafeBERT,
    GatedDualNoBnCafeBERT,
    GatedDualNoGateCafeBERT,
    GatedDualNoInteractionCafeBERT,
)
from gated_dual_ema_msd.compatibility.root_models.base import BaseNLIModel
from gated_dual_ema_msd.compatibility.root_models.baselines import (
    MeanPoolCafeBERT,
    ParameterMatchedCLSCafeBERT,
    SimpleRelationCafeBERT,
    VanillaCafeBERT,
    VanillaMSDCafeBERT,
)
from gated_dual_ema_msd.compatibility.root_models.gated_dual import GatedDualCafeBERT

MODEL_REGISTRY: Dict[str, Type[BaseNLIModel]] = {
    # Proposed Architecture
    "gated_dual_msd": GatedDualCafeBERT,
    "gated_dual": GatedDualCafeBERT,
    # Baselines
    "vanilla_cls": VanillaCafeBERT,
    "mean_pool": MeanPoolCafeBERT,
    "parameter_matched_cls": ParameterMatchedCLSCafeBERT,
    "simple_relation": SimpleRelationCafeBERT,
    "vanilla_msd": VanillaMSDCafeBERT,
    # Ablations
    "ablation_mean_pool": GatedDualMeanPoolCafeBERT,
    "ablation_no_interaction": GatedDualNoInteractionCafeBERT,
    "ablation_no_gate": GatedDualNoGateCafeBERT,
    "ablation_no_bottleneck": GatedDualNoBnCafeBERT,
}


def register_model(name: str) -> Callable[[Type[BaseNLIModel]], Type[BaseNLIModel]]:
    """Decorator to register a new NLI model architecture in the factory."""
    def decorator(cls: Type[BaseNLIModel]) -> Type[BaseNLIModel]:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def create_nli_model(model_type: str, **kwargs: Any) -> BaseNLIModel:
    """Factory method to instantiate an NLI model by its registered type name."""
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available types: {available}"
        )
    model_cls = MODEL_REGISTRY[model_type]
    return model_cls(**kwargs)
