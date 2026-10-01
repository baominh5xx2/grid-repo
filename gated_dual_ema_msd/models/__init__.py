"""Model package exports."""
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
from gated_dual_ema_msd.models.factory import (
    MODEL_REGISTRY,
    create_nli_model,
    register_model,
)
from gated_dual_ema_msd.models.gated_dual import GatedDualCafeBERT
from gated_dual_ema_msd.models.heads import (
    BaseClassifierHead,
    MultiSampleDropoutHead,
    StandardClassifierHead,
)
from gated_dual_ema_msd.models.poolers import (
    AttentiveSegmentPooler,
    BasePooler,
    ClsPooler,
    MeanPooler,
)

__all__ = [
    "BaseNLIModel",
    "GatedDualCafeBERT",
    "VanillaCafeBERT",
    "MeanPoolCafeBERT",
    "ParameterMatchedCLSCafeBERT",
    "SimpleRelationCafeBERT",
    "VanillaMSDCafeBERT",
    "GatedDualMeanPoolCafeBERT",
    "GatedDualNoInteractionCafeBERT",
    "GatedDualNoGateCafeBERT",
    "GatedDualNoBnCafeBERT",
    "MODEL_REGISTRY",
    "create_nli_model",
    "register_model",
    "BasePooler",
    "ClsPooler",
    "MeanPooler",
    "AttentiveSegmentPooler",
    "BaseClassifierHead",
    "StandardClassifierHead",
    "MultiSampleDropoutHead",
]
