"""Modular, Object-Oriented NLI Models Package."""
from gated_dual_ema_msd.compatibility.root_models.ablations import (
    GatedDualMeanPoolCafeBERT,
    GatedDualNoBnCafeBERT,
    GatedDualNoGateCafeBERT,
    GatedDualNoInteractionCafeBERT,
)
from gated_dual_ema_msd.compatibility.root_models.base import BaseNLIModel, ID2LABEL, LABEL2ID, LABELS
from gated_dual_ema_msd.compatibility.root_models.baselines import (
    MeanPoolCafeBERT,
    ParameterMatchedCLSCafeBERT,
    SimpleRelationCafeBERT,
    VanillaCafeBERT,
    VanillaMSDCafeBERT,
)
from gated_dual_ema_msd.compatibility.root_models.factory import (
    MODEL_REGISTRY,
    create_nli_model,
    register_model,
)
from gated_dual_ema_msd.compatibility.root_models.gated_dual import GatedDualCafeBERT
from gated_dual_ema_msd.compatibility.root_models.heads import (
    BaseClassifierHead,
    MultiSampleDropoutHead,
    StandardClassifierHead,
)
from gated_dual_ema_msd.compatibility.root_models.loss import FocalLoss, NLILossCriteria
from gated_dual_ema_msd.compatibility.root_models.output import NLIModelOutput
from gated_dual_ema_msd.compatibility.root_models.poolers import (
    AttentiveSegmentPooler,
    BasePooler,
    ClsPooler,
    MeanPooler,
)

__all__ = [
    "BaseNLIModel",
    "NLIModelOutput",
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
    "NLILossCriteria",
    "FocalLoss",
    "LABELS",
    "LABEL2ID",
    "ID2LABEL",
]
