"""gated_dual_ema_msd canonical package."""
__version__ = "0.1.0"

from gated_dual_ema_msd.models.base import BaseNLIModel
from gated_dual_ema_msd.models.factory import (
    MODEL_REGISTRY,
    create_nli_model,
    register_model,
)
from gated_dual_ema_msd.models.gated_dual import GatedDualCafeBERT
from gated_dual_ema_msd.models.baselines import (
    VanillaCafeBERT,
    MeanPoolCafeBERT,
)
from gated_dual_ema_msd.training.ema import ModelEMA
from gated_dual_ema_msd.training.direct import DirectTrainer

__all__ = [
    "BaseNLIModel",
    "GatedDualCafeBERT",
    "VanillaCafeBERT",
    "MeanPoolCafeBERT",
    "create_nli_model",
    "register_model",
    "MODEL_REGISTRY",
    "ModelEMA",
    "DirectTrainer",
]
