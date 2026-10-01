"""Proposed Method: Gated-Dual CafeBERT with Attentive Bottleneck & MSD.

Inherits directly from FlatCafeBERT (pool_mode='gated_dual', use_msd=True).
Guarantees 100% bit-for-bit mathematical and random state identity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from gated_dual_ema_msd.models.flat_cafebert import FlatCafeBERT


class GatedDualCafeBERT(FlatCafeBERT):
    """Grand Master Architecture (Bit-for-bit exact 100% clone of FlatCafeBERT):

    1. CafeBERT-Large Backbone (1024-d)
    2. Premise & Hypothesis Attentive Attention Scorers -> u, v
    3. Relational Bottleneck Head: 4096 -> 128 -> 1024 (GELU + Dropout)
    4. Learned Residual Gate: Sigmoid(W_g [h_cls, r] + b_g) with b_g = -1.0
    5. Multi-Sample Dropout (MSD 5-path [0.1, 0.2, 0.3, 0.4, 0.5])
    """
    def __init__(
        self,
        model_name: str = "uitnlp/CafeBERT",
        revision: str = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d",
        num_labels: int = 3,
        dropout: float = 0.1,
        label_smoothing: float = 0.02,
        relation_hidden: int = 128,
        gate_bias: float = -1.0,
        use_multi_sample_dropout: bool = True,
        msd_dropouts: Optional[List[float]] = None,
        class_weights: Optional[List[float]] = None,
        sep_token_id: int = 2,
    ):
        super().__init__(
            model_name=model_name,
            fallback_models=[],
            dropout=dropout,
            num_labels=num_labels,
            label_smoothing=label_smoothing,
            revision=revision,
            class_weights=class_weights,
            pool_mode="gated_dual",
            segment_pooling="attentive",
            relation_hidden=relation_hidden,
            gate_bias=gate_bias,
            head_architecture="standard",
            relation_features_mode="standard",
            use_multi_sample_dropout=use_multi_sample_dropout,
            msd_dropouts=msd_dropouts or [0.1, 0.2, 0.3, 0.4, 0.5],
            classifier_type="linear",
            classifier_hidden=256,
            sep_token_id=sep_token_id,
        )

    @property
    def model_type(self) -> str:
        return "gated_dual_msd"
